from __future__ import annotations

import itertools
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)
from torch.utils.data import DataLoader, Dataset

from .data_sources import sha256_file
from .features import FeatureConfig
from .model import OPEN_STRING_PITCHES, PITCH_COUNT, PITCH_MIN, STRING_COUNT
from .onset_position import (
    OnsetPositionConfig,
    OnsetPositionDataset,
    OnsetPositionNet,
    candidate_position_queries,
    candidate_position_query_weights,
)


@dataclass(frozen=True)
class CandidateSequenceConfig:
    sequence_length: int = 32
    sequence_stride: int = 16
    token_size: int = 64
    hidden_size: int = 64
    output_hidden_size: int = 64
    dropout: float = 0.1
    maximum_duration: float = 2.0
    maximum_group_interval: float = 2.0
    minimum_log_probability: float = -12.0

    @classmethod
    def from_dict(
        cls,
        value: dict[str, object],
    ) -> "CandidateSequenceConfig":
        return cls(
            sequence_length=int(value.get("sequence_length", 32)),
            sequence_stride=int(value.get("sequence_stride", 16)),
            token_size=int(value.get("token_size", 64)),
            hidden_size=int(value.get("hidden_size", 64)),
            output_hidden_size=int(
                value.get("output_hidden_size", 64)
            ),
            dropout=float(value.get("dropout", 0.1)),
            maximum_duration=float(value.get("maximum_duration", 2.0)),
            maximum_group_interval=float(
                value.get("maximum_group_interval", 2.0)
            ),
            minimum_log_probability=float(
                value.get("minimum_log_probability", -12.0)
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateSequenceGroup:
    annotation_index: int
    onset: float
    pitches: Tensor
    confidences: Tensor
    durations: Tensor
    base_string_logits: Tensor
    valid_strings: Tensor
    string_targets: Tensor
    target_weights: Tensor
    base_activity_logits: Tensor = field(
        default_factory=lambda: torch.empty(0)
    )
    activity_targets: Tensor = field(
        default_factory=lambda: torch.empty(0)
    )
    repeat_targets: Tensor = field(
        default_factory=lambda: torch.empty(0)
    )


def validate_sequence_config(config: CandidateSequenceConfig) -> None:
    if config.sequence_length <= 0:
        raise ValueError("sequence length must be positive")
    if not 0 < config.sequence_stride <= config.sequence_length:
        raise ValueError(
            "sequence stride must be in 1..sequence_length"
        )
    if min(
        config.token_size,
        config.hidden_size,
        config.output_hidden_size,
    ) <= 0:
        raise ValueError("sequence model dimensions must be positive")
    if not 0 <= config.dropout < 1:
        raise ValueError("sequence dropout must be in [0, 1)")
    if (
        not math.isfinite(config.maximum_duration)
        or config.maximum_duration <= 0
        or not math.isfinite(config.maximum_group_interval)
        or config.maximum_group_interval <= 0
    ):
        raise ValueError(
            "sequence duration and interval limits must be positive"
        )
    if (
        not math.isfinite(config.minimum_log_probability)
        or config.minimum_log_probability >= 0
    ):
        raise ValueError(
            "minimum sequence log probability must be finite and negative"
        )


class CandidateSequenceStringModel(nn.Module):
    def __init__(
        self,
        config: CandidateSequenceConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or CandidateSequenceConfig()
        validate_sequence_config(self.config)
        token_feature_count = 3 + STRING_COUNT * 2
        self.token_encoder = nn.Sequential(
            nn.Linear(token_feature_count, self.config.token_size),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.token_size, self.config.token_size),
            nn.SiLU(),
        )
        self.sequence_encoder = nn.GRU(
            input_size=self.config.token_size * 2 + 2,
            hidden_size=self.config.hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.residual_head = nn.Sequential(
            nn.Linear(
                self.config.token_size + self.config.hidden_size * 2,
                self.config.output_hidden_size,
            ),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.output_hidden_size, STRING_COUNT),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(
        self,
        pitches: Tensor,
        confidences: Tensor,
        durations: Tensor,
        base_string_logits: Tensor,
        valid_strings: Tensor,
        candidate_mask: Tensor,
        group_mask: Tensor,
        group_onsets: Tensor,
    ) -> Tensor:
        if pitches.ndim != 3:
            raise ValueError(
                "sequence pitches must have shape [batch, groups, candidates]"
            )
        batch_size, group_count, candidate_count = pitches.shape
        candidate_shape = (batch_size, group_count, candidate_count)
        if (
            confidences.shape != candidate_shape
            or durations.shape != candidate_shape
            or candidate_mask.shape != candidate_shape
        ):
            raise ValueError("sequence candidate tensors must align")
        if base_string_logits.shape != candidate_shape + (STRING_COUNT,):
            raise ValueError("sequence string logits must have six classes")
        if valid_strings.shape != base_string_logits.shape:
            raise ValueError("sequence valid-string masks must align")
        if group_mask.shape != (batch_size, group_count):
            raise ValueError("sequence group mask must align")
        if group_onsets.shape != group_mask.shape:
            raise ValueError("sequence group onsets must align")
        if bool((candidate_mask & ~group_mask.unsqueeze(2)).any()):
            raise ValueError("padded sequence groups cannot contain candidates")
        lengths = group_mask.sum(dim=1)
        if bool((lengths <= 0).any()):
            raise ValueError("every sequence must contain at least one group")

        normalized_pitches = (
            pitches.to(base_string_logits.dtype) / max(1, PITCH_COUNT - 1)
        )
        normalized_durations = durations.clamp(
            0,
            self.config.maximum_duration,
        ) / self.config.maximum_duration
        normalized_logits = base_string_logits.clamp(
            self.config.minimum_log_probability,
            0,
        ) / abs(self.config.minimum_log_probability)
        token_features = torch.cat(
            (
                normalized_pitches.unsqueeze(3),
                confidences.unsqueeze(3),
                normalized_durations.unsqueeze(3),
                normalized_logits,
                valid_strings.to(base_string_logits.dtype),
            ),
            dim=3,
        )
        token_embeddings = self.token_encoder(token_features)
        token_embeddings = token_embeddings * candidate_mask.unsqueeze(
            3
        ).to(token_embeddings.dtype)
        candidate_counts = candidate_mask.sum(dim=2).clamp(min=1)
        mean_embedding = token_embeddings.sum(dim=2) / (
            candidate_counts.unsqueeze(2).to(token_embeddings.dtype)
        )
        maximum_embedding = token_embeddings.masked_fill(
            ~candidate_mask.unsqueeze(3),
            torch.finfo(token_embeddings.dtype).min,
        ).amax(dim=2)
        maximum_embedding = torch.where(
            group_mask.unsqueeze(2),
            maximum_embedding,
            torch.zeros_like(maximum_embedding),
        )

        previous_gap = torch.full_like(
            group_onsets,
            self.config.maximum_group_interval,
        )
        next_gap = torch.full_like(
            group_onsets,
            self.config.maximum_group_interval,
        )
        if group_count > 1:
            gaps = (group_onsets[:, 1:] - group_onsets[:, :-1]).clamp(
                0,
                self.config.maximum_group_interval,
            )
            adjacent_valid = group_mask[:, 1:] & group_mask[:, :-1]
            previous_gap[:, 1:] = torch.where(
                adjacent_valid,
                gaps,
                previous_gap[:, 1:],
            )
            next_gap[:, :-1] = torch.where(
                adjacent_valid,
                gaps,
                next_gap[:, :-1],
            )
        interval_features = torch.stack(
            (
                previous_gap / self.config.maximum_group_interval,
                next_gap / self.config.maximum_group_interval,
            ),
            dim=2,
        )
        group_embeddings = torch.cat(
            (mean_embedding, maximum_embedding, interval_features),
            dim=2,
        )
        packed = pack_padded_sequence(
            group_embeddings,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_context, _hidden = self.sequence_encoder(packed)
        group_context, _lengths = pad_packed_sequence(
            packed_context,
            batch_first=True,
            total_length=group_count,
        )
        candidate_context = group_context.unsqueeze(2).expand(
            -1,
            -1,
            candidate_count,
            -1,
        )
        residuals = self.residual_head(
            torch.cat((token_embeddings, candidate_context), dim=3)
        )
        return residuals * candidate_mask.unsqueeze(3).to(
            residuals.dtype
        )


class CandidateSequenceActivityModel(nn.Module):
    def __init__(
        self,
        config: CandidateSequenceConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or CandidateSequenceConfig()
        validate_sequence_config(self.config)
        token_feature_count = 4 + STRING_COUNT * 2
        self.token_encoder = nn.Sequential(
            nn.Linear(token_feature_count, self.config.token_size),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.token_size, self.config.token_size),
            nn.SiLU(),
        )
        self.sequence_encoder = nn.GRU(
            input_size=self.config.token_size * 2 + 2,
            hidden_size=self.config.hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.residual_head = nn.Sequential(
            nn.Linear(
                self.config.token_size + self.config.hidden_size * 2,
                self.config.output_hidden_size,
            ),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.output_hidden_size, 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(
        self,
        pitches: Tensor,
        confidences: Tensor,
        durations: Tensor,
        base_string_logits: Tensor,
        valid_strings: Tensor,
        base_activity_logits: Tensor,
        candidate_mask: Tensor,
        group_mask: Tensor,
        group_onsets: Tensor,
    ) -> Tensor:
        if pitches.ndim != 3:
            raise ValueError(
                "sequence pitches must have shape [batch, groups, candidates]"
            )
        batch_size, group_count, candidate_count = pitches.shape
        candidate_shape = (batch_size, group_count, candidate_count)
        if (
            confidences.shape != candidate_shape
            or durations.shape != candidate_shape
            or base_activity_logits.shape != candidate_shape
            or candidate_mask.shape != candidate_shape
        ):
            raise ValueError("sequence activity candidate tensors must align")
        if base_string_logits.shape != candidate_shape + (STRING_COUNT,):
            raise ValueError("sequence string logits must have six classes")
        if valid_strings.shape != base_string_logits.shape:
            raise ValueError("sequence valid-string masks must align")
        if group_mask.shape != (batch_size, group_count):
            raise ValueError("sequence group mask must align")
        if group_onsets.shape != group_mask.shape:
            raise ValueError("sequence group onsets must align")
        if bool((candidate_mask & ~group_mask.unsqueeze(2)).any()):
            raise ValueError("padded sequence groups cannot contain candidates")
        lengths = group_mask.sum(dim=1)
        if bool((lengths <= 0).any()):
            raise ValueError("every sequence must contain at least one group")

        normalized_pitches = (
            pitches.to(base_string_logits.dtype) / max(1, PITCH_COUNT - 1)
        )
        normalized_durations = durations.clamp(
            0,
            self.config.maximum_duration,
        ) / self.config.maximum_duration
        normalized_string_logits = base_string_logits.clamp(
            self.config.minimum_log_probability,
            0,
        ) / abs(self.config.minimum_log_probability)
        normalized_activity_logits = base_activity_logits.clamp(
            self.config.minimum_log_probability,
            abs(self.config.minimum_log_probability),
        ) / abs(self.config.minimum_log_probability)
        token_features = torch.cat(
            (
                normalized_pitches.unsqueeze(3),
                confidences.unsqueeze(3),
                normalized_durations.unsqueeze(3),
                normalized_activity_logits.unsqueeze(3),
                normalized_string_logits,
                valid_strings.to(base_string_logits.dtype),
            ),
            dim=3,
        )
        token_embeddings = self.token_encoder(token_features)
        token_embeddings = token_embeddings * candidate_mask.unsqueeze(
            3
        ).to(token_embeddings.dtype)
        candidate_counts = candidate_mask.sum(dim=2).clamp(min=1)
        mean_embedding = token_embeddings.sum(dim=2) / (
            candidate_counts.unsqueeze(2).to(token_embeddings.dtype)
        )
        maximum_embedding = token_embeddings.masked_fill(
            ~candidate_mask.unsqueeze(3),
            torch.finfo(token_embeddings.dtype).min,
        ).amax(dim=2)
        maximum_embedding = torch.where(
            group_mask.unsqueeze(2),
            maximum_embedding,
            torch.zeros_like(maximum_embedding),
        )

        previous_gap = torch.full_like(
            group_onsets,
            self.config.maximum_group_interval,
        )
        next_gap = torch.full_like(
            group_onsets,
            self.config.maximum_group_interval,
        )
        if group_count > 1:
            gaps = (group_onsets[:, 1:] - group_onsets[:, :-1]).clamp(
                0,
                self.config.maximum_group_interval,
            )
            adjacent_valid = group_mask[:, 1:] & group_mask[:, :-1]
            previous_gap[:, 1:] = torch.where(
                adjacent_valid,
                gaps,
                previous_gap[:, 1:],
            )
            next_gap[:, :-1] = torch.where(
                adjacent_valid,
                gaps,
                next_gap[:, :-1],
            )
        interval_features = torch.stack(
            (
                previous_gap / self.config.maximum_group_interval,
                next_gap / self.config.maximum_group_interval,
            ),
            dim=2,
        )
        group_embeddings = torch.cat(
            (mean_embedding, maximum_embedding, interval_features),
            dim=2,
        )
        packed = pack_padded_sequence(
            group_embeddings,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_context, _hidden = self.sequence_encoder(packed)
        group_context, _lengths = pad_packed_sequence(
            packed_context,
            batch_first=True,
            total_length=group_count,
        )
        candidate_context = group_context.unsqueeze(2).expand(
            -1,
            -1,
            candidate_count,
            -1,
        )
        residuals = self.residual_head(
            torch.cat((token_embeddings, candidate_context), dim=3)
        ).squeeze(3)
        return residuals * candidate_mask.to(residuals.dtype)


class CandidateSequenceConfidenceModel(CandidateSequenceActivityModel):
    def __init__(
        self,
        config: CandidateSequenceConfig | None = None,
        *,
        identity_accept_probability: float = 0.99,
    ) -> None:
        if (
            not math.isfinite(identity_accept_probability)
            or not 0.5 < identity_accept_probability < 1
        ):
            raise ValueError(
                "identity accept probability must be between 0.5 and 1"
            )
        super().__init__(config)
        self.identity_accept_probability = identity_accept_probability
        identity_logit = math.log(
            identity_accept_probability
            / (1 - identity_accept_probability)
        )
        nn.init.constant_(self.residual_head[-1].bias, identity_logit)


class CandidateSequenceJointModel(nn.Module):
    def __init__(
        self,
        config: CandidateSequenceConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or CandidateSequenceConfig()
        validate_sequence_config(self.config)
        token_feature_count = 4 + STRING_COUNT * 2
        self.token_encoder = nn.Sequential(
            nn.Linear(token_feature_count, self.config.token_size),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.token_size, self.config.token_size),
            nn.SiLU(),
        )
        self.sequence_encoder = nn.GRU(
            input_size=self.config.token_size * 2 + 2,
            hidden_size=self.config.hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.residual_head = nn.Sequential(
            nn.Linear(
                self.config.token_size + self.config.hidden_size * 2,
                self.config.output_hidden_size,
            ),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(
                self.config.output_hidden_size,
                STRING_COUNT + 1,
            ),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(
        self,
        pitches: Tensor,
        confidences: Tensor,
        durations: Tensor,
        base_string_logits: Tensor,
        valid_strings: Tensor,
        base_activity_logits: Tensor,
        candidate_mask: Tensor,
        group_mask: Tensor,
        group_onsets: Tensor,
    ) -> Tensor:
        if pitches.ndim != 3:
            raise ValueError(
                "sequence pitches must have shape [batch, groups, candidates]"
            )
        batch_size, group_count, candidate_count = pitches.shape
        candidate_shape = (batch_size, group_count, candidate_count)
        if (
            confidences.shape != candidate_shape
            or durations.shape != candidate_shape
            or base_activity_logits.shape != candidate_shape
            or candidate_mask.shape != candidate_shape
        ):
            raise ValueError("sequence joint candidate tensors must align")
        if base_string_logits.shape != candidate_shape + (STRING_COUNT,):
            raise ValueError("sequence string logits must have six classes")
        if valid_strings.shape != base_string_logits.shape:
            raise ValueError("sequence valid-string masks must align")
        if group_mask.shape != (batch_size, group_count):
            raise ValueError("sequence group mask must align")
        if group_onsets.shape != group_mask.shape:
            raise ValueError("sequence group onsets must align")
        if bool((candidate_mask & ~group_mask.unsqueeze(2)).any()):
            raise ValueError("padded sequence groups cannot contain candidates")
        lengths = group_mask.sum(dim=1)
        if bool((lengths <= 0).any()):
            raise ValueError("every sequence must contain at least one group")

        normalized_pitches = (
            pitches.to(base_string_logits.dtype) / max(1, PITCH_COUNT - 1)
        )
        normalized_durations = durations.clamp(
            0,
            self.config.maximum_duration,
        ) / self.config.maximum_duration
        normalized_string_logits = base_string_logits.clamp(
            self.config.minimum_log_probability,
            0,
        ) / abs(self.config.minimum_log_probability)
        normalized_activity_logits = base_activity_logits.clamp(
            self.config.minimum_log_probability,
            abs(self.config.minimum_log_probability),
        ) / abs(self.config.minimum_log_probability)
        token_features = torch.cat(
            (
                normalized_pitches.unsqueeze(3),
                confidences.unsqueeze(3),
                normalized_durations.unsqueeze(3),
                normalized_activity_logits.unsqueeze(3),
                normalized_string_logits,
                valid_strings.to(base_string_logits.dtype),
            ),
            dim=3,
        )
        token_embeddings = self.token_encoder(token_features)
        token_embeddings = token_embeddings * candidate_mask.unsqueeze(
            3
        ).to(token_embeddings.dtype)
        candidate_counts = candidate_mask.sum(dim=2).clamp(min=1)
        mean_embedding = token_embeddings.sum(dim=2) / (
            candidate_counts.unsqueeze(2).to(token_embeddings.dtype)
        )
        maximum_embedding = token_embeddings.masked_fill(
            ~candidate_mask.unsqueeze(3),
            torch.finfo(token_embeddings.dtype).min,
        ).amax(dim=2)
        maximum_embedding = torch.where(
            group_mask.unsqueeze(2),
            maximum_embedding,
            torch.zeros_like(maximum_embedding),
        )

        previous_gap = torch.full_like(
            group_onsets,
            self.config.maximum_group_interval,
        )
        next_gap = torch.full_like(
            group_onsets,
            self.config.maximum_group_interval,
        )
        if group_count > 1:
            gaps = (group_onsets[:, 1:] - group_onsets[:, :-1]).clamp(
                0,
                self.config.maximum_group_interval,
            )
            adjacent_valid = group_mask[:, 1:] & group_mask[:, :-1]
            previous_gap[:, 1:] = torch.where(
                adjacent_valid,
                gaps,
                previous_gap[:, 1:],
            )
            next_gap[:, :-1] = torch.where(
                adjacent_valid,
                gaps,
                next_gap[:, :-1],
            )
        interval_features = torch.stack(
            (
                previous_gap / self.config.maximum_group_interval,
                next_gap / self.config.maximum_group_interval,
            ),
            dim=2,
        )
        group_embeddings = torch.cat(
            (mean_embedding, maximum_embedding, interval_features),
            dim=2,
        )
        packed = pack_padded_sequence(
            group_embeddings,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_context, _hidden = self.sequence_encoder(packed)
        group_context, _lengths = pad_packed_sequence(
            packed_context,
            batch_first=True,
            total_length=group_count,
        )
        candidate_context = group_context.unsqueeze(2).expand(
            -1,
            -1,
            candidate_count,
            -1,
        )
        residuals = self.residual_head(
            torch.cat((token_embeddings, candidate_context), dim=3)
        )
        return residuals * candidate_mask.unsqueeze(3).to(
            residuals.dtype
        )


def refine_candidate_position_logits(
    base_logits: Tensor,
    string_residuals: Tensor,
    valid_strings: Tensor,
) -> Tensor:
    if (
        base_logits.ndim != 2
        or base_logits.shape[1] != STRING_COUNT + 1
    ):
        raise ValueError(
            "candidate position logits must contain none plus six strings"
        )
    if string_residuals.shape != (base_logits.shape[0], STRING_COUNT):
        raise ValueError("candidate sequence residuals must have six strings")
    if valid_strings.shape != string_residuals.shape:
        raise ValueError("candidate valid-string mask must align")
    if not bool(torch.count_nonzero(string_residuals)):
        return base_logits
    active_log_probability = torch.logsumexp(
        base_logits[:, 1:],
        dim=1,
        keepdim=True,
    )
    conditional_log_probabilities = functional.log_softmax(
        (base_logits[:, 1:] + string_residuals).masked_fill(
            ~valid_strings,
            -30.0,
        ),
        dim=1,
    )
    return torch.cat(
        (
            base_logits[:, :1],
            active_log_probability + conditional_log_probabilities,
        ),
        dim=1,
    )


def sequence_window_ranges(
    group_count: int,
    *,
    sequence_length: int,
    sequence_stride: int,
) -> list[tuple[int, int]]:
    if group_count <= 0:
        return []
    if sequence_length <= 0:
        raise ValueError("sequence length must be positive")
    if not 0 < sequence_stride <= sequence_length:
        raise ValueError("sequence stride must be in 1..sequence_length")
    if group_count <= sequence_length:
        return [(0, group_count)]
    final_start = group_count - sequence_length
    starts = list(range(0, final_start + 1, sequence_stride))
    if starts[-1] != final_start:
        starts.append(final_start)
    return [(start, start + sequence_length) for start in starts]


def candidate_sequence_activity_audit(
    dataset: OnsetPositionDataset,
    *,
    repeat_interval: float = 0.75,
    sequence_length: int = 32,
    sequence_stride: int = 16,
) -> dict[str, object]:
    if not math.isfinite(repeat_interval) or repeat_interval <= 0:
        raise ValueError("candidate repeat interval must be positive")
    repeated_event_indexes: set[tuple[int, int]] = set()
    for annotation_index, annotation in enumerate(dataset.annotations):
        previous_by_position: dict[tuple[int, int], float] = {}
        for event_index, event in enumerate(annotation.events):
            position = (event.string, event.fret)
            previous = previous_by_position.get(position)
            if (
                previous is not None
                and 0 < event.onset - previous <= repeat_interval
            ):
                repeated_event_indexes.add(
                    (annotation_index, event_index)
                )
            previous_by_position[position] = event.onset

    counters: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    tracks_by_dataset: dict[str, set[int]] = defaultdict(set)
    groups_by_annotation: dict[int, int] = defaultdict(int)
    for group in dataset.groups:
        annotation = dataset.annotations[group.annotation_index]
        source = str(annotation.provenance.get("dataset", "unknown"))
        tracks_by_dataset[source].add(group.annotation_index)
        groups_by_annotation[group.annotation_index] += 1
        references_by_pitch: dict[int, list[int]] = defaultdict(list)
        for event_index in group.event_indexes:
            event = annotation.events[event_index]
            pitch = OPEN_STRING_PITCHES[event.string - 1] + event.fret
            references_by_pitch[pitch].append(event_index)
        input_pitches = list(
            group.input_pitches
            if group.input_pitches is not None
            else references_by_pitch
        )
        repeated_pitches = {
            pitch
            for pitch, event_indexes in references_by_pitch.items()
            if any(
                (group.annotation_index, event_index)
                in repeated_event_indexes
                for event_index in event_indexes
            )
        }
        active_candidates = sum(
            int(pitch in references_by_pitch)
            for pitch in input_pitches
        )
        repeated_candidates = sum(
            int(pitch in repeated_pitches)
            for pitch in input_pitches
        )
        values = counters[source]
        values["groups"] += 1
        values["active_groups"] += int(bool(group.event_indexes))
        values["inactive_groups"] += int(not group.event_indexes)
        values["candidates"] += len(input_pitches)
        values["active_candidates"] += active_candidates
        values["inactive_candidates"] += (
            len(input_pitches) - active_candidates
        )
        values["repeated_candidates"] += repeated_candidates
        values["matched_reference_events"] += len(group.event_indexes)
        values["repeated_reference_events"] += sum(
            int(
                (group.annotation_index, event_index)
                in repeated_event_indexes
            )
            for event_index in group.event_indexes
        )
    for annotation_index, group_count in groups_by_annotation.items():
        annotation = dataset.annotations[annotation_index]
        source = str(annotation.provenance.get("dataset", "unknown"))
        counters[source]["windows"] += len(
            sequence_window_ranges(
                group_count,
                sequence_length=sequence_length,
                sequence_stride=sequence_stride,
            )
        )

    datasets: dict[str, dict[str, int]] = {}
    for source, values in sorted(counters.items()):
        datasets[source] = {
            "tracks": len(tracks_by_dataset[source]),
            **dict(values),
        }
    total_keys = (
        "tracks",
        "groups",
        "windows",
        "active_groups",
        "inactive_groups",
        "candidates",
        "active_candidates",
        "inactive_candidates",
        "repeated_candidates",
        "matched_reference_events",
        "repeated_reference_events",
    )
    return {
        "repeat_interval": repeat_interval,
        "sequence_length": sequence_length,
        "sequence_stride": sequence_stride,
        "totals": {
            key: sum(values.get(key, 0) for values in datasets.values())
            for key in total_keys
        },
        "datasets": datasets,
        "filtered_negative_groups": dataset.filtered_negative_groups,
    }


def precompute_candidate_sequence_groups(
    dataset: OnsetPositionDataset,
    model: OnsetPositionNet,
    device: torch.device,
    *,
    batch_size: int,
    repeat_interval: float = 0.75,
) -> list[CandidateSequenceGroup]:
    if batch_size <= 0:
        raise ValueError("candidate sequence batch size must be positive")
    if not math.isfinite(repeat_interval) or repeat_interval <= 0:
        raise ValueError("candidate repeat interval must be positive")
    if (
        model.candidate_head is None
        or model.config.candidate_position_objective
        != "spectral_residual"
    ):
        raise ValueError(
            "candidate sequences require a spectral residual position model"
        )
    if model.config.candidate_temporal_string_residual:
        raise ValueError(
            "candidate sequences require a local position checkpoint"
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    result: list[CandidateSequenceGroup] = []
    repeated_event_indexes: set[tuple[int, int]] = set()
    for annotation_index, annotation in enumerate(dataset.annotations):
        previous_by_position: dict[tuple[int, int], float] = {}
        for event_index, event in enumerate(annotation.events):
            position = (event.string, event.fret)
            previous = previous_by_position.get(position)
            if (
                previous is not None
                and 0 < event.onset - previous <= repeat_interval
            ):
                repeated_event_indexes.add(
                    (annotation_index, event_index)
                )
            previous_by_position[position] = event.onset
    group_offset = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            pitch_masks = batch["pitch_mask"].to(device)
            encoded = model.encode(features, pitch_masks)
            (
                query_batch_indexes,
                query_pitch_indexes,
                targets,
            ) = candidate_position_queries(
                batch["event_pitches"].to(device),
                batch["event_strings"].to(device),
                pitch_masks,
                categorical=True,
            )
            query_weights = candidate_position_query_weights(
                batch["event_pitches"].to(device),
                batch["candidate_target_weights"].to(device),
                pitch_masks,
            )
            logits = model.classify_candidate_positions(
                encoded[query_batch_indexes],
                query_pitch_indexes,
                features=features[query_batch_indexes],
                candidate_confidences=pitch_masks[
                    query_batch_indexes,
                    query_pitch_indexes,
                ],
            )
            activity_logits = model.classify_candidate_activity(
                encoded[query_batch_indexes],
                query_pitch_indexes,
                features=features[query_batch_indexes],
                candidate_confidences=pitch_masks[
                    query_batch_indexes,
                    query_pitch_indexes,
                ],
                candidate_position_logits=logits,
            )
            conditional_logits = functional.log_softmax(
                logits[:, 1:],
                dim=1,
            )
            valid_strings = model.valid_candidate_positions[
                query_pitch_indexes,
                1:,
            ]
            for local_index in range(features.shape[0]):
                rows = query_batch_indexes == local_index
                group = dataset.groups[group_offset + local_index]
                input_pitches = list(group.input_pitches or ())
                input_durations = list(group.input_durations or ())
                duration_by_pitch: dict[int, float] = {}
                for pitch, duration in zip(
                    input_pitches,
                    input_durations,
                ):
                    duration_by_pitch[pitch] = max(
                        duration_by_pitch.get(pitch, 0.0),
                        duration,
                    )
                local_pitch_indexes = query_pitch_indexes[rows].detach().cpu()
                local_pitches = local_pitch_indexes + PITCH_MIN
                annotation = dataset.annotations[group.annotation_index]
                repeated_pitches = {
                    OPEN_STRING_PITCHES[
                        annotation.events[event_index].string - 1
                    ]
                    + annotation.events[event_index].fret
                    for event_index in group.event_indexes
                    if (
                        group.annotation_index,
                        event_index,
                    )
                    in repeated_event_indexes
                }
                result.append(
                    CandidateSequenceGroup(
                        annotation_index=group.annotation_index,
                        onset=float(group.onset or 0.0),
                        pitches=local_pitch_indexes,
                        confidences=pitch_masks[
                            local_index,
                            query_pitch_indexes[rows],
                        ].detach().cpu(),
                        durations=torch.tensor(
                            [
                                duration_by_pitch.get(int(pitch), 0.0)
                                for pitch in local_pitches
                            ],
                            dtype=torch.float32,
                        ),
                        base_string_logits=conditional_logits[
                            rows
                        ].detach().cpu(),
                        valid_strings=valid_strings[rows].detach().cpu(),
                        string_targets=targets[
                            rows,
                            1:,
                        ].detach().cpu(),
                        target_weights=query_weights[rows].detach().cpu(),
                        base_activity_logits=activity_logits[
                            rows
                        ].detach().cpu(),
                        activity_targets=targets[
                            rows,
                            1:,
                        ].sum(dim=1).detach().cpu(),
                        repeat_targets=torch.tensor(
                            [
                                float(int(pitch) in repeated_pitches)
                                for pitch in local_pitches
                            ],
                            dtype=torch.float32,
                        ),
                    )
                )
            group_offset += features.shape[0]
    if len(result) != len(dataset):
        raise RuntimeError("candidate sequence groups are incomplete")
    return result


class CandidateSequenceWindowDataset(
    Dataset[list[CandidateSequenceGroup]]
):
    def __init__(
        self,
        groups: list[CandidateSequenceGroup],
        config: CandidateSequenceConfig,
    ) -> None:
        validate_sequence_config(config)
        self.groups = groups
        self.config = config
        group_indexes_by_annotation: dict[int, list[int]] = {}
        for group_index, group in enumerate(groups):
            group_indexes_by_annotation.setdefault(
                group.annotation_index,
                [],
            ).append(group_index)
        self.windows: list[tuple[int, ...]] = []
        for group_indexes in group_indexes_by_annotation.values():
            for start, end in sequence_window_ranges(
                len(group_indexes),
                sequence_length=config.sequence_length,
                sequence_stride=config.sequence_stride,
            ):
                self.windows.append(tuple(group_indexes[start:end]))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> list[CandidateSequenceGroup]:
        return [self.groups[item] for item in self.windows[index]]


def collate_candidate_sequence_windows(
    windows: list[list[CandidateSequenceGroup]],
) -> dict[str, Tensor]:
    if not windows or any(not window for window in windows):
        raise ValueError("candidate sequence windows cannot be empty")
    batch_size = len(windows)
    group_count = max(len(window) for window in windows)
    candidate_count = max(
        group.pitches.shape[0]
        for window in windows
        for group in window
    )
    pitches = torch.zeros(
        batch_size,
        group_count,
        candidate_count,
        dtype=torch.long,
    )
    confidences = torch.zeros_like(pitches, dtype=torch.float32)
    durations = torch.zeros_like(confidences)
    base_string_logits = torch.zeros(
        batch_size,
        group_count,
        candidate_count,
        STRING_COUNT,
    )
    valid_strings = torch.zeros_like(
        base_string_logits,
        dtype=torch.bool,
    )
    string_targets = torch.zeros_like(base_string_logits)
    target_weights = torch.ones_like(confidences)
    base_activity_logits = torch.zeros_like(confidences)
    activity_targets = torch.zeros_like(confidences)
    repeat_targets = torch.zeros_like(confidences)
    candidate_mask = torch.zeros_like(pitches, dtype=torch.bool)
    group_mask = torch.zeros(
        batch_size,
        group_count,
        dtype=torch.bool,
    )
    group_onsets = torch.zeros(
        batch_size,
        group_count,
        dtype=torch.float32,
    )
    for batch_index, window in enumerate(windows):
        for group_index, group in enumerate(window):
            count = group.pitches.shape[0]
            pitches[batch_index, group_index, :count] = group.pitches
            confidences[batch_index, group_index, :count] = (
                group.confidences
            )
            durations[batch_index, group_index, :count] = group.durations
            base_string_logits[
                batch_index,
                group_index,
                :count,
            ] = group.base_string_logits
            valid_strings[
                batch_index,
                group_index,
                :count,
            ] = group.valid_strings
            string_targets[
                batch_index,
                group_index,
                :count,
            ] = group.string_targets
            target_weights[
                batch_index,
                group_index,
                :count,
            ] = group.target_weights
            if group.base_activity_logits.numel():
                base_activity_logits[
                    batch_index,
                    group_index,
                    :count,
                ] = group.base_activity_logits
            if group.activity_targets.numel():
                activity_targets[
                    batch_index,
                    group_index,
                    :count,
                ] = group.activity_targets
            else:
                activity_targets[
                    batch_index,
                    group_index,
                    :count,
                ] = (group.string_targets.sum(dim=1) > 0).to(
                    activity_targets.dtype
                )
            if group.repeat_targets.numel():
                repeat_targets[
                    batch_index,
                    group_index,
                    :count,
                ] = group.repeat_targets
            candidate_mask[batch_index, group_index, :count] = True
            group_mask[batch_index, group_index] = True
            group_onsets[batch_index, group_index] = group.onset
    return {
        "pitches": pitches,
        "confidences": confidences,
        "durations": durations,
        "base_string_logits": base_string_logits,
        "valid_strings": valid_strings,
        "string_targets": string_targets,
        "target_weights": target_weights,
        "base_activity_logits": base_activity_logits,
        "activity_targets": activity_targets,
        "repeat_targets": repeat_targets,
        "candidate_mask": candidate_mask,
        "group_mask": group_mask,
        "group_onsets": group_onsets,
    }


def candidate_sequence_loss(
    base_string_logits: Tensor,
    string_residuals: Tensor,
    string_targets: Tensor,
    target_weights: Tensor,
    candidate_mask: Tensor,
    valid_strings: Tensor,
) -> tuple[Tensor, int, int]:
    if base_string_logits.shape != string_residuals.shape:
        raise ValueError("base logits and sequence residuals must align")
    if string_targets.shape != base_string_logits.shape:
        raise ValueError("sequence string targets must align")
    if valid_strings.shape != base_string_logits.shape:
        raise ValueError("sequence valid-string masks must align")
    if target_weights.shape != candidate_mask.shape:
        raise ValueError("sequence target weights must align")
    active_mask = candidate_mask & (string_targets.sum(dim=3) > 0)
    refined_log_probabilities = functional.log_softmax(
        (base_string_logits + string_residuals).masked_fill(
            ~valid_strings,
            -30.0,
        ),
        dim=3,
    )
    per_candidate_loss = -(
        string_targets * refined_log_probabilities
    ).sum(dim=3)
    weighted_loss = per_candidate_loss * target_weights
    total = int(active_mask.sum().detach().cpu())
    if total == 0:
        return string_residuals.sum() * 0, 0, 0
    loss = weighted_loss[active_mask].mean()
    predictions = refined_log_probabilities.argmax(dim=3)
    matches = torch.gather(
        string_targets,
        3,
        predictions.unsqueeze(3),
    ).squeeze(3) > 0
    correct = int((matches & active_mask).sum().detach().cpu())
    return loss, correct, total


@lru_cache(maxsize=STRING_COUNT)
def unique_string_assignments(
    candidate_count: int,
) -> tuple[tuple[int, ...], ...]:
    if not 1 <= candidate_count <= STRING_COUNT:
        raise ValueError("unique assignments require one to six candidates")
    return tuple(
        itertools.permutations(range(STRING_COUNT), candidate_count)
    )


def candidate_group_structured_objective(
    string_logits: Tensor,
    valid_strings: Tensor,
    string_targets: Tensor,
    supervised_mask: Tensor,
) -> tuple[Tensor | None, Tensor, bool]:
    if (
        string_logits.ndim != 2
        or string_logits.shape[1] != STRING_COUNT
    ):
        raise ValueError("group string logits must have shape [items, 6]")
    if (
        valid_strings.shape != string_logits.shape
        or string_targets.shape != string_logits.shape
        or supervised_mask.shape != string_logits.shape[:1]
    ):
        raise ValueError("structured group tensors must align")
    if string_logits.shape[0] == 0:
        raise ValueError("structured group cannot be empty")
    if bool((~valid_strings.any(dim=1)).any()):
        raise ValueError("every structured candidate needs a playable string")
    target_mask = string_targets > 0
    if bool((supervised_mask & ~target_mask.any(dim=1)).any()):
        raise ValueError("supervised candidates need string targets")
    if bool((target_mask & ~valid_strings).any()):
        raise ValueError("structured string targets must be playable")

    candidate_count = string_logits.shape[0]
    if candidate_count == 1:
        predictions = string_logits.masked_fill(
            ~valid_strings,
            -torch.inf,
        ).argmax(dim=1)
        allowed = valid_strings & (
            ~supervised_mask.unsqueeze(1) | target_mask
        )
        denominator = torch.logsumexp(
            string_logits.masked_fill(~valid_strings, -torch.inf),
            dim=1,
        ).sum()
        numerator = torch.logsumexp(
            string_logits.masked_fill(~allowed, -torch.inf),
            dim=1,
        ).sum()
        return denominator - numerator, predictions, True
    if candidate_count <= STRING_COUNT:
        assignments = torch.tensor(
            unique_string_assignments(candidate_count),
            device=string_logits.device,
            dtype=torch.long,
        )
        rows = torch.arange(
            candidate_count,
            device=string_logits.device,
        ).unsqueeze(0)
        feasible = valid_strings[rows, assignments].all(dim=1)
        if bool(feasible.any()):
            feasible_assignments = assignments[feasible]
            feasible_scores = string_logits[
                rows,
                feasible_assignments,
            ].sum(dim=1)
            best_assignment = feasible_assignments[
                feasible_scores.argmax()
            ]
            correct = (
                ~supervised_mask.unsqueeze(0)
                | target_mask[rows, feasible_assignments]
            ).all(dim=1)
            if not bool(correct.any()):
                return None, best_assignment, True
            loss = torch.logsumexp(feasible_scores, dim=0) - torch.logsumexp(
                feasible_scores[correct],
                dim=0,
            )
            return loss, best_assignment, True

    allowed = valid_strings & (
        ~supervised_mask.unsqueeze(1) | target_mask
    )
    predictions = string_logits.masked_fill(
        ~valid_strings,
        -torch.inf,
    ).argmax(dim=1)
    if bool((~allowed.any(dim=1)).any()):
        return None, predictions, False
    denominator = torch.logsumexp(
        string_logits.masked_fill(~valid_strings, -torch.inf),
        dim=1,
    ).sum()
    numerator = torch.logsumexp(
        string_logits.masked_fill(~allowed, -torch.inf),
        dim=1,
    ).sum()
    return denominator - numerator, predictions, False


def candidate_sequence_structured_loss(
    base_string_logits: Tensor,
    string_residuals: Tensor,
    string_targets: Tensor,
    target_weights: Tensor,
    base_activity_logits: Tensor,
    repeat_targets: Tensor,
    candidate_mask: Tensor,
    group_mask: Tensor,
    valid_strings: Tensor,
    *,
    multi_candidate_group_weight: float,
    repeat_group_weight: float,
) -> tuple[Tensor, dict[str, int | float]]:
    candidate_shape = candidate_mask.shape
    string_shape = candidate_shape + (STRING_COUNT,)
    if (
        base_string_logits.shape != string_shape
        or string_residuals.shape != string_shape
        or string_targets.shape != string_shape
        or valid_strings.shape != string_shape
        or target_weights.shape != candidate_shape
        or base_activity_logits.shape != candidate_shape
        or repeat_targets.shape != candidate_shape
        or group_mask.shape != candidate_shape[:2]
    ):
        raise ValueError("structured sequence tensors must align")
    if (
        not math.isfinite(multi_candidate_group_weight)
        or multi_candidate_group_weight < 1
        or not math.isfinite(repeat_group_weight)
        or repeat_group_weight < 1
    ):
        raise ValueError("structured group weights must be at least one")

    refined_logits = base_string_logits + string_residuals
    weighted_losses: list[Tensor] = []
    loss_weights: list[float] = []
    counts: defaultdict[str, int] = defaultdict(int)
    for batch_index in range(candidate_shape[0]):
        for group_index in range(candidate_shape[1]):
            if not bool(group_mask[batch_index, group_index]):
                continue
            present = candidate_mask[batch_index, group_index]
            accepted = present & (
                base_activity_logits[batch_index, group_index] >= 0
            )
            accepted_indexes = torch.nonzero(
                accepted,
                as_tuple=False,
            ).flatten()
            if not accepted_indexes.numel():
                continue
            local_targets = string_targets[
                batch_index,
                group_index,
                accepted_indexes,
            ]
            supervised = local_targets.sum(dim=1) > 0
            if not bool(supervised.any()):
                continue
            local_repeat = repeat_targets[
                batch_index,
                group_index,
                accepted_indexes,
            ] > 0
            group_loss, predictions, used_unique_assignment = (
                candidate_group_structured_objective(
                    refined_logits[
                        batch_index,
                        group_index,
                        accepted_indexes,
                    ],
                    valid_strings[
                        batch_index,
                        group_index,
                        accepted_indexes,
                    ],
                    local_targets,
                    supervised,
                )
            )
            matches = torch.gather(
                local_targets > 0,
                1,
                predictions.unsqueeze(1),
            ).squeeze(1)
            supervised_matches = matches[supervised]
            repeat_supervised = supervised & local_repeat
            counts["supervised_groups"] += 1
            counts["groups_with_multiple_accepted"] += int(
                accepted_indexes.numel() > 1
            )
            counts["unique_assignment_groups"] += int(
                used_unique_assignment
            )
            counts["fallback_independent_groups"] += int(
                not used_unique_assignment
            )
            counts["supervised_candidates"] += int(supervised.sum())
            counts["correct_candidates"] += int(
                supervised_matches.sum().detach().cpu()
            )
            counts["wrong_string"] += int(
                (~supervised_matches).sum().detach().cpu()
            )
            counts["exact_groups"] += int(
                bool(supervised_matches.all())
            )
            counts["repeat_true_positive"] += int(
                (matches & repeat_supervised).sum().detach().cpu()
            )
            counts["repeat_false_negative"] += int(
                (~matches & repeat_supervised).sum().detach().cpu()
            )
            if group_loss is None:
                counts["unsatisfiable_groups"] += 1
                continue
            local_weights = target_weights[
                batch_index,
                group_index,
                accepted_indexes,
            ][supervised]
            group_weight = float(local_weights.mean().detach().cpu())
            if accepted_indexes.numel() > 1:
                group_weight *= multi_candidate_group_weight
            if bool(repeat_supervised.any()):
                group_weight *= repeat_group_weight
            weighted_losses.append(group_loss * group_weight)
            loss_weights.append(group_weight)

    if not weighted_losses:
        loss = string_residuals.sum() * 0
    else:
        loss = torch.stack(weighted_losses).sum() / sum(loss_weights)
    counts["loss_weight_millis"] = round(sum(loss_weights) * 1000)
    return loss, dict(counts)


def refine_candidate_activity_logits(
    base_activity_logits: Tensor,
    activity_residuals: Tensor,
) -> Tensor:
    if base_activity_logits.shape != activity_residuals.shape:
        raise ValueError("base activity logits and residuals must align")
    return base_activity_logits + activity_residuals


def candidate_sequence_activity_loss(
    base_activity_logits: Tensor,
    activity_residuals: Tensor,
    activity_targets: Tensor,
    repeat_targets: Tensor,
    candidate_mask: Tensor,
    *,
    activity_positive_weight: float,
    repeat_positive_weight: float,
) -> tuple[Tensor, dict[str, int]]:
    candidate_shape = candidate_mask.shape
    if (
        base_activity_logits.shape != candidate_shape
        or activity_residuals.shape != candidate_shape
        or activity_targets.shape != candidate_shape
        or repeat_targets.shape != candidate_shape
    ):
        raise ValueError("sequence activity tensors must align")
    if (
        not math.isfinite(activity_positive_weight)
        or activity_positive_weight <= 0
        or not math.isfinite(repeat_positive_weight)
        or repeat_positive_weight < 1
    ):
        raise ValueError(
            "sequence activity weights must be positive and repeat at least one"
        )
    if bool(
        (
            candidate_mask
            & (
                ((activity_targets != 0) & (activity_targets != 1))
                | ((repeat_targets != 0) & (repeat_targets != 1))
                | (repeat_targets > activity_targets)
            )
        ).any()
    ):
        raise ValueError("sequence activity targets must be binary and valid")
    refined_logits = refine_candidate_activity_logits(
        base_activity_logits,
        activity_residuals,
    )
    per_candidate_loss = functional.binary_cross_entropy_with_logits(
        refined_logits,
        activity_targets,
        reduction="none",
    )
    weights = torch.where(
        activity_targets > 0,
        torch.full_like(activity_targets, activity_positive_weight),
        torch.ones_like(activity_targets),
    )
    weights = torch.where(
        repeat_targets > 0,
        weights * repeat_positive_weight,
        weights,
    )
    total = int(candidate_mask.sum().detach().cpu())
    if total == 0:
        return activity_residuals.sum() * 0, {
            "candidates": 0,
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0,
            "repeat_true_positive": 0,
            "repeat_false_negative": 0,
        }
    loss = (per_candidate_loss * weights)[candidate_mask].mean()
    predictions = refined_logits >= 0
    references = activity_targets > 0
    repeated = repeat_targets > 0
    return loss, {
        "candidates": total,
        "true_positive": int(
            (predictions & references & candidate_mask).sum().detach().cpu()
        ),
        "false_positive": int(
            (predictions & ~references & candidate_mask).sum().detach().cpu()
        ),
        "false_negative": int(
            (~predictions & references & candidate_mask).sum().detach().cpu()
        ),
        "repeat_true_positive": int(
            (predictions & repeated & candidate_mask).sum().detach().cpu()
        ),
        "repeat_false_negative": int(
            (~predictions & repeated & candidate_mask).sum().detach().cpu()
        ),
    }


def candidate_joint_log_probabilities(
    base_activity_logits: Tensor,
    base_string_logits: Tensor,
    valid_strings: Tensor,
) -> Tensor:
    if base_string_logits.shape[:-1] != base_activity_logits.shape:
        raise ValueError("joint activity and string logits must align")
    if base_string_logits.shape[-1] != STRING_COUNT:
        raise ValueError("joint string logits must have six classes")
    if valid_strings.shape != base_string_logits.shape:
        raise ValueError("joint valid-string masks must align")
    conditional_log_probabilities = functional.log_softmax(
        base_string_logits.masked_fill(~valid_strings, -30.0),
        dim=-1,
    )
    return torch.cat(
        (
            functional.logsigmoid(-base_activity_logits).unsqueeze(-1),
            functional.logsigmoid(base_activity_logits).unsqueeze(-1)
            + conditional_log_probabilities,
        ),
        dim=-1,
    )


def refine_candidate_joint_logits(
    base_activity_logits: Tensor,
    base_string_logits: Tensor,
    valid_strings: Tensor,
    joint_residuals: Tensor,
) -> tuple[Tensor, Tensor]:
    expected_shape = base_activity_logits.shape + (STRING_COUNT + 1,)
    if joint_residuals.shape != expected_shape:
        raise ValueError("joint decision residuals must have seven classes")
    base_joint_logits = candidate_joint_log_probabilities(
        base_activity_logits,
        base_string_logits,
        valid_strings,
    )
    joint_valid = torch.cat(
        (
            torch.ones_like(
                base_activity_logits.unsqueeze(-1),
                dtype=torch.bool,
            ),
            valid_strings,
        ),
        dim=-1,
    )
    refined_joint_logits = functional.log_softmax(
        (base_joint_logits + joint_residuals).masked_fill(
            ~joint_valid,
            -30.0,
        ),
        dim=-1,
    )
    active_log_probability = torch.logsumexp(
        refined_joint_logits[..., 1:],
        dim=-1,
    )
    activity_logits = (
        active_log_probability - refined_joint_logits[..., 0]
    )
    return refined_joint_logits, activity_logits


def candidate_sequence_joint_loss(
    base_activity_logits: Tensor,
    base_string_logits: Tensor,
    valid_strings: Tensor,
    joint_residuals: Tensor,
    activity_targets: Tensor,
    string_targets: Tensor,
    repeat_targets: Tensor,
    candidate_mask: Tensor,
    *,
    wrong_string_weight: float,
    repeat_positive_weight: float,
) -> tuple[Tensor, dict[str, int]]:
    candidate_shape = candidate_mask.shape
    if (
        base_activity_logits.shape != candidate_shape
        or base_string_logits.shape
        != candidate_shape + (STRING_COUNT,)
        or valid_strings.shape != base_string_logits.shape
        or joint_residuals.shape
        != candidate_shape + (STRING_COUNT + 1,)
        or activity_targets.shape != candidate_shape
        or string_targets.shape != base_string_logits.shape
        or repeat_targets.shape != candidate_shape
    ):
        raise ValueError("sequence joint tensors must align")
    if (
        not math.isfinite(wrong_string_weight)
        or wrong_string_weight < 1
        or not math.isfinite(repeat_positive_weight)
        or repeat_positive_weight < 1
    ):
        raise ValueError("sequence joint weights must be at least one")
    reference_active = activity_targets > 0
    string_target_mask = string_targets > 0
    if bool(
        (
            candidate_mask
            & (
                ((activity_targets != 0) & (activity_targets != 1))
                | ((repeat_targets != 0) & (repeat_targets != 1))
                | (repeat_targets > activity_targets)
                | (
                    reference_active
                    != string_target_mask.any(dim=-1)
                )
            )
        ).any()
    ):
        raise ValueError("sequence joint targets must be binary and aligned")
    if bool(
        (
            candidate_mask.unsqueeze(-1)
            & string_target_mask
            & ~valid_strings
        ).any()
    ):
        raise ValueError("sequence joint targets must be playable")

    refined_joint_logits, activity_logits = refine_candidate_joint_logits(
        base_activity_logits,
        base_string_logits,
        valid_strings,
        joint_residuals,
    )
    target_mask = torch.cat(
        (
            (~reference_active).unsqueeze(-1),
            string_target_mask,
        ),
        dim=-1,
    )
    target_log_probability = torch.logsumexp(
        refined_joint_logits.masked_fill(~target_mask, -torch.inf),
        dim=-1,
    )
    per_candidate_loss = -target_log_probability

    base_string_predictions = base_string_logits.masked_fill(
        ~valid_strings,
        -30.0,
    ).argmax(dim=-1)
    base_string_matches = torch.gather(
        string_target_mask,
        -1,
        base_string_predictions.unsqueeze(-1),
    ).squeeze(-1)
    base_wrong_string = reference_active & ~base_string_matches
    weights = torch.ones_like(per_candidate_loss)
    weights = torch.where(
        base_wrong_string,
        weights * wrong_string_weight,
        weights,
    )
    weights = torch.where(
        repeat_targets > 0,
        weights * repeat_positive_weight,
        weights,
    )
    total = int(candidate_mask.sum().detach().cpu())
    if total == 0:
        return joint_residuals.sum() * 0, {
            "candidates": 0,
            "joint_true_positive": 0,
            "joint_false_positive": 0,
            "joint_false_negative": 0,
            "activity_true_positive": 0,
            "activity_false_positive": 0,
            "activity_false_negative": 0,
            "wrong_string": 0,
            "base_wrong_string": 0,
            "repeat_true_positive": 0,
            "repeat_false_negative": 0,
        }
    loss = (per_candidate_loss * weights)[candidate_mask].mean()

    predicted_active = activity_logits >= 0
    string_predictions = refined_joint_logits[..., 1:].argmax(dim=-1)
    string_matches = torch.gather(
        string_target_mask,
        -1,
        string_predictions.unsqueeze(-1),
    ).squeeze(-1)
    correct_position = predicted_active & reference_active & string_matches
    wrong_string = predicted_active & reference_active & ~string_matches
    joint_false_positive = predicted_active & (
        ~reference_active | ~string_matches
    )
    joint_false_negative = reference_active & (
        ~predicted_active | ~string_matches
    )
    repeated = repeat_targets > 0
    return loss, {
        "candidates": total,
        "joint_true_positive": int(
            (correct_position & candidate_mask).sum().detach().cpu()
        ),
        "joint_false_positive": int(
            (joint_false_positive & candidate_mask).sum().detach().cpu()
        ),
        "joint_false_negative": int(
            (joint_false_negative & candidate_mask).sum().detach().cpu()
        ),
        "activity_true_positive": int(
            (
                predicted_active
                & reference_active
                & candidate_mask
            ).sum().detach().cpu()
        ),
        "activity_false_positive": int(
            (
                predicted_active
                & ~reference_active
                & candidate_mask
            ).sum().detach().cpu()
        ),
        "activity_false_negative": int(
            (
                ~predicted_active
                & reference_active
                & candidate_mask
            ).sum().detach().cpu()
        ),
        "wrong_string": int(
            (wrong_string & candidate_mask).sum().detach().cpu()
        ),
        "base_wrong_string": int(
            (base_wrong_string & candidate_mask).sum().detach().cpu()
        ),
        "repeat_true_positive": int(
            (
                correct_position
                & repeated
                & candidate_mask
            ).sum().detach().cpu()
        ),
        "repeat_false_negative": int(
            (
                ~correct_position
                & repeated
                & candidate_mask
            ).sum().detach().cpu()
        ),
    }


def combine_candidate_activity_confidence_logits(
    base_activity_logits: Tensor,
    confidence_logits: Tensor,
) -> Tensor:
    if base_activity_logits.shape != confidence_logits.shape:
        raise ValueError("activity and confidence logits must align")
    return torch.minimum(base_activity_logits, confidence_logits)


def candidate_sequence_confidence_loss(
    base_activity_logits: Tensor,
    base_string_logits: Tensor,
    valid_strings: Tensor,
    confidence_logits: Tensor,
    activity_targets: Tensor,
    string_targets: Tensor,
    repeat_targets: Tensor,
    candidate_mask: Tensor,
    *,
    repeat_positive_weight: float,
) -> tuple[Tensor, dict[str, int]]:
    candidate_shape = candidate_mask.shape
    if (
        base_activity_logits.shape != candidate_shape
        or base_string_logits.shape
        != candidate_shape + (STRING_COUNT,)
        or valid_strings.shape != base_string_logits.shape
        or confidence_logits.shape != candidate_shape
        or activity_targets.shape != candidate_shape
        or string_targets.shape != base_string_logits.shape
        or repeat_targets.shape != candidate_shape
    ):
        raise ValueError("sequence confidence tensors must align")
    if (
        not math.isfinite(repeat_positive_weight)
        or repeat_positive_weight < 1
    ):
        raise ValueError("repeat positive weight must be at least one")
    reference_active = activity_targets > 0
    string_target_mask = string_targets > 0
    if bool(
        (
            candidate_mask
            & (
                ((activity_targets != 0) & (activity_targets != 1))
                | ((repeat_targets != 0) & (repeat_targets != 1))
                | (repeat_targets > activity_targets)
                | (
                    reference_active
                    != string_target_mask.any(dim=-1)
                )
            )
        ).any()
    ):
        raise ValueError(
            "sequence confidence targets must be binary and aligned"
        )
    if bool(
        (
            candidate_mask.unsqueeze(-1)
            & string_target_mask
            & ~valid_strings
        ).any()
    ):
        raise ValueError("sequence confidence targets must be playable")

    base_string_predictions = base_string_logits.masked_fill(
        ~valid_strings,
        -30.0,
    ).argmax(dim=-1)
    base_string_matches = torch.gather(
        string_target_mask,
        -1,
        base_string_predictions.unsqueeze(-1),
    ).squeeze(-1)
    correctness_targets = (
        reference_active & base_string_matches
    ).to(confidence_logits.dtype)
    supervised_mask = candidate_mask & reference_active
    per_candidate_loss = functional.binary_cross_entropy_with_logits(
        confidence_logits,
        correctness_targets,
        reduction="none",
    )
    weights = torch.where(
        repeat_targets > 0,
        torch.full_like(
            per_candidate_loss,
            repeat_positive_weight,
        ),
        torch.ones_like(per_candidate_loss),
    )
    supervised_total = int(supervised_mask.sum().detach().cpu())
    if supervised_total == 0:
        loss = confidence_logits.sum() * 0
    else:
        loss = (per_candidate_loss * weights)[supervised_mask].mean()

    predicted_correct = confidence_logits >= 0
    accepted = (
        combine_candidate_activity_confidence_logits(
            base_activity_logits,
            confidence_logits,
        )
        >= 0
    )
    correct_position = accepted & reference_active & base_string_matches
    wrong_string = accepted & reference_active & ~base_string_matches
    joint_false_positive = accepted & (
        ~reference_active | ~base_string_matches
    )
    joint_false_negative = reference_active & (
        ~accepted | ~base_string_matches
    )
    repeated = repeat_targets > 0
    return loss, {
        "candidates": int(candidate_mask.sum().detach().cpu()),
        "supervised_active": supervised_total,
        "correctness_true_positive": int(
            (
                predicted_correct
                & base_string_matches
                & supervised_mask
            ).sum().detach().cpu()
        ),
        "correctness_false_positive": int(
            (
                predicted_correct
                & ~base_string_matches
                & supervised_mask
            ).sum().detach().cpu()
        ),
        "correctness_false_negative": int(
            (
                ~predicted_correct
                & base_string_matches
                & supervised_mask
            ).sum().detach().cpu()
        ),
        "joint_true_positive": int(
            (correct_position & candidate_mask).sum().detach().cpu()
        ),
        "joint_false_positive": int(
            (joint_false_positive & candidate_mask).sum().detach().cpu()
        ),
        "joint_false_negative": int(
            (joint_false_negative & candidate_mask).sum().detach().cpu()
        ),
        "wrong_string": int(
            (wrong_string & candidate_mask).sum().detach().cpu()
        ),
        "base_wrong_string": int(
            (
                reference_active
                & ~base_string_matches
                & candidate_mask
            ).sum().detach().cpu()
        ),
        "repeat_true_positive": int(
            (
                correct_position
                & repeated
                & candidate_mask
            ).sum().detach().cpu()
        ),
        "repeat_false_negative": int(
            (
                ~correct_position
                & repeated
                & candidate_mask
            ).sum().detach().cpu()
        ),
    }


def infer_candidate_sequence_residuals(
    model: CandidateSequenceStringModel,
    groups: list[CandidateSequenceGroup],
    device: torch.device,
    *,
    batch_size: int = 32,
) -> list[Tensor]:
    if not groups:
        return []
    if batch_size <= 0:
        raise ValueError("sequence inference batch size must be positive")
    annotation_indexes = {group.annotation_index for group in groups}
    if len(annotation_indexes) != 1:
        raise ValueError(
            "sequence inference groups must belong to one annotation"
        )
    ranges = sequence_window_ranges(
        len(groups),
        sequence_length=model.config.sequence_length,
        sequence_stride=model.config.sequence_stride,
    )
    sums = [
        torch.zeros(
            group.pitches.shape[0],
            STRING_COUNT,
            dtype=torch.float32,
        )
        for group in groups
    ]
    counts = torch.zeros(len(groups), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(ranges), batch_size):
            range_batch = ranges[offset : offset + batch_size]
            windows = [
                groups[start:end] for start, end in range_batch
            ]
            batch = {
                key: value.to(device)
                for key, value in collate_candidate_sequence_windows(
                    windows
                ).items()
            }
            residuals = model(
                batch["pitches"],
                batch["confidences"],
                batch["durations"],
                batch["base_string_logits"],
                batch["valid_strings"],
                batch["candidate_mask"],
                batch["group_mask"],
                batch["group_onsets"],
            ).cpu()
            for batch_index, (start, end) in enumerate(range_batch):
                for local_index, group_index in enumerate(
                    range(start, end)
                ):
                    count = groups[group_index].pitches.shape[0]
                    sums[group_index] += residuals[
                        batch_index,
                        local_index,
                        :count,
                    ]
                    counts[group_index] += 1
    return [
        residual_sum / counts[index].clamp(min=1)
        for index, residual_sum in enumerate(sums)
    ]


def infer_candidate_sequence_activity_residuals(
    model: CandidateSequenceActivityModel,
    groups: list[CandidateSequenceGroup],
    device: torch.device,
    *,
    batch_size: int = 32,
) -> list[Tensor]:
    if not groups:
        return []
    if batch_size <= 0:
        raise ValueError("sequence inference batch size must be positive")
    annotation_indexes = {group.annotation_index for group in groups}
    if len(annotation_indexes) != 1:
        raise ValueError(
            "sequence inference groups must belong to one annotation"
        )
    ranges = sequence_window_ranges(
        len(groups),
        sequence_length=model.config.sequence_length,
        sequence_stride=model.config.sequence_stride,
    )
    sums = [
        torch.zeros(group.pitches.shape[0], dtype=torch.float32)
        for group in groups
    ]
    counts = torch.zeros(len(groups), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(ranges), batch_size):
            range_batch = ranges[offset : offset + batch_size]
            windows = [groups[start:end] for start, end in range_batch]
            batch = {
                key: value.to(device)
                for key, value in collate_candidate_sequence_windows(
                    windows
                ).items()
            }
            residuals = model(
                batch["pitches"],
                batch["confidences"],
                batch["durations"],
                batch["base_string_logits"],
                batch["valid_strings"],
                batch["base_activity_logits"],
                batch["candidate_mask"],
                batch["group_mask"],
                batch["group_onsets"],
            ).cpu()
            for batch_index, (start, end) in enumerate(range_batch):
                for local_index, group_index in enumerate(range(start, end)):
                    count = groups[group_index].pitches.shape[0]
                    sums[group_index] += residuals[
                        batch_index,
                        local_index,
                        :count,
                    ]
                    counts[group_index] += 1
    return [
        residual_sum / counts[index].clamp(min=1)
        for index, residual_sum in enumerate(sums)
    ]


def infer_candidate_sequence_joint_residuals(
    model: CandidateSequenceJointModel,
    groups: list[CandidateSequenceGroup],
    device: torch.device,
    *,
    batch_size: int = 32,
) -> list[Tensor]:
    if not groups:
        return []
    if batch_size <= 0:
        raise ValueError("sequence inference batch size must be positive")
    annotation_indexes = {group.annotation_index for group in groups}
    if len(annotation_indexes) != 1:
        raise ValueError(
            "sequence inference groups must belong to one annotation"
        )
    ranges = sequence_window_ranges(
        len(groups),
        sequence_length=model.config.sequence_length,
        sequence_stride=model.config.sequence_stride,
    )
    sums = [
        torch.zeros(
            group.pitches.shape[0],
            STRING_COUNT + 1,
            dtype=torch.float32,
        )
        for group in groups
    ]
    counts = torch.zeros(len(groups), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(ranges), batch_size):
            range_batch = ranges[offset : offset + batch_size]
            windows = [groups[start:end] for start, end in range_batch]
            batch = {
                key: value.to(device)
                for key, value in collate_candidate_sequence_windows(
                    windows
                ).items()
            }
            residuals = model(
                batch["pitches"],
                batch["confidences"],
                batch["durations"],
                batch["base_string_logits"],
                batch["valid_strings"],
                batch["base_activity_logits"],
                batch["candidate_mask"],
                batch["group_mask"],
                batch["group_onsets"],
            ).cpu()
            for batch_index, (start, end) in enumerate(range_batch):
                for local_index, group_index in enumerate(range(start, end)):
                    count = groups[group_index].pitches.shape[0]
                    sums[group_index] += residuals[
                        batch_index,
                        local_index,
                        :count,
                    ]
                    counts[group_index] += 1
    return [
        residual_sum / counts[index].clamp(min=1)
        for index, residual_sum in enumerate(sums)
    ]


def infer_candidate_sequence_confidence_logits(
    model: CandidateSequenceConfidenceModel,
    groups: list[CandidateSequenceGroup],
    device: torch.device,
    *,
    batch_size: int = 32,
) -> list[Tensor]:
    return infer_candidate_sequence_activity_residuals(
        model,
        groups,
        device,
        batch_size=batch_size,
    )


def load_candidate_sequence_model(
    checkpoint_path: Path,
    position_checkpoint_path: Path,
    position_config: OnsetPositionConfig,
    feature_config: FeatureConfig,
    device: torch.device,
) -> tuple[CandidateSequenceStringModel, dict[str, object]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if int(checkpoint.get("format_version", 0)) != 1:
        raise ValueError("Unsupported candidate sequence checkpoint format")
    metadata = dict(checkpoint.get("metadata", {}))
    base_checkpoint = dict(metadata.get("base_checkpoint", {}))
    if base_checkpoint.get("sha256") != sha256_file(
        position_checkpoint_path
    ):
        raise ValueError(
            "Candidate sequence model targets a different position checkpoint"
        )
    expected_position_config = OnsetPositionConfig.from_dict(
        dict(checkpoint.get("position_config", {}))
    )
    if expected_position_config != position_config:
        raise ValueError(
            "Candidate sequence and position configurations differ"
        )
    expected_feature_config = FeatureConfig.from_dict(
        dict(checkpoint.get("feature_config", {}))
    )
    if expected_feature_config != feature_config:
        raise ValueError(
            "Candidate sequence and feature configurations differ"
        )
    sequence_config = CandidateSequenceConfig.from_dict(
        dict(checkpoint.get("sequence_config", {}))
    )
    model = CandidateSequenceStringModel(sequence_config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, metadata


def load_candidate_sequence_activity_model(
    checkpoint_path: Path,
    position_checkpoint_path: Path,
    string_sequence_checkpoint_path: Path,
    position_config: OnsetPositionConfig,
    feature_config: FeatureConfig,
    device: torch.device,
) -> tuple[CandidateSequenceActivityModel, dict[str, object]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if int(checkpoint.get("format_version", 0)) != 1:
        raise ValueError(
            "Unsupported candidate sequence activity checkpoint format"
        )
    metadata = dict(checkpoint.get("metadata", {}))
    base_checkpoint = dict(metadata.get("base_checkpoint", {}))
    if base_checkpoint.get("sha256") != sha256_file(
        position_checkpoint_path
    ):
        raise ValueError(
            "Candidate sequence activity model targets a different "
            "position checkpoint"
        )
    string_checkpoint = dict(
        metadata.get("string_sequence_checkpoint", {})
    )
    if string_checkpoint.get("sha256") != sha256_file(
        string_sequence_checkpoint_path
    ):
        raise ValueError(
            "Candidate sequence activity model targets a different "
            "string sequence checkpoint"
        )
    expected_position_config = OnsetPositionConfig.from_dict(
        dict(checkpoint.get("position_config", {}))
    )
    if expected_position_config != position_config:
        raise ValueError(
            "Candidate sequence activity and position configurations differ"
        )
    expected_feature_config = FeatureConfig.from_dict(
        dict(checkpoint.get("feature_config", {}))
    )
    if expected_feature_config != feature_config:
        raise ValueError(
            "Candidate sequence activity and feature configurations differ"
        )
    sequence_config = CandidateSequenceConfig.from_dict(
        dict(checkpoint.get("sequence_config", {}))
    )
    model = CandidateSequenceActivityModel(sequence_config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, metadata


def load_candidate_sequence_joint_model(
    checkpoint_path: Path,
    position_checkpoint_path: Path,
    string_sequence_checkpoint_path: Path,
    position_config: OnsetPositionConfig,
    feature_config: FeatureConfig,
    device: torch.device,
) -> tuple[CandidateSequenceJointModel, dict[str, object]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if int(checkpoint.get("format_version", 0)) != 1:
        raise ValueError(
            "Unsupported candidate sequence joint checkpoint format"
        )
    metadata = dict(checkpoint.get("metadata", {}))
    base_checkpoint = dict(metadata.get("base_checkpoint", {}))
    if base_checkpoint.get("sha256") != sha256_file(
        position_checkpoint_path
    ):
        raise ValueError(
            "Candidate sequence joint model targets a different "
            "position checkpoint"
        )
    string_checkpoint = dict(
        metadata.get("string_sequence_checkpoint", {})
    )
    if string_checkpoint.get("sha256") != sha256_file(
        string_sequence_checkpoint_path
    ):
        raise ValueError(
            "Candidate sequence joint model targets a different "
            "string sequence checkpoint"
        )
    expected_position_config = OnsetPositionConfig.from_dict(
        dict(checkpoint.get("position_config", {}))
    )
    if expected_position_config != position_config:
        raise ValueError(
            "Candidate sequence joint and position configurations differ"
        )
    expected_feature_config = FeatureConfig.from_dict(
        dict(checkpoint.get("feature_config", {}))
    )
    if expected_feature_config != feature_config:
        raise ValueError(
            "Candidate sequence joint and feature configurations differ"
        )
    sequence_config = CandidateSequenceConfig.from_dict(
        dict(checkpoint.get("sequence_config", {}))
    )
    model = CandidateSequenceJointModel(sequence_config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, metadata


def load_candidate_sequence_confidence_model(
    checkpoint_path: Path,
    position_checkpoint_path: Path,
    string_sequence_checkpoint_path: Path,
    position_config: OnsetPositionConfig,
    feature_config: FeatureConfig,
    device: torch.device,
) -> tuple[CandidateSequenceConfidenceModel, dict[str, object]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if int(checkpoint.get("format_version", 0)) != 1:
        raise ValueError(
            "Unsupported candidate sequence confidence checkpoint format"
        )
    metadata = dict(checkpoint.get("metadata", {}))
    base_checkpoint = dict(metadata.get("base_checkpoint", {}))
    if base_checkpoint.get("sha256") != sha256_file(
        position_checkpoint_path
    ):
        raise ValueError(
            "Candidate sequence confidence model targets a different "
            "position checkpoint"
        )
    string_checkpoint = dict(
        metadata.get("string_sequence_checkpoint", {})
    )
    if string_checkpoint.get("sha256") != sha256_file(
        string_sequence_checkpoint_path
    ):
        raise ValueError(
            "Candidate sequence confidence model targets a different "
            "string sequence checkpoint"
        )
    expected_position_config = OnsetPositionConfig.from_dict(
        dict(checkpoint.get("position_config", {}))
    )
    if expected_position_config != position_config:
        raise ValueError(
            "Candidate sequence confidence and position configurations differ"
        )
    expected_feature_config = FeatureConfig.from_dict(
        dict(checkpoint.get("feature_config", {}))
    )
    if expected_feature_config != feature_config:
        raise ValueError(
            "Candidate sequence confidence and feature configurations differ"
        )
    sequence_config = CandidateSequenceConfig.from_dict(
        dict(checkpoint.get("sequence_config", {}))
    )
    identity_accept_probability = float(
        metadata.get("identity_accept_probability", 0.0)
    )
    model = CandidateSequenceConfidenceModel(
        sequence_config,
        identity_accept_probability=identity_accept_probability,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, metadata

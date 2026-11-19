from __future__ import annotations

import math
from dataclasses import asdict, dataclass
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
from .model import PITCH_COUNT, PITCH_MIN, STRING_COUNT
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


def precompute_candidate_sequence_groups(
    dataset: OnsetPositionDataset,
    model: OnsetPositionNet,
    device: torch.device,
    *,
    batch_size: int,
) -> list[CandidateSequenceGroup]:
    if batch_size <= 0:
        raise ValueError("candidate sequence batch size must be positive")
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

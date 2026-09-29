from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


STRING_COUNT = 6
FRET_COUNT = 25
FRET_STATE_COUNT = FRET_COUNT + 1
OPEN_STRING_PITCHES = (64, 59, 55, 50, 45, 40)
PITCH_MIN = 40
PITCH_MAX = 88
PITCH_COUNT = PITCH_MAX - PITCH_MIN + 1


@dataclass(frozen=True)
class ModelConfig:
    convolution_channels: tuple[int, int, int] = (24, 48, 96)
    frequency_bins: int = 216
    frequency_encoding: str = "position-aware"
    frequency_projection_channels: int = 8
    recurrent_size: int = 96
    recurrent_layers: int = 2
    dropout: float = 0.15
    pitch_conditioned_position: bool = False
    pitch_embedding_size: int = 24

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ModelConfig":
        channels = value.get("convolution_channels", [24, 48, 96])
        return cls(
            convolution_channels=tuple(int(channel) for channel in channels),
            frequency_bins=int(value.get("frequency_bins", 216)),
            frequency_encoding=str(value.get("frequency_encoding", "mean")),
            frequency_projection_channels=int(
                value.get("frequency_projection_channels", 8)
            ),
            recurrent_size=int(value.get("recurrent_size", 96)),
            recurrent_layers=int(value.get("recurrent_layers", 2)),
            dropout=float(value.get("dropout", 0.15)),
            pitch_conditioned_position=bool(
                value.get("pitch_conditioned_position", False)
            ),
            pitch_embedding_size=int(value.get("pitch_embedding_size", 24)),
        )

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["convolution_channels"] = list(self.convolution_channels)
        return result


class StringFretNet(nn.Module):
    """CQT encoder with separate activity and re-articulation heads."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        channels = (1, *self.config.convolution_channels)
        convolution_layers: list[nn.Module] = []
        for input_channels, output_channels in zip(channels, channels[1:]):
            convolution_layers.extend(
                [
                    nn.Conv2d(
                        input_channels,
                        output_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.BatchNorm2d(output_channels),
                    nn.SiLU(),
                    nn.MaxPool2d(kernel_size=(2, 1)),
                ]
            )
        self.encoder = nn.Sequential(*convolution_layers)
        if self.config.frequency_encoding == "mean":
            self.frequency_projection: nn.Module = nn.Identity()
            temporal_input_size = self.config.convolution_channels[-1]
        elif self.config.frequency_encoding == "position-aware":
            self.frequency_projection = nn.Sequential(
                nn.Conv2d(
                    self.config.convolution_channels[-1],
                    self.config.frequency_projection_channels,
                    kernel_size=1,
                ),
                nn.SiLU(),
            )
            pooled_frequency_bins = self.config.frequency_bins // 8
            temporal_input_size = (
                self.config.frequency_projection_channels
                * pooled_frequency_bins
            )
        else:
            raise ValueError(
                f"Unknown frequency encoding {self.config.frequency_encoding!r}"
            )
        self.temporal = nn.GRU(
            input_size=temporal_input_size,
            hidden_size=self.config.recurrent_size,
            num_layers=self.config.recurrent_layers,
            batch_first=True,
            bidirectional=True,
            dropout=(
                self.config.dropout
                if self.config.recurrent_layers > 1
                else 0.0
            ),
        )
        temporal_size = self.config.recurrent_size * 2
        self.dropout = nn.Dropout(self.config.dropout)
        self.fret_head = nn.Linear(
            temporal_size,
            STRING_COUNT * FRET_STATE_COUNT,
        )
        self.onset_head = nn.Linear(
            temporal_size,
            STRING_COUNT,
        )
        if self.config.pitch_conditioned_position:
            self.pitch_embedding: nn.Module | None = nn.Embedding(
                PITCH_COUNT,
                self.config.pitch_embedding_size,
            )
            self.position_head: nn.Module | None = nn.Sequential(
                nn.Linear(
                    temporal_size + self.config.pitch_embedding_size,
                    temporal_size,
                ),
                nn.SiLU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(temporal_size, STRING_COUNT),
            )
            valid_positions = torch.zeros(
                PITCH_COUNT,
                STRING_COUNT,
                dtype=torch.bool,
            )
            for pitch_index in range(PITCH_COUNT):
                pitch = PITCH_MIN + pitch_index
                for string_index, open_pitch in enumerate(OPEN_STRING_PITCHES):
                    valid_positions[pitch_index, string_index] = (
                        0 <= pitch - open_pitch < FRET_COUNT
                    )
            self.register_buffer(
                "valid_pitch_string_positions",
                valid_positions,
                persistent=False,
            )
        else:
            self.pitch_embedding = None
            self.position_head = None

    def forward(
        self,
        features: Tensor,
        *,
        position_queries: tuple[Tensor, Tensor, Tensor] | None = None,
        include_all_positions: bool = False,
    ) -> dict[str, Tensor]:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, frames, bins]")
        if features.shape[-1] != self.config.frequency_bins:
            raise ValueError(
                f"Expected {self.config.frequency_bins} frequency bins, "
                f"got {features.shape[-1]}"
            )
        encoded = self.encoder(features.transpose(1, 2).unsqueeze(1))
        if self.config.frequency_encoding == "mean":
            temporal_input = encoded.mean(dim=2).transpose(1, 2)
        else:
            projected = self.frequency_projection(encoded)
            temporal_input = projected.permute(0, 3, 1, 2).flatten(2)
        temporal_output, _state = self.temporal(temporal_input)
        temporal_output = self.dropout(temporal_output)
        batch_size, frames, _channels = temporal_output.shape
        fret_logits = self.fret_head(temporal_output).reshape(
            batch_size,
            frames,
            STRING_COUNT,
            FRET_STATE_COUNT,
        )
        onset_logits = self.onset_head(temporal_output).reshape(
            batch_size,
            frames,
            STRING_COUNT,
        )
        outputs = {
            "fret_logits": fret_logits,
            "onset_logits": onset_logits,
        }
        if (
            self.position_head is not None
            and self.pitch_embedding is not None
            and position_queries is not None
        ):
            batch_indexes, frame_indexes, pitch_indexes = position_queries
            position_input = torch.cat(
                (
                    temporal_output[batch_indexes, frame_indexes],
                    self.pitch_embedding(pitch_indexes),
                ),
                dim=-1,
            )
            position_logits = self.position_head(position_input)
            outputs["position_logits"] = position_logits.masked_fill(
                ~self.valid_pitch_string_positions[pitch_indexes],
                -30.0,
            )
        elif (
            self.position_head is not None
            and self.pitch_embedding is not None
            and include_all_positions
        ):
            pitch_embeddings = self.pitch_embedding(
                torch.arange(PITCH_COUNT, device=features.device)
            )
            position_input = torch.cat(
                (
                    temporal_output.unsqueeze(2).expand(
                        -1,
                        -1,
                        PITCH_COUNT,
                        -1,
                    ),
                    pitch_embeddings.view(
                        1,
                        1,
                        PITCH_COUNT,
                        self.config.pitch_embedding_size,
                    ).expand(batch_size, frames, -1, -1),
                ),
                dim=-1,
            )
            position_logits = self.position_head(position_input)
            outputs["position_logits"] = position_logits.masked_fill(
                ~self.valid_pitch_string_positions.view(
                    1,
                    1,
                    PITCH_COUNT,
                    STRING_COUNT,
                ),
                -30.0,
            )
        return outputs


def transcription_loss(
    outputs: dict[str, Tensor],
    fret_targets: Tensor,
    onset_targets: Tensor,
    valid_frames: Tensor,
    *,
    onset_positive_weight: float = 20.0,
    attack_fret_weight: float = 2.0,
    attack_fret_frames: int = 6,
    fret_distance_weight: float = 0.0,
    pitch_consistency_weight: float = 0.0,
    pitch_positive_weight: float = 5.0,
    position_loss_weight: float = 0.0,
    string_loss_weights: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    fret_logits = outputs["fret_logits"]
    onset_logits = outputs["onset_logits"]
    frame_mask = valid_frames.to(fret_logits.dtype).unsqueeze(-1)
    if fret_distance_weight < 0:
        raise ValueError("fret_distance_weight must be non-negative")
    if pitch_consistency_weight < 0:
        raise ValueError("pitch_consistency_weight must be non-negative")
    if pitch_positive_weight < 1:
        raise ValueError("pitch_positive_weight must be at least one")
    if position_loss_weight < 0:
        raise ValueError("position_loss_weight must be non-negative")
    if string_loss_weights is None:
        string_weights = torch.ones(
            STRING_COUNT,
            device=fret_logits.device,
            dtype=fret_logits.dtype,
        )
    else:
        if string_loss_weights.shape != (STRING_COUNT,):
            raise ValueError("string_loss_weights must contain six values")
        string_weights = string_loss_weights.to(
            device=fret_logits.device,
            dtype=fret_logits.dtype,
        )

    class_weights = torch.ones(
        FRET_STATE_COUNT,
        device=fret_logits.device,
        dtype=fret_logits.dtype,
    )
    class_weights[0] = 0.25
    fret_loss_values = functional.cross_entropy(
        fret_logits.reshape(-1, FRET_STATE_COUNT),
        fret_targets.reshape(-1),
        weight=class_weights,
        reduction="none",
    ).reshape_as(fret_targets)
    attack_mask = torch.zeros_like(onset_targets)
    for offset in range(max(1, attack_fret_frames)):
        if offset == 0:
            attack_mask = torch.maximum(attack_mask, onset_targets)
        else:
            attack_mask[:, offset:] = torch.maximum(
                attack_mask[:, offset:],
                onset_targets[:, :-offset],
            )
    attack_mask = attack_mask * (fret_targets > 0).to(fret_logits.dtype)
    active_string_weights = torch.where(
        fret_targets > 0,
        string_weights.view(1, 1, STRING_COUNT),
        torch.ones_like(fret_targets, dtype=fret_logits.dtype),
    )
    fret_weights = frame_mask * active_string_weights * (
        1 + attack_mask * (attack_fret_weight - 1)
    )
    fret_loss = (fret_loss_values * fret_weights).sum() / (
        fret_weights.sum()
    ).clamp_min(1)

    active_mask = (
        (fret_targets > 0).to(fret_logits.dtype)
        * frame_mask
        * string_weights.view(1, 1, STRING_COUNT)
    )
    active_probabilities = torch.softmax(fret_logits, dim=-1)[..., 1:]
    conditional_probabilities = active_probabilities / (
        active_probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    )
    fret_values = torch.arange(
        FRET_COUNT,
        device=fret_logits.device,
        dtype=fret_logits.dtype,
    )
    expected_frets = (
        conditional_probabilities * fret_values
    ).sum(dim=-1)
    target_frets = (fret_targets - 1).clamp_min(0).to(fret_logits.dtype)
    fret_distance_loss = (
        (
            (expected_frets - target_frets).abs()
            / max(1, FRET_COUNT - 1)
        )
        * active_mask
    ).sum() / active_mask.sum().clamp_min(1)

    pitch_scores = torch.zeros(
        *fret_logits.shape[:2],
        PITCH_COUNT,
        device=fret_logits.device,
        dtype=fret_logits.dtype,
    )
    pitch_targets = torch.zeros_like(pitch_scores)
    for string_index, open_pitch in enumerate(OPEN_STRING_PITCHES):
        pitch_indexes = (
            torch.arange(
                FRET_COUNT,
                device=fret_logits.device,
            )
            + open_pitch
            - PITCH_MIN
        )
        pitch_scores.scatter_add_(
            -1,
            pitch_indexes.view(1, 1, -1).expand(
                *fret_logits.shape[:2],
                -1,
            ),
            active_probabilities[:, :, string_index],
        )
        target_active = fret_targets[:, :, string_index] > 0
        target_pitch_indexes = (
            fret_targets[:, :, string_index]
            - 1
            + open_pitch
            - PITCH_MIN
        ).clamp(0, PITCH_COUNT - 1)
        pitch_targets.scatter_add_(
            -1,
            target_pitch_indexes.unsqueeze(-1),
            target_active.to(fret_logits.dtype).unsqueeze(-1),
        )
    pitch_probabilities = 1 - torch.exp(-pitch_scores)
    pitch_targets = pitch_targets.clamp_max(1)
    pitch_loss_values = functional.binary_cross_entropy(
        pitch_probabilities.clamp(1e-6, 1 - 1e-6),
        pitch_targets,
        reduction="none",
    )
    pitch_weights = (
        1 + pitch_targets * (pitch_positive_weight - 1)
    )
    pitch_frame_mask = valid_frames.to(fret_logits.dtype).unsqueeze(-1)
    pitch_consistency_loss = (
        pitch_loss_values * pitch_weights * pitch_frame_mask
    ).sum() / (
        pitch_weights * pitch_frame_mask
    ).sum().clamp_min(1)

    position_logits = outputs.get("position_logits")
    position_event_mask = (
        (onset_targets > 0)
        & (fret_targets > 0)
        & valid_frames.unsqueeze(-1)
    )
    if position_logits is None:
        if position_loss_weight > 0:
            raise ValueError(
                "position_loss_weight requires a pitch-conditioned position head"
            )
        position_loss = torch.zeros((), device=fret_logits.device)
    elif position_event_mask.any():
        batch_indexes, frame_indexes, string_indexes = torch.nonzero(
            position_event_mask,
            as_tuple=True,
        )
        open_pitches = torch.tensor(
            OPEN_STRING_PITCHES,
            device=fret_logits.device,
        )
        pitch_indexes = (
            fret_targets[batch_indexes, frame_indexes, string_indexes]
            - 1
            + open_pitches[string_indexes]
            - PITCH_MIN
        )
        event_position_logits = (
            position_logits[
                batch_indexes,
                frame_indexes,
                pitch_indexes,
            ]
            if position_logits.ndim == 4
            else position_logits
        )
        if event_position_logits.shape != (batch_indexes.numel(), STRING_COUNT):
            raise ValueError(
                "position_logits do not match the active onset events"
            )
        position_loss = functional.cross_entropy(
            event_position_logits,
            string_indexes,
            weight=string_weights,
        )
    else:
        position_loss = position_logits.sum() * 0

    onset_loss_values = functional.binary_cross_entropy_with_logits(
        onset_logits,
        onset_targets,
        reduction="none",
    )
    onset_weights = (
        1 + onset_targets * (onset_positive_weight - 1)
    ) * torch.where(
        onset_targets > 0,
        string_weights.view(1, 1, STRING_COUNT),
        torch.ones_like(onset_targets),
    )
    onset_loss = (onset_loss_values * onset_weights * frame_mask).sum() / (
        frame_mask.sum() * STRING_COUNT
    ).clamp_min(1)

    total = (
        fret_loss
        + onset_loss
        + fret_distance_loss * fret_distance_weight
        + pitch_consistency_loss * pitch_consistency_weight
        + position_loss * position_loss_weight
    )
    return total, {
        "loss": total.detach(),
        "fret_loss": fret_loss.detach(),
        "onset_loss": onset_loss.detach(),
        "fret_distance_loss": fret_distance_loss.detach(),
        "pitch_consistency_loss": pitch_consistency_loss.detach(),
        "position_loss": position_loss.detach(),
    }

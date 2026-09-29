from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .features import FeatureConfig, extract_file_cqt
from .model import PITCH_COUNT, ModelConfig, StringFretNet
from .schema import LabeledEvent


@dataclass(frozen=True)
class LoadedModel:
    model: StringFretNet
    feature_config: FeatureConfig
    metadata: dict[str, object]


def load_checkpoint(path: Path, device: torch.device) -> LoadedModel:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_config = ModelConfig.from_dict(checkpoint.get("model_config", {}))
    feature_config = FeatureConfig.from_dict(
        checkpoint.get("feature_config", {})
    )
    model = StringFretNet(model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return LoadedModel(
        model=model,
        feature_config=feature_config,
        metadata=dict(checkpoint.get("metadata", {})),
    )


def _chunk_starts(frame_count: int, chunk_frames: int, overlap_frames: int) -> list[int]:
    if frame_count <= chunk_frames:
        return [0]
    step = max(1, chunk_frames - overlap_frames)
    starts = list(range(0, frame_count - chunk_frames + 1, step))
    final_start = frame_count - chunk_frames
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def predict_probabilities(
    loaded: LoadedModel,
    features: np.ndarray,
    *,
    device: torch.device,
    chunk_frames: int = 512,
    overlap_frames: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    fret_probabilities, onset_probabilities, _position_probabilities = (
        predict_probabilities_with_position(
            loaded,
            features,
            device=device,
            chunk_frames=chunk_frames,
            overlap_frames=overlap_frames,
        )
    )
    return fret_probabilities, onset_probabilities


def predict_probabilities_with_position(
    loaded: LoadedModel,
    features: np.ndarray,
    *,
    device: torch.device,
    chunk_frames: int = 512,
    overlap_frames: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    frame_count = features.shape[0]
    fret_sum = np.zeros((frame_count, 6, 26), dtype=np.float32)
    onset_sum = np.zeros((frame_count, 6), dtype=np.float32)
    position_sum = (
        np.zeros((frame_count, PITCH_COUNT, 6), dtype=np.float32)
        if loaded.model.config.pitch_conditioned_position
        else None
    )
    counts = np.zeros((frame_count, 1), dtype=np.float32)
    with torch.no_grad():
        for start in _chunk_starts(frame_count, chunk_frames, overlap_frames):
            end = min(frame_count, start + chunk_frames)
            batch = torch.from_numpy(features[start:end]).unsqueeze(0).to(device)
            outputs = loaded.model(
                batch,
                include_all_positions=position_sum is not None,
            )
            fret_probabilities = (
                torch.softmax(outputs["fret_logits"], dim=-1)[0]
                .detach()
                .cpu()
                .numpy()
            )
            onset_probabilities = (
                torch.sigmoid(outputs["onset_logits"])[0]
                .detach()
                .cpu()
                .numpy()
            )
            fret_sum[start:end] += fret_probabilities
            onset_sum[start:end] += onset_probabilities
            if position_sum is not None:
                position_sum[start:end] += (
                    torch.softmax(outputs["position_logits"], dim=-1)[0]
                    .detach()
                    .cpu()
                    .numpy()
                )
            counts[start:end] += 1
    return (
        fret_sum / counts[:, :, None],
        onset_sum / counts,
        (
            position_sum / counts[:, :, None]
            if position_sum is not None
            else None
        ),
    )


def decode_events(
    fret_probabilities: np.ndarray,
    onset_probabilities: np.ndarray,
    feature_config: FeatureConfig,
    *,
    onset_threshold: float = 0.60,
    activity_threshold: float = 0.25,
    minimum_duration: float = 0.045,
    minimum_onset_gap: float = 0.035,
    activity_context: float = 0.046,
    activity_delay: float = 0.046,
    minimum_event_confidence: float = 0.42,
) -> list[LabeledEvent]:
    fret_states = np.argmax(fret_probabilities, axis=-1)
    frame_count = fret_probabilities.shape[0]
    seconds_per_frame = 1 / feature_config.frames_per_second
    minimum_gap_frames = max(
        1,
        round(minimum_onset_gap * feature_config.frames_per_second),
    )
    activity_context_frames = max(
        1,
        round(activity_context * feature_config.frames_per_second),
    )
    activity_delay_frames = max(
        0,
        round(activity_delay * feature_config.frames_per_second),
    )
    events: list[LabeledEvent] = []
    for string_index in range(6):
        onset_frames: list[tuple[int, int, float, float]] = []
        for frame in range(frame_count):
            onset_confidence = float(onset_probabilities[frame, string_index])
            previous = (
                float(onset_probabilities[frame - 1, string_index])
                if frame > 0
                else 0.0
            )
            following = (
                float(onset_probabilities[frame + 1, string_index])
                if frame + 1 < frame_count
                else 0.0
            )
            if (
                onset_confidence < onset_threshold
                or onset_confidence < previous
                or onset_confidence < following
            ):
                continue
            activity_start = min(
                frame_count - 1,
                frame + activity_delay_frames,
            )
            activity_end = min(
                frame_count,
                activity_start + activity_context_frames,
            )
            activity_probabilities = np.mean(
                fret_probabilities[
                    activity_start:activity_end,
                    string_index,
                ],
                axis=0,
            )
            active_state = int(np.argmax(activity_probabilities))
            if active_state == 0:
                active_state = (
                    int(np.argmax(activity_probabilities[1:])) + 1
                )
            fret = active_state - 1
            fret_activity = float(activity_probabilities[active_state])
            event_confidence = (
                onset_confidence * fret_activity
            ) ** 0.5
            if (
                fret_activity < activity_threshold
                or event_confidence < minimum_event_confidence
            ):
                continue
            candidate = (frame, fret, onset_confidence, fret_activity)
            if (
                onset_frames
                and frame - onset_frames[-1][0] < minimum_gap_frames
            ):
                (
                    _previous_frame,
                    _previous_fret,
                    previous_confidence,
                    previous_activity,
                ) = onset_frames[-1]
                if (
                    onset_confidence * fret_activity
                    > previous_confidence * previous_activity
                ):
                    onset_frames[-1] = candidate
                continue
            onset_frames.append(candidate)

        for event_index, (
            frame,
            fret,
            onset_confidence,
            activity_confidence,
        ) in enumerate(onset_frames):
            next_onset_frame = (
                onset_frames[event_index + 1][0]
                if event_index + 1 < len(onset_frames)
                else frame_count
            )
            active_state = fret + 1
            end_frame = frame + 1
            while (
                end_frame < next_onset_frame
                and end_frame < frame_count
                and fret_states[end_frame, string_index] == active_state
            ):
                end_frame += 1
            onset = frame * seconds_per_frame
            offset = max(
                onset + minimum_duration,
                end_frame * seconds_per_frame,
            )
            events.append(
                LabeledEvent(
                    onset=round(onset, 4),
                    offset=round(offset, 4),
                    string=string_index + 1,
                    fret=fret,
                    technique="unknown",
                    confidence=round(
                        (onset_confidence * activity_confidence) ** 0.5,
                        4,
                    ),
                )
            )
    return sorted(events, key=lambda event: (event.onset, event.string))


def transcribe_audio(
    audio_path: Path,
    checkpoint_path: Path,
    *,
    device_name: str = "cpu",
    onset_threshold: float = 0.60,
    activity_threshold: float = 0.25,
) -> tuple[list[LabeledEvent], dict[str, object]]:
    device = torch.device(device_name)
    loaded = load_checkpoint(checkpoint_path, device)
    features = extract_file_cqt(audio_path, loaded.feature_config)
    fret_probabilities, onset_probabilities = predict_probabilities(
        loaded,
        features,
        device=device,
    )
    return (
        decode_events(
            fret_probabilities,
            onset_probabilities,
            loaded.feature_config,
            onset_threshold=onset_threshold,
            activity_threshold=activity_threshold,
        ),
        loaded.metadata,
    )

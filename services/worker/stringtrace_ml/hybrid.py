from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass

import numpy as np

from .features import FeatureConfig
from .model import FRET_COUNT, OPEN_STRING_PITCHES, PITCH_COUNT, PITCH_MIN
from .schema import LabeledEvent


@dataclass(frozen=True)
class PositionAssignmentConfig:
    group_tolerance: float = 0.045
    activity_delay: float = 0.023
    activity_context: float = 0.046
    onset_context: float = 0.023
    onset_weight: float = 1.0
    position_head_weight: float = 1.0
    continuity_weight: float = 0.5

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class PitchEvent:
    onset: float
    offset: float
    pitch: int
    confidence: float = 1.0
    technique: str = "unknown"


def candidate_positions(pitch: int) -> list[tuple[int, int]]:
    return [
        (string, pitch - open_pitch)
        for string, open_pitch in enumerate(OPEN_STRING_PITCHES, start=1)
        if 0 <= pitch - open_pitch < FRET_COUNT
    ]


def group_pitch_events(
    events: list[PitchEvent],
    tolerance: float,
) -> list[list[PitchEvent]]:
    groups: list[list[PitchEvent]] = []
    for event in sorted(events, key=lambda item: (item.onset, item.pitch)):
        if not groups or event.onset - groups[-1][0].onset > tolerance:
            groups.append([event])
        else:
            groups[-1].append(event)
    return groups


def _position_evidence(
    *,
    onset: float,
    pitch: int,
    string: int,
    fret: int,
    fret_probabilities: np.ndarray,
    onset_probabilities: np.ndarray,
    position_probabilities: np.ndarray | None,
    feature_config: FeatureConfig,
    config: PositionAssignmentConfig,
) -> tuple[float, float, float]:
    frames_per_second = feature_config.frames_per_second
    frame_count = fret_probabilities.shape[0]
    activity_start = max(
        0,
        min(
            frame_count - 1,
            round((onset + config.activity_delay) * frames_per_second),
        ),
    )
    activity_frames = max(
        1,
        round(config.activity_context * frames_per_second),
    )
    activity_end = min(frame_count, activity_start + activity_frames)
    activity = float(
        fret_probabilities[
            activity_start:activity_end,
            string - 1,
            fret + 1,
        ].mean()
    )

    onset_frame = max(
        0,
        min(frame_count - 1, round(onset * frames_per_second)),
    )
    onset_radius = max(
        0,
        round(config.onset_context * frames_per_second),
    )
    onset_start = max(0, onset_frame - onset_radius)
    onset_end = min(frame_count, onset_frame + onset_radius + 1)
    onset_evidence = float(
        onset_probabilities[onset_start:onset_end, string - 1].max()
    )
    position_evidence = (
        float(
            position_probabilities[
                onset_start:onset_end,
                pitch - PITCH_MIN,
                string - 1,
            ].max()
        )
        if position_probabilities is not None
        else 1.0
    )
    return activity, onset_evidence, position_evidence


def _continuity_cost(
    pitches: list[int],
    positions: tuple[tuple[int, int], ...],
    previous_positions: list[tuple[int, int]],
) -> float:
    cost = 0.0
    fretted = [fret for _string, fret in positions if fret > 0]
    if fretted:
        cost += max(0, max(fretted) - min(fretted) - 4)

    ordered = sorted(zip(pitches, positions), key=lambda item: item[0])
    cost += sum(
        right_position[0] >= left_position[0]
        for (_left_pitch, left_position), (_right_pitch, right_position)
        in zip(ordered, ordered[1:])
    )
    if previous_positions:
        previous_mean_fret = sum(
            fret for _string, fret in previous_positions
        ) / len(previous_positions)
        current_mean_fret = sum(
            fret for _string, fret in positions
        ) / len(positions)
        cost += abs(current_mean_fret - previous_mean_fret) * 0.25
    return cost


def _position_combinations(
    candidate_sets: list[list[tuple[int, int]]],
) -> list[tuple[tuple[int, int], ...]]:
    if len(candidate_sets) > len(OPEN_STRING_PITCHES):
        return [
            tuple(
                min(candidates, key=lambda position: (position[1], position[0]))
                for candidates in candidate_sets
            )
        ]
    combinations = [
        positions
        for positions in itertools.product(*candidate_sets)
        if len({position[0] for position in positions}) == len(positions)
    ]
    if combinations:
        return combinations
    return list(itertools.product(*candidate_sets))


def assign_pitch_events(
    pitch_events: list[PitchEvent],
    fret_probabilities: np.ndarray,
    onset_probabilities: np.ndarray,
    feature_config: FeatureConfig,
    *,
    position_probabilities: np.ndarray | None = None,
    config: PositionAssignmentConfig | None = None,
) -> list[LabeledEvent]:
    resolved_config = config or PositionAssignmentConfig()
    if fret_probabilities.ndim != 3 or fret_probabilities.shape[1:] != (6, 26):
        raise ValueError("fret_probabilities must have shape [frames, 6, 26]")
    if onset_probabilities.shape != fret_probabilities.shape[:2]:
        raise ValueError("onset_probabilities must have shape [frames, 6]")
    if (
        position_probabilities is not None
        and position_probabilities.shape
        != (fret_probabilities.shape[0], PITCH_COUNT, 6)
    ):
        raise ValueError(
            "position_probabilities must have shape [frames, 49, 6]"
        )

    assigned: list[LabeledEvent] = []
    previous_positions: list[tuple[int, int]] = []
    for group in group_pitch_events(
        pitch_events,
        resolved_config.group_tolerance,
    ):
        pitches = [event.pitch for event in group]
        candidate_sets = [candidate_positions(pitch) for pitch in pitches]
        if any(not candidates for candidates in candidate_sets):
            continue
        combinations = _position_combinations(candidate_sets)
        scored: list[
            tuple[
                float,
                tuple[tuple[int, int], ...],
                list[tuple[float, float, float]],
            ]
        ] = []
        for positions in combinations:
            evidence = [
                _position_evidence(
                    onset=event.onset,
                    pitch=pitch,
                    string=string,
                    fret=fret,
                    fret_probabilities=fret_probabilities,
                    onset_probabilities=onset_probabilities,
                    position_probabilities=position_probabilities,
                    feature_config=feature_config,
                    config=resolved_config,
                )
                for event, pitch, (string, fret) in zip(
                    group,
                    pitches,
                    positions,
                )
            ]
            score = sum(
                math.log(max(activity, 1e-8))
                + resolved_config.onset_weight
                * math.log(max(onset, 1e-8))
                + resolved_config.position_head_weight
                * math.log(max(position, 1e-8))
                for activity, onset, position in evidence
            )
            score -= resolved_config.continuity_weight * _continuity_cost(
                pitches,
                positions,
                previous_positions,
            )
            scored.append((score, positions, evidence))
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_positions, best_evidence = scored[0]
        score_margin = (
            best_score - scored[1][0]
            if len(scored) > 1
            else 4.0
        )
        assignment_confidence = 1 / (1 + math.exp(-score_margin))
        for event, (string, fret), (activity, onset, position) in zip(
            group,
            best_positions,
            best_evidence,
        ):
            confidence = (
                max(0.0, min(1.0, event.confidence))
                * max(0.0, min(1.0, activity))
                * max(0.0, min(1.0, onset))
                * max(0.0, min(1.0, position))
                * assignment_confidence
            ) ** 0.2
            assigned.append(
                LabeledEvent(
                    onset=event.onset,
                    offset=event.offset,
                    string=string,
                    fret=fret,
                    technique=event.technique,
                    confidence=round(confidence, 4),
                )
            )
        previous_positions = list(best_positions)
    return sorted(assigned, key=lambda event: (event.onset, event.string))

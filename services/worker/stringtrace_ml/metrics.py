from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import asdict, dataclass
from typing import Callable

from .schema import (
    ChordRegion,
    ExcludedRange,
    LabeledEvent,
    STANDARD_TUNING_LIST,
)


@dataclass(frozen=True)
class EventMetrics:
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    mean_onset_error_ms: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def _pitch(event: LabeledEvent) -> int:
    return STANDARD_TUNING_LIST[event.string - 1] + event.fret


def _exclude_events(
    events: list[LabeledEvent],
    excluded_ranges: list[ExcludedRange],
) -> tuple[list[LabeledEvent], int]:
    retained = [
        event
        for event in events
        if not any(
            excluded_range.start <= event.onset < excluded_range.end
            for excluded_range in excluded_ranges
        )
    ]
    return retained, len(events) - len(retained)


def _match(
    reference: list[LabeledEvent],
    predicted: list[LabeledEvent],
    compatible: Callable[[LabeledEvent, LabeledEvent], bool],
    onset_tolerance: float,
) -> EventMetrics:
    unmatched = set(range(len(reference)))
    onset_errors: list[float] = []
    true_positive = 0
    for prediction in sorted(predicted, key=lambda event: event.onset):
        candidates = [
            index
            for index in unmatched
            if compatible(reference[index], prediction)
            and abs(reference[index].onset - prediction.onset) <= onset_tolerance
        ]
        if not candidates:
            continue
        matched_index = min(
            candidates,
            key=lambda index: abs(reference[index].onset - prediction.onset),
        )
        unmatched.remove(matched_index)
        true_positive += 1
        onset_errors.append(
            abs(reference[matched_index].onset - prediction.onset) * 1000
        )
    false_positive = len(predicted) - true_positive
    false_negative = len(reference) - true_positive
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return EventMetrics(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_onset_error_ms=(
            sum(onset_errors) / len(onset_errors)
            if onset_errors
            else None
        ),
    )


def _consume_matches(
    reference: list[LabeledEvent],
    predicted: list[LabeledEvent],
    unmatched_reference: set[int],
    unmatched_predicted: set[int],
    compatible: Callable[[LabeledEvent, LabeledEvent], bool],
    onset_tolerance: float,
) -> list[tuple[int, int]]:
    matches: list[tuple[int, int]] = []
    for predicted_index in sorted(
        unmatched_predicted,
        key=lambda index: predicted[index].onset,
    ):
        candidates = [
            reference_index
            for reference_index in unmatched_reference
            if compatible(reference[reference_index], predicted[predicted_index])
            and abs(
                reference[reference_index].onset
                - predicted[predicted_index].onset
            )
            <= onset_tolerance
        ]
        if not candidates:
            continue
        reference_index = min(
            candidates,
            key=lambda index: abs(
                reference[index].onset - predicted[predicted_index].onset
            ),
        )
        unmatched_reference.remove(reference_index)
        unmatched_predicted.remove(predicted_index)
        matches.append((reference_index, predicted_index))
    return matches


def decoder_error_breakdown(
    reference: list[LabeledEvent],
    predicted: list[LabeledEvent],
    *,
    onset_tolerance: float = 0.05,
) -> dict[str, object]:
    unmatched_reference = set(range(len(reference)))
    unmatched_predicted = set(range(len(predicted)))
    categories = (
        (
            "exact_tablature",
            lambda ref, pred: ref.string == pred.string and ref.fret == pred.fret,
        ),
        (
            "same_pitch_wrong_string",
            lambda ref, pred: (
                _pitch(ref) == _pitch(pred) and ref.string != pred.string
            ),
        ),
        (
            "same_string_wrong_fret",
            lambda ref, pred: ref.string == pred.string,
        ),
        ("wrong_string_and_pitch", lambda _ref, _pred: True),
    )
    counts: dict[str, int] = {}
    string_confusion = [[0 for _predicted in range(6)] for _reference in range(6)]
    for name, compatible in categories:
        matches = _consume_matches(
            reference,
            predicted,
            unmatched_reference,
            unmatched_predicted,
            compatible,
            onset_tolerance,
        )
        counts[name] = len(matches)
        for reference_index, predicted_index in matches:
            reference_string = reference[reference_index].string - 1
            predicted_string = predicted[predicted_index].string - 1
            string_confusion[reference_string][predicted_string] += 1
    counts["spurious_onset"] = len(unmatched_predicted)
    counts["missed_onset"] = len(unmatched_reference)
    return {
        **counts,
        "string_confusion": string_confusion,
    }


def _beat_duration(onset: float, beats: list[float]) -> float:
    position = bisect_right(beats, onset)
    if position == 0:
        return beats[1] - beats[0]
    if position >= len(beats):
        return beats[-1] - beats[-2]
    return beats[position] - beats[position - 1]


def evaluate_beat_relative_onsets(
    reference: list[LabeledEvent],
    predicted: list[LabeledEvent],
    beats: list[float],
    *,
    tolerance_fraction: float = 0.125,
) -> dict[str, int | float | None]:
    unmatched = set(range(len(reference)))
    onset_errors: list[float] = []
    beat_errors: list[float] = []
    true_positive = 0
    for prediction in sorted(predicted, key=lambda event: event.onset):
        candidates = []
        for index in unmatched:
            beat_duration = _beat_duration(reference[index].onset, beats)
            onset_error = abs(reference[index].onset - prediction.onset)
            if onset_error <= beat_duration * tolerance_fraction:
                candidates.append((index, onset_error, beat_duration))
        if not candidates:
            continue
        matched_index, onset_error, beat_duration = min(
            candidates,
            key=lambda candidate: candidate[1],
        )
        unmatched.remove(matched_index)
        true_positive += 1
        onset_errors.append(onset_error * 1000)
        beat_errors.append(onset_error / beat_duration)
    false_positive = len(predicted) - true_positive
    false_negative = len(reference) - true_positive
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
        "mean_onset_error_ms": (
            sum(onset_errors) / len(onset_errors)
            if onset_errors
            else None
        ),
        "mean_beat_error": (
            sum(beat_errors) / len(beat_errors)
            if beat_errors
            else None
        ),
        "tolerance_fraction": tolerance_fraction,
    }


def evaluate_beats(
    reference: list[float],
    predicted: list[float],
    *,
    tolerance: float = 0.07,
) -> dict[str, int | float | None]:
    eligible_reference = sorted(
        float(beat)
        for beat in reference
        if math.isfinite(float(beat)) and float(beat) >= 0
    )
    eligible_predicted = sorted(
        float(beat)
        for beat in predicted
        if math.isfinite(float(beat)) and float(beat) >= 0
    )
    unmatched = set(range(len(eligible_reference)))
    errors: list[float] = []
    for prediction in eligible_predicted:
        candidates = [
            index
            for index in unmatched
            if abs(eligible_reference[index] - prediction) <= tolerance
        ]
        if not candidates:
            continue
        matched = min(
            candidates,
            key=lambda index: abs(eligible_reference[index] - prediction),
        )
        unmatched.remove(matched)
        errors.append(abs(eligible_reference[matched] - prediction) * 1000)
    true_positive = len(errors)
    false_positive = len(eligible_predicted) - true_positive
    false_negative = len(eligible_reference) - true_positive
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    sorted_errors = sorted(errors)

    def percentile(fraction: float) -> float | None:
        if not sorted_errors:
            return None
        position = (len(sorted_errors) - 1) * fraction
        lower = int(position)
        upper = min(len(sorted_errors) - 1, lower + 1)
        weight = position - lower
        return (
            sorted_errors[lower] * (1 - weight)
            + sorted_errors[upper] * weight
        )

    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
        "mean_error_ms": (
            sum(sorted_errors) / len(sorted_errors)
            if sorted_errors
            else None
        ),
        "median_error_ms": percentile(0.5),
        "p95_error_ms": percentile(0.95),
        "tolerance_ms": tolerance * 1000,
    }


def _chord_root(chord: str) -> str:
    normalized = chord.strip()
    if normalized in {"", "N.C.", "NC"}:
        return "N.C."
    if len(normalized) >= 2 and normalized[1] in {"#", "b"}:
        return normalized[:2]
    return normalized[:1]


def evaluate_chords(
    reference: list[ChordRegion],
    predicted: list[ChordRegion],
) -> dict[str, int | float]:
    boundaries = sorted(
        {
            boundary
            for region in [*reference, *predicted]
            for boundary in (region.start, region.end)
        }
    )
    reference_duration = 0.0
    predicted_duration = 0.0
    exact_duration = 0.0
    root_duration = 0.0
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        midpoint = (start + end) / 2
        reference_chord = next(
            (
                region.chord
                for region in reference
                if region.start <= midpoint < region.end
            ),
            None,
        )
        predicted_chord = next(
            (
                region.chord
                for region in predicted
                if region.start <= midpoint < region.end
            ),
            None,
        )
        duration = end - start
        if reference_chord is not None:
            reference_duration += duration
        if predicted_chord is not None:
            predicted_duration += duration
        if reference_chord is None or predicted_chord is None:
            continue
        if reference_chord == predicted_chord:
            exact_duration += duration
        if _chord_root(reference_chord) == _chord_root(predicted_chord):
            root_duration += duration
    overlap_denominator = max(reference_duration, predicted_duration, 1e-8)
    return {
        "reference_regions": len(reference),
        "predicted_regions": len(predicted),
        "reference_duration": reference_duration,
        "predicted_duration": predicted_duration,
        "coverage": min(1.0, predicted_duration / max(reference_duration, 1e-8)),
        "exact_accuracy": exact_duration / overlap_denominator,
        "root_accuracy": root_duration / overlap_denominator,
    }


def evaluate_events(
    reference: list[LabeledEvent],
    predicted: list[LabeledEvent],
    *,
    onset_tolerance: float = 0.05,
    beats: list[float] | None = None,
    excluded_ranges: list[ExcludedRange] | None = None,
    include_diagnostics: bool = True,
) -> dict[str, object]:
    ranges = excluded_ranges or []
    eligible_reference, excluded_reference_count = _exclude_events(
        reference,
        ranges,
    )
    eligible_predicted, excluded_predicted_count = _exclude_events(
        predicted,
        ranges,
    )
    repeated_reference = [
        event
        for index, event in enumerate(eligible_reference)
        if any(
            previous.string == event.string
            and previous.fret == event.fret
            and 0 < event.onset - previous.onset <= 0.75
            for previous in eligible_reference[:index]
        )
    ]
    technique_reference = [
        event
        for event in eligible_reference
        if event.technique != "unknown"
    ]
    metrics: dict[str, object] = {
        "exclusions": {
            "range_count": len(ranges),
            "duration_seconds": sum(
                excluded_range.end - excluded_range.start
                for excluded_range in ranges
            ),
            "excluded_reference_events": excluded_reference_count,
            "excluded_predicted_events": excluded_predicted_count,
        },
        "onset": _match(
            eligible_reference,
            eligible_predicted,
            lambda _reference, _prediction: True,
            onset_tolerance,
        ).to_dict(),
        "pitch": _match(
            eligible_reference,
            eligible_predicted,
            lambda reference_event, predicted_event: (
                _pitch(reference_event) == _pitch(predicted_event)
            ),
            onset_tolerance,
        ).to_dict(),
        "tablature": _match(
            eligible_reference,
            eligible_predicted,
            lambda reference_event, predicted_event: (
                reference_event.string == predicted_event.string
                and reference_event.fret == predicted_event.fret
            ),
            onset_tolerance,
        ).to_dict(),
        "repeated_tablature": _match(
            repeated_reference,
            eligible_predicted,
            lambda reference_event, predicted_event: (
                reference_event.string == predicted_event.string
                and reference_event.fret == predicted_event.fret
            ),
            onset_tolerance,
        ).to_dict(),
        "technique": _match(
            technique_reference,
            eligible_predicted,
            lambda reference_event, predicted_event: (
                reference_event.string == predicted_event.string
                and reference_event.fret == predicted_event.fret
                and reference_event.technique == predicted_event.technique
            ),
            onset_tolerance,
        ).to_dict(),
    }
    if include_diagnostics:
        metrics["decoder_errors"] = decoder_error_breakdown(
            eligible_reference,
            eligible_predicted,
            onset_tolerance=onset_tolerance,
        )
    if beats is not None and len(beats) >= 2:
        metrics["beat_relative_onset"] = evaluate_beat_relative_onsets(
            eligible_reference,
            eligible_predicted,
            beats,
        )
    return metrics

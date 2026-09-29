from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import librosa
import soundfile

from .reference_review import (
    load_approved_review,
    load_project,
    measure_clip,
    relative_beats,
    select_project_measures,
    sha256_file,
)
from .schema import (
    ExcludedRange,
    LabeledEvent,
    TrackAnnotation,
    load_manifest,
    write_manifest,
)

CONDITION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def _event_interval(
    note: dict[str, Any],
    clip_start: float,
    clip_end: float,
) -> tuple[float, float]:
    source_onset = float(note["at"])
    source_offset = source_onset + float(note["duration"])
    onset = max(0.0, source_onset - clip_start)
    offset = min(clip_end - clip_start, source_offset - clip_start)
    return round(onset, 6), round(max(onset + 0.01, offset), 6)


def source_verified_events(
    selected_measures: list[dict[str, Any]],
    clip_start: float,
    clip_end: float,
) -> tuple[list[LabeledEvent], bool]:
    events: list[LabeledEvent] = []
    technique_reviewed = True
    for measure in selected_measures:
        for beat in measure["beats"]:
            for note in beat["notes"]:
                position_source = str(note.get("positionSource", ""))
                if not position_source.startswith("source-score"):
                    raise RuntimeError(
                        "Selected measures contain non-source-verified notes; "
                        "use a complete review"
                    )
                technique_source = str(note.get("techniqueSource", ""))
                technique_is_verified = technique_source.startswith("source-score")
                technique_reviewed = (
                    technique_reviewed and technique_is_verified
                )
                onset, offset = _event_interval(note, clip_start, clip_end)
                events.append(
                    LabeledEvent(
                        onset=onset,
                        offset=offset,
                        string=int(note["string"]),
                        fret=int(note["fret"]),
                        technique=(
                            str(note.get("technique", "unknown"))
                            if technique_is_verified
                            else "unknown"
                        ),
                        confidence=1.0,
                    )
                )
    if not events:
        raise RuntimeError("No eligible source-verified notes were found")
    events.sort(key=lambda event: (event.onset, event.string))
    return events, technique_reviewed


def reviewed_events(review: dict[str, Any]) -> list[LabeledEvent]:
    technique_reviewed = bool(review["checks"]["technique"])
    events = [
        LabeledEvent(
            onset=float(value["onset"]),
            offset=float(value["offset"]),
            string=int(value["string"]),
            fret=int(value["fret"]),
            technique=(
                str(value.get("technique", "unknown"))
                if technique_reviewed
                else "unknown"
            ),
            confidence=1.0,
        )
        for value in review["events"]
    ]
    events.sort(key=lambda event: (event.onset, event.string))
    return events


def export_reference(
    *,
    project_path: Path,
    audio_path: Path,
    output: Path,
    measure_numbers: list[int] | None = None,
    review_path: Path | None = None,
    capo: int | None = None,
    allow_quarantined_source: bool = False,
    render_audio_path: Path | None = None,
    render_audio_offset: float = 0.0,
    audio_condition: str | None = None,
) -> TrackAnnotation:
    if not math.isfinite(render_audio_offset) or abs(render_audio_offset) > 1:
        raise ValueError("Render audio offset must be finite and within one second")
    if render_audio_path is None and (
        render_audio_offset != 0 or audio_condition is not None
    ):
        raise ValueError(
            "Render audio offset and condition require --render-audio"
        )
    if render_audio_path is not None:
        if review_path is None:
            raise ValueError("Derived audio export requires an approved review")
        if (
            audio_condition is None
            or not CONDITION_PATTERN.fullmatch(audio_condition)
        ):
            raise ValueError(
                "Derived audio export requires a safe audio condition"
            )
        if not render_audio_path.is_file():
            raise FileNotFoundError(render_audio_path)
    project = load_project(project_path)
    review = (
        load_approved_review(review_path, project_path, audio_path)
        if review_path is not None
        else None
    )
    if review is not None:
        selected_numbers = [int(number) for number in review["measure_numbers"]]
    else:
        if not allow_quarantined_source:
            raise ValueError(
                "Direct source-score export is disabled for evaluation; "
                "provide an approved --review or explicitly use quarantined "
                "research mode"
            )
        if not measure_numbers:
            raise ValueError("--measure is required without --review")
        selected_numbers = sorted(set(measure_numbers))
    selected_measures = select_project_measures(project, selected_numbers)
    clip_start, clip_end = measure_clip(selected_measures)

    if review is not None:
        if (
            abs(float(review["source_start"]) - clip_start) > 1e-6
            or abs(float(review["source_end"]) - clip_end) > 1e-6
        ):
            raise ValueError("Review clip no longer matches the selected measures")
        events = reviewed_events(review)
        excluded_ranges = [
            ExcludedRange.from_dict(value)
            for value in review.get("excluded_ranges", [])
        ]
        track_id = str(review["review_id"])
        resolved_capo = int(review["capo"])
        provenance: dict[str, object] = {
            "dataset": "stringtrace-reviewed-reference",
            "annotation_source": "complete-human-review",
            "verification_status": "approved-human-review",
            "review_schema_version": int(review["schema_version"]),
            "review_content_sha256": str(
                review["approval"]["content_sha256"]
            ),
            "project_sha256": str(review["project_sha256"]),
            "source_audio_sha256": str(review["source_audio_sha256"]),
            "technique_reviewed": bool(review["checks"]["technique"]),
            "alignment_audit": {
                "status": "approved",
                "method": "complete-human-review",
                "timing_reviewed": bool(review["checks"]["timing"]),
                "reviewer_alias": str(
                    review["approval"]["reviewer_alias"]
                ),
                "reviewed_at": str(review["approval"]["reviewed_at"]),
                "excluded_ranges": [
                    {
                        "start": excluded_range.start,
                        "end": excluded_range.end,
                        "reason": excluded_range.reason,
                    }
                    for excluded_range in excluded_ranges
                ],
                "excluded_duration_seconds": sum(
                    excluded_range.end - excluded_range.start
                    for excluded_range in excluded_ranges
                ),
            },
        }
        if render_audio_path is not None:
            provenance.update(
                {
                    "dataset": "stringtrace-derived-reviewed-reference",
                    "annotation_source": (
                        "approved-human-review-transferred-to-synchronized-audio"
                    ),
                    "verification_status": "derived-from-approved-human-review",
                    "audio_condition": audio_condition,
                    "derived_audio_sha256": sha256_file(render_audio_path),
                }
            )
            provenance["alignment_audit"]["derived_audio"] = {
                "status": "offset-compensated",
                "method": "onset-envelope-cross-correlation",
                "offset_seconds": render_audio_offset,
            }
        else:
            provenance["audio_condition"] = "source-mix"
    else:
        events, _technique_reviewed = source_verified_events(
            selected_measures,
            clip_start,
            clip_end,
        )
        events = [
            LabeledEvent(
                onset=event.onset,
                offset=event.offset,
                string=event.string,
                fret=event.fret,
                technique="unknown",
                confidence=event.confidence,
            )
            for event in events
        ]
        measure_suffix = "-".join(str(number) for number in selected_numbers)
        track_id = f"{project['id']}-m{measure_suffix}"
        excluded_ranges = []
        resolved_capo = (
            int(capo)
            if capo is not None
            else int(project["tab"].get("capo", 0))
        )
        provenance = {
            "dataset": "stringtrace-source-reference",
            "annotation_source": "source-score-reference-v1",
            "verification_status": "quarantined",
            "project_sha256": sha256_file(project_path),
            "source_audio_sha256": sha256_file(audio_path),
            "technique_reviewed": False,
            "alignment_audit": {
                "status": "not-reviewed",
                "method": "source-score-timing-only",
                "reason": (
                    "Source-score events have not been acoustically aligned "
                    "to the exported audio"
                ),
            },
        }

    output = output.resolve()
    audio_directory = output / "audio"
    audio_directory.mkdir(parents=True, exist_ok=True)
    audio_output = audio_directory / f"{track_id}.wav"
    rendered_source = render_audio_path or audio_path
    audio, _sample_rate = librosa.load(
        rendered_source,
        sr=22050,
        mono=True,
        offset=clip_start + render_audio_offset,
        duration=clip_end - clip_start,
    )
    soundfile.write(audio_output, audio, 22050, subtype="PCM_16")
    provenance.update(
        {
            "project_id": str(project["id"]),
            "source_kind": str(project.get("source", {}).get("kind", "unknown")),
            "measure_numbers": selected_numbers,
            "source_start": round(clip_start, 6),
            "source_end": round(clip_end, 6),
        }
    )
    annotation = TrackAnnotation(
        track_id=track_id,
        group_id=str(project["id"]),
        audio=str(audio_output.relative_to(output)),
        split="test",
        duration=round(clip_end - clip_start, 6),
        events=events,
        excluded_ranges=excluded_ranges,
        capo=resolved_capo,
        bpm=float(project["bpm"]),
        beats=relative_beats(selected_measures, clip_start, clip_end),
        time_signature=(4, 4),
        provenance=provenance,
    )
    annotation.validate()
    manifest_path = output / "manifest.jsonl"
    existing = load_manifest(manifest_path) if manifest_path.exists() else []
    retained = [
        item
        for item in existing
        if item.track_id != annotation.track_id
    ]
    write_manifest(manifest_path, [*retained, annotation])
    return annotation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export approved or source-verified excerpts as held-out data.",
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measure", type=int, action="append")
    parser.add_argument("--review", type=Path)
    parser.add_argument("--capo", type=int)
    parser.add_argument(
        "--render-audio",
        type=Path,
        help="Optional synchronized derivative audio used for the exported clip.",
    )
    parser.add_argument(
        "--render-offset",
        type=float,
        default=0.0,
        help="Seconds added to source timestamps when reading --render-audio.",
    )
    parser.add_argument(
        "--audio-condition",
        help="Safe label for a synchronized derivative audio condition.",
    )
    parser.add_argument(
        "--quarantined-source-score",
        action="store_true",
        help=(
            "Allow a source-score-only export marked as quarantined and "
            "ineligible for evaluation."
        ),
    )
    arguments = parser.parse_args()
    if arguments.review is not None and arguments.measure:
        parser.error("--measure cannot be combined with --review")
    if arguments.review is not None and arguments.capo is not None:
        parser.error("--capo is fixed by an approved review")
    if arguments.review is not None and arguments.quarantined_source_score:
        parser.error("--quarantined-source-score cannot be combined with --review")
    if arguments.render_audio is None and (
        arguments.render_offset != 0 or arguments.audio_condition is not None
    ):
        parser.error("--render-offset/--audio-condition require --render-audio")

    annotation = export_reference(
        project_path=arguments.project,
        audio_path=arguments.audio,
        output=arguments.output,
        measure_numbers=arguments.measure,
        review_path=arguments.review,
        capo=arguments.capo,
        allow_quarantined_source=arguments.quarantined_source_score,
        render_audio_path=arguments.render_audio,
        render_audio_offset=arguments.render_offset,
        audio_condition=arguments.audio_condition,
    )
    print(
        json.dumps(
            {
                "track_id": annotation.track_id,
                "events": len(annotation.events),
                "duration": annotation.duration,
                "split": annotation.split,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

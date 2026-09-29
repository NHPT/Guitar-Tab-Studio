from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pretty_midi
import soundfile
import torch

from stringtrace_ml.candidate_activity import (
    CandidateActivityConfig,
    CandidateActivityDataset,
    CandidateActivityHead,
    candidate_sampling_weights,
    candidate_temporal_context,
)
from stringtrace_ml.compose_training_manifest import (
    compose_training_manifests,
)
from stringtrace_ml.convert_technique_position_evaluation import (
    convert_technique_position_evaluation,
)
from stringtrace_ml.convert_technique_positions import (
    convert_technique_positions,
)
from stringtrace_ml.dataset import GuitarTabDataset
from stringtrace_ml.download_guitarset import (
    DownloadSpec,
    assemble_download,
    byte_ranges,
    part_path,
    validate_content_range,
)
from stringtrace_ml.evaluate import model_identity
from stringtrace_ml.evaluate_onset_position import (
    assign_detected_events,
    merge_predictions,
    parse_arguments,
    unmatched_onset_candidates,
)
from stringtrace_ml.evaluate_technique import (
    expected_calibration_error,
    select_confidence_gates,
    sha256_file,
)
from stringtrace_ml.features import FeatureConfig
from stringtrace_ml.import_guitarset import parse_jams
from stringtrace_ml.import_guitar_techs import (
    merge_gesture_fragments,
    midi_events,
    shifted_events,
)
from stringtrace_ml.import_idmt_techniques import parse_idmt_annotation
from stringtrace_ml.inference import (
    LoadedModel,
    decode_events,
    predict_probabilities,
    predict_probabilities_with_position,
)
from stringtrace_ml.hybrid import (
    PitchEvent,
    PositionAssignmentConfig,
    assign_pitch_events,
)
from stringtrace_ml.gate import evaluate_gate
from stringtrace_ml.import_agpt_techniques import normalized_pitch
from stringtrace_ml.metrics import evaluate_events
from stringtrace_ml.model import (
    OPEN_STRING_PITCHES,
    PITCH_MIN,
    ModelConfig,
    StringFretNet,
    transcription_loss,
)
from stringtrace_ml.onset_position import (
    OnsetPositionConfig,
    OnsetPositionDataset,
    OnsetPositionNet,
    OnsetGroupIndex,
    augment_pitch_candidates,
    assign_pitch_group,
    basic_pitch_cache_path,
    extract_onset_patch,
    group_event_indexes,
    pitch_mask,
)
from stringtrace_ml.prepare_reference import export_reference
from stringtrace_ml.prepare_guitarset_stems import (
    derived_annotation,
    relocated_source_annotation,
    stratified_selection,
    track_family,
)
from stringtrace_ml.prepare_mixture_stems import (
    mix_source_audio,
    procedural_accompaniment,
    stable_track_seed,
)
from stringtrace_ml.reference_coverage import build_coverage_report
from stringtrace_ml.reference_queue import (
    build_review_queue,
    priority_strings_from_coverage,
)
from stringtrace_ml.reference_review import (
    approve_review,
    create_review,
    load_approved_review,
)
from stringtrace_ml.residual_onset_review import (
    _seed_review_events,
    export_approved_residual_reviews,
    normalize_review_audio,
    paired_review_source,
    reference_candidate_difficulty,
    select_review_windows,
)
from stringtrace_ml.schema import (
    ExcludedRange,
    LabeledEvent,
    TrackAnnotation,
    load_manifest,
    reject_quarantined_references,
    write_manifest,
)
from stringtrace_ml.technique_schema import TechniqueEvent
from stringtrace_ml.technique_descriptors import (
    TechniqueDescriptorConfig,
    summarize_event_descriptors,
)
from stringtrace_ml.technique_gate import build_technique_gate
from stringtrace_ml.technique_inference import predict_techniques
from stringtrace_ml.technique_model import (
    TechniqueEventDataset,
    TechniqueFeatureConfig,
    TechniqueModelConfig,
    TechniqueNet,
)
from stringtrace_ml.technique_ovr import select_binary_threshold
from stringtrace_ml.technique_schema import (
    TechniqueTrack,
    write_technique_manifest,
)
from stringtrace_ml.train import (
    best_compatible_validation_loss,
    best_compatible_validation_metric,
    configure_trainable_parameters,
    parse_dataset_sampling_weights as parse_window_dataset_sampling_weights,
)
from stringtrace_ml.train_candidate_activity import (
    parse_dataset_sampling_weights as parse_candidate_dataset_sampling_weights,
)


class StringTraceMlTest(unittest.TestCase):
    def temporary_directory(self) -> tempfile.TemporaryDirectory[str]:
        return tempfile.TemporaryDirectory(dir=Path(__file__).parent)

    def test_residual_review_selection_uses_train_split_and_group_diversity(
        self,
    ) -> None:
        def annotation(
            track_id: str,
            group_id: str,
            split: str,
            onsets: list[float],
        ) -> TrackAnnotation:
            return TrackAnnotation(
                track_id=track_id,
                group_id=group_id,
                audio=f"{track_id}.wav",
                split=split,
                duration=20.0,
                events=[
                    LabeledEvent(
                        onset=onset,
                        offset=onset + 0.2,
                        string=2,
                        fret=3,
                    )
                    for onset in onsets
                ],
                provenance={
                    "dataset": "Guitar-TECHS",
                    "audio_condition": (
                        "procedural-mixture-demucs-guitar-stem"
                    ),
                },
            )

        annotations = [
            annotation(
                "group-a-direct",
                "group-a",
                "train",
                [1.0, 1.4, 1.8, 8.0],
            ),
            annotation(
                "group-a-mic",
                "group-a",
                "train",
                [2.0, 8.0],
            ),
            annotation(
                "group-b-direct",
                "group-b",
                "train",
                [3.0, 3.5],
            ),
            annotation(
                "validation-only",
                "group-c",
                "validation",
                [1.0, 1.2, 1.4, 1.6],
            ),
        ]

        selected = select_review_windows(
            annotations,
            dataset="Guitar-TECHS",
            audio_condition="procedural-mixture-demucs-guitar-stem",
            clip_seconds=4.0,
            repeat_interval=0.75,
            limit=2,
        )

        self.assertEqual(
            {
                annotations[item.annotation_index].group_id
                for item in selected
            },
            {"group-a", "group-b"},
        )
        self.assertTrue(
            all(
                annotations[item.annotation_index].split == "train"
                for item in selected
            )
        )

    def test_residual_review_source_pair_requires_sample_alignment(
        self,
    ) -> None:
        source = TrackAnnotation(
            track_id="source-track",
            group_id="source-group",
            audio="source.wav",
            split="train",
            duration=1.0,
            events=[],
        )
        derived = TrackAnnotation(
            track_id="derived-track",
            group_id="source-group",
            audio="derived.wav",
            split="train",
            duration=1.0,
            events=[],
            provenance={
                "derived_from_track_id": source.track_id,
                "annotation_alignment": "sample-aligned-no-offset",
            },
        )

        self.assertIs(
            paired_review_source(
                derived,
                {
                    source.track_id: source,
                    derived.track_id: derived,
                },
            ),
            source,
        )
        derived.provenance["annotation_alignment"] = "unknown"
        with self.assertRaisesRegex(ValueError, "not declared sample-aligned"):
            paired_review_source(
                derived,
                {
                    source.track_id: source,
                    derived.track_id: derived,
                },
            )

    def test_residual_review_selection_prioritizes_missing_candidates(
        self,
    ) -> None:
        def annotation(
            track_id: str,
            group_id: str,
            onset: float,
            fret: int,
        ) -> TrackAnnotation:
            return TrackAnnotation(
                track_id=track_id,
                group_id=group_id,
                audio=f"{track_id}.wav",
                split="train",
                duration=10.0,
                events=[
                    LabeledEvent(
                        onset=onset,
                        offset=onset + 0.2,
                        string=1,
                        fret=fret,
                    )
                ],
                provenance={
                    "dataset": "AG-PT-set",
                    "audio_condition": (
                        "procedural-mixture-demucs-guitar-stem"
                    ),
                },
            )

        annotations = [
            annotation("matched", "player-1", 1.0, 0),
            annotation("missing", "player-2", 2.0, 2),
        ]
        candidate_events = {
            "matched": [
                {
                    "onset": 1.01,
                    "pitch": 64,
                    "confidence": 0.9,
                }
            ],
            "missing": [],
        }

        selected = select_review_windows(
            annotations,
            dataset="AG-PT-set",
            audio_condition="procedural-mixture-demucs-guitar-stem",
            clip_seconds=4.0,
            repeat_interval=0.75,
            limit=1,
            candidate_events_by_track_id=candidate_events,
        )

        self.assertEqual(
            annotations[selected[0].annotation_index].track_id,
            "missing",
        )
        self.assertEqual(selected[0].missing_candidate_events, 1)
        self.assertEqual(selected[0].weak_candidate_events, 0)

    def test_residual_review_candidate_difficulty_marks_weak_matches(
        self,
    ) -> None:
        annotation = TrackAnnotation(
            track_id="difficulty",
            group_id="player-1",
            audio="difficulty.wav",
            split="train",
            duration=2.0,
            events=[
                LabeledEvent(0.2, 0.4, 1, 0),
                LabeledEvent(1.0, 1.2, 1, 2),
            ],
        )

        missing, weak = reference_candidate_difficulty(
            annotation,
            [
                {
                    "onset": 0.21,
                    "pitch": 64,
                    "confidence": 0.7,
                }
            ],
            onset_tolerance=0.05,
            weak_confidence_threshold=0.75,
        )

        self.assertEqual(missing, frozenset({1}))
        self.assertEqual(weak, frozenset({0}))

    def test_residual_review_filters_low_confidence_seed_candidates(
        self,
    ) -> None:
        seeded = _seed_review_events(
            [
                LabeledEvent(0.2, 0.4, 1, 0, confidence=0.4),
                LabeledEvent(0.8, 1.0, 1, 3, confidence=0.6),
            ],
            [],
            [64, 59, 55, 50, 45, 40],
            onset_tolerance=0.05,
            minimum_candidate_confidence=0.5,
        )

        self.assertEqual(len(seeded), 1)
        self.assertEqual(seeded[0]["fret"], 3)

    def test_residual_review_audio_normalization_preserves_samples(
        self,
    ) -> None:
        audio = np.full(1000, 0.01, dtype=np.float32)
        normalized, gain_db = normalize_review_audio(audio)

        self.assertEqual(normalized.shape, audio.shape)
        self.assertLessEqual(
            float(np.max(np.abs(normalized))),
            10 ** (-1.0 / 20.0),
        )
        normalized_rms = float(
            np.sqrt(np.mean(np.square(normalized)))
        )
        self.assertAlmostEqual(
            20.0 * np.log10(normalized_rms),
            -20.0,
            places=1,
        )
        self.assertAlmostEqual(gain_db, 20.0, places=5)

        quiet = np.full(1000, 0.0001, dtype=np.float32)
        normalized_quiet, quiet_gain_db = normalize_review_audio(quiet)
        quiet_rms = float(
            np.sqrt(np.mean(np.square(normalized_quiet)))
        )
        self.assertAlmostEqual(
            20.0 * np.log10(quiet_rms),
            -20.0,
            places=1,
        )
        self.assertAlmostEqual(quiet_gain_db, 60.0, places=5)

    def test_exports_only_approved_residual_reviews_as_training_data(
        self,
    ) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            source_audio = root / "source.wav"
            soundfile.write(
                source_audio,
                np.zeros(22050, dtype=np.float32),
                22050,
            )
            source_manifest = root / "source.jsonl"
            write_manifest(
                source_manifest,
                [
                    TrackAnnotation(
                        track_id="source-track",
                        group_id="source-group",
                        audio=source_audio.name,
                        split="train",
                        duration=1.0,
                        events=[],
                    )
                ],
            )
            review_directory = root / "reviews"
            review_directory.mkdir()
            review_audio = review_directory / "residual-onset-01.wav"
            soundfile.write(
                review_audio,
                np.full(22050, 0.25, dtype=np.float32),
                22050,
            )
            training_audio = (
                review_directory / "residual-onset-01.training.wav"
            )
            soundfile.write(
                training_audio,
                np.zeros(22050, dtype=np.float32),
                22050,
            )
            review_path = review_directory / "residual-onset-01.json"
            review = {
                "schema_version": 1,
                "status": "draft",
                "review_kind": "residual-onset-training",
                "display_title": "P1 test",
                "display_subtitle": "Training stem",
                "review_id": "residual-onset-source-track-clip-01",
                "project_id": "source-track",
                "project_sha256": sha256_file(source_manifest),
                "source_audio_sha256": sha256_file(source_audio),
                "review_audio": review_audio.name,
                "review_audio_sha256": sha256_file(review_audio),
                "training_audio": training_audio.name,
                "training_audio_sha256": sha256_file(training_audio),
                "source_kind": "residual-onset-training",
                "measure_numbers": [1],
                "source_start": 0.0,
                "source_end": 1.0,
                "duration": 1.0,
                "capo": 0,
                "bpm": 60.0,
                "beats": [0.0],
                "time_signature": [4, 4],
                "events": [
                    {
                        "onset": 0.2,
                        "offset": 0.5,
                        "string": 2,
                        "fret": 3,
                        "technique": "pick",
                        "confidence": 1.0,
                    }
                ],
                "excluded_ranges": [],
                "checks": {
                    "timing": False,
                    "string_fret": False,
                    "completeness": False,
                    "technique": False,
                },
                "approval": None,
                "training_provenance": {
                    "source_manifest": source_manifest.name,
                    "source_manifest_sha256": sha256_file(
                        source_manifest
                    ),
                    "source_track_id": "source-track",
                    "review_source_track_id": "source-track-clean",
                    "source_group_id": "source-group",
                    "source_split": "train",
                    "source_dataset": "Guitar-TECHS",
                    "source_license": "CC-BY-4.0",
                    "audio_condition": (
                        "procedural-mixture-demucs-guitar-stem"
                    ),
                    "training_source_audio_sha256": sha256_file(
                        source_audio
                    ),
                    "audio_alignment": "sample-aligned-no-offset",
                    "review_audio_processing": (
                        "rms-normalized--20-dbfs-soft-limited--1-dbfs"
                    ),
                    "review_audio_gain_db": 12.0,
                },
            }
            review_path.write_text(
                json.dumps(review, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "must be approved",
            ):
                export_approved_residual_reviews(
                    review_directory,
                    source_manifest,
                    root / "blocked" / "manifest.jsonl",
                    root / "blocked-report.json",
                )
            approve_review(
                review_path,
                reviewer_alias="reviewer-1",
                timing_reviewed=True,
                string_fret_reviewed=False,
                completeness_reviewed=True,
            )

            output_manifest = root / "export" / "manifest.jsonl"
            output_report = root / "report.json"
            annotations, report = export_approved_residual_reviews(
                review_directory,
                source_manifest,
                output_manifest,
                output_report,
            )

            self.assertEqual(len(annotations), 1)
            self.assertEqual(annotations[0].split, "train")
            self.assertEqual(len(annotations[0].events), 1)
            self.assertEqual(
                annotations[0].resolve_audio(output_manifest),
                training_audio,
            )
            self.assertEqual(
                annotations[0].provenance["training_audio_sha256"],
                sha256_file(training_audio),
            )
            self.assertEqual(
                annotations[0].provenance["review_audio_sha256"],
                sha256_file(review_audio),
            )
            self.assertTrue(
                annotations[0]
                .provenance["human_verified_onset_complete"]
            )
            self.assertFalse(
                annotations[0]
                .provenance["human_verified_string_fret"]
            )
            self.assertEqual(report["test_tracks"], 0)

    def test_compose_training_manifests_relocates_audio_and_reports_counts(
        self,
    ) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            first_manifest = root / "first" / "manifest.jsonl"
            second_manifest = root / "second" / "manifest.jsonl"
            output_manifest = root / "combined" / "manifest.jsonl"
            first_manifest.parent.mkdir()
            second_manifest.parent.mkdir()
            (first_manifest.parent / "train.wav").write_bytes(b"train")
            (second_manifest.parent / "validation.wav").write_bytes(
                b"validation"
            )
            write_manifest(
                first_manifest,
                [
                    TrackAnnotation(
                        track_id="train-track",
                        group_id="train-group",
                        audio="train.wav",
                        split="train",
                        duration=1.0,
                        events=[],
                        provenance={"dataset": "first"},
                    )
                ],
            )
            write_manifest(
                second_manifest,
                [
                    TrackAnnotation(
                        track_id="validation-track",
                        group_id="validation-group",
                        audio="validation.wav",
                        split="validation",
                        duration=1.0,
                        events=[],
                        provenance={"dataset": "second"},
                    )
                ],
            )

            annotations, report = compose_training_manifests(
                [first_manifest, second_manifest],
                output_manifest,
            )
            loaded = load_manifest(output_manifest)
            resolved_audio = [
                annotation.resolve_audio(output_manifest).read_bytes()
                for annotation in loaded
            ]

        self.assertEqual(len(annotations), 2)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(report["splits"], {"train": 1, "validation": 1})
        self.assertEqual(report["datasets"], {"first": 1, "second": 1})
        self.assertEqual(report["test_tracks"], 0)
        self.assertEqual(report["group_split_leaks"], 0)
        self.assertEqual(resolved_audio, [b"train", b"validation"])

    def test_compose_training_manifests_rejects_test_and_group_leakage(
        self,
    ) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            output_manifest = root / "combined" / "manifest.jsonl"

            def manifest(
                name: str,
                *,
                track_id: str,
                group_id: str,
                split: str,
            ) -> Path:
                path = root / name / "manifest.jsonl"
                path.parent.mkdir()
                (path.parent / "audio.wav").write_bytes(b"audio")
                write_manifest(
                    path,
                    [
                        TrackAnnotation(
                            track_id=track_id,
                            group_id=group_id,
                            audio="audio.wav",
                            split=split,
                            duration=1.0,
                            events=[],
                        )
                    ],
                )
                return path

            test_manifest = manifest(
                "test",
                track_id="test-track",
                group_id="test-group",
                split="test",
            )
            with self.assertRaisesRegex(ValueError, "rejects test split"):
                compose_training_manifests(
                    [test_manifest],
                    output_manifest,
                )

            train_manifest = manifest(
                "train",
                track_id="train-track",
                group_id="shared-group",
                split="train",
            )
            validation_manifest = manifest(
                "validation",
                track_id="validation-track",
                group_id="shared-group",
                split="validation",
            )
            with self.assertRaisesRegex(ValueError, "Groups cross"):
                compose_training_manifests(
                    [train_manifest, validation_manifest],
                    output_manifest,
                )

    def test_convert_technique_positions_filters_split_license_and_frets(
        self,
    ) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            source_path = root / "techniques.jsonl"
            output_path = root / "positions" / "manifest.jsonl"
            audit_path = root / "reports" / "conversion.json"
            audio_path = root / "audio.wav"
            soundfile.write(
                audio_path,
                np.zeros(22050, dtype=np.float32),
                22050,
            )

            def track(
                track_id: str,
                *,
                dataset: str,
                license_name: str,
                split: str,
                events: list[TechniqueEvent],
            ) -> TechniqueTrack:
                return TechniqueTrack(
                    track_id=track_id,
                    group_id=f"group-{track_id}",
                    audio=audio_path.name,
                    split=split,
                    duration=1.0,
                    events=events,
                    provenance={
                        "dataset": dataset,
                        "license": license_name,
                    },
                )

            write_technique_manifest(
                source_path,
                [
                    track(
                        "approved",
                        dataset="AG-PT-set",
                        license_name="CC-BY-4.0",
                        split="train",
                        events=[
                            TechniqueEvent(
                                0.1,
                                0.2,
                                64,
                                1,
                                "pick",
                            ),
                            TechniqueEvent(
                                0.3,
                                0.4,
                                90,
                                1,
                                "harmonic",
                            ),
                        ],
                    ),
                    track(
                        "held-out",
                        dataset="AG-PT-set",
                        license_name="CC-BY-4.0",
                        split="validation",
                        events=[
                            TechniqueEvent(
                                0.1,
                                0.2,
                                65,
                                1,
                                "pick",
                            )
                        ],
                    ),
                    track(
                        "restricted",
                        dataset="IDMT-SMT-GUITAR_V2",
                        license_name="CC-BY-4.0",
                        split="train",
                        events=[
                            TechniqueEvent(
                                0.1,
                                0.2,
                                40,
                                6,
                                "pick",
                            )
                        ],
                    ),
                ],
            )

            converted, audit = convert_technique_positions(
                source_path,
                output_path,
                audit_path,
            )
            loaded = load_manifest(output_path)

            self.assertEqual(
                [annotation.track_id for annotation in converted],
                ["approved"],
            )
            self.assertEqual(
                [event.fret for event in loaded[0].events],
                [0],
            )
            self.assertEqual(loaded[0].events[0].technique, "pick")
            self.assertEqual(loaded[0].split, "train")
            self.assertEqual(
                loaded[0].resolve_audio(output_path),
                audio_path.resolve(),
            )
            self.assertNotIn(str(root.resolve()), audit_path.read_text())
            counts = audit["counts"]
            self.assertEqual(
                counts["filtered_invalid_fret_events_by_dataset"],
                {"AG-PT-set": 1},
            )
            self.assertEqual(
                counts["rejected_tracks_by_dataset_and_reason"],
                {
                    "AG-PT-set:non_training_split": 1,
                    "IDMT-SMT-GUITAR_V2:dataset_not_approved": 1,
                },
            )
            self.assertEqual(
                counts["declared_license_mismatch_tracks_by_dataset"],
                {"IDMT-SMT-GUITAR_V2": 1},
            )

    def test_convert_technique_position_evaluation_is_held_out_only(
        self,
    ) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            source_path = root / "techniques.jsonl"
            output_path = root / "evaluation" / "manifest.jsonl"
            audit_path = root / "reports" / "evaluation.json"
            audio_path = root / "audio.wav"
            soundfile.write(
                audio_path,
                np.zeros(22050, dtype=np.float32),
                22050,
            )

            def track(
                track_id: str,
                *,
                dataset: str,
                split: str,
            ) -> TechniqueTrack:
                return TechniqueTrack(
                    track_id=track_id,
                    group_id=f"group-{track_id}",
                    audio=audio_path.name,
                    split=split,
                    duration=1.0,
                    events=[
                        TechniqueEvent(0.1, 0.2, 64, 1, "pick")
                    ],
                    provenance={
                        "dataset": dataset,
                        "license": (
                            "CC-BY-4.0"
                            if dataset != "IDMT-SMT-GUITAR_V2"
                            else "CC-BY-NC-ND-4.0"
                        ),
                    },
                )

            write_technique_manifest(
                source_path,
                [
                    track(
                        "approved-validation",
                        dataset="AG-PT-set",
                        split="validation",
                    ),
                    track(
                        "approved-train",
                        dataset="AG-PT-set",
                        split="train",
                    ),
                    track(
                        "restricted-validation",
                        dataset="IDMT-SMT-GUITAR_V2",
                        split="validation",
                    ),
                ],
            )

            converted, audit = convert_technique_position_evaluation(
                source_path,
                output_path,
                audit_path,
                source_split="validation",
            )
            loaded = load_manifest(output_path)

            self.assertEqual(len(converted), 1)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].split, "validation")
            self.assertTrue(loaded[0].provenance["evaluation_only"])
            self.assertTrue(audit["training_prohibited"])
            self.assertEqual(
                audit["counts"][
                    "rejected_tracks_by_dataset_and_reason"
                ],
                {"IDMT-SMT-GUITAR_V2:dataset_not_approved": 1},
            )
            with self.assertRaisesRegex(
                ValueError,
                "validation or test",
            ):
                convert_technique_position_evaluation(
                    source_path,
                    output_path,
                    audit_path,
                    source_split="train",
                )

    def test_onset_position_patch_and_model_shapes(self) -> None:
        feature_config = FeatureConfig()
        model_config = OnsetPositionConfig(frames_before=2, frames_after=12)
        features = np.linspace(
            0,
            1,
            30 * feature_config.n_bins,
            dtype=np.float32,
        ).reshape(30, feature_config.n_bins)
        patch = extract_onset_patch(
            features,
            onset=0.1,
            feature_config=feature_config,
            config=model_config,
        )
        model = OnsetPositionNet(feature_config, model_config)
        output = model(
            torch.from_numpy(patch).unsqueeze(0),
            torch.from_numpy(pitch_mask([64, 67])).unsqueeze(0),
        )

        self.assertEqual(patch.shape, (3, 14, 216))
        self.assertEqual(tuple(output.shape), (1, 6, 26))

    def test_zero_initialized_onset_adapter_preserves_base_output(
        self,
    ) -> None:
        feature_config = FeatureConfig()
        base_config = OnsetPositionConfig(dropout=0)
        adapter_config = OnsetPositionConfig(dropout=0, adapter_size=16)
        base = OnsetPositionNet(feature_config, base_config).eval()
        adapted = OnsetPositionNet(feature_config, adapter_config).eval()
        incompatible = adapted.load_state_dict(
            base.state_dict(),
            strict=False,
        )
        features = torch.randn(
            2,
            3,
            adapter_config.frame_count,
            feature_config.n_bins,
        )
        pitches = torch.rand(2, 49)

        base_output = base(features, pitches)
        adapted_output = adapted(features, pitches)

        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(
            all(
                name.startswith("adapter.")
                for name in incompatible.missing_keys
            )
        )
        self.assertTrue(torch.equal(base_output, adapted_output))
        self.assertEqual(adapter_config.to_dict()["adapter_size"], 16)
        self.assertEqual(adapter_config.to_dict()["adapter_scale"], 1.0)

        with torch.no_grad():
            adapted.adapter[-1].weight.normal_()
            adapted.adapter[-1].bias.normal_()
        adapted.adapter_scale = 0
        self.assertTrue(
            torch.equal(base_output, adapted(features, pitches))
        )
        adapted.adapter_scale = 1
        self.assertFalse(
            torch.equal(base_output, adapted(features, pitches))
        )

    def test_onset_position_grouping_splits_duplicate_strings(self) -> None:
        groups = group_event_indexes(
            [0.0, 0.01, 0.02, 0.2],
            [1, 2, 1, 3],
            tolerance=0.045,
        )

        self.assertEqual(groups, [(0, 1), (2,), (3,)])

    def test_onset_position_pitch_mask_preserves_maximum_confidence(
        self,
    ) -> None:
        mask = pitch_mask([64, 64, 66], [0.3, 0.8, 1.2])

        self.assertAlmostEqual(float(mask[64 - PITCH_MIN]), 0.8)
        self.assertAlmostEqual(float(mask[66 - PITCH_MIN]), 1.0)

    def test_candidate_augmentation_drops_cached_pitch_candidates(
        self,
    ) -> None:
        with patch(
            "stringtrace_ml.onset_position.random.random",
            return_value=0.0,
        ):
            pitches, confidences = augment_pitch_candidates(
                [64, 66],
                [0.8, 0.6],
                [64, 67],
                drop_probability=1.0,
                add_probability=0.0,
            )

        self.assertEqual(len(pitches), 1)
        self.assertIn(pitches[0], (64, 67))
        self.assertEqual(confidences, [1.0])

    def test_candidate_augmentation_keeps_empty_negative_group_safe(
        self,
    ) -> None:
        with patch(
            "stringtrace_ml.onset_position.random.random",
            return_value=0.0,
        ):
            pitches, confidences = augment_pitch_candidates(
                [64],
                [0.8],
                [],
                drop_probability=1.0,
                add_probability=1.0,
            )

        self.assertEqual(pitches, [])
        self.assertEqual(confidences, [])

    def test_guitarset_stem_selection_spans_each_stratum(self) -> None:
        annotations = [
            TrackAnnotation(
                track_id=f"{player}_{style}{take}-120-C_{role}",
                group_id=f"player-{player}",
                audio=f"{player}-{style}-{role}-{take}.wav",
                split="train",
                duration=1.0,
                events=[],
            )
            for player in ("00", "01")
            for style in ("BN", "Rock")
            for role in ("comp", "solo")
            for take in (1, 2)
        ]

        selected = stratified_selection(annotations, 8)

        self.assertEqual(len(selected), 8)
        self.assertEqual(
            {
                (
                    annotation.group_id,
                    track_family(annotation.track_id),
                    annotation.track_id.rsplit("_", 1)[-1],
                )
                for annotation in selected
            },
            {
                (f"player-{player}", style, role)
                for player in ("00", "01")
                for style in ("BN", "Rock")
                for role in ("comp", "solo")
            },
        )

    def test_derived_guitarset_stem_records_provenance(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            source_manifest = root / "source" / "manifest.jsonl"
            output_manifest = root / "derived" / "manifest.jsonl"
            source_manifest.parent.mkdir()
            output_manifest.parent.mkdir()
            source_audio = source_manifest.parent / "source.wav"
            stem_audio = output_manifest.parent / "audio" / "track.wav"
            stem_audio.parent.mkdir()
            source_audio.write_bytes(b"source-audio")
            stem_audio.write_bytes(b"guitar-stem")
            annotation = TrackAnnotation(
                track_id="track",
                group_id="player",
                audio=source_audio.name,
                split="train",
                duration=1.0,
                events=[],
                provenance={"dataset": "GuitarSet"},
            )

            derived = derived_annotation(
                annotation,
                stem_audio,
                source_manifest,
                output_manifest,
                model="htdemucs_6s",
            )
            relocated = relocated_source_annotation(
                annotation,
                source_manifest,
                output_manifest,
            )

        self.assertEqual(derived.track_id, "track-demucs-guitar")
        self.assertEqual(derived.audio, "audio/track.wav")
        self.assertEqual(relocated.audio, "../source/source.wav")
        self.assertEqual(
            derived.provenance["derived_from_track_id"],
            "track",
        )
        self.assertEqual(
            derived.provenance["audio_condition"],
            "demucs-guitar-stem",
        )
        self.assertEqual(
            derived.provenance["annotation_alignment"],
            "sample-aligned-no-offset",
        )

    def test_procedural_accompaniment_is_deterministic_and_stereo(
        self,
    ) -> None:
        first = procedural_accompaniment(8000, 8000, seed=42)
        second = procedural_accompaniment(8000, 8000, seed=42)
        different = procedural_accompaniment(8000, 8000, seed=43)

        self.assertEqual(first.shape, (8000, 2))
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, different))
        self.assertGreater(float(np.sqrt(np.mean(np.square(first)))), 0)
        self.assertNotEqual(stable_track_seed(42, "a"), stable_track_seed(42, "b"))

    def test_procedural_mix_respects_target_snr(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            source_path = root / "source.wav"
            mixture_path = root / "mixture.wav"
            sample_rate = 8000
            time = np.arange(sample_rate, dtype=np.float32) / sample_rate
            soundfile.write(
                source_path,
                0.1 * np.sin(2 * np.pi * 220 * time),
                sample_rate,
            )

            metadata = mix_source_audio(
                source_path,
                mixture_path,
                seed=42,
                snr_db=3.0,
            )
            mixture, loaded_rate = soundfile.read(
                mixture_path,
                dtype="float32",
                always_2d=True,
            )

        self.assertEqual(loaded_rate, sample_rate)
        self.assertEqual(mixture.shape, (sample_rate, 2))
        self.assertAlmostEqual(float(metadata["realized_snr_db"]), 3.0, places=4)
        self.assertLessEqual(float(np.max(np.abs(mixture))), 0.981)
        self.assertEqual(len(str(metadata["mixture_sha256"])), 64)

    def test_onset_position_candidate_groups_map_complete_reference_chord(
        self,
    ) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            audio_path = root / "source.wav"
            manifest_path = root / "manifest.jsonl"
            candidate_cache = root / "candidate-cache"
            candidate_cache.mkdir()
            soundfile.write(
                audio_path,
                np.zeros(22050, dtype=np.float32),
                22050,
            )
            write_manifest(
                manifest_path,
                [
                    TrackAnnotation(
                        track_id="candidate-map",
                        group_id="candidate-map",
                        audio=audio_path.name,
                        split="train",
                        duration=1.0,
                        events=[
                            LabeledEvent(0.100, 0.3, 1, 0),
                            LabeledEvent(0.125, 0.3, 2, 5),
                            LabeledEvent(0.140, 0.3, 1, 2),
                        ],
                        provenance={"dataset": "Sparse"},
                    )
                ],
            )
            cache_path = basic_pitch_cache_path(
                audio_path,
                candidate_cache,
                onset_threshold=0.35,
                frame_threshold=0.25,
                minimum_note_length=70,
            )
            cache_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "events": [
                            {
                                "onset": 0.110,
                                "pitch": 64,
                                "confidence": 0.8,
                            },
                            {
                                "onset": 0.130,
                                "pitch": 66,
                                "confidence": 0.6,
                            },
                            {
                                "onset": 0.500,
                                "pitch": 69,
                                "confidence": 0.4,
                            },
                        ],
                        "dropped": 0,
                    }
                ),
                encoding="utf-8",
            )

            dataset = OnsetPositionDataset(
                manifest_path,
                split="train",
                config=OnsetPositionConfig(
                    use_candidate_confidence=True
                ),
                candidate_cache_directory=candidate_cache,
                candidate_onset_threshold=0.35,
                candidate_frame_threshold=0.25,
                candidate_minimum_note_length=70,
            )

            self.assertEqual(len(dataset.groups), 2)
            self.assertEqual(dataset.groups[0].event_indexes, (0, 1))
            self.assertEqual(dataset.groups[0].input_pitches, (64, 66))
            self.assertEqual(
                dataset.groups[0].input_confidences,
                (0.8, 0.6),
            )
            self.assertEqual(dataset.groups[1].event_indexes, ())

            positive_only_dataset = OnsetPositionDataset(
                manifest_path,
                split="train",
                candidate_cache_directory=candidate_cache,
                candidate_onset_threshold=0.35,
                candidate_frame_threshold=0.25,
                candidate_minimum_note_length=70,
                positive_only_datasets={"Sparse"},
            )

            self.assertEqual(len(positive_only_dataset.groups), 1)
            self.assertEqual(
                positive_only_dataset.filtered_negative_groups,
                1,
            )

    def test_onset_position_repeat_sampling_weights_rearticulations(
        self,
    ) -> None:
        with self.temporary_directory() as temporary:
            manifest_path = Path(temporary) / "manifest.jsonl"
            write_manifest(
                manifest_path,
                [
                    TrackAnnotation(
                        track_id="repeat-sampling",
                        group_id="repeat-sampling",
                        audio="unused.wav",
                        split="train",
                        duration=2.0,
                        events=[
                            LabeledEvent(0.1, 0.3, 2, 1),
                            LabeledEvent(0.5, 0.7, 2, 1),
                            LabeledEvent(0.6, 0.8, 3, 2),
                            LabeledEvent(1.4, 1.6, 2, 1),
                        ],
                    )
                ],
            )
            dataset = OnsetPositionDataset(
                manifest_path,
                split="train",
            )

            weights = dataset.repeat_sample_weights(
                interval=0.75,
                bonus=0.75,
            )

            self.assertEqual(weights, [1.0, 1.75, 1.0, 1.0])

    def test_candidate_activity_dataset_labels_repeated_groups(
        self,
    ) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            audio_path = root / "source.wav"
            manifest_path = root / "manifest.jsonl"
            candidate_cache = root / "candidate-cache"
            feature_cache = root / "feature-cache"
            candidate_cache.mkdir()
            soundfile.write(
                audio_path,
                np.zeros(22050 * 2, dtype=np.float32),
                22050,
            )
            write_manifest(
                manifest_path,
                [
                    TrackAnnotation(
                        track_id="activity",
                        group_id="activity",
                        audio=audio_path.name,
                        split="train",
                        duration=2.0,
                        events=[
                            LabeledEvent(0.1, 0.2, 2, 5),
                            LabeledEvent(0.5, 0.6, 2, 5),
                        ],
                    )
                ],
            )
            cache_path = basic_pitch_cache_path(
                audio_path,
                candidate_cache,
                onset_threshold=0.35,
                frame_threshold=0.25,
                minimum_note_length=70,
            )
            cache_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "events": [
                            {"onset": 0.1, "pitch": 64},
                            {"onset": 0.5, "pitch": 64},
                            {"onset": 1.5, "pitch": 67},
                        ],
                        "dropped": 0,
                    }
                ),
                encoding="utf-8",
            )
            dataset = CandidateActivityDataset(
                manifest_path,
                split="train",
                feature_config=FeatureConfig(),
                position_config=OnsetPositionConfig(),
                activity_config=CandidateActivityConfig(),
                feature_cache_directory=feature_cache,
                candidate_cache_directory=candidate_cache,
                candidate_onset_threshold=0.35,
                candidate_frame_threshold=0.25,
                candidate_minimum_note_length=70,
            )

            self.assertEqual(
                dataset.targets,
                [(1.0, 0.0), (1.0, 1.0), (0.0, 0.0)],
            )
            self.assertEqual(
                dataset.target_counts(),
                {
                    "groups": 3,
                    "active": 2,
                    "inactive": 1,
                    "repeated": 1,
                },
            )

    def test_candidate_temporal_context_tracks_same_pitch_history(
        self,
    ) -> None:
        context = candidate_temporal_context(
            [0.1, 0.5, 1.5],
            [[64], [64, 67], [69]],
            [[0.8], [0.6, 0.8], [0.5]],
            repeat_interval=0.75,
            maximum_interval=2.0,
        )

        np.testing.assert_allclose(
            context,
            np.asarray(
                [
                    [1.0, 1.0, 0.0, 0.0, 1 / 6, 0.8],
                    [0.2, 0.2, 1.0, 1.0, 2 / 6, 0.7],
                    [0.5, 1.0, 0.0, 0.0, 1 / 6, 0.5],
                ],
                dtype=np.float32,
            ),
        )
        head = CandidateActivityHead(
            OnsetPositionConfig(),
            CandidateActivityConfig(
                hidden_size=8,
                dropout=0,
                temporal_feature_count=6,
            ),
        )
        output = head(
            torch.zeros((3, 384)),
            torch.from_numpy(context),
        )
        self.assertEqual(tuple(output.shape), (3, 2))
        with self.assertRaisesRegex(ValueError, "temporal context"):
            head(torch.zeros((3, 384)))

    def test_candidate_activity_dataset_sampling_weights(
        self,
    ) -> None:
        annotations = [
            TrackAnnotation(
                track_id="base",
                group_id="base",
                audio="base.wav",
                split="train",
                duration=1.0,
                events=[],
                provenance={"dataset": "GuitarSet"},
            ),
            TrackAnnotation(
                track_id="manual",
                group_id="manual",
                audio="manual.wav",
                split="train",
                duration=1.0,
                events=[],
                provenance={"dataset": "Residual-Onset-Manual"},
            ),
        ]
        groups = [
            OnsetGroupIndex(annotation_index=0, event_indexes=()),
            OnsetGroupIndex(annotation_index=1, event_indexes=()),
        ]

        self.assertEqual(
            candidate_sampling_weights(
                annotations,
                groups,
                {"Residual-Onset-Manual": 8.0},
            ),
            [1.0, 8.0],
        )
        self.assertEqual(
            parse_candidate_dataset_sampling_weights(
                ["Residual-Onset-Manual=8", "GuitarSet=0.5"]
            ),
            {
                "Residual-Onset-Manual": 8.0,
                "GuitarSet": 0.5,
            },
        )
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            candidate_sampling_weights(
                annotations,
                groups,
                {"Residual-Onset-Manual": 0.0},
            )
        self.assertEqual(
            parse_window_dataset_sampling_weights(
                ["Residual-Onset-Manual=16"]
            ),
            {"Residual-Onset-Manual": 16.0},
        )

    def test_candidate_activity_head_lowers_repeat_direct_threshold(
        self,
    ) -> None:
        feature_config = FeatureConfig()
        position_config = OnsetPositionConfig(dropout=0)
        position_model = OnsetPositionNet(
            feature_config,
            position_config,
        ).eval()
        activity_head = CandidateActivityHead(
            position_config,
            CandidateActivityConfig(hidden_size=8, dropout=0),
        ).eval()
        with torch.no_grad():
            for parameter in position_model.parameters():
                parameter.zero_()
            position_model.output[-1].bias.fill_(-10)
            position_model.output[-1].bias[0] = 0
            position_model.output[-1].bias[1] = 0.1
            position_model.output[-1].bias[2] = 0
            for string_index in range(1, 6):
                position_model.output[-1].bias[string_index * 26] = 10
            for parameter in activity_head.parameters():
                parameter.zero_()
            activity_head.network[-1].bias.fill_(10)
        events = [
            {
                "onset": 0.1,
                "offset": 0.3,
                "pitch": 64,
                "confidence": 1.0,
            }
        ]
        features = np.zeros((30, feature_config.n_bins), dtype=np.float32)

        baseline = assign_detected_events(
            events,
            features,
            position_model,
            feature_config,
            position_config,
            torch.device("cpu"),
            mode="direct",
            minimum_direct_probability=0.5,
        )
        repeated = assign_detected_events(
            events,
            features,
            position_model,
            feature_config,
            position_config,
            torch.device("cpu"),
            mode="direct",
            minimum_direct_probability=0.5,
            activity_head=activity_head,
            minimum_repeat_probability=0.5,
            repeat_minimum_direct_probability=0.3,
        )

        self.assertEqual(baseline, [])
        self.assertEqual(
            [(event.string, event.fret) for event in repeated],
            [(1, 0)],
        )

    def test_onset_position_assignment_obeys_pitch_and_unique_strings(
        self,
    ) -> None:
        probabilities = np.full((6, 26), 1e-4, dtype=np.float32)
        probabilities[0, 1] = 0.9
        probabilities[1, 6] = 0.8
        probabilities[2, 10] = 0.95

        positions = assign_pitch_group(probabilities, [64, 64])

        self.assertEqual(
            {position[0] for position in positions},
            {0, 2},
        )
        self.assertTrue(all(
            OPEN_STRING_PITCHES[string] + fret == 64
            for string, fret in positions
        ))

    def test_onset_position_combined_confidence_can_abstain(self) -> None:
        feature_config = FeatureConfig()
        model_config = OnsetPositionConfig()
        model = OnsetPositionNet(feature_config, model_config)
        for parameter in model.parameters():
            parameter.data.zero_()
        model.output[-1].bias.data[1] = 5
        events = [
            {
                "onset": 0.1,
                "offset": 0.3,
                "pitch": 64,
                "confidence": 0.25,
            }
        ]
        features = np.zeros((30, feature_config.n_bins), dtype=np.float32)

        accepted = assign_detected_events(
            events,
            features,
            model,
            feature_config,
            model_config,
            torch.device("cpu"),
            mode="direct",
            minimum_direct_probability=0.5,
            minimum_output_confidence=0.4,
        )
        rejected = assign_detected_events(
            events,
            features,
            model,
            feature_config,
            model_config,
            torch.device("cpu"),
            mode="direct",
            minimum_direct_probability=0.5,
            minimum_output_confidence=0.5,
        )

        self.assertEqual(len(accepted), 1)
        self.assertEqual(rejected, [])

    def test_onset_fusion_excludes_an_entire_matched_group(self) -> None:
        primary = [
            {
                "onset": 0.1,
                "offset": 0.3,
                "pitch": 64,
                "confidence": 0.9,
            }
        ]
        candidates = [
            {
                "onset": 0.145,
                "offset": 0.3,
                "pitch": 64,
                "confidence": 0.8,
            },
            {
                "onset": 0.18,
                "offset": 0.3,
                "pitch": 67,
                "confidence": 0.7,
            },
            {
                "onset": 0.4,
                "offset": 0.6,
                "pitch": 69,
                "confidence": 0.85,
            },
        ]

        unmatched = unmatched_onset_candidates(
            primary,
            candidates,
            group_tolerance=0.045,
            match_tolerance=0.05,
        )

        self.assertEqual(unmatched, [candidates[2]])

    def test_onset_fusion_merge_deduplicates_tablature_events(self) -> None:
        primary = [LabeledEvent(0.1, 0.3, 2, 5, confidence=0.8)]
        supplemental = [
            LabeledEvent(0.14, 0.4, 2, 5, confidence=0.9),
            LabeledEvent(0.14, 0.4, 3, 5, confidence=0.9),
        ]

        merged = merge_predictions(
            primary,
            supplemental,
            onset_tolerance=0.05,
        )

        self.assertEqual(len(merged), 2)
        self.assertEqual(
            [(event.string, event.fret) for event in merged],
            [(2, 5), (3, 5)],
        )

    def test_onset_fusion_cli_rejects_invalid_probability(self) -> None:
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parse_arguments(
                [
                    "--manifest",
                    "manifest.jsonl",
                    "--checkpoint",
                    "position.pt",
                    "--onset-fusion-checkpoint",
                    "frame.pt",
                    "--onset-fusion-onset-threshold",
                    "1.1",
                ]
            )

    def test_onset_position_cli_rejects_negative_adapter_scale(self) -> None:
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parse_arguments(
                [
                    "--manifest",
                    "manifest.jsonl",
                    "--checkpoint",
                    "position.pt",
                    "--adapter-scale",
                    "-0.1",
                ]
            )

    def reference_fixture(self, root: Path) -> tuple[Path, Path]:
        project_path = root / "project.json"
        audio_path = root / "source.wav"
        project_path.write_text(
            json.dumps(
                {
                    "id": "project-1",
                    "bpm": 120,
                    "source": {
                        "kind": "upload",
                        "url": "https://example.invalid/?private=value",
                    },
                    "tab": {
                        "capo": 2,
                        "measures": [
                            {
                                "number": 1,
                                "start": 0.0,
                                "duration": 1.0,
                                "beats": [
                                    {
                                        "at": 0.0,
                                        "notes": [
                                            {
                                                "id": "note-1",
                                                "at": 0.1,
                                                "duration": 0.2,
                                                "string": 2,
                                                "fret": 5,
                                                "technique": "hammer-on",
                                                "positionSource": (
                                                    "source-score-reference-v1"
                                                ),
                                                "techniqueSource": (
                                                    "source-score-reference-v1"
                                                ),
                                            }
                                        ],
                                    }
                                ],
                            },
                            {
                                "number": 2,
                                "start": 1.0,
                                "duration": 1.0,
                                "beats": [
                                    {
                                        "at": 1.0,
                                        "notes": [
                                            {
                                                "id": "note-2",
                                                "at": 1.2,
                                                "duration": 0.2,
                                                "string": 1,
                                                "fret": 12,
                                                "technique": "slide",
                                                "positionSource": (
                                                    "playable-optimizer-v2"
                                                ),
                                                "techniqueSource": (
                                                    "transition-heuristic-v2"
                                                ),
                                            }
                                        ],
                                    }
                                ],
                            },
                        ],
                    },
                }
            ),
            encoding="utf-8",
        )
        soundfile.write(
            audio_path,
            np.zeros(44_100, dtype=np.float32),
            22_050,
            subtype="PCM_16",
        )
        return project_path, audio_path

    def test_reference_review_requires_explicit_complete_approval(self) -> None:
        with self.temporary_directory() as directory:
            root = Path(directory)
            project_path, audio_path = self.reference_fixture(root)
            review_path = root / "review.json"
            review = create_review(
                project_path,
                audio_path,
                review_path,
                [2],
            )

            self.assertEqual(review["status"], "draft")
            self.assertEqual(review["events"][0]["technique"], "unknown")
            serialized = review_path.read_text(encoding="utf-8")
            self.assertNotIn(str(root), serialized)
            self.assertNotIn("private=value", serialized)
            review["excluded_ranges"] = [
                {"start": 0.2, "end": 0.5, "reason": "uncertain"},
                {"start": 0.4, "end": 0.7, "reason": "overlap"},
            ]
            review_path.write_text(json.dumps(review), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not overlap"):
                approve_review(
                    review_path,
                    reviewer_alias="reviewer-1",
                    timing_reviewed=True,
                    string_fret_reviewed=True,
                    completeness_reviewed=True,
                )
            review["excluded_ranges"] = []
            review_path.write_text(json.dumps(review), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Cannot approve"):
                approve_review(
                    review_path,
                    reviewer_alias="reviewer-1",
                    timing_reviewed=True,
                    string_fret_reviewed=True,
                    completeness_reviewed=False,
                )

            approved = approve_review(
                review_path,
                reviewer_alias="reviewer-1",
                timing_reviewed=True,
                string_fret_reviewed=True,
                completeness_reviewed=True,
            )
            self.assertEqual(approved["status"], "approved")
            load_approved_review(review_path, project_path, audio_path)

            approved["events"][0]["fret"] = 13
            review_path.write_text(json.dumps(approved), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed after approval"):
                load_approved_review(review_path, project_path, audio_path)

    def test_reviewed_export_is_test_only_and_drops_unreviewed_technique(
        self,
    ) -> None:
        with self.temporary_directory() as directory:
            root = Path(directory)
            project_path, audio_path = self.reference_fixture(root)
            review_path = root / "review.json"
            review = create_review(
                project_path,
                audio_path,
                review_path,
                [2],
            )
            review["events"][0]["technique"] = "slide"
            review["excluded_ranges"] = [
                {
                    "start": 0.6,
                    "end": 0.8,
                    "reason": "inaudible light strum",
                }
            ]
            review_path.write_text(
                json.dumps(review, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            approve_review(
                review_path,
                reviewer_alias="reviewer-1",
                timing_reviewed=True,
                string_fret_reviewed=True,
                completeness_reviewed=True,
            )

            output = root / "reference"
            annotation = export_reference(
                project_path=project_path,
                audio_path=audio_path,
                output=output,
                review_path=review_path,
            )

            self.assertEqual(annotation.split, "test")
            self.assertEqual(annotation.events[0].technique, "unknown")
            self.assertEqual(
                annotation.excluded_ranges,
                [ExcludedRange(0.6, 0.8, "inaudible light strum")],
            )
            self.assertEqual(annotation.group_id, "project-1")
            self.assertEqual(
                annotation.provenance["verification_status"],
                "approved-human-review",
            )
            self.assertEqual(
                annotation.provenance["alignment_audit"]["status"],
                "approved",
            )
            self.assertEqual(
                annotation.provenance["alignment_audit"]["excluded_ranges"],
                [
                    {
                        "start": 0.6,
                        "end": 0.8,
                        "reason": "inaudible light strum",
                    }
                ],
            )
            provenance = json.dumps(annotation.provenance)
            self.assertNotIn(str(root), provenance)
            self.assertNotIn("private=value", provenance)
            self.assertEqual(
                load_manifest(output / "manifest.jsonl")[0],
                annotation,
            )

    def test_reviewed_export_can_render_a_synchronized_audio_condition(
        self,
    ) -> None:
        with self.temporary_directory() as directory:
            root = Path(directory)
            project_path, audio_path = self.reference_fixture(root)
            review_path = root / "review.json"
            create_review(
                project_path,
                audio_path,
                review_path,
                [2],
            )
            approve_review(
                review_path,
                reviewer_alias="reviewer-1",
                timing_reviewed=True,
                string_fret_reviewed=True,
                completeness_reviewed=True,
            )
            sample_rate = 22_050
            rendered_audio = np.zeros(sample_rate * 3, dtype=np.float32)
            rendered_audio[
                int(1.05 * sample_rate) : int(2.05 * sample_rate)
            ] = 0.25
            rendered_audio_path = root / "guitar.wav"
            soundfile.write(rendered_audio_path, rendered_audio, sample_rate)

            output = root / "derived-reference"
            annotation = export_reference(
                project_path=project_path,
                audio_path=audio_path,
                output=output,
                review_path=review_path,
                render_audio_path=rendered_audio_path,
                render_audio_offset=0.05,
                audio_condition="guitar-stem",
            )

            exported_audio, exported_rate = soundfile.read(
                annotation.resolve_audio(output / "manifest.jsonl"),
            )
            self.assertEqual(exported_rate, sample_rate)
            self.assertAlmostEqual(float(np.mean(exported_audio)), 0.25, places=3)
            self.assertEqual(
                annotation.provenance["verification_status"],
                "derived-from-approved-human-review",
            )
            self.assertEqual(
                annotation.provenance["audio_condition"],
                "guitar-stem",
            )
            self.assertEqual(
                annotation.provenance["alignment_audit"]["derived_audio"],
                {
                    "status": "offset-compensated",
                    "method": "onset-envelope-cross-correlation",
                    "offset_seconds": 0.05,
                },
            )

    def test_generated_project_notes_cannot_bypass_review(self) -> None:
        with self.temporary_directory() as directory:
            root = Path(directory)
            project_path, audio_path = self.reference_fixture(root)
            with self.assertRaisesRegex(
                ValueError,
                "Direct source-score export is disabled",
            ):
                export_reference(
                    project_path=project_path,
                    audio_path=audio_path,
                    output=root / "reference",
                    measure_numbers=[1],
                )
            with self.assertRaisesRegex(RuntimeError, "use a complete review"):
                export_reference(
                    project_path=project_path,
                    audio_path=audio_path,
                    output=root / "reference",
                    measure_numbers=[2],
                    allow_quarantined_source=True,
                )

    def test_source_score_research_export_is_quarantined(self) -> None:
        with self.temporary_directory() as directory:
            root = Path(directory)
            project_path, audio_path = self.reference_fixture(root)
            annotation = export_reference(
                project_path=project_path,
                audio_path=audio_path,
                output=root / "reference",
                measure_numbers=[1],
                allow_quarantined_source=True,
            )

            self.assertEqual(
                annotation.provenance["verification_status"],
                "quarantined",
            )
            self.assertEqual(
                annotation.provenance["alignment_audit"]["status"],
                "not-reviewed",
            )
            self.assertFalse(annotation.provenance["technique_reviewed"])
            self.assertEqual(annotation.events[0].technique, "unknown")
            with self.assertRaisesRegex(
                ValueError,
                "Evaluation is blocked for quarantined references",
            ):
                reject_quarantined_references([annotation])

    def test_reference_coverage_reports_hard_event_gaps_by_condition(
        self,
    ) -> None:
        with self.temporary_directory() as directory:
            root = Path(directory)
            events = [
                LabeledEvent(0.10, 0.15, 1, 0, "unknown"),
                LabeledEvent(0.18, 0.24, 1, 12, "unknown"),
                LabeledEvent(0.40, 0.50, 2, 13, "unknown"),
                LabeledEvent(0.60, 0.70, 2, 3, "unknown"),
            ]
            manifests = []
            for label in ("source-mix", "guitar-stem"):
                manifest = root / f"{label}.jsonl"
                write_manifest(
                    manifest,
                    [
                        TrackAnnotation(
                            track_id=label,
                            group_id="project-1",
                            audio=f"{label}.wav",
                            split="test",
                            duration=1.0,
                            events=events,
                            excluded_ranges=[
                                ExcludedRange(
                                    0.35,
                                    0.55,
                                    "inaudible passage",
                                )
                            ],
                            provenance={
                                "source_kind": "upload",
                                "annotation_source": "complete-human-review",
                                "verification_status": "approved-human-review",
                                "alignment_audit": {
                                    "status": "approved",
                                },
                                "technique_reviewed": False,
                            },
                        )
                    ],
                )
                manifests.append((label, manifest))

            report = build_coverage_report(manifests)
            source = report["datasets"]["source-mix"]
            self.assertEqual(source["events"], 4)
            self.assertEqual(source["frets"]["high_fret_events"], 2)
            self.assertEqual(
                source["same_string_intervals"]["fast_intervals"],
                1,
            )
            self.assertEqual(source["missing_strings"], [3, 4, 5, 6])
            self.assertEqual(source["techniques"]["reviewed_events"], 0)
            self.assertEqual(source["evaluation_eligible_tracks"], 1)
            self.assertEqual(source["evaluation_eligible_events"], 3)
            self.assertEqual(source["evaluation_excluded_events"], 1)
            self.assertEqual(source["evaluation_excluded_range_count"], 1)
            self.assertEqual(
                source["evaluation_excluded_duration_seconds"],
                0.2,
            )
            self.assertEqual(
                source["evaluation_eligible_high_fret_events"],
                1,
            )
            self.assertEqual(source["quarantined_tracks"], 0)
            self.assertNotIn(str(root), json.dumps(report))

    def test_review_queue_prioritizes_gaps_without_overlapping_excerpts(
        self,
    ) -> None:
        with self.temporary_directory() as directory:
            root = Path(directory)
            project_path = root / "project.json"

            def note(
                identifier: str,
                onset: float,
                string: int,
                fret: int,
                confidence: float,
                source: str = "playable-optimizer-v2",
            ) -> dict[str, object]:
                return {
                    "id": identifier,
                    "at": onset,
                    "duration": 0.1,
                    "string": string,
                    "fret": fret,
                    "confidence": confidence,
                    "positionConfidence": confidence,
                    "positionSource": source,
                }

            measure_notes = [
                [
                    note(
                        "verified",
                        0.1,
                        2,
                        1,
                        1.0,
                        "source-score-reference-v1",
                    )
                ],
                [note("ordinary", 1.1, 2, 1, 0.9)],
                [
                    note("hard-1", 2.1, 6, 12, 0.2),
                    note("hard-2", 2.18, 6, 14, 0.2),
                ],
                [note("ordinary-2", 3.1, 2, 2, 0.9)],
                [note("medium", 4.1, 6, 3, 0.5)],
            ]
            project_path.write_text(
                json.dumps(
                    {
                        "id": "project-queue",
                        "source": {
                            "kind": "upload",
                            "url": "https://example.invalid/?private=value",
                        },
                        "tab": {
                            "measures": [
                                {
                                    "number": index + 1,
                                    "start": float(index),
                                    "duration": 1.0,
                                    "beats": [
                                        {
                                            "at": float(index),
                                            "notes": notes,
                                        }
                                    ],
                                }
                                for index, notes in enumerate(measure_notes)
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            coverage = {
                "datasets": {
                    "source-mix": {"missing_strings": [3, 6]},
                    "guitar-stem": {"missing_strings": [6]},
                }
            }

            self.assertEqual(priority_strings_from_coverage(coverage), [6])
            report = build_review_queue(
                project_path,
                coverage_report=coverage,
                excerpt_measures=2,
                limit=2,
            )

            candidates = report["candidates"]
            self.assertEqual(candidates[0]["measure_numbers"], [2, 3])
            self.assertEqual(
                candidates[0]["signals"]["fast_same_string_intervals"],
                1,
            )
            self.assertEqual(
                candidates[0]["signals"]["priority_string_events"],
                2,
            )
            selected = [
                set(candidate["measure_numbers"])
                for candidate in candidates
            ]
            self.assertFalse(selected[0] & selected[1])
            self.assertNotIn(1, set.union(*selected))
            serialized = json.dumps(report)
            self.assertNotIn(str(root), serialized)
            self.assertNotIn("private=value", serialized)

    def test_range_download_layout_and_assembly_are_verified(self) -> None:
        with self.temporary_directory() as directory:
            root = Path(directory)
            content = b"0123456789"
            spec = DownloadSpec(
                name="fixture.zip",
                size=len(content),
                md5="781e5e245d69b566979b86e28d23f2c7",
            )
            ranges = byte_ranges(spec.size, 4)
            self.assertEqual(ranges, [(0, 3), (4, 7), (8, 9)])
            parts = root / "parts"
            parts.mkdir()
            for start, end in ranges:
                part_path(parts, start, end).write_bytes(content[start : end + 1])

            destination = root / spec.name
            assemble_download(spec, destination, parts, ranges)

            self.assertEqual(destination.read_bytes(), content)
            validate_content_range(
                "bytes 4-7/10",
                start=4,
                end=7,
                total=10,
            )
            with self.assertRaisesRegex(RuntimeError, "Unexpected Content-Range"):
                validate_content_range(
                    "bytes 4-8/10",
                    start=4,
                    end=7,
                    total=10,
                )

    def test_report_model_identity_excludes_local_paths(self) -> None:
        identity = model_identity(
            {
                "name": "model",
                "version": "v2",
                "kind": "trained",
                "epoch": 12,
                "manifest": "/private/training/manifest.jsonl",
                "checkpoint": "/private/checkpoint.pt",
            }
        )

        self.assertEqual(identity["epoch"], 12)
        self.assertNotIn("manifest", identity)
        self.assertNotIn("checkpoint", identity)

    def test_changed_loss_config_resets_resume_comparison(self) -> None:
        history = [{"validation": {"loss": 0.5}}]
        previous = {
            "training_config": {
                "loss": {
                    "attack_fret_weight": 1.0,
                    "attack_fret_frames": 0,
                },
                "sampling": {},
            }
        }
        current = {
            "loss": {
                "attack_fret_weight": 2.0,
                "attack_fret_frames": 6,
            },
            "sampling": {},
        }

        self.assertEqual(
            best_compatible_validation_loss(history, previous, current),
            float("inf"),
        )
        self.assertEqual(
            best_compatible_validation_loss(
                history,
                {"training_config": current},
                current,
            ),
            0.5,
        )

    def test_validation_metric_direction_and_compatibility(self) -> None:
        history = [
            {
                "validation": {
                    "loss": 0.7,
                    "frame_tablature_f1": 0.4,
                }
            },
            {
                "validation": {
                    "loss": 0.6,
                    "frame_tablature_f1": 0.5,
                }
            },
        ]
        training_config = {"loss": {}, "selection_metric": "metric"}
        metadata = {"training_config": training_config}

        self.assertEqual(
            best_compatible_validation_metric(
                history,
                metadata,
                training_config,
                "loss",
            ),
            0.6,
        )
        self.assertEqual(
            best_compatible_validation_metric(
                history,
                metadata,
                training_config,
                "frame_tablature_f1",
            ),
            0.5,
        )
        self.assertEqual(
            best_compatible_validation_metric(
                history,
                {},
                training_config,
                "frame_tablature_f1",
            ),
            float("-inf"),
        )

    def test_manifest_rejects_group_leakage(self) -> None:
        with self.temporary_directory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            first = TrackAnnotation(
                "first",
                "same-performer",
                "first.wav",
                "train",
                1.0,
                [],
            )
            second = TrackAnnotation(
                "second",
                "same-performer",
                "second.wav",
                "test",
                1.0,
                [],
            )
            with self.assertRaisesRegex(ValueError, "Data leakage"):
                write_manifest(manifest, [first, second])
            manifest.write_text(
                json.dumps(first.to_dict()) + "\n" + json.dumps(second.to_dict()) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Data leakage"):
                load_manifest(manifest, split="train")

    def test_dataset_keeps_repeated_same_fret_onsets(self) -> None:
        with self.temporary_directory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            write_manifest(
                manifest,
                [
                    TrackAnnotation(
                        track_id="repeat",
                        group_id="player-1",
                        audio="repeat.wav",
                        split="train",
                        duration=1.0,
                        events=[
                            LabeledEvent(0.1, 0.4, 2, 1),
                            LabeledEvent(0.5, 0.8, 2, 1),
                        ],
                    )
                ],
            )
            fake_features = np.zeros((87, 216), dtype=np.float32)
            with patch(
                "stringtrace_ml.dataset.extract_file_cqt",
                return_value=fake_features,
            ):
                dataset = GuitarTabDataset(
                    manifest,
                    split="train",
                    window_seconds=1.0,
                )
                item = dataset[0]
            onset_frames = torch.nonzero(
                item["onset_targets"][:, 1],
            ).flatten()
            self.assertEqual(onset_frames.numel(), 2)
            self.assertGreater(
                int((item["fret_targets"][:, 1] == 2).sum()),
                20,
            )

    def test_dataset_weights_fast_and_high_fret_windows(self) -> None:
        with self.temporary_directory() as directory:
            manifest = Path(directory) / "manifest.jsonl"
            write_manifest(
                manifest,
                [
                    TrackAnnotation(
                        track_id="hard-windows",
                        group_id="player-1",
                        audio="unused.wav",
                        split="train",
                        duration=8.0,
                        events=[
                            LabeledEvent(0.1, 0.3, 2, 1),
                            LabeledEvent(0.2, 0.4, 2, 1),
                            LabeledEvent(4.5, 4.8, 1, 13),
                        ],
                    )
                ],
            )
            dataset = GuitarTabDataset(
                manifest,
                split="train",
                window_seconds=4.0,
                window_hop_seconds=4.0,
            )

            weights = dataset.difficulty_sample_weights(
                fast_bonus=0.5,
                high_fret_bonus=0.75,
            )

            self.assertEqual(weights, [1.5, 1.75])

    def test_dataset_balances_underrepresented_strings(self) -> None:
        with self.temporary_directory() as directory:
            manifest = Path(directory) / "manifest.jsonl"
            write_manifest(
                manifest,
                [
                    TrackAnnotation(
                        track_id="string-balance",
                        group_id="player-1",
                        audio="unused.wav",
                        split="train",
                        duration=1.0,
                        events=[
                            LabeledEvent(0.1, 0.2, 1, 1),
                            LabeledEvent(0.2, 0.3, 2, 1),
                            LabeledEvent(0.3, 0.4, 2, 2),
                            LabeledEvent(0.4, 0.5, 2, 3),
                            LabeledEvent(0.5, 0.6, 2, 4),
                        ],
                    )
                ],
            )
            dataset = GuitarTabDataset(
                manifest,
                split="train",
                window_seconds=1.0,
            )

            weights = dataset.active_string_weights(exponent=0.5)

            self.assertGreater(weights[0], weights[1])
            self.assertAlmostEqual(sum(weights) / 6, 1.0)

    def test_dataset_pitch_shift_moves_features_and_fret_targets(self) -> None:
        with self.temporary_directory() as directory:
            manifest = Path(directory) / "manifest.jsonl"
            write_manifest(
                manifest,
                [
                    TrackAnnotation(
                        track_id="shift",
                        group_id="player-1",
                        audio="unused.wav",
                        split="train",
                        duration=1.0,
                        events=[LabeledEvent(0.1, 0.5, 2, 3)],
                    )
                ],
            )
            features = np.zeros((87, 216), dtype=np.float32)
            features[:, 10] = 1
            with (
                patch(
                    "stringtrace_ml.dataset.extract_file_cqt",
                    return_value=features,
                ),
                patch(
                    "stringtrace_ml.dataset.random.random",
                    return_value=0,
                ),
                patch(
                    "stringtrace_ml.dataset.random.randint",
                    return_value=2,
                ),
            ):
                dataset = GuitarTabDataset(
                    manifest,
                    split="train",
                    window_seconds=1.0,
                    pitch_shift_probability=1.0,
                )
                item = dataset[0]

            self.assertEqual(item["pitch_shift"], 2)
            self.assertEqual(float(item["features"][0, 16]), 1.0)
            self.assertEqual(float(item["features"][0, 10]), 0.0)
            self.assertTrue(bool((item["fret_targets"][:, 1] == 6).any()))

    def test_dataset_persists_full_track_feature_windows(self) -> None:
        with self.temporary_directory() as directory:
            root = Path(directory)
            audio = root / "track.wav"
            audio.touch()
            manifest = root / "manifest.jsonl"
            write_manifest(
                manifest,
                [
                    TrackAnnotation(
                        track_id="cached",
                        group_id="player-1",
                        audio=str(audio),
                        split="train",
                        duration=6.0,
                        events=[],
                    )
                ],
            )
            full_features = np.zeros((520, 216), dtype=np.float32)
            cache = root / "cache"
            with patch(
                "stringtrace_ml.dataset.extract_file_cqt",
                return_value=full_features,
            ) as extract:
                dataset = GuitarTabDataset(
                    manifest,
                    split="train",
                    window_seconds=4.0,
                    window_hop_seconds=2.0,
                    feature_cache_directory=cache,
                )
                dataset[0]
                dataset[1]
                self.assertEqual(extract.call_count, 1)
                cache_files = list(cache.glob("*.npy"))
                self.assertEqual(len(cache_files), 1)
                self.assertEqual(
                    np.load(cache_files[0], allow_pickle=False).dtype,
                    np.float16,
                )

            with patch(
                "stringtrace_ml.dataset.extract_file_cqt",
            ) as extract:
                restored = GuitarTabDataset(
                    manifest,
                    split="train",
                    window_seconds=4.0,
                    window_hop_seconds=2.0,
                    feature_cache_directory=cache,
                )
                restored_item = restored[0]
                restored[1]
                extract.assert_not_called()
                self.assertEqual(
                    restored_item["features"].dtype,
                    torch.float32,
                )

    def test_model_outputs_string_fret_and_onset_heads(self) -> None:
        model = StringFretNet()
        features = torch.rand(2, 32, 216)
        outputs = model(features)

        self.assertEqual(model.temporal.input_size, 216)
        self.assertEqual(outputs["fret_logits"].shape, (2, 32, 6, 26))
        self.assertEqual(outputs["onset_logits"].shape, (2, 32, 6))
        loss, values = transcription_loss(
            outputs,
            torch.zeros(2, 32, 6, dtype=torch.long),
            torch.zeros(2, 32, 6),
            torch.ones(2, 32, dtype=torch.bool),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(values["loss"], 0)

    def test_onset_head_only_freezes_every_other_model_parameter(self) -> None:
        model = StringFretNet(
            ModelConfig(pitch_conditioned_position=True)
        )

        configure_trainable_parameters(model, onset_head_only=True)

        trainable = {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertEqual(
            trainable,
            {"onset_head.weight", "onset_head.bias"},
        )

    def test_pitch_conditioned_head_predicts_a_string_for_each_pitch(self) -> None:
        model = StringFretNet(
            ModelConfig(pitch_conditioned_position=True)
        )
        outputs = model(
            torch.rand(2, 8, 216),
            include_all_positions=True,
        )

        self.assertEqual(outputs["position_logits"].shape, (2, 8, 49, 6))
        self.assertLess(
            float(outputs["position_logits"][0, 0, 0, 0].detach()),
            -20,
        )
        queried = model(
            torch.rand(2, 8, 216),
            position_queries=(
                torch.tensor([0, 1]),
                torch.tensor([2, 3]),
                torch.tensor([24, 19]),
            ),
        )
        self.assertEqual(queried["position_logits"].shape, (2, 6))

    def test_position_loss_rewards_the_correct_string_for_a_known_pitch(
        self,
    ) -> None:
        fret_targets = torch.zeros(1, 1, 6, dtype=torch.long)
        fret_targets[0, 0, 0] = 1
        onset_targets = torch.zeros(1, 1, 6)
        onset_targets[0, 0, 0] = 1
        valid_frames = torch.ones(1, 1, dtype=torch.bool)
        base_outputs = {
            "fret_logits": torch.zeros(1, 1, 6, 26),
            "onset_logits": torch.zeros(1, 1, 6),
        }
        correct_positions = torch.zeros(1, 1, 49, 6)
        correct_positions[0, 0, 24, 0] = 5
        wrong_positions = torch.zeros(1, 1, 49, 6)
        wrong_positions[0, 0, 24, 1] = 5

        _correct_loss, correct_values = transcription_loss(
            base_outputs | {"position_logits": correct_positions},
            fret_targets,
            onset_targets,
            valid_frames,
            position_loss_weight=1.0,
        )
        _wrong_loss, wrong_values = transcription_loss(
            base_outputs | {"position_logits": wrong_positions},
            fret_targets,
            onset_targets,
            valid_frames,
            position_loss_weight=1.0,
        )

        self.assertLess(
            correct_values["position_loss"],
            wrong_values["position_loss"],
        )

    def test_loss_emphasizes_active_frets_after_an_attack(self) -> None:
        outputs = {
            "fret_logits": torch.zeros(1, 8, 6, 26),
            "onset_logits": torch.zeros(1, 8, 6),
        }
        fret_targets = torch.zeros(1, 8, 6, dtype=torch.long)
        fret_targets[:, 2:7, 0] = 3
        onset_targets = torch.zeros(1, 8, 6)
        onset_targets[:, 2, 0] = 1
        valid_frames = torch.ones(1, 8, dtype=torch.bool)

        unweighted, _values = transcription_loss(
            outputs,
            fret_targets,
            onset_targets,
            valid_frames,
            attack_fret_weight=1.0,
        )
        weighted, _values = transcription_loss(
            outputs,
            fret_targets,
            onset_targets,
            valid_frames,
            attack_fret_weight=3.0,
            attack_fret_frames=4,
        )

        self.assertGreater(weighted, unweighted)

    def test_fret_distance_loss_penalizes_distant_errors(self) -> None:
        fret_targets = torch.zeros(1, 1, 6, dtype=torch.long)
        fret_targets[0, 0, 0] = 3
        onset_targets = torch.zeros(1, 1, 6)
        valid_frames = torch.ones(1, 1, dtype=torch.bool)
        near_logits = torch.zeros(1, 1, 6, 26)
        near_logits[0, 0, 0, 4] = 5
        far_logits = torch.zeros(1, 1, 6, 26)
        far_logits[0, 0, 0, 13] = 5
        onset_logits = torch.zeros(1, 1, 6)

        near_loss, near_values = transcription_loss(
            {
                "fret_logits": near_logits,
                "onset_logits": onset_logits,
            },
            fret_targets,
            onset_targets,
            valid_frames,
            attack_fret_weight=1.0,
            fret_distance_weight=1.0,
        )
        far_loss, far_values = transcription_loss(
            {
                "fret_logits": far_logits,
                "onset_logits": onset_logits,
            },
            fret_targets,
            onset_targets,
            valid_frames,
            attack_fret_weight=1.0,
            fret_distance_weight=1.0,
        )

        self.assertGreater(
            far_values["fret_distance_loss"],
            near_values["fret_distance_loss"],
        )
        self.assertGreater(far_loss, near_loss)

    def test_pitch_consistency_accepts_equivalent_string_positions(self) -> None:
        fret_targets = torch.zeros(1, 1, 6, dtype=torch.long)
        fret_targets[0, 0, 0] = 1
        onset_targets = torch.zeros(1, 1, 6)
        valid_frames = torch.ones(1, 1, dtype=torch.bool)
        same_pitch_logits = torch.zeros(1, 1, 6, 26)
        same_pitch_logits[..., 0] = 5
        same_pitch_logits[0, 0, 1, 6] = 10
        wrong_pitch_logits = torch.zeros(1, 1, 6, 26)
        wrong_pitch_logits[..., 0] = 5
        wrong_pitch_logits[0, 0, 1, 7] = 10
        onset_logits = torch.zeros(1, 1, 6)

        _same_loss, same_values = transcription_loss(
            {
                "fret_logits": same_pitch_logits,
                "onset_logits": onset_logits,
            },
            fret_targets,
            onset_targets,
            valid_frames,
            attack_fret_weight=1.0,
            pitch_consistency_weight=1.0,
        )
        _wrong_loss, wrong_values = transcription_loss(
            {
                "fret_logits": wrong_pitch_logits,
                "onset_logits": onset_logits,
            },
            fret_targets,
            onset_targets,
            valid_frames,
            attack_fret_weight=1.0,
            pitch_consistency_weight=1.0,
        )

        self.assertLess(
            same_values["pitch_consistency_loss"],
            wrong_values["pitch_consistency_loss"],
        )

    def test_legacy_model_config_preserves_mean_frequency_encoder(self) -> None:
        config = ModelConfig.from_dict(
            {
                "convolution_channels": [24, 48, 96],
                "recurrent_size": 96,
                "recurrent_layers": 2,
                "dropout": 0.15,
            }
        )
        model = StringFretNet(config)

        self.assertEqual(config.frequency_encoding, "mean")
        self.assertEqual(model.temporal.input_size, 96)
        outputs = model(torch.rand(1, 8, 216))
        self.assertEqual(outputs["fret_logits"].shape, (1, 8, 6, 26))

    def test_decoder_emits_two_attacks_on_the_same_string_and_fret(self) -> None:
        fret_probabilities = np.zeros((20, 6, 26), dtype=np.float32)
        fret_probabilities[..., 0] = 1
        fret_probabilities[:, 1, 0] = 0.05
        fret_probabilities[:, 1, 2] = 0.95
        onset_probabilities = np.zeros((20, 6), dtype=np.float32)
        onset_probabilities[3, 1] = 0.9
        onset_probabilities[10, 1] = 0.88

        events = decode_events(
            fret_probabilities,
            onset_probabilities,
            FeatureConfig(),
        )

        self.assertEqual(
            [(event.string, event.fret) for event in events],
            [(2, 1), (2, 1)],
        )
        self.assertLess(events[0].offset, events[1].onset + 0.001)

    def test_pitch_first_assignment_uses_acoustics_and_unique_strings(self) -> None:
        feature_config = FeatureConfig()
        fret_probabilities = np.full(
            (20, 6, 26),
            0.001,
            dtype=np.float32,
        )
        onset_probabilities = np.full((20, 6), 0.001, dtype=np.float32)
        frame = round(0.1 * feature_config.frames_per_second)
        fret_probabilities[frame : frame + 2, 0, 1] = 0.70
        fret_probabilities[frame : frame + 2, 1, 6] = 0.80
        fret_probabilities[frame : frame + 2, 1, 1] = 0.90
        onset_probabilities[frame, 0] = 0.75
        onset_probabilities[frame, 1] = 0.90
        pitch_events = [
            PitchEvent(0.1, 0.3, 64, confidence=0.9),
            PitchEvent(0.1, 0.3, 59, confidence=0.9),
        ]

        assigned = assign_pitch_events(
            pitch_events,
            fret_probabilities,
            onset_probabilities,
            feature_config,
            config=PositionAssignmentConfig(
                activity_delay=0,
                activity_context=0.023,
                onset_context=0,
                continuity_weight=0,
            ),
        )

        self.assertEqual(
            [(event.string, event.fret) for event in assigned],
            [(1, 0), (2, 0)],
        )

    def test_decoder_suppresses_adjacent_peaks_on_the_same_string(self) -> None:
        fret_probabilities = np.zeros((20, 6, 26), dtype=np.float32)
        fret_probabilities[..., 0] = 1
        fret_probabilities[:, 1, 0] = 0.05
        fret_probabilities[:, 1, 2] = 0.95
        onset_probabilities = np.zeros((20, 6), dtype=np.float32)
        onset_probabilities[3, 1] = 0.88
        onset_probabilities[5, 1] = 0.92

        events = decode_events(
            fret_probabilities,
            onset_probabilities,
            FeatureConfig(),
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].string, 2)
        self.assertEqual(events[0].fret, 1)
        self.assertAlmostEqual(
            events[0].onset,
            5 / FeatureConfig().frames_per_second,
            places=4,
        )

    def test_decoder_uses_post_onset_activity_to_choose_the_fret(self) -> None:
        fret_probabilities = np.zeros((12, 6, 26), dtype=np.float32)
        fret_probabilities[..., 0] = 1
        fret_probabilities[3, 0] = 0
        fret_probabilities[3, 0, 2] = 0.9
        fret_probabilities[4:6, 0] = 0
        fret_probabilities[4:6, 0, 4] = 0.9
        onset_probabilities = np.zeros((12, 6), dtype=np.float32)
        onset_probabilities[3, 0] = 0.92

        events = decode_events(
            fret_probabilities,
            onset_probabilities,
            FeatureConfig(),
            activity_delay=0.0,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].fret, 3)

    def test_decoder_can_delay_activity_sampling_past_the_pick_attack(self) -> None:
        fret_probabilities = np.zeros((12, 6, 26), dtype=np.float32)
        fret_probabilities[..., 0] = 1
        fret_probabilities[3, 0] = 0
        fret_probabilities[3, 0, 2] = 0.99
        fret_probabilities[4:6, 0] = 0
        fret_probabilities[4:6, 0, 4] = 0.8
        onset_probabilities = np.zeros((12, 6), dtype=np.float32)
        onset_probabilities[3, 0] = 0.92

        immediate = decode_events(
            fret_probabilities,
            onset_probabilities,
            FeatureConfig(),
            activity_context=0.023,
            activity_delay=0.0,
        )
        delayed = decode_events(
            fret_probabilities,
            onset_probabilities,
            FeatureConfig(),
            activity_context=0.023,
            activity_delay=0.012,
        )

        self.assertEqual(immediate[0].fret, 1)
        self.assertEqual(delayed[0].fret, 3)

    def test_decoder_can_require_combined_event_confidence(self) -> None:
        fret_probabilities = np.zeros((12, 6, 26), dtype=np.float32)
        fret_probabilities[..., 0] = 1
        fret_probabilities[4:8, 0, 0] = 0.7
        fret_probabilities[4:8, 0, 2] = 0.3
        onset_probabilities = np.zeros((12, 6), dtype=np.float32)
        onset_probabilities[3, 0] = 0.65

        events = decode_events(
            fret_probabilities,
            onset_probabilities,
            FeatureConfig(),
            onset_threshold=0.6,
            activity_threshold=0.2,
            activity_context=0.046,
            activity_delay=0.012,
            minimum_event_confidence=0.5,
        )

        self.assertEqual(events, [])

    def test_chunked_inference_preserves_frame_shapes(self) -> None:
        model = StringFretNet()
        feature_config = FeatureConfig()
        fret_probabilities, onset_probabilities = predict_probabilities(
            LoadedModel(model, feature_config, {}),
            np.zeros((40, 216), dtype=np.float32),
            device=torch.device("cpu"),
            chunk_frames=24,
            overlap_frames=8,
        )

        self.assertEqual(fret_probabilities.shape, (40, 6, 26))
        self.assertEqual(onset_probabilities.shape, (40, 6))

        position_model = StringFretNet(
            ModelConfig(pitch_conditioned_position=True)
        )
        _, _, position_probabilities = predict_probabilities_with_position(
            LoadedModel(position_model, feature_config, {}),
            np.zeros((40, 216), dtype=np.float32),
            device=torch.device("cpu"),
            chunk_frames=24,
            overlap_frames=8,
        )
        self.assertEqual(position_probabilities.shape, (40, 49, 6))

    def test_metrics_separate_pitch_from_tablature_accuracy(self) -> None:
        reference = [LabeledEvent(0.1, 0.4, 2, 1)]
        same_pitch_wrong_string = [LabeledEvent(0.1, 0.4, 3, 5)]

        metrics = evaluate_events(reference, same_pitch_wrong_string)

        self.assertEqual(metrics["pitch"]["f1"], 1.0)
        self.assertEqual(metrics["tablature"]["f1"], 0.0)
        self.assertEqual(
            metrics["decoder_errors"]["same_pitch_wrong_string"],
            1,
        )

    def test_metrics_keep_absolute_frets_when_a_capo_is_display_only(self) -> None:
        reference = [LabeledEvent(0.1, 0.4, 5, 1)]
        same_position = [LabeledEvent(0.1, 0.4, 5, 1)]
        shifted_position = [LabeledEvent(0.1, 0.4, 5, 2)]

        self.assertEqual(
            evaluate_events(reference, same_position)["tablature"]["f1"],
            1.0,
        )
        self.assertEqual(
            evaluate_events(reference, shifted_position)["tablature"]["f1"],
            0.0,
        )

    def test_metrics_measure_timing_relative_to_the_beat(self) -> None:
        reference = [LabeledEvent(0.25, 0.4, 2, 1)]
        predicted = [LabeledEvent(0.28, 0.4, 2, 1)]

        metrics = evaluate_events(
            reference,
            predicted,
            beats=[0.0, 0.5, 1.0],
        )

        self.assertEqual(metrics["beat_relative_onset"]["f1"], 1.0)
        self.assertAlmostEqual(
            metrics["beat_relative_onset"]["mean_beat_error"],
            0.06,
        )

    def test_metrics_measure_repeated_same_fret_attacks_separately(self) -> None:
        reference = [
            LabeledEvent(0.1, 0.3, 2, 1),
            LabeledEvent(0.5, 0.7, 2, 1),
        ]
        predicted = [
            LabeledEvent(0.1, 0.3, 2, 1),
            LabeledEvent(0.5, 0.7, 2, 1),
        ]

        metrics = evaluate_events(reference, predicted)

        self.assertEqual(metrics["repeated_tablature"]["recall"], 1.0)

    def test_metrics_exclude_reference_and_predictions_by_onset(self) -> None:
        reference = [
            LabeledEvent(0.1, 0.3, 2, 1),
            LabeledEvent(0.55, 0.7, 3, 2),
        ]
        predicted = [
            LabeledEvent(0.1, 0.3, 2, 1),
            LabeledEvent(0.6, 0.75, 4, 7),
        ]

        metrics = evaluate_events(
            reference,
            predicted,
            excluded_ranges=[
                ExcludedRange(0.5, 0.7, "inaudible light strum"),
            ],
        )

        self.assertEqual(metrics["tablature"]["f1"], 1.0)
        self.assertEqual(metrics["exclusions"]["range_count"], 1)
        self.assertEqual(metrics["exclusions"]["excluded_reference_events"], 1)
        self.assertEqual(metrics["exclusions"]["excluded_predicted_events"], 1)

    def test_production_gate_rejects_a_small_or_inaccurate_report(self) -> None:
        report = {
            "track_count": 1,
            "aggregate": {
                "onset": {"f1": 0.9},
                "tablature": {"f1": 0.7},
                "repeated_tablature": {"recall": 0.8},
            },
        }
        baseline = {"metrics": {"tablature": {"f1": 0.6}}}

        result = evaluate_gate(
            report,
            baseline,
            minimum_tracks=100,
            minimum_onset_f1=0.8,
            minimum_tablature_f1=0.75,
            minimum_improvement=0.15,
            minimum_repeated_recall=0.85,
        )

        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["minimum_tracks"])
        self.assertFalse(result["checks"]["tablature_f1"])
        self.assertFalse(result["checks"]["baseline_improvement"])

    def test_guitar_techs_midi_preserves_string_and_applies_alignment(
        self,
    ) -> None:
        with self.temporary_directory() as directory:
            midi_path = Path(directory) / "technique.mid"
            midi = pretty_midi.PrettyMIDI()
            high_e = pretty_midi.Instrument(program=0, name="e")
            high_e.notes.append(
                pretty_midi.Note(
                    velocity=100,
                    pitch=69,
                    start=0.1,
                    end=0.4,
                )
            )
            low_e = pretty_midi.Instrument(program=0, name="E")
            low_e.notes.append(
                pretty_midi.Note(
                    velocity=100,
                    pitch=43,
                    start=0.5,
                    end=0.8,
                )
            )
            midi.instruments.extend((high_e, low_e))
            midi.write(str(midi_path))

            events = midi_events(midi_path, "vibrato")
            shifted = shifted_events(events, -0.02, 1.0)

            self.assertEqual(
                [(event.string, event.pitch) for event in shifted],
                [(1, 69), (6, 43)],
            )
            self.assertAlmostEqual(shifted[0].onset, 0.08)
            self.assertTrue(
                all(
                    isinstance(event, TechniqueEvent)
                    and event.technique == "vibrato"
                    for event in shifted
                )
            )

    def test_agpt_frequency_labels_are_converted_to_midi(self) -> None:
        self.assertEqual(normalized_pitch("440"), 69)
        self.assertEqual(normalized_pitch("494"), 71)
        self.assertEqual(normalized_pitch("64"), 64)

    def test_guitar_techs_merges_contiguous_pitch_fragments(self) -> None:
        events = [
            TechniqueEvent(0.0, 0.2, 60, 3, "bend"),
            TechniqueEvent(0.18, 0.4, 61, 3, "bend"),
            TechniqueEvent(0.19, 0.25, 68, 2, "bend"),
            TechniqueEvent(0.8, 1.1, 64, 2, "bend"),
        ]

        merged = merge_gesture_fragments(events)

        self.assertEqual(len(merged), 2)
        self.assertEqual(
            (merged[0].onset, merged[0].offset, merged[0].pitch),
            (0.0, 0.4, 60),
        )
        self.assertEqual(merged[0].string, 3)

    def test_technique_dataset_extracts_fixed_event_patches(self) -> None:
        with self.temporary_directory() as directory:
            root = Path(directory)
            manifest = root / "techniques.jsonl"
            audio = root / "audio.wav"
            audio.touch()
            write_technique_manifest(
                manifest,
                [
                    TechniqueTrack(
                        track_id="technique-track",
                        group_id="player-1",
                        audio="audio.wav",
                        split="train",
                        duration=1.0,
                        events=[
                            TechniqueEvent(
                                onset=0.02,
                                offset=0.4,
                                pitch=64,
                                string=1,
                                technique="vibrato",
                            )
                        ],
                    )
                ],
            )
            with patch(
                "stringtrace_ml.technique_model.extract_log_cqt",
                return_value=np.ones((180, 216), dtype=np.float32),
            ):
                dataset = TechniqueEventDataset(
                    manifest,
                    split="train",
                )
                item = dataset[0]

            self.assertEqual(item["features"].shape, (138, 144))
            self.assertEqual(item["dynamics"].shape, (138, 8))
            self.assertEqual(item["summary"].shape, (878,))
            self.assertTrue(
                torch.equal(
                    item["dynamics"][60:],
                    torch.zeros_like(item["dynamics"][60:]),
                )
            )
            self.assertEqual(int(item["label"]), 5)
            logits = TechniqueNet()(
                item["features"].unsqueeze(0),
                item["dynamics"].unsqueeze(0),
            )
            self.assertEqual(logits.shape, (1, 6))

    def test_technique_confidence_gate_abstains_below_precision_target(
        self,
    ) -> None:
        probabilities = np.asarray(
            [
                [0.95, 0.01, 0.01, 0.01, 0.01, 0.01],
                [0.90, 0.02, 0.02, 0.02, 0.02, 0.02],
                [0.85, 0.03, 0.03, 0.03, 0.03, 0.03],
                [0.60, 0.20, 0.05, 0.05, 0.05, 0.05],
                [0.10, 0.70, 0.05, 0.05, 0.05, 0.05],
                [0.10, 0.65, 0.10, 0.05, 0.05, 0.05],
            ],
            dtype=np.float32,
        )
        references = np.asarray([0, 0, 0, 1, 1, 1])

        report = select_confidence_gates(
            probabilities,
            references,
            minimum_precision=0.90,
            minimum_predictions=3,
            minimum_class_f1=0.65,
        )

        self.assertTrue(report["gates"]["pick"]["supported"])
        self.assertAlmostEqual(
            report["gates"]["pick"]["threshold"],
            0.85,
        )
        self.assertFalse(report["gates"]["bend"]["supported"])
        self.assertEqual(report["selective"]["accepted"], 3)
        self.assertEqual(report["selective"]["unknown"], 3)
        self.assertLessEqual(
            expected_calibration_error(probabilities, references),
            1,
        )

    def test_technique_descriptor_summary_has_fixed_size(self) -> None:
        config = TechniqueDescriptorConfig()
        descriptors = np.ones((100, 90), dtype=np.float32)

        summary = summarize_event_descriptors(
            descriptors,
            onset=0.25,
            config=config,
        )

        self.assertEqual(summary.shape, (180,))
        self.assertTrue(np.isfinite(summary).all())

    def test_technique_production_gate_disables_unverified_labels(self) -> None:
        checkpoint = {"name": "model.pt", "sha256": "abc"}
        validation = {
            "split": "validation",
            "checkpoint": checkpoint,
            "calibration": {"temperature": 1.25, "ece": 0.04},
            "gates": {
                label: {
                    "supported": label in {"bend", "pinch-harmonic"},
                    "threshold": 0.8,
                    "validation_precision": 0.95,
                }
                for label in (
                    "pick",
                    "bend",
                    "harmonic",
                    "palm-mute",
                    "pinch-harmonic",
                    "vibrato",
                )
            },
        }
        test_classes = {
            label: {
                "support": 30 if label == "bend" else 0,
                "precision": 0.96 if label == "bend" else 0.0,
                "f1": 0.8 if label == "bend" else 0.0,
            }
            for label in validation["gates"]
        }
        test = {
            "split": "test",
            "checkpoint": checkpoint,
            "selective": {"per_class": test_classes},
        }

        gate = build_technique_gate(validation, test)

        self.assertEqual(gate["approved_labels"], ["bend"])
        self.assertTrue(gate["gates"]["bend"]["supported"])
        self.assertFalse(gate["gates"]["pinch-harmonic"]["supported"])
        self.assertFalse(gate["all_labels_approved"])

    def test_binary_technique_gate_requires_precision_in_each_domain(
        self,
    ) -> None:
        scores = np.asarray([0.9, 0.8, 0.7, 0.95, 0.85, 0.75])
        references = np.asarray([1, 1, 0, 1, 0, 0])
        domains = np.asarray(["a", "a", "a", "b", "b", "b"])

        gate = select_binary_threshold(
            scores,
            references,
            domains,
            1,
            minimum_precision=0.9,
            minimum_predictions=2,
            minimum_domain_predictions=1,
        )

        self.assertTrue(gate["supported"])
        self.assertAlmostEqual(gate["threshold"], 0.9)
        self.assertEqual(
            gate["validation_domains"]["b"]["false_positive"],
            0,
        )

    def test_technique_inference_marks_only_gate_approved_label(self) -> None:
        with self.temporary_directory() as directory:
            root = Path(directory)
            checkpoint_path = root / "technique.pt"
            gate_path = root / "gate.json"
            feature_config = TechniqueFeatureConfig()
            model_config = TechniqueModelConfig()
            model = TechniqueNet(feature_config, model_config)
            for parameter in model.parameters():
                parameter.data.zero_()
            model.classifier[-1].bias.data[1] = 5
            torch.save(
                {
                    "format_version": 3,
                    "model_state": model.state_dict(),
                    "model_config": model_config.to_dict(),
                    "feature_config": feature_config.to_dict(),
                    "metadata": {
                        "name": "test-technique-model",
                        "version": "test-v1",
                    },
                },
                checkpoint_path,
            )
            gates = {
                label: {
                    "supported": label == "bend",
                    "threshold": 0.5 if label == "bend" else None,
                }
                for label in (
                    "pick",
                    "bend",
                    "harmonic",
                    "palm-mute",
                    "pinch-harmonic",
                    "vibrato",
                )
            }
            gate_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "technique-production-gate-v1",
                        "checkpoint": {
                            "name": checkpoint_path.name,
                            "sha256": sha256_file(checkpoint_path),
                        },
                        "calibration": {"temperature": 1.0},
                        "approved_labels": ["bend"],
                        "gates": gates,
                    }
                ),
                encoding="utf-8",
            )
            event = type(
                "Event",
                (),
                {"start": 0.1, "end": 0.5, "pitch": 64},
            )()
            with patch(
                "stringtrace_ml.technique_inference.extract_log_cqt",
                return_value=np.ones((100, 216), dtype=np.float32),
            ):
                decisions, metadata = predict_techniques(
                    root / "unused.wav",
                    [event],
                    checkpoint_path,
                    gate_path,
                    device=torch.device("cpu"),
                )

            self.assertEqual(decisions[0].technique, "bend")
            self.assertGreater(decisions[0].confidence, 0.5)
            self.assertEqual(metadata["approved_labels"], ["bend"])

    def test_idmt_import_maps_styles_and_reverses_string_number(self) -> None:
        with self.temporary_directory() as directory:
            root = Path(directory)
            audio_directory = root / "audio"
            audio_directory.mkdir()
            audio_path = audio_directory / "AR_test.wav"
            soundfile.write(audio_path, np.zeros(22050), 22050)
            annotation_path = root / "AR_test.xml"
            annotation_path.write_text(
                """<?xml version="1.0" encoding="UTF-8"?>
<instrumentRecording>
  <globalParameter>
    <audioFileName>\\AR_test.wav</audioFileName>
    <instrumentModel>Test Guitar</instrumentModel>
    <recordingArtist>Test Player</recordingArtist>
  </globalParameter>
  <transcription>
    <event>
      <onsetSec>0.1</onsetSec><offsetSec>0.8</offsetSec>
      <pitch>45</pitch><stringNumber>1</stringNumber>
      <excitationStyle>MU</excitationStyle>
      <expressionStyle>NO</expressionStyle>
    </event>
    <event>
      <onsetSec>0.2</onsetSec><offsetSec>0.9</offsetSec>
      <pitch>64</pitch><stringNumber>6</stringNumber>
      <excitationStyle>PK</excitationStyle>
      <expressionStyle>HA</expressionStyle>
    </event>
  </transcription>
</instrumentRecording>
""",
                encoding="utf-8",
            )

            track = parse_idmt_annotation(
                annotation_path,
                audio_directory,
                root / "manifest.jsonl",
            )

            self.assertIsNotNone(track)
            assert track is not None
            self.assertEqual(track.split, "test")
            self.assertEqual(
                [
                    (event.string, event.technique)
                    for event in track.events
                ],
                [(6, "palm-mute"), (1, "harmonic")],
            )

    def test_guitarset_import_maps_each_annotation_to_its_string(self) -> None:
        with self.temporary_directory() as directory:
            root = Path(directory)
            track_id = "00_BN1-120-C_comp"
            (root / f"{track_id}_mic.wav").touch()
            annotations = []
            pitches = [40, 45, 50, 55, 59, 64]
            for source_index, pitch in enumerate(pitches):
                annotations.append(
                    {
                        "namespace": "note_midi",
                        "annotation_metadata": {
                            "data_source": str(source_index),
                        },
                        "data": [
                            {
                                "time": 0.1,
                                "duration": 0.2,
                                "value": pitch,
                                "confidence": None,
                            }
                        ],
                    }
                )
            jams_path = root / f"{track_id}.jams"
            jams_path.write_text(
                json.dumps(
                    {
                        "file_metadata": {"duration": 1.0},
                        "annotations": annotations,
                    }
                ),
                encoding="utf-8",
            )

            annotation = parse_jams(jams_path, root, "mic")

            self.assertEqual(annotation.split, "train")
            self.assertEqual(
                [(event.string, event.fret) for event in annotation.events],
                [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)],
            )
            reloaded_manifest = root / "manifest.jsonl"
            write_manifest(reloaded_manifest, [annotation])
            self.assertEqual(load_manifest(reloaded_manifest)[0].track_id, track_id)


if __name__ == "__main__":
    unittest.main()

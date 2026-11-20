from __future__ import annotations

import hashlib
import json
import pickle
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pretty_midi
import soundfile
import torch

from stringtrace_ml.beat_reference import (
    adjudicate_reviews as adjudicate_beat_reviews,
    compare_reviews as compare_beat_reviews,
    create_review as create_beat_review,
    evaluate_gate as evaluate_beat_gate,
    evaluate_review as evaluate_beat_review,
    submit_review as submit_beat_review,
    validate_review as validate_beat_review,
)
from stringtrace_ml.analyze_candidate_sequence_confidence import (
    distribution_summary as confidence_distribution_summary,
    separation_metrics as confidence_separation_metrics,
)
from stringtrace_ml.candidate_activity import (
    CandidateActivityConfig,
    CandidateActivityDataset,
    CandidateActivityHead,
    candidate_group_counts,
    candidate_sampling_weights,
    candidate_temporal_context,
)
from stringtrace_ml.candidate_margin import (
    calibration_partition_keys,
    candidate_string_probability_margins,
    collect_margin_calibration_values,
    load_candidate_margin_calibration,
    select_margin_threshold,
    stratified_calibration_partition,
)
from stringtrace_ml.candidate_sequence import (
    CandidateSequenceActivityModel,
    CandidateSequenceConfidenceModel,
    CandidateSequenceConfig,
    CandidateSequenceGroup,
    CandidateSequenceJointModel,
    CandidateSequenceStringModel,
    CandidateSequenceWindowDataset,
    candidate_joint_log_probabilities,
    candidate_sequence_activity_audit,
    candidate_sequence_activity_loss,
    candidate_sequence_confidence_loss,
    candidate_group_structured_objective,
    candidate_sequence_joint_loss,
    candidate_sequence_loss,
    candidate_sequence_structured_loss,
    collate_candidate_sequence_windows,
    combine_candidate_activity_confidence_logits,
    refine_candidate_activity_logits,
    refine_candidate_joint_logits,
    refine_candidate_position_logits,
    sequence_window_ranges,
)
from stringtrace_ml.checkpoint_governance import (
    evaluate_checkpoint_governance,
    load_checkpoint_file,
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
from stringtrace_ml.data_sources import (
    audit_records,
    load_source_ledger,
    manifest_identity,
)
from stringtrace_ml.dataset import GuitarTabDataset, augment_cqt_features
from stringtrace_ml.download_guitarset import (
    DownloadSpec,
    assemble_download,
    byte_ranges,
    part_path,
    validate_content_range,
)
from stringtrace_ml.evaluate import model_identity
from stringtrace_ml.evaluate_onset_position import (
    BASIC_PITCH_CACHE_VERSION,
    aggregate_stage_traces,
    assign_detected_events,
    direct_pitch_onset_candidates,
    merge_candidate_identities,
    merge_predictions,
    parse_arguments,
    unmatched_onset_candidates,
    unmatched_pitch_candidates,
)
from stringtrace_ml.evaluate_technique import (
    expected_calibration_error,
    select_confidence_gates,
    sha256_file,
)
from stringtrace_ml.features import FeatureConfig
from stringtrace_ml.import_guitarset import parse_jams
from stringtrace_ml.import_guitar_techs import (
    discover_recordings,
    merge_gesture_fragments,
    midi_events,
    recording_events,
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
from stringtrace_ml.generate_synthetic import (
    build_training_track,
    generate_symbolic_dataset,
    load_midi_symbolic,
)
from stringtrace_ml.import_agpt_techniques import normalized_pitch
from stringtrace_ml.metrics import (
    evaluate_beats,
    evaluate_chords,
    evaluate_events,
)
from stringtrace_ml.manifest_isolation import audit_manifest_isolation
from stringtrace_ml.model import (
    OPEN_STRING_PITCHES,
    PITCH_MIN,
    ModelConfig,
    StringFretNet,
    build_pitch_onset_targets,
    transcription_loss,
)
from stringtrace_ml.onset_position import (
    OnsetPositionConfig,
    OnsetPositionDataset,
    OnsetPositionNet,
    OnsetGroupIndex,
    augment_pitch_candidates,
    assign_candidate_pitch_group,
    assign_hierarchical_candidate_pitch_group,
    assign_pitch_group,
    basic_pitch_cache_path,
    candidate_harmonic_cqt_indexes,
    candidate_position_loss,
    candidate_position_probabilities,
    candidate_position_query_weights,
    candidate_position_queries,
    candidate_string_temporal_context,
    extract_onset_patch,
    group_event_indexes,
    hierarchical_candidate_probabilities,
    normalize_candidate_attack_evidence,
    pitch_mask,
)
from stringtrace_ml.onset_position_bundle import (
    BUNDLE_KIND,
    build_research_bundle,
    load_research_bundle,
)
from stringtrace_ml.prepare_candidate_cache import (
    collect_candidate_audio,
    prepare_candidate_caches,
    validate_candidate_cache,
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
from stringtrace_ml.scan_candidate_none_bias import (
    candidate_bias_metrics,
    parse_arguments as parse_candidate_none_bias_arguments,
)
from stringtrace_ml.reference_coverage import build_coverage_report
from stringtrace_ml.reference_queue import (
    analyze_model_disagreements,
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
    STANDARD_TUNING_LIST,
    ChordRegion,
    ExcludedRange,
    LabeledEvent,
    TrackAnnotation,
    load_manifest,
    reject_quarantined_references,
    write_manifest,
)
from stringtrace_ml.score_gate import evaluate_score_gate
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
from stringtrace_ml.train_candidate_sequence_activity import (
    window_sampling_plan,
)
from stringtrace_ml.train_candidate_sequence_v3 import (
    EXPECTED_BASELINE,
    candidate_cache_set_identity,
    epoch_is_eligible,
    epoch_selection_key,
    run_training_epoch as run_v3_training_epoch,
    select_v3_tracks,
    validate_candidate_cache_set,
    validate_epoch_zero,
    verify_artifact_hashes,
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
            self.assertEqual(
                annotations[0].provenance["source_id"],
                "guitar-techs-zenodo-14963133",
            )
            self.assertEqual(
                annotations[0].provenance["license"],
                "CC-BY-4.0",
            )
            composed, composition_report = compose_training_manifests(
                [output_manifest],
                root / "composed" / "manifest.jsonl",
            )
            self.assertEqual(len(composed), 1)
            self.assertEqual(
                composition_report["source_policy_violations"],
                0,
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
                        provenance={
                            "dataset": "first",
                            "source_id": "stringtrace-synthetic-v1",
                            "source_version": "1",
                            "rights_basis": "original",
                        },
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
                        provenance={
                            "dataset": "second",
                            "source_id": "stringtrace-synthetic-v1",
                            "source_version": "1",
                            "rights_basis": "original",
                        },
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

    def test_source_ledger_audit_is_fail_closed_and_path_safe(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            audio_path = root / "audio.wav"
            audio_path.write_bytes(b"audio")
            manifest_path = root / "manifest.jsonl"
            synthetic = TrackAnnotation(
                track_id="synthetic-track",
                group_id="synthetic-group",
                audio=audio_path.name,
                split="train",
                duration=1.0,
                events=[],
                provenance={
                    "dataset": "stringtrace-symbolic-synthetic",
                    "source_id": "stringtrace-synthetic-v1",
                    "source_version": "1",
                    "rights_basis": "original",
                },
            )
            write_manifest(manifest_path, [synthetic])
            ledger = load_source_ledger()

            report = audit_records(
                [synthetic],
                purpose="training",
                ledger=ledger,
            )
            identity = manifest_identity(
                manifest_path,
                [synthetic],
                purpose="training",
                ledger=ledger,
            )
            restricted = TrackAnnotation(
                track_id="idmt-track",
                group_id="idmt-group",
                audio=audio_path.name,
                split="train",
                duration=1.0,
                events=[],
                provenance={
                    "dataset": "IDMT-SMT-GUITAR_V2",
                    "source_id": "idmt-smt-guitar-v2-zenodo-7544110",
                    "source_version": "2",
                    "license": "CC-BY-NC-ND-4.0",
                },
            )
            rejected = audit_records(
                [restricted],
                purpose="training",
                ledger=ledger,
            )

        self.assertTrue(report["passed"])
        self.assertEqual(
            identity["sources"],
            {"stringtrace-synthetic-v1": 1},
        )
        self.assertFalse(rejected["passed"])
        self.assertIn(
            "purpose_not_allowed:"
            "idmt-smt-guitar-v2-zenodo-7544110:training",
            rejected["violations"],
        )
        self.assertNotIn(str(root), json.dumps(identity))

    def test_manifest_isolation_detects_scoped_and_audio_overlap(self) -> None:
        reference = TrackAnnotation(
            track_id="train-track",
            group_id="train-group",
            audio="train.wav",
            split="train",
            duration=1.0,
            events=[],
            provenance={
                "dataset": "source-a",
                "source_id": "source-a-v1",
                "player_id": "player-1",
                "audio_sha256": "a" * 64,
            },
        )
        isolated = TrackAnnotation(
            track_id="validation-track",
            group_id="validation-group",
            audio="validation.wav",
            split="validation",
            duration=1.0,
            events=[],
            provenance={
                "dataset": "source-b",
                "source_id": "source-b-v1",
                "player_id": "player-1",
                "audio_sha256": "b" * 64,
            },
        )
        overlapping = TrackAnnotation(
            track_id="other-track",
            group_id="other-group",
            audio="other.wav",
            split="validation",
            duration=1.0,
            events=[],
            provenance={
                "dataset": "source-a",
                "source_id": "source-a-v1",
                "player_id": "player-1",
                "source_audio_sha256": "a" * 64,
            },
        )

        isolated_report = audit_manifest_isolation(
            [reference],
            [isolated],
        )
        overlap_report = audit_manifest_isolation(
            [reference],
            [overlapping],
        )

        self.assertTrue(isolated_report["passed"])
        self.assertFalse(overlap_report["passed"])
        self.assertEqual(
            overlap_report["overlaps"]["performer"]["count"],
            1,
        )
        self.assertEqual(
            overlap_report["overlaps"]["audio_sha256"]["count"],
            1,
        )

    def test_candidate_cache_collection_deduplicates_and_limits_root(
        self,
    ) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            audio_path = root / "shared.wav"
            audio_path.touch()
            manifest_path = root / "manifest.jsonl"
            write_manifest(
                manifest_path,
                [
                    TrackAnnotation(
                        track_id="train",
                        group_id="train",
                        audio=audio_path.name,
                        split="train",
                        duration=1.0,
                        events=[],
                    ),
                    TrackAnnotation(
                        track_id="validation",
                        group_id="validation",
                        audio=audio_path.name,
                        split="validation",
                        duration=1.0,
                        events=[],
                    ),
                ],
            )

            audio_paths, reports = collect_candidate_audio(
                [manifest_path],
                splits=frozenset({"train"}),
                allowed_root=root,
            )

            self.assertEqual(audio_paths, [audio_path])
            self.assertEqual(reports[0]["records"], 2)
            self.assertEqual(reports[0]["selected_records"], 1)
            with self.assertRaisesRegex(ValueError, "outside allowed root"):
                collect_candidate_audio(
                    [manifest_path],
                    allowed_root=root / "nested",
                )

    def test_candidate_cache_validation_rejects_malformed_event(self) -> None:
        with self.temporary_directory() as temporary:
            cache_path = Path(temporary) / "cache.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "events": [
                            {
                                "onset": 0.1,
                                "offset": 0.2,
                                "pitch": 64,
                                "confidence": 2.0,
                            }
                        ],
                        "dropped": 0,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "out of range"):
                validate_candidate_cache(cache_path)

    def test_candidate_cache_preparation_counts_existing_and_generated(
        self,
    ) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            cache_directory = root / "cache"
            cache_directory.mkdir()
            existing_audio = root / "existing.wav"
            generated_audio = root / "generated.wav"
            existing_audio.touch()
            generated_audio.touch()

            def write_cache(audio_path: Path) -> Path:
                cache_path = basic_pitch_cache_path(
                    audio_path,
                    cache_directory,
                    onset_threshold=0.35,
                    frame_threshold=0.25,
                    minimum_note_length=70.0,
                )
                cache_path.write_text(
                    json.dumps(
                        {
                            "version": 1,
                            "events": [
                                {
                                    "onset": 0.1,
                                    "offset": 0.2,
                                    "pitch": 64,
                                    "confidence": 0.8,
                                }
                            ],
                            "dropped": 1,
                        }
                    ),
                    encoding="utf-8",
                )
                return cache_path

            write_cache(existing_audio)

            def generate(
                audio_path: Path,
                *_args: object,
                **_kwargs: object,
            ) -> tuple[list[object], int]:
                write_cache(audio_path)
                return [], 1

            with patch(
                "stringtrace_ml.prepare_candidate_cache.basic_pitch_events",
                side_effect=generate,
            ) as mocked:
                summary = prepare_candidate_caches(
                    [existing_audio, generated_audio],
                    cache_directory,
                    onset_threshold=0.35,
                    frame_threshold=0.25,
                    minimum_note_length=70.0,
                )

            self.assertEqual(mocked.call_count, 1)
            self.assertEqual(summary["audio_files"], 2)
            self.assertEqual(summary["existing"], 1)
            self.assertEqual(summary["generated"], 1)
            self.assertEqual(summary["events"], 2)
            self.assertEqual(summary["dropped"], 2)
            self.assertEqual(len(str(summary["cache_set_sha256"])), 64)

    def test_checkpoint_governance_requires_current_release_sources(
        self,
    ) -> None:
        ledger = load_source_ledger()
        compliant = {
            "metadata": {
                "training_data": {
                    "name": "manifest.jsonl",
                    "sha256": "a" * 64,
                    "records": 8,
                    "sources": {
                        "stringtrace-synthetic-v1": 8,
                    },
                    "source_ledger": {
                        "version": ledger.ledger_version,
                        "sha256": ledger.sha256,
                    },
                }
            }
        }
        passed = evaluate_checkpoint_governance(
            compliant,
            release_channel="production",
        )
        legacy = evaluate_checkpoint_governance(
            {"metadata": {"manifest_name": "manifest.jsonl"}},
            release_channel="production",
        )
        path_leak = evaluate_checkpoint_governance(
            {
                **compliant,
                "debug_path": "/private/training/audio.wav",
            },
            release_channel="production",
        )
        validation_only = evaluate_checkpoint_governance(
            {
                "metadata": {
                    "validation_data": compliant["metadata"][
                        "training_data"
                    ]
                }
            },
            release_channel="production",
        )
        empty_counts = evaluate_checkpoint_governance(
            {
                "metadata": {
                    "training_data": {
                        **compliant["metadata"]["training_data"],
                        "records": 0,
                        "sources": {
                            "stringtrace-synthetic-v1": 0,
                        },
                    }
                }
            },
            release_channel="production",
        )
        mismatched_counts = evaluate_checkpoint_governance(
            {
                "metadata": {
                    "training_data": {
                        **compliant["metadata"]["training_data"],
                        "records": 8,
                        "sources": {
                            "stringtrace-synthetic-v1": 7,
                        },
                    }
                }
            },
            release_channel="production",
        )

        self.assertTrue(passed["passed"])
        self.assertFalse(legacy["passed"])
        self.assertIn(
            "missing_training_data_identity",
            legacy["violations"],
        )
        self.assertFalse(path_leak["passed"])
        self.assertIn(
            "absolute_local_path_embedded",
            path_leak["violations"],
        )
        self.assertFalse(validation_only["passed"])
        self.assertIn(
            "missing_training_data_identity",
            validation_only["violations"],
        )
        self.assertFalse(empty_counts["passed"])
        self.assertIn("invalid_record_count", empty_counts["violations"])
        self.assertIn("invalid_source_count", empty_counts["violations"])
        self.assertFalse(mismatched_counts["passed"])
        self.assertIn(
            "source_count_total_mismatch",
            mismatched_counts["violations"],
        )

    def test_checkpoint_governance_loads_plain_pickle_checkpoints(
        self,
    ) -> None:
        with self.temporary_directory() as directory:
            checkpoint_path = Path(directory) / "technique-svm.pkl"
            checkpoint = {
                "format_version": 1,
                "metadata": {"name": "stringtrace-technique-svm"},
            }
            with checkpoint_path.open("wb") as stream:
                pickle.dump(checkpoint, stream)

            self.assertEqual(
                load_checkpoint_file(checkpoint_path),
                checkpoint,
            )

    def test_symbolic_synthesis_is_deterministic_and_uses_absolute_frets(
        self,
    ) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            source.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "rights_basis": "original",
                        "split": "train",
                        "bpm": 120,
                        "time_signature": [4, 4],
                        "capo": 2,
                        "events": [
                            {
                                "onset_beats": 0,
                                "duration_beats": 0.5,
                                "string": 6,
                                "fret": 2,
                                "velocity": 90,
                            },
                            {
                                "onset_beats": 1,
                                "duration_beats": 0.5,
                                "string": 5,
                                "fret": 4,
                                "velocity": 100,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            first, first_report = generate_symbolic_dataset(
                [source],
                root / "first",
                seed=42,
                sample_rate=8000,
            )
            second, second_report = generate_symbolic_dataset(
                [source],
                root / "second",
                seed=42,
                sample_rate=8000,
            )
            first_audio = first[0].resolve_audio(
                root / "first" / "manifest.jsonl"
            )
            second_audio = second[0].resolve_audio(
                root / "second" / "manifest.jsonl"
            )
            first_audio_sha256 = sha256_file(first_audio)
            second_audio_sha256 = sha256_file(second_audio)
            duplicate = root / "duplicate.json"
            duplicate_value = json.loads(source.read_text(encoding="utf-8"))
            duplicate_value["split"] = "validation"
            duplicate.write_text(
                json.dumps(duplicate_value),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "cross dataset records or splits",
            ):
                generate_symbolic_dataset(
                    [source, duplicate],
                    root / "rejected",
                    seed=42,
                    sample_rate=8000,
                )

        self.assertEqual(first[0].capo, 2)
        self.assertEqual(
            [(event.string, event.fret) for event in first[0].events],
            [(6, 2), (5, 4)],
        )
        self.assertEqual(first[0].track_id, second[0].track_id)
        self.assertEqual(first_audio_sha256, second_audio_sha256)
        self.assertEqual(
            first_report["manifest"]["sha256"],
            second_report["manifest"]["sha256"],
        )
        self.assertNotIn(str(root), json.dumps(first_report))

    def test_training_synthesis_covers_strings_frets_and_repeats(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            first = build_training_track(
                0,
                root / "first",
                8000,
                np.random.default_rng(42),
            )
            second = build_training_track(
                0,
                root / "second",
                8000,
                np.random.default_rng(42),
            )
            first_audio = first.resolve_audio(root / "first" / "manifest.jsonl")
            second_audio = second.resolve_audio(
                root / "second" / "manifest.jsonl"
            )
            repeated = Counter(
                (event.string, event.fret)
                for event in first.events
            )
            first_audio_sha256 = sha256_file(first_audio)
            second_audio_sha256 = sha256_file(second_audio)

        self.assertEqual(first.split, "train")
        self.assertEqual(
            {event.string for event in first.events},
            {1, 2, 3, 4, 5, 6},
        )
        self.assertGreaterEqual(max(event.fret for event in first.events), 20)
        self.assertTrue(any(count > 1 for count in repeated.values()))
        self.assertEqual(first_audio_sha256, second_audio_sha256)
        self.assertEqual(
            first.provenance["generator"],
            "physical-fret-training-v1",
        )

    def test_midi_symbolic_assignment_is_playable_and_deterministic(
        self,
    ) -> None:
        with self.temporary_directory() as temporary:
            path = Path(temporary) / "source.mid"
            midi = pretty_midi.PrettyMIDI(initial_tempo=100)
            instrument = pretty_midi.Instrument(program=25)
            instrument.notes.extend(
                [
                    pretty_midi.Note(
                        velocity=90,
                        pitch=52,
                        start=0,
                        end=0.5,
                    ),
                    pretty_midi.Note(
                        velocity=90,
                        pitch=55,
                        start=0,
                        end=0.5,
                    ),
                ]
            )
            midi.instruments.append(instrument)
            midi.write(str(path))

            first = load_midi_symbolic(
                path,
                default_split="train",
                rights_basis="original",
            )
            second = load_midi_symbolic(
                path,
                default_split="train",
                rights_basis="original",
            )

        self.assertEqual(first, second)
        self.assertEqual(len(first.events), 2)
        self.assertEqual(len({event.string for event in first.events}), 2)
        self.assertEqual(
            sorted(
                STANDARD_TUNING_LIST[event.string - 1] + event.fret
                for event in first.events
            ),
            [52, 55],
        )

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
                            provenance={
                                "dataset": "fixture",
                                "source_id": "stringtrace-synthetic-v1",
                                "source_version": "1",
                                "rights_basis": "original",
                            },
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
                        "source_id": {
                            "Guitar-TECHS": (
                                "guitar-techs-zenodo-14963133"
                            ),
                            "IDMT-SMT-GUITAR_V2": (
                                "idmt-smt-guitar-v2-zenodo-7544110"
                            ),
                        }[dataset],
                    },
                )

            write_technique_manifest(
                source_path,
                [
                    track(
                        "approved",
                        dataset="Guitar-TECHS",
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
                        dataset="Guitar-TECHS",
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
                {"Guitar-TECHS": 1},
            )
            self.assertEqual(
                counts["rejected_tracks_by_dataset_and_reason"],
                {
                    "Guitar-TECHS:non_selected_split": 1,
                    "IDMT-SMT-GUITAR_V2:license_mismatch": 1,
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
                        "source_id": {
                            "Guitar-TECHS": (
                                "guitar-techs-zenodo-14963133"
                            ),
                            "IDMT-SMT-GUITAR_V2": (
                                "idmt-smt-guitar-v2-zenodo-7544110"
                            ),
                            "AG-PT-set": (
                                "agpt-set-zenodo-10159492"
                            ),
                        }[dataset],
                    },
                )

            write_technique_manifest(
                source_path,
                [
                    track(
                        "approved-validation",
                        dataset="Guitar-TECHS",
                        split="validation",
                    ),
                    track(
                        "approved-train",
                        dataset="Guitar-TECHS",
                        split="train",
                    ),
                    track(
                        "restricted-validation",
                        dataset="IDMT-SMT-GUITAR_V2",
                        split="validation",
                    ),
                    track(
                        "approved-agpt-validation",
                        dataset="AG-PT-set",
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

            self.assertEqual(len(converted), 3)
            self.assertEqual(len(loaded), 3)
            self.assertTrue(
                all(track.split == "validation" for track in loaded)
            )
            self.assertTrue(
                all(track.provenance["evaluation_only"] for track in loaded)
            )
            self.assertTrue(audit["training_prohibited"])
            self.assertEqual(
                audit["counts"][
                    "rejected_tracks_by_dataset_and_reason"
                ],
                {},
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

    def test_candidate_position_head_masks_unplayable_strings(self) -> None:
        feature_config = FeatureConfig()
        model_config = OnsetPositionConfig(
            dropout=0,
            candidate_position_head=True,
        )
        model = OnsetPositionNet(feature_config, model_config).eval()
        features = torch.rand(
            2,
            3,
            model_config.frame_count,
            feature_config.n_bins,
        )
        encoded = model.encode(features, torch.rand(2, 49))

        logits = model.classify_candidate_positions(
            encoded,
            torch.tensor([0, 24]),
        )

        self.assertEqual(tuple(logits.shape), (2, 6))
        self.assertTrue(torch.all(logits[0, :5] < -20))
        self.assertGreater(float(logits[0, 5].detach()), -20)
        self.assertTrue(torch.all(logits[1] > -20))

    def test_categorical_candidate_head_preserves_none_and_masks_strings(
        self,
    ) -> None:
        feature_config = FeatureConfig()
        model_config = OnsetPositionConfig(
            dropout=0,
            candidate_position_head=True,
            candidate_position_objective="categorical",
        )
        model = OnsetPositionNet(feature_config, model_config).eval()
        features = torch.rand(
            2,
            3,
            model_config.frame_count,
            feature_config.n_bins,
        )
        encoded = model.encode(features, torch.rand(2, 49))

        logits = model.classify_candidate_positions(
            encoded,
            torch.tensor([0, 24]),
        )

        self.assertEqual(tuple(logits.shape), (2, 7))
        self.assertTrue(torch.all(logits[:, 0] > -20))
        self.assertTrue(torch.all(logits[0, 1:6] < -20))
        self.assertGreater(float(logits[0, 6].detach()), -20)
        self.assertTrue(torch.all(logits[1, 1:] > -20))

    def test_candidate_harmonic_indexes_follow_cqt_pitch_bins(self) -> None:
        indexes, valid = candidate_harmonic_cqt_indexes(
            FeatureConfig(),
            harmonic_count=5,
            radius=1,
        )

        self.assertEqual(tuple(indexes.shape), (49, 5, 3))
        self.assertEqual(indexes[24, 0].tolist(), [71, 72, 73])
        self.assertEqual(valid[24, 0].tolist(), [True, True, True])
        self.assertEqual(indexes[24, 1].tolist(), [107, 108, 109])
        self.assertEqual(valid[0, 0].tolist(), [False, True, True])

    def test_factorized_candidate_head_uses_local_spectral_evidence(
        self,
    ) -> None:
        torch.manual_seed(7)
        feature_config = FeatureConfig()
        model_config = OnsetPositionConfig(
            convolution_channels=(4, 6, 8),
            hidden_size=16,
            dropout=0,
            candidate_position_head=True,
            candidate_position_objective="factorized",
            candidate_spectral_evidence=True,
            candidate_spectral_size=8,
            candidate_string_embedding_size=4,
            candidate_position_hidden_size=8,
        )
        model = OnsetPositionNet(feature_config, model_config).eval()
        pitch_masks = torch.zeros((2, 49))
        pitch_masks[0, 0] = 0.8
        pitch_masks[1, 24] = 0.6
        base_features = torch.zeros(
            2,
            3,
            model_config.frame_count,
            feature_config.n_bins,
        )
        spectral_features = base_features.clone()
        spectral_features[0, :, :, :6] = 1
        encoded = model.encode(base_features, pitch_masks).detach()
        pitch_indexes = torch.tensor([0, 24])
        confidences = torch.tensor([0.8, 0.6])

        base_logits = model.classify_candidate_positions(
            encoded,
            pitch_indexes,
            features=base_features,
            candidate_confidences=confidences,
        )
        spectral_logits = model.classify_candidate_positions(
            encoded,
            pitch_indexes,
            features=spectral_features,
            candidate_confidences=confidences,
        )

        self.assertEqual(tuple(base_logits.shape), (2, 7))
        self.assertTrue(
            torch.allclose(
                base_logits.exp().sum(dim=1),
                torch.ones(2),
                atol=1e-5,
            )
        )
        self.assertTrue(torch.all(base_logits[0, 1:6] < -20))
        self.assertGreater(float(base_logits[0, 6].detach()), -20)
        self.assertFalse(
            torch.allclose(base_logits[0], spectral_logits[0])
        )

        targets = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        loss, correct, total = candidate_position_loss(
            spectral_logits,
            targets,
            objective="factorized",
        )
        loss.backward()

        self.assertGreater(float(loss.detach()), 0)
        self.assertEqual(total, 1)
        self.assertIn(correct, (0, 1))
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in model.candidate_head.parameters()
            )
        )

    def test_spectral_residual_head_preserves_v2_at_initialization(
        self,
    ) -> None:
        torch.manual_seed(11)
        feature_config = FeatureConfig()
        shared = {
            "convolution_channels": (4, 6, 8),
            "hidden_size": 16,
            "dropout": 0,
            "candidate_position_head": True,
        }
        base_config = OnsetPositionConfig(
            **shared,
            candidate_position_objective="categorical",
        )
        residual_config = OnsetPositionConfig(
            **shared,
            candidate_position_objective="spectral_residual",
            candidate_spectral_evidence=True,
            candidate_spectral_size=8,
            candidate_string_embedding_size=4,
            candidate_position_hidden_size=8,
        )
        base_model = OnsetPositionNet(
            feature_config,
            base_config,
        ).eval()
        residual_model = OnsetPositionNet(
            feature_config,
            residual_config,
        ).eval()
        incompatible = residual_model.load_state_dict(
            base_model.state_dict(),
            strict=False,
        )
        self.assertTrue(
            all(
                name.startswith("candidate_residual_head.")
                for name in incompatible.missing_keys
            )
        )
        self.assertEqual(incompatible.unexpected_keys, [])

        features = torch.rand(
            3,
            3,
            base_config.frame_count,
            feature_config.n_bins,
        )
        pitch_masks = torch.rand(3, 49)
        pitch_indexes = torch.tensor([0, 24, 30])
        confidences = pitch_masks[torch.arange(3), pitch_indexes]
        base_encoded = base_model.encode(features, pitch_masks)
        residual_encoded = residual_model.encode(features, pitch_masks)
        base_probabilities = candidate_position_probabilities(
            base_model.classify_candidate_positions(
                base_encoded,
                pitch_indexes,
            ),
            objective="categorical",
        )
        residual_logits = residual_model.classify_candidate_positions(
            residual_encoded,
            pitch_indexes,
            features=features,
            candidate_confidences=confidences,
        )
        residual_probabilities = candidate_position_probabilities(
            residual_logits,
            objective="spectral_residual",
        )

        self.assertTrue(
            torch.allclose(
                base_probabilities,
                residual_probabilities,
                atol=1e-5,
            )
        )
        targets = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        loss, _correct, total = candidate_position_loss(
            residual_logits,
            targets,
            objective="spectral_residual",
        )
        loss.backward()

        self.assertEqual(total, 2)
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in (
                    residual_model.candidate_residual_head.parameters()
                )
            )
        )

    def test_activity_residual_preserves_v4_at_initialization(
        self,
    ) -> None:
        torch.manual_seed(17)
        feature_config = FeatureConfig()
        shared = {
            "convolution_channels": (4, 6, 8),
            "hidden_size": 16,
            "dropout": 0,
            "candidate_position_head": True,
            "candidate_position_objective": "spectral_residual",
            "candidate_spectral_evidence": True,
            "candidate_spectral_size": 8,
            "candidate_string_embedding_size": 4,
            "candidate_position_hidden_size": 8,
            "candidate_activity_hidden_size": 8,
        }
        base_model = OnsetPositionNet(
            feature_config,
            OnsetPositionConfig(**shared),
        ).eval()
        activity_model = OnsetPositionNet(
            feature_config,
            OnsetPositionConfig(
                **shared,
                candidate_activity_residual=True,
            ),
        ).eval()
        incompatible = activity_model.load_state_dict(
            base_model.state_dict(),
            strict=False,
        )
        self.assertTrue(
            all(
                name.startswith("candidate_activity_residual_head.")
                for name in incompatible.missing_keys
            )
        )
        self.assertEqual(incompatible.unexpected_keys, [])

        features = torch.rand(
            3,
            3,
            base_model.config.frame_count,
            feature_config.n_bins,
        )
        pitch_masks = torch.rand(3, 49)
        pitch_indexes = torch.tensor([0, 24, 30])
        confidences = pitch_masks[torch.arange(3), pitch_indexes]
        base_encoded = base_model.encode(features, pitch_masks)
        activity_encoded = activity_model.encode(features, pitch_masks)
        base_position_logits = base_model.classify_candidate_positions(
            base_encoded,
            pitch_indexes,
            features=features,
            candidate_confidences=confidences,
        )
        activity_position_logits = (
            activity_model.classify_candidate_positions(
                activity_encoded,
                pitch_indexes,
                features=features,
                candidate_confidences=confidences,
            )
        )
        base_activity_logits = base_model.classify_candidate_activity(
            base_encoded,
            pitch_indexes,
            features=features,
            candidate_confidences=confidences,
            candidate_position_logits=base_position_logits,
        )
        activity_logits = activity_model.classify_candidate_activity(
            activity_encoded,
            pitch_indexes,
            features=features,
            candidate_confidences=confidences,
            candidate_position_logits=activity_position_logits,
        )

        self.assertTrue(
            torch.allclose(
                base_position_logits,
                activity_position_logits,
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                base_activity_logits,
                activity_logits,
                atol=1e-5,
            )
        )
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            activity_logits,
            torch.tensor([1.0, 0.0, 1.0]),
        )
        loss.backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in (
                    activity_model
                    .candidate_activity_residual_head
                    .parameters()
                )
            )
        )

    def test_temporal_string_residual_preserves_activity_v1_at_initialization(
        self,
    ) -> None:
        torch.manual_seed(23)
        feature_config = FeatureConfig()
        shared = {
            "convolution_channels": (4, 6, 8),
            "hidden_size": 16,
            "dropout": 0,
            "candidate_position_head": True,
            "candidate_position_objective": "spectral_residual",
            "candidate_spectral_evidence": True,
            "candidate_spectral_size": 8,
            "candidate_string_embedding_size": 4,
            "candidate_position_hidden_size": 8,
            "candidate_activity_residual": True,
            "candidate_activity_hidden_size": 8,
        }
        activity_model = OnsetPositionNet(
            feature_config,
            OnsetPositionConfig(**shared),
        ).eval()
        temporal_model = OnsetPositionNet(
            feature_config,
            OnsetPositionConfig(
                **shared,
                candidate_temporal_string_residual=True,
                candidate_temporal_hidden_size=8,
            ),
        ).eval()
        incompatible = temporal_model.load_state_dict(
            activity_model.state_dict(),
            strict=False,
        )
        self.assertTrue(
            all(
                name.startswith("candidate_temporal_residual_head.")
                for name in incompatible.missing_keys
            )
        )
        self.assertEqual(incompatible.unexpected_keys, [])

        features = torch.rand(
            3,
            3,
            activity_model.config.frame_count,
            feature_config.n_bins,
        )
        pitch_masks = torch.rand(3, 49)
        pitch_indexes = torch.tensor([0, 24, 30])
        confidences = pitch_masks[torch.arange(3), pitch_indexes]
        temporal_context = torch.rand(3, 6, 20)
        activity_encoded = activity_model.encode(features, pitch_masks)
        temporal_encoded = temporal_model.encode(features, pitch_masks)
        activity_position_logits = (
            activity_model.classify_candidate_positions(
                activity_encoded,
                pitch_indexes,
                features=features,
                candidate_confidences=confidences,
            )
        )
        temporal_position_logits = (
            temporal_model.classify_candidate_positions(
                temporal_encoded,
                pitch_indexes,
                features=features,
                candidate_confidences=confidences,
                temporal_context=temporal_context,
            )
        )
        activity_logits = activity_model.classify_candidate_activity(
            activity_encoded,
            pitch_indexes,
            features=features,
            candidate_confidences=confidences,
            candidate_position_logits=activity_position_logits,
        )
        temporal_activity_logits = (
            temporal_model.classify_candidate_activity(
                temporal_encoded,
                pitch_indexes,
                features=features,
                candidate_confidences=confidences,
                candidate_position_logits=temporal_position_logits,
            )
        )

        self.assertTrue(
            torch.allclose(
                activity_position_logits,
                temporal_position_logits,
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                activity_logits,
                temporal_activity_logits,
                atol=1e-5,
            )
        )
        targets = torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ]
        )
        loss, _correct, _total = candidate_position_loss(
            temporal_position_logits,
            targets,
            objective="spectral_residual",
        )
        loss.backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in (
                    temporal_model
                    .candidate_temporal_residual_head
                    .parameters()
                )
            )
        )

    def test_candidate_sequence_zero_initialization_preserves_local_logits(
        self,
    ) -> None:
        torch.manual_seed(29)
        model = CandidateSequenceStringModel(
            CandidateSequenceConfig(
                sequence_length=4,
                sequence_stride=2,
                token_size=8,
                hidden_size=6,
                output_hidden_size=7,
                dropout=0,
            )
        )
        pitches = torch.tensor(
            [
                [
                    [0, 4, 0],
                    [6, 9, 12],
                    [15, 0, 0],
                    [20, 24, 30],
                ]
            ]
        )
        candidate_mask = torch.tensor(
            [
                [
                    [True, True, False],
                    [True, True, True],
                    [True, False, False],
                    [True, True, True],
                ]
            ]
        )
        group_mask = torch.ones(1, 4, dtype=torch.bool)
        valid_strings = torch.ones(1, 4, 3, 6, dtype=torch.bool)
        base_string_logits = torch.log_softmax(
            torch.randn(1, 4, 3, 6),
            dim=3,
        )
        residuals = model(
            pitches,
            torch.rand(1, 4, 3),
            torch.rand(1, 4, 3),
            base_string_logits,
            valid_strings,
            candidate_mask,
            group_mask,
            torch.tensor([[0.1, 0.4, 0.7, 1.1]]),
        )
        self.assertTrue(torch.equal(residuals, torch.zeros_like(residuals)))

        base_joint = torch.log_softmax(torch.randn(12, 7), dim=1)
        refined = refine_candidate_position_logits(
            base_joint,
            residuals.reshape(12, 6),
            valid_strings.reshape(12, 6),
        )
        self.assertTrue(torch.allclose(base_joint, refined, atol=1e-6))
        self.assertTrue(
            torch.allclose(
                torch.logsumexp(base_joint[:, 1:], dim=1),
                torch.logsumexp(refined[:, 1:], dim=1),
                atol=1e-6,
            )
        )
        changed = refine_candidate_position_logits(
            base_joint,
            torch.randn(12, 6),
            valid_strings.reshape(12, 6),
        )
        self.assertTrue(torch.equal(base_joint[:, 0], changed[:, 0]))
        self.assertTrue(
            torch.allclose(
                torch.logsumexp(base_joint[:, 1:], dim=1),
                torch.logsumexp(changed[:, 1:], dim=1),
                atol=1e-6,
            )
        )

    def test_candidate_sequence_activity_zero_initialization_preserves_logits(
        self,
    ) -> None:
        torch.manual_seed(31)
        model = CandidateSequenceActivityModel(
            CandidateSequenceConfig(
                sequence_length=3,
                sequence_stride=2,
                token_size=8,
                hidden_size=6,
                output_hidden_size=7,
                dropout=0,
            )
        )
        group = CandidateSequenceGroup(
            annotation_index=0,
            onset=0.1,
            pitches=torch.tensor([0, 4]),
            confidences=torch.tensor([0.8, 0.7]),
            durations=torch.tensor([0.2, 0.3]),
            base_string_logits=torch.log_softmax(
                torch.randn(2, 6),
                dim=1,
            ),
            valid_strings=torch.ones(2, 6, dtype=torch.bool),
            string_targets=torch.zeros(2, 6),
            target_weights=torch.ones(2),
            base_activity_logits=torch.tensor([-0.4, 0.6]),
            activity_targets=torch.tensor([0.0, 1.0]),
            repeat_targets=torch.tensor([0.0, 1.0]),
        )
        batch = collate_candidate_sequence_windows([[group]])
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
        )

        self.assertTrue(torch.equal(residuals, torch.zeros_like(residuals)))
        self.assertTrue(
            torch.equal(
                refine_candidate_activity_logits(
                    batch["base_activity_logits"],
                    residuals,
                ),
                batch["base_activity_logits"],
            )
        )

    def test_candidate_sequence_activity_loss_weights_repeated_positives(
        self,
    ) -> None:
        base_logits = torch.zeros(1, 1, 2)
        residuals = torch.zeros(1, 1, 2, requires_grad=True)
        targets = torch.ones(1, 1, 2)
        repeats = torch.tensor([[[0.0, 1.0]]])
        mask = torch.ones(1, 1, 2, dtype=torch.bool)

        loss, counts = candidate_sequence_activity_loss(
            base_logits,
            residuals,
            targets,
            repeats,
            mask,
            activity_positive_weight=1.0,
            repeat_positive_weight=4.0,
        )

        self.assertAlmostEqual(
            float(loss.detach()),
            float(torch.log(torch.tensor(2.0))) * 2.5,
            places=6,
        )
        self.assertEqual(counts["true_positive"], 2)
        self.assertEqual(counts["repeat_true_positive"], 1)
        loss.backward()
        self.assertIsNotNone(residuals.grad)
        assert residuals.grad is not None
        self.assertGreater(
            abs(float(residuals.grad[0, 0, 1])),
            abs(float(residuals.grad[0, 0, 0])),
        )

    def test_candidate_sequence_confidence_identity_preserves_activity(
        self,
    ) -> None:
        model = CandidateSequenceConfidenceModel(
            CandidateSequenceConfig(
                sequence_length=3,
                sequence_stride=2,
                token_size=8,
                hidden_size=6,
                output_hidden_size=7,
                dropout=0,
            ),
            identity_accept_probability=0.99,
        )
        group = CandidateSequenceGroup(
            annotation_index=0,
            onset=0.1,
            pitches=torch.tensor([0, 4]),
            confidences=torch.tensor([0.8, 0.7]),
            durations=torch.tensor([0.2, 0.3]),
            base_string_logits=torch.log_softmax(
                torch.randn(2, 6),
                dim=1,
            ),
            valid_strings=torch.ones(2, 6, dtype=torch.bool),
            string_targets=torch.zeros(2, 6),
            target_weights=torch.ones(2),
            base_activity_logits=torch.tensor([-0.4, 0.6]),
            activity_targets=torch.tensor([0.0, 1.0]),
            repeat_targets=torch.tensor([0.0, 1.0]),
        )
        batch = collate_candidate_sequence_windows([[group]])
        confidence_logits = model(
            batch["pitches"],
            batch["confidences"],
            batch["durations"],
            batch["base_string_logits"],
            batch["valid_strings"],
            batch["base_activity_logits"],
            batch["candidate_mask"],
            batch["group_mask"],
            batch["group_onsets"],
        )

        self.assertTrue(bool((confidence_logits > 0).all()))
        self.assertTrue(
            torch.equal(
                combine_candidate_activity_confidence_logits(
                    batch["base_activity_logits"],
                    confidence_logits,
                ),
                batch["base_activity_logits"],
            )
        )

    def test_candidate_sequence_confidence_loss_rejects_wrong_top_string(
        self,
    ) -> None:
        base_activity_logits = torch.tensor([2.0, 2.0, 2.0, -2.0])
        base_string_logits = torch.tensor(
            [
                [4.0, -4.0, -4.0, -4.0, -4.0, -4.0],
                [-4.0, 4.0, -4.0, -4.0, -4.0, -4.0],
                [4.0, -4.0, -4.0, -4.0, -4.0, -4.0],
                [4.0, -4.0, -4.0, -4.0, -4.0, -4.0],
            ]
        )
        valid_strings = torch.ones(4, 6, dtype=torch.bool)
        confidence_logits = torch.tensor(
            [2.0, -2.0, 2.0, 2.0],
            requires_grad=True,
        )
        activity_targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
        string_targets = torch.tensor(
            [
                [1.0, 0, 0, 0, 0, 0],
                [1.0, 0, 0, 0, 0, 0],
                [0.0, 0, 0, 0, 0, 0],
                [0.0, 0, 0, 0, 0, 0],
            ]
        )
        repeats = torch.tensor([1.0, 0.0, 0.0, 0.0])
        mask = torch.ones(4, dtype=torch.bool)

        loss, counts = candidate_sequence_confidence_loss(
            base_activity_logits,
            base_string_logits,
            valid_strings,
            confidence_logits,
            activity_targets,
            string_targets,
            repeats,
            mask,
            repeat_positive_weight=1.5,
        )

        self.assertEqual(counts["supervised_active"], 2)
        self.assertEqual(counts["correctness_true_positive"], 1)
        self.assertEqual(counts["wrong_string"], 0)
        self.assertEqual(counts["base_wrong_string"], 1)
        self.assertEqual(counts["joint_true_positive"], 1)
        self.assertEqual(counts["joint_false_positive"], 1)
        self.assertEqual(counts["joint_false_negative"], 1)
        self.assertEqual(counts["repeat_true_positive"], 1)
        loss.backward()
        self.assertIsNotNone(confidence_logits.grad)

        inactive_logits = torch.tensor([2.0, 2.0], requires_grad=True)
        inactive_loss, inactive_counts = (
            candidate_sequence_confidence_loss(
                torch.tensor([2.0, -2.0]),
                base_string_logits[:2],
                valid_strings[:2],
                inactive_logits,
                torch.zeros(2),
                torch.zeros(2, 6),
                torch.zeros(2),
                torch.ones(2, dtype=torch.bool),
                repeat_positive_weight=1.5,
            )
        )
        self.assertEqual(float(inactive_loss.detach()), 0.0)
        self.assertEqual(inactive_counts["supervised_active"], 0)
        self.assertEqual(inactive_counts["joint_false_positive"], 1)
        inactive_loss.backward()
        assert inactive_logits.grad is not None
        self.assertTrue(torch.equal(inactive_logits.grad, torch.zeros(2)))

    def test_candidate_sequence_confidence_separability_metrics(
        self,
    ) -> None:
        targets = np.array([0, 0, 1, 1], dtype=np.int64)
        separated_scores = np.array([-2.0, -1.0, 1.0, 2.0])

        separated = confidence_separation_metrics(
            targets,
            separated_scores,
        )
        tied = confidence_separation_metrics(
            targets,
            np.zeros(4),
        )
        distribution = confidence_distribution_summary(
            separated_scores,
        )

        self.assertEqual(separated["auroc"], 1.0)
        self.assertEqual(separated["average_precision"], 1.0)
        self.assertEqual(
            separated["zero_threshold_correct_retention"],
            1.0,
        )
        self.assertEqual(
            separated["zero_threshold_wrong_rejection"],
            1.0,
        )
        self.assertEqual(tied["auroc"], 0.5)
        self.assertEqual(tied["average_precision"], 0.5)
        self.assertEqual(distribution["count"], 4)
        self.assertEqual(distribution["mean"], 0.0)
        self.assertEqual(distribution["fraction_below_zero"], 0.5)

    def test_candidate_string_probability_margin_masks_unplayable_strings(
        self,
    ) -> None:
        logits = torch.log(
            torch.tensor(
                [
                    [0.6, 0.4, 0.5, 0.5, 0.5, 0.5],
                    [0.1, 0.9, 0.5, 0.5, 0.5, 0.5],
                ]
            )
        )
        valid_strings = torch.tensor(
            [
                [True, True, False, False, False, False],
                [False, True, False, False, False, False],
            ]
        )

        margins = candidate_string_probability_margins(
            logits,
            valid_strings,
        )

        self.assertTrue(
            torch.allclose(margins, torch.tensor([0.2, 1.0]), atol=1e-6)
        )

    def test_margin_calibration_partition_is_grouped_and_deterministic(
        self,
    ) -> None:
        annotations = [
            TrackAnnotation(
                track_id=f"a-{index}",
                group_id=group_id,
                audio=f"a-{index}.wav",
                split="train",
                duration=1.0,
                events=[],
                provenance={"dataset": dataset},
            )
            for index, (dataset, group_id) in enumerate(
                [
                    ("A", "a1"),
                    ("A", "a1"),
                    ("A", "a2"),
                    ("A", "a3"),
                    ("A", "a4"),
                    ("B", "b1"),
                    ("B", "b2"),
                ]
            )
        ]

        selected, counts = calibration_partition_keys(
            annotations,
            fraction=0.5,
            seed=20261120,
        )
        reversed_selected, reversed_counts = calibration_partition_keys(
            list(reversed(annotations)),
            fraction=0.5,
            seed=20261120,
        )

        self.assertEqual(selected, reversed_selected)
        self.assertEqual(counts, reversed_counts)
        self.assertEqual(
            counts,
            {
                "A": {
                    "available_groups": 4,
                    "selected_groups": 2,
                },
                "B": {
                    "available_groups": 2,
                    "selected_groups": 1,
                },
            },
        )
        self.assertEqual(
            Counter(dataset for dataset, _group_id in selected),
            Counter({"A": 2, "B": 1}),
        )

    def test_margin_calibration_partition_can_stratify_tracks_by_group(
        self,
    ) -> None:
        annotations = [
            TrackAnnotation(
                track_id=f"{group_id}-{track_index}",
                group_id=group_id,
                audio=f"{group_id}-{track_index}.wav",
                split="train",
                duration=1.0,
                events=[],
                provenance={"dataset": dataset},
            )
            for dataset, group_id, track_index in [
                ("A", "player-1", 0),
                ("A", "player-1", 1),
                ("A", "player-2", 0),
                ("A", "player-2", 1),
                ("B", "group-1", 0),
                ("B", "group-2", 0),
            ]
        ]

        selected, keys, counts = stratified_calibration_partition(
            annotations,
            fraction=0.5,
            seed=20261121,
            stratify_by_group_datasets={"A"},
        )

        selected_a_groups = {
            annotations[index].group_id
            for index in selected
            if annotations[index].provenance["dataset"] == "A"
        }
        self.assertEqual(selected_a_groups, {"player-1", "player-2"})
        self.assertEqual(len(selected_a_groups), 2)
        self.assertEqual(dict(counts["A"])["selected_tracks"], 2)
        self.assertTrue(
            all(key[0] == "track" for key in keys if key[1] == "A")
        )

    def test_v3_track_selection_excludes_p3_from_calibration(self) -> None:
        def annotation(
            track_id: str,
            dataset: str,
            group_id: str,
        ) -> TrackAnnotation:
            return TrackAnnotation(
                track_id=track_id,
                group_id=group_id,
                audio=f"{track_id}.wav",
                split="train",
                duration=1.0,
                events=[],
                provenance={"dataset": dataset},
            )

        annotations = [
            annotation("g00-a", "GuitarSet", "guitarset-player-00"),
            annotation("g00-b", "GuitarSet", "guitarset-player-00"),
            annotation("g01-a", "GuitarSet", "guitarset-player-01"),
            annotation("g01-b", "GuitarSet", "guitarset-player-01"),
            annotation("agpt", "AG-PT-set", "agpt-player"),
            annotation("p1", "Guitar-TECHS", "guitar-techs-P1"),
            annotation(
                "synthetic",
                "stringtrace-symbolic-synthetic",
                "synthetic-group",
            ),
            annotation("p3-direct", "Guitar-TECHS", "guitar-techs-P3"),
            annotation("p3-mic", "Guitar-TECHS", "guitar-techs-P3"),
        ]

        selection = select_v3_tracks(
            annotations,
            fraction=0.5,
            seed=20261122,
        )

        self.assertEqual(
            selection.p3_track_ids,
            frozenset({"p3-direct", "p3-mic"}),
        )
        self.assertTrue(
            selection.p3_track_ids.isdisjoint(
                selection.calibration_track_ids
            )
        )
        self.assertTrue(
            all("guitar-techs-P3" not in key for key in selection.partition_keys)
        )
        gradient_annotations = [
            item
            for item in annotations
            if item.track_id in selection.gradient_track_ids
        ]
        self.assertEqual(
            {
                (
                    item.provenance["dataset"],
                    item.group_id == "guitar-techs-P3",
                )
                for item in gradient_annotations
            },
            {("GuitarSet", False), ("Guitar-TECHS", True)},
        )
        self.assertFalse(
            {"agpt", "p1", "synthetic"} & selection.gradient_track_ids
        )

    def test_v3_candidate_cache_set_identity_is_fail_closed(self) -> None:
        with self.temporary_directory() as directory:
            root = Path(directory)
            cache = root / "cache"
            cache.mkdir()
            annotations = [
                TrackAnnotation(
                    track_id=f"track-{index}",
                    group_id="guitar-techs-P3",
                    audio=f"track-{index}.wav",
                    split="train",
                    duration=1.0,
                    events=[],
                    provenance={"dataset": "Guitar-TECHS"},
                )
                for index in range(2)
            ]
            manifest = root / "manifest.jsonl"
            write_manifest(manifest, annotations)
            identities: list[tuple[str, str]] = []
            for index, annotation in enumerate(annotations):
                audio = annotation.resolve_audio(manifest)
                audio.touch()
                cache_path = basic_pitch_cache_path(
                    audio,
                    cache,
                    onset_threshold=0.35,
                    frame_threshold=0.25,
                    minimum_note_length=70.0,
                )
                cache_path.write_text(
                    json.dumps(
                        {
                            "version": BASIC_PITCH_CACHE_VERSION,
                            "events": [
                                {
                                    "onset": 0.1,
                                    "offset": 0.2,
                                    "pitch": 64 + index,
                                    "confidence": 0.9,
                                }
                            ],
                            "dropped": index,
                        }
                    ),
                    encoding="utf-8",
                )
                identities.append((cache_path.name, sha256_file(cache_path)))
            digest = hashlib.sha256()
            for name, cache_sha256 in sorted(identities):
                digest.update(f"{name}:{cache_sha256}\n".encode())
            expected = {
                "audio_files": 2,
                "events": 2,
                "dropped": 1,
                "cache_set_sha256": digest.hexdigest(),
            }

            actual = candidate_cache_set_identity(manifest, cache)

            self.assertEqual(actual, expected)
            validate_candidate_cache_set(actual, expected)
            with self.assertRaisesRegex(ValueError, "cache_set_sha256"):
                validate_candidate_cache_set(
                    actual,
                    {**expected, "cache_set_sha256": "0" * 64},
                )

    def test_v3_epoch_gate_and_tie_break_are_frozen(self) -> None:
        baseline = dict(EXPECTED_BASELINE)
        eligible = {
            **baseline,
            "aggregate_accuracy": (
                float(baseline["aggregate_accuracy"]) + 0.001
            ),
            "aggregate_wrong_strings": (
                int(baseline["aggregate_wrong_strings"]) - 1
            ),
            "agpt_accuracy": float(baseline["agpt_accuracy"]) - 0.004,
            "guitar_techs_p1_accuracy": (
                float(baseline["guitar_techs_p1_accuracy"]) + 0.004
            ),
        }
        eligible["cross_source_macro_accuracy"] = (
            float(eligible["agpt_accuracy"])
            + float(eligible["guitar_techs_p1_accuracy"])
        ) / 2

        validate_epoch_zero(baseline)
        with self.assertRaisesRegex(ValueError, "aggregate_wrong_strings"):
            validate_epoch_zero(
                {**baseline, "aggregate_wrong_strings": 3373}
            )
        self.assertTrue(epoch_is_eligible(eligible))
        self.assertGreater(
            epoch_selection_key(1, eligible),
            epoch_selection_key(2, eligible),
        )
        self.assertFalse(
            epoch_is_eligible(
                {
                    **eligible,
                    "agpt_accuracy": (
                        float(baseline["agpt_accuracy"]) - 0.005001
                    ),
                }
            )
        )
        self.assertFalse(
            epoch_is_eligible(
                {
                    **eligible,
                    "aggregate_accuracy": baseline["aggregate_accuracy"],
                }
            )
        )

    def test_v3_training_epoch_uses_original_candidate_loss(self) -> None:
        model = CandidateSequenceStringModel(
            CandidateSequenceConfig(
                sequence_length=2,
                sequence_stride=1,
                token_size=8,
                hidden_size=6,
                output_hidden_size=7,
                dropout=0,
            )
        )
        group = CandidateSequenceGroup(
            annotation_index=0,
            onset=0.1,
            pitches=torch.tensor([12]),
            confidences=torch.ones(1),
            durations=torch.ones(1),
            base_string_logits=torch.log_softmax(torch.randn(1, 6), dim=1),
            valid_strings=torch.ones(1, 6, dtype=torch.bool),
            string_targets=torch.tensor([[1.0, 0, 0, 0, 0, 0]]),
            target_weights=torch.ones(1),
        )
        batch = collate_candidate_sequence_windows([[group]])
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        with (
            patch(
                "stringtrace_ml.train_candidate_sequence_v3."
                "candidate_sequence_loss",
                wraps=candidate_sequence_loss,
            ) as loss,
            patch(
                "stringtrace_ml.train_candidate_sequence_v3."
                "torch.nn.utils.clip_grad_norm_",
                wraps=torch.nn.utils.clip_grad_norm_,
            ) as clip,
        ):
            metrics = run_v3_training_epoch(
                model,
                [batch],
                torch.device("cpu"),
                optimizer,
            )

        loss.assert_called_once()
        clip.assert_called_once()
        self.assertEqual(metrics["window_active_candidates"], 1)

    def test_v3_artifact_hash_mismatch_fails_closed(self) -> None:
        with self.temporary_directory() as directory:
            root = Path(directory)
            artifact = root / "artifact.json"
            artifact.write_text('{"version": 1}\n', encoding="utf-8")
            specs = {
                "artifact": (
                    artifact.name,
                    sha256_file(artifact),
                )
            }
            report = verify_artifact_hashes(root, specs)
            self.assertEqual(
                report["artifact"]["sha256"],
                sha256_file(artifact),
            )

            artifact.write_text('{"version": 2}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_artifact_hashes(root, specs)

    def test_margin_calibration_post_assignment_uses_group_decisions(
        self,
    ) -> None:
        probabilities = torch.tensor(
            [
                [0.163, 0.163, 0.163, 0.163, 0.184, 0.164],
                [0.0925, 0.0925, 0.0925, 0.30, 0.33, 0.0925],
            ]
        )
        group = CandidateSequenceGroup(
            annotation_index=0,
            onset=0.1,
            pitches=torch.tensor([64 - PITCH_MIN, 64 - PITCH_MIN]),
            confidences=torch.ones(2),
            durations=torch.ones(2),
            base_string_logits=torch.log(probabilities),
            valid_strings=torch.ones(2, 6, dtype=torch.bool),
            string_targets=torch.tensor(
                [
                    [0.0, 0, 0, 0, 0, 0],
                    [0.0, 0, 0, 1, 0, 0],
                ]
            ),
            target_weights=torch.ones(2),
            base_activity_logits=torch.ones(2),
            activity_targets=torch.tensor([0.0, 1.0]),
            repeat_targets=torch.tensor([0.0, 1.0]),
        )

        independent = collect_margin_calibration_values([group])
        assigned = collect_margin_calibration_values(
            [group],
            assignment_order="post_assignment",
        )

        self.assertEqual(
            independent["top_string_matches"].tolist(),
            [False, False],
        )
        self.assertEqual(
            assigned["top_string_matches"].tolist(),
            [False, True],
        )

    def test_margin_calibration_selects_best_eligible_threshold(
        self,
    ) -> None:
        selection = select_margin_threshold(
            {
                "base_activity_logits": np.ones(4),
                "string_margins": np.array([0.9, 0.8, 0.1, 0.2]),
                "reference_active": np.array(
                    [True, True, True, False]
                ),
                "top_string_matches": np.array(
                    [True, True, False, False]
                ),
                "repeated": np.array([False, True, False, False]),
            }
        )

        self.assertTrue(selection["calibration_gate_passed"])
        self.assertEqual(selection["eligible_thresholds"], 2)
        selected = dict(selection["selected"])
        self.assertAlmostEqual(float(selected["threshold"]), 0.5)
        self.assertEqual(selected["wrong_string"], 0)
        self.assertEqual(selected["repeat_true_positive"], 1)
        self.assertAlmostEqual(float(selected["joint_f1"]), 0.8)

    def test_margin_calibration_falls_back_when_no_threshold_is_eligible(
        self,
    ) -> None:
        selection = select_margin_threshold(
            {
                "base_activity_logits": np.ones(2),
                "string_margins": np.array([0.1, 0.9]),
                "reference_active": np.array([True, True]),
                "top_string_matches": np.array([True, False]),
                "repeated": np.array([False, False]),
            }
        )

        self.assertFalse(selection["calibration_gate_passed"])
        self.assertEqual(selection["eligible_thresholds"], 0)
        self.assertEqual(selection["selected"], selection["baseline"])
        self.assertEqual(dict(selection["selected"])["threshold"], 0.0)

    def test_margin_calibration_loader_enforces_frozen_inputs(self) -> None:
        with self.temporary_directory() as raw_root:
            root = Path(raw_root)
            position_path = root / "position.pt"
            sequence_path = root / "sequence.pt"
            artifact_path = root / "calibration.json"
            position_path.write_bytes(b"position")
            sequence_path.write_bytes(b"sequence")
            artifact = {
                "schema_version": 1,
                "kind": "candidate-string-margin-calibration",
                "version": "candidate-sequence-margin-v1",
                "training_only": True,
                "governed_validation_used": False,
                "cross_source_used": False,
                "calibration_gate_passed": True,
                "position_checkpoint": {
                    "sha256": sha256_file(position_path),
                },
                "string_sequence_checkpoint": {
                    "sha256": sha256_file(sequence_path),
                },
                "candidate_frontend": {
                    "kind": "spotify-basic-pitch-0.4",
                    "onset_threshold": 0.35,
                    "frame_threshold": 0.25,
                    "minimum_note_length": 70.0,
                },
                "selected_threshold": 0.2,
                "selected": {
                    "threshold": 0.2,
                },
            }
            artifact_path.write_text(
                json.dumps(artifact),
                encoding="utf-8",
            )

            loaded = load_candidate_margin_calibration(
                artifact_path,
                position_checkpoint_path=position_path,
                string_sequence_checkpoint_path=sequence_path,
                candidate_onset_threshold=0.35,
                candidate_frame_threshold=0.25,
                candidate_minimum_note_length=70.0,
            )
            self.assertEqual(loaded["selected_threshold"], 0.2)

            artifact["governed_validation_used"] = True
            artifact_path.write_text(
                json.dumps(artifact),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "provenance"):
                load_candidate_margin_calibration(
                    artifact_path,
                    position_checkpoint_path=position_path,
                    string_sequence_checkpoint_path=sequence_path,
                    candidate_onset_threshold=0.35,
                    candidate_frame_threshold=0.25,
                    candidate_minimum_note_length=70.0,
                )

            artifact["governed_validation_used"] = False
            artifact_path.write_text(
                json.dumps(artifact),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "frame_threshold"):
                load_candidate_margin_calibration(
                    artifact_path,
                    position_checkpoint_path=position_path,
                    string_sequence_checkpoint_path=sequence_path,
                    candidate_onset_threshold=0.35,
                    candidate_frame_threshold=0.3,
                    candidate_minimum_note_length=70.0,
                )

            artifact["position_checkpoint"]["sha256"] = "wrong"
            artifact_path.write_text(
                json.dumps(artifact),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "position checkpoint"):
                load_candidate_margin_calibration(
                    artifact_path,
                    position_checkpoint_path=position_path,
                    string_sequence_checkpoint_path=sequence_path,
                    candidate_onset_threshold=0.35,
                    candidate_frame_threshold=0.25,
                    candidate_minimum_note_length=70.0,
                )

    def test_margin_v2_calibration_loader_enforces_assignment_semantics(
        self,
    ) -> None:
        with self.temporary_directory() as raw_root:
            root = Path(raw_root)
            position_path = root / "position.pt"
            sequence_path = root / "sequence.pt"
            artifact_path = root / "calibration.json"
            position_path.write_bytes(b"position")
            sequence_path.write_bytes(b"sequence")
            artifact = {
                "schema_version": 2,
                "kind": "candidate-string-margin-calibration",
                "version": "candidate-sequence-margin-v2",
                "assignment_order": "post_assignment",
                "calibration_match_mode": (
                    "activity_then_full_group_assignment"
                ),
                "training_only": True,
                "governed_validation_used": False,
                "cross_source_used": False,
                "calibration_gate_passed": True,
                "position_checkpoint": {
                    "sha256": sha256_file(position_path),
                },
                "string_sequence_checkpoint": {
                    "sha256": sha256_file(sequence_path),
                },
                "candidate_frontend": {
                    "kind": "spotify-basic-pitch-0.4",
                    "onset_threshold": 0.35,
                    "frame_threshold": 0.25,
                    "minimum_note_length": 70.0,
                },
                "selected_threshold": 0.2,
                "selected": {"threshold": 0.2},
            }
            artifact_path.write_text(
                json.dumps(artifact),
                encoding="utf-8",
            )

            loaded = load_candidate_margin_calibration(
                artifact_path,
                position_checkpoint_path=position_path,
                string_sequence_checkpoint_path=sequence_path,
                candidate_onset_threshold=0.35,
                candidate_frame_threshold=0.25,
                candidate_minimum_note_length=70.0,
            )
            self.assertEqual(loaded["assignment_order"], "post_assignment")

            artifact["assignment_order"] = "pre_assignment"
            artifact_path.write_text(
                json.dumps(artifact),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "assignment order"):
                load_candidate_margin_calibration(
                    artifact_path,
                    position_checkpoint_path=position_path,
                    string_sequence_checkpoint_path=sequence_path,
                    candidate_onset_threshold=0.35,
                    candidate_frame_threshold=0.25,
                    candidate_minimum_note_length=70.0,
                )

    def test_candidate_sequence_joint_zero_initialization_preserves_decision(
        self,
    ) -> None:
        torch.manual_seed(37)
        model = CandidateSequenceJointModel(
            CandidateSequenceConfig(
                sequence_length=3,
                sequence_stride=2,
                token_size=8,
                hidden_size=6,
                output_hidden_size=7,
                dropout=0,
            )
        )
        group = CandidateSequenceGroup(
            annotation_index=0,
            onset=0.1,
            pitches=torch.tensor([0, 4]),
            confidences=torch.tensor([0.8, 0.7]),
            durations=torch.tensor([0.2, 0.3]),
            base_string_logits=torch.log_softmax(
                torch.randn(2, 6),
                dim=1,
            ),
            valid_strings=torch.ones(2, 6, dtype=torch.bool),
            string_targets=torch.tensor(
                [
                    [1.0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            ),
            target_weights=torch.ones(2),
            base_activity_logits=torch.tensor([0.6, -0.4]),
            activity_targets=torch.tensor([1.0, 0.0]),
            repeat_targets=torch.tensor([0.0, 0.0]),
        )
        batch = collate_candidate_sequence_windows([[group]])
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
        )
        base_joint = candidate_joint_log_probabilities(
            batch["base_activity_logits"],
            batch["base_string_logits"],
            batch["valid_strings"],
        )
        refined_joint, refined_activity = refine_candidate_joint_logits(
            batch["base_activity_logits"],
            batch["base_string_logits"],
            batch["valid_strings"],
            residuals,
        )

        self.assertTrue(torch.equal(residuals, torch.zeros_like(residuals)))
        self.assertTrue(
            torch.allclose(base_joint, refined_joint, atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(
                batch["base_activity_logits"],
                refined_activity,
                atol=1e-6,
            )
        )

    def test_candidate_sequence_joint_weights_and_corrects_wrong_string(
        self,
    ) -> None:
        base_activity_logits = torch.tensor([2.0, 2.0])
        base_string_logits = torch.tensor(
            [
                [4.0, -4.0, -4.0, -4.0, -4.0, -4.0],
                [-4.0, 4.0, -4.0, -4.0, -4.0, -4.0],
            ]
        )
        valid_strings = torch.ones(2, 6, dtype=torch.bool)
        activity_targets = torch.ones(2)
        string_targets = torch.tensor(
            [
                [1.0, 0, 0, 0, 0, 0],
                [1.0, 0, 0, 0, 0, 0],
            ]
        )
        repeats = torch.zeros(2)
        mask = torch.ones(2, dtype=torch.bool)
        residuals = torch.zeros(2, 7)

        unweighted_loss, baseline_counts = candidate_sequence_joint_loss(
            base_activity_logits,
            base_string_logits,
            valid_strings,
            residuals,
            activity_targets,
            string_targets,
            repeats,
            mask,
            wrong_string_weight=1.0,
            repeat_positive_weight=1.0,
        )
        weighted_loss, weighted_counts = candidate_sequence_joint_loss(
            base_activity_logits,
            base_string_logits,
            valid_strings,
            residuals,
            activity_targets,
            string_targets,
            repeats,
            mask,
            wrong_string_weight=2.0,
            repeat_positive_weight=1.0,
        )
        corrected_residuals = residuals.clone()
        corrected_residuals[1, 1] = 12.0
        _corrected_loss, corrected_counts = (
            candidate_sequence_joint_loss(
                base_activity_logits,
                base_string_logits,
                valid_strings,
                corrected_residuals,
                activity_targets,
                string_targets,
                repeats,
                mask,
                wrong_string_weight=2.0,
                repeat_positive_weight=1.0,
            )
        )

        self.assertEqual(baseline_counts["wrong_string"], 1)
        self.assertEqual(weighted_counts["base_wrong_string"], 1)
        self.assertGreater(float(weighted_loss), float(unweighted_loss))
        self.assertEqual(corrected_counts["wrong_string"], 0)
        self.assertEqual(corrected_counts["joint_true_positive"], 2)

    def test_candidate_sequence_joint_loss_has_no_target_probability_floor(
        self,
    ) -> None:
        loss, _counts = candidate_sequence_joint_loss(
            torch.tensor([10.0]),
            torch.tensor(
                [[-80.0, 80.0, -80.0, -80.0, -80.0, -80.0]]
            ),
            torch.ones(1, 6, dtype=torch.bool),
            torch.zeros(1, 7),
            torch.ones(1),
            torch.tensor([[1.0, 0, 0, 0, 0, 0]]),
            torch.zeros(1),
            torch.ones(1, dtype=torch.bool),
            wrong_string_weight=1.0,
            repeat_positive_weight=1.0,
        )

        self.assertGreater(float(loss), 100.0)

    def test_candidate_sequence_masks_padding_and_backpropagates(
        self,
    ) -> None:
        model = CandidateSequenceStringModel(
            CandidateSequenceConfig(
                sequence_length=3,
                sequence_stride=2,
                token_size=8,
                hidden_size=6,
                output_hidden_size=7,
                dropout=0,
            )
        )
        group = CandidateSequenceGroup(
            annotation_index=0,
            onset=0.1,
            pitches=torch.arange(11),
            confidences=torch.ones(11),
            durations=torch.full((11,), 0.2),
            base_string_logits=torch.log_softmax(
                torch.randn(11, 6),
                dim=1,
            ),
            valid_strings=torch.ones(11, 6, dtype=torch.bool),
            string_targets=torch.nn.functional.one_hot(
                torch.arange(11) % 6,
                num_classes=6,
            ).float(),
            target_weights=torch.ones(11),
        )
        batch = collate_candidate_sequence_windows(
            [[group], [group, group]]
        )
        self.assertEqual(batch["pitches"].shape, (2, 2, 11))
        residuals = model(
            batch["pitches"],
            batch["confidences"],
            batch["durations"],
            batch["base_string_logits"],
            batch["valid_strings"],
            batch["candidate_mask"],
            batch["group_mask"],
            batch["group_onsets"],
        )
        self.assertTrue(
            torch.equal(
                residuals[0, 1],
                torch.zeros_like(residuals[0, 1]),
            )
        )
        loss, correct, total = candidate_sequence_loss(
            batch["base_string_logits"],
            residuals,
            batch["string_targets"],
            batch["target_weights"],
            batch["candidate_mask"],
            batch["valid_strings"],
        )
        self.assertGreaterEqual(correct, 0)
        self.assertEqual(total, 33)
        loss.backward()
        self.assertIsNotNone(model.residual_head[-1].weight.grad)

    def test_structured_group_objective_scores_unique_assignments(
        self,
    ) -> None:
        logits = torch.tensor(
            [
                [0.0, 0, 0, 0, 3.0, 0],
                [0.0, 0, 0, 2.0, 4.0, 0],
            ],
            requires_grad=True,
        )
        targets = torch.tensor(
            [
                [0.0, 0, 0, 0, 1, 0],
                [0.0, 0, 0, 1, 0, 0],
            ]
        )
        loss, predictions, used_unique = (
            candidate_group_structured_objective(
                logits,
                torch.ones_like(logits, dtype=torch.bool),
                targets,
                torch.ones(2, dtype=torch.bool),
            )
        )
        assert loss is not None
        improved_logits = logits.detach().clone()
        improved_logits[0, 4] += 2
        improved_logits[1, 3] += 2
        improved_loss, _predictions, _used_unique = (
            candidate_group_structured_objective(
                improved_logits,
                torch.ones_like(logits, dtype=torch.bool),
                targets,
                torch.ones(2, dtype=torch.bool),
            )
        )
        assert improved_loss is not None

        self.assertTrue(used_unique)
        self.assertEqual(predictions.tolist(), [4, 3])
        self.assertLess(
            float(improved_loss.detach()),
            float(loss.detach()),
        )
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_structured_group_objective_matches_large_group_fallback(
        self,
    ) -> None:
        logits = torch.randn(7, 6)
        targets = torch.nn.functional.one_hot(
            torch.arange(7) % 6,
            num_classes=6,
        ).float()

        loss, predictions, used_unique = (
            candidate_group_structured_objective(
                logits,
                torch.ones_like(logits, dtype=torch.bool),
                targets,
                torch.ones(7, dtype=torch.bool),
            )
        )

        assert loss is not None
        self.assertFalse(used_unique)
        self.assertTrue(torch.equal(predictions, logits.argmax(dim=1)))
        self.assertTrue(torch.isfinite(loss))

    def test_structured_group_objective_detects_reference_collision(
        self,
    ) -> None:
        loss, _predictions, used_unique = (
            candidate_group_structured_objective(
                torch.zeros(2, 6),
                torch.ones(2, 6, dtype=torch.bool),
                torch.tensor(
                    [
                        [1.0, 0, 0, 0, 0, 0],
                        [1.0, 0, 0, 0, 0, 0],
                    ]
                ),
                torch.ones(2, dtype=torch.bool),
            )
        )

        self.assertTrue(used_unique)
        self.assertIsNone(loss)

    def test_structured_sequence_loss_uses_frozen_activity_acceptance(
        self,
    ) -> None:
        base_logits = torch.zeros(1, 1, 3, 6)
        residuals = torch.zeros_like(base_logits, requires_grad=True)
        targets = torch.tensor(
            [
                [
                    [
                        [0.0, 0, 0, 0, 1, 0],
                        [0.0, 0, 0, 1, 0, 0],
                        [0.0, 0, 1, 0, 0, 0],
                    ]
                ]
            ]
        ).reshape(1, 1, 3, 6)
        loss, counts = candidate_sequence_structured_loss(
            base_logits,
            residuals,
            targets,
            torch.ones(1, 1, 3),
            torch.tensor([[[1.0, 1.0, -1.0]]]),
            torch.tensor([[[0.0, 1.0, 0.0]]]),
            torch.ones(1, 1, 3, dtype=torch.bool),
            torch.ones(1, 1, dtype=torch.bool),
            torch.ones(1, 1, 3, 6, dtype=torch.bool),
            multi_candidate_group_weight=2.0,
            repeat_group_weight=1.5,
        )

        self.assertEqual(counts["supervised_candidates"], 2)
        self.assertEqual(counts["groups_with_multiple_accepted"], 1)
        self.assertEqual(counts["unique_assignment_groups"], 1)
        self.assertNotIn("unsatisfiable_groups", counts)
        loss.backward()
        self.assertIsNotNone(residuals.grad)

    def test_candidate_attack_evidence_removes_baseline_and_scale(
        self,
    ) -> None:
        evidence = torch.tensor(
            [
                [
                    2.0,
                    4.0,
                    4.0,
                    8.0,
                    6.0,
                    12.0,
                    1.0,
                    2.0,
                    2.0,
                    4.0,
                    3.0,
                    6.0,
                    -1.0,
                    -2.0,
                    1.0,
                    2.0,
                    2.0,
                    4.0,
                ],
                [
                    4.0,
                    8.0,
                    8.0,
                    16.0,
                    12.0,
                    24.0,
                    2.0,
                    4.0,
                    4.0,
                    8.0,
                    6.0,
                    12.0,
                    -2.0,
                    -4.0,
                    2.0,
                    4.0,
                    4.0,
                    8.0,
                ],
            ]
        )

        normalized = normalize_candidate_attack_evidence(
            evidence,
            frame_count=3,
            frames_before=1,
            harmonic_count=2,
            radius=0,
        )

        self.assertEqual(normalized.shape, evidence.shape)
        self.assertTrue(
            torch.allclose(normalized[0], normalized[1])
        )
        shaped = normalized[0].reshape(3, 3, 2, 1)
        self.assertTrue(
            torch.equal(shaped[0, 0], torch.zeros(2, 1))
        )
        self.assertAlmostEqual(float(shaped[0].abs().max()), 1.0)
        self.assertAlmostEqual(float(shaped[1].abs().max()), 1.0)
        self.assertAlmostEqual(float(shaped[2].abs().max()), 1.0)

    def test_candidate_attack_evidence_validates_shape(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "feature count",
        ):
            normalize_candidate_attack_evidence(
                torch.zeros(1, 17),
                frame_count=3,
                frames_before=1,
                harmonic_count=2,
                radius=0,
            )

    def test_candidate_sequence_windows_cover_track_end_without_truncation(
        self,
    ) -> None:
        ranges = sequence_window_ranges(
            70,
            sequence_length=32,
            sequence_stride=16,
        )
        self.assertEqual(
            ranges,
            [(0, 32), (16, 48), (32, 64), (38, 70)],
        )
        covered = {
            index
            for start, end in ranges
            for index in range(start, end)
        }
        self.assertEqual(covered, set(range(70)))
        self.assertEqual(
            sequence_window_ranges(
                11,
                sequence_length=32,
                sequence_stride=16,
            ),
            [(0, 11)],
        )

    def test_candidate_sequence_activity_sampling_weights_windows(
        self,
    ) -> None:
        def group(annotation_index: int) -> CandidateSequenceGroup:
            return CandidateSequenceGroup(
                annotation_index=annotation_index,
                onset=0.1,
                pitches=torch.tensor([0]),
                confidences=torch.tensor([1.0]),
                durations=torch.tensor([0.1]),
                base_string_logits=torch.zeros(1, 6),
                valid_strings=torch.ones(1, 6, dtype=torch.bool),
                string_targets=torch.zeros(1, 6),
                target_weights=torch.ones(1),
            )

        annotations = [
            TrackAnnotation(
                track_id="guitarset",
                group_id="guitarset",
                audio="guitarset.wav",
                split="train",
                duration=1.0,
                events=[],
                provenance={"dataset": "GuitarSet"},
            ),
            TrackAnnotation(
                track_id="agpt",
                group_id="agpt",
                audio="agpt.wav",
                split="train",
                duration=1.0,
                events=[],
                provenance={"dataset": "AG-PT-set"},
            ),
        ]
        windows = CandidateSequenceWindowDataset(
            [group(0), group(1)],
            CandidateSequenceConfig(),
        )

        weights, plan = window_sampling_plan(
            windows,
            annotations,
            {"AG-PT-set": 0.5},
        )

        self.assertEqual(weights, [1.0, 0.5])
        self.assertAlmostEqual(
            plan["expected_dataset_share"]["AG-PT-set"],
            1 / 3,
        )
        self.assertAlmostEqual(
            plan["expected_dataset_share"]["GuitarSet"],
            2 / 3,
        )

    def test_candidate_position_queries_merge_unison_targets(self) -> None:
        batch_indexes, pitch_indexes, targets = (
            candidate_position_queries(
                torch.tensor([[64, 64, 67, -1, -1, -1]]),
                torch.tensor([[0, 2, 1, -1, -1, -1]]),
            )
        )

        self.assertEqual(batch_indexes.tolist(), [0, 0])
        self.assertEqual(pitch_indexes.tolist(), [24, 27])
        self.assertEqual(
            targets.tolist(),
            [
                [1, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
            ],
        )
        loss, correct, total = candidate_position_loss(
            torch.tensor(
                [
                    [3.0, -2.0, 2.0, -2.0, -2.0, -2.0],
                    [-2.0, 3.0, -2.0, -2.0, -2.0, -2.0],
                ]
            ),
            targets,
        )
        self.assertGreater(float(loss), 0)
        self.assertEqual((correct, total), (3, 3))

    def test_candidate_position_queries_skip_missing_frontend_pitch(
        self,
    ) -> None:
        mask = torch.zeros((1, 49))
        mask[0, 24] = 1

        batch_indexes, pitch_indexes, targets = (
            candidate_position_queries(
                torch.tensor([[64, 67, -1, -1, -1, -1]]),
                torch.tensor([[0, 1, -1, -1, -1, -1]]),
                mask,
            )
        )

        self.assertEqual(batch_indexes.tolist(), [0])
        self.assertEqual(pitch_indexes.tolist(), [24])
        self.assertEqual(targets[0].tolist(), [1, 0, 0, 0, 0, 0])

    def test_candidate_position_queries_include_unmatched_negatives(
        self,
    ) -> None:
        mask = torch.zeros((1, 49))
        mask[0, 24] = 1
        mask[0, 27] = 0.6

        _batch_indexes, pitch_indexes, targets = (
            candidate_position_queries(
                torch.tensor([[64, -1, -1, -1, -1, -1]]),
                torch.tensor([[0, -1, -1, -1, -1, -1]]),
                mask,
            )
        )

        self.assertEqual(pitch_indexes.tolist(), [24, 27])
        self.assertEqual(targets[0].tolist(), [1, 0, 0, 0, 0, 0])
        self.assertEqual(targets[1].tolist(), [0, 0, 0, 0, 0, 0])

    def test_categorical_candidate_queries_include_none_and_soft_unison(
        self,
    ) -> None:
        mask = torch.zeros((1, 49))
        mask[0, 24] = 1
        mask[0, 27] = 0.6

        _batch_indexes, pitch_indexes, targets = (
            candidate_position_queries(
                torch.tensor([[64, 64, -1, -1, -1, -1]]),
                torch.tensor([[0, 2, -1, -1, -1, -1]]),
                mask,
                categorical=True,
            )
        )

        self.assertEqual(pitch_indexes.tolist(), [24, 27])
        self.assertEqual(
            targets[0].tolist(),
            [0, 0.5, 0, 0.5, 0, 0, 0],
        )
        self.assertEqual(targets[1].tolist(), [1, 0, 0, 0, 0, 0, 0])
        loss, correct, total = candidate_position_loss(
            torch.tensor(
                [
                    [-2.0, 3.0, -2.0, 2.0, -2.0, -2.0, -2.0],
                    [3.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
                ]
            ),
            targets,
        )
        self.assertGreater(float(loss), 0)
        self.assertEqual((correct, total), (1, 1))

    def test_candidate_position_probabilities_apply_none_logit_bias(
        self,
    ) -> None:
        logits = torch.zeros((1, 7))

        baseline = candidate_position_probabilities(
            logits,
            objective="categorical",
        )
        retained = candidate_position_probabilities(
            logits,
            objective="categorical",
            none_logit_bias=-2.0,
        )

        self.assertAlmostEqual(float(baseline[0, 0]), 1 / 7)
        self.assertLess(float(retained[0, 0]), float(baseline[0, 0]))
        self.assertGreater(float(retained[0, 1]), float(baseline[0, 1]))
        with self.assertRaisesRegex(
            ValueError,
            "requires a categorical objective",
        ):
            candidate_position_probabilities(
                torch.zeros((1, 6)),
                objective="binary",
                none_logit_bias=-1.0,
            )

    def test_hierarchical_candidate_probabilities_separate_activity(
        self,
    ) -> None:
        logits = torch.log(
            torch.tensor(
                [
                    [0.4, 0.3, 0.1, 0.05, 0.05, 0.05, 0.05],
                    [0.4, 0.05, 0.05, 0.05, 0.05, 0.1, 0.3],
                ]
            )
        )

        activity, strings = hierarchical_candidate_probabilities(
            logits,
            objective="spectral_residual",
        )

        self.assertTrue(
            torch.allclose(activity, torch.tensor([0.6, 0.6]))
        )
        self.assertTrue(
            torch.allclose(strings.sum(dim=1), torch.ones(2))
        )
        self.assertNotEqual(
            int(strings[0].argmax()),
            int(strings[1].argmax()),
        )
        overridden_activity, overridden_strings = (
            hierarchical_candidate_probabilities(
                logits,
                objective="spectral_residual",
                activity_logits=torch.tensor([-2.0, 2.0]),
            )
        )
        self.assertTrue(
            torch.allclose(
                overridden_activity,
                torch.sigmoid(torch.tensor([-2.0, 2.0])),
            )
        )
        self.assertTrue(
            torch.allclose(
                overridden_strings.sum(dim=1),
                torch.ones(2),
            )
        )
        self.assertTrue(torch.allclose(strings, overridden_strings))
        with self.assertRaisesRegex(
            ValueError,
            "require a categorical objective",
        ):
            hierarchical_candidate_probabilities(
                torch.zeros((1, 6)),
                objective="binary",
            )

    def test_candidate_none_bias_metrics_separate_rejection_and_strings(
        self,
    ) -> None:
        logits = torch.tensor(
            [
                [3.0, 2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
                [3.0, -2.0, 2.0, -2.0, -2.0, -2.0, -2.0],
                [3.0, -2.0, -2.0, 2.0, -2.0, -2.0, -2.0],
            ]
        )
        targets = torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        baseline = candidate_bias_metrics(
            logits,
            targets,
            none_logit_bias=0.0,
        )
        retained = candidate_bias_metrics(
            logits,
            targets,
            none_logit_bias=-2.0,
        )

        self.assertEqual(baseline["counts"]["false_rejects"], 2)
        self.assertEqual(retained["counts"]["false_rejects"], 0)
        self.assertEqual(retained["counts"]["false_accepts"], 1)
        self.assertEqual(retained["counts"]["wrong_strings"], 1)
        self.assertAlmostEqual(retained["active_string_accuracy"], 0.5)
        self.assertAlmostEqual(
            retained["oracle_active_string_accuracy"],
            0.5,
        )
        self.assertAlmostEqual(
            retained["conditional_accepted_string_accuracy"],
            0.5,
        )

    def test_candidate_none_bias_cli_defaults_and_validation(self) -> None:
        arguments = parse_candidate_none_bias_arguments(
            [
                "--manifest",
                "manifest.jsonl",
                "--checkpoint",
                "model.pt",
                "--output",
                "report.json",
            ]
        )

        self.assertIn(0.0, arguments.biases)
        self.assertIn(-1.0, arguments.biases)
        with self.assertRaises(SystemExit):
            parse_candidate_none_bias_arguments(
                [
                    "--manifest",
                    "manifest.jsonl",
                    "--checkpoint",
                    "model.pt",
                    "--output",
                    "report.json",
                    "--biases",
                    "nan",
                ]
            )

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
            self.assertEqual(
                dataset.groups[0].input_durations,
                (0.0, 0.0),
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
            raw_audit = candidate_sequence_activity_audit(dataset)
            governed_audit = candidate_sequence_activity_audit(
                positive_only_dataset
            )
            self.assertEqual(raw_audit["totals"]["groups"], 2)
            self.assertEqual(raw_audit["totals"]["windows"], 1)
            self.assertEqual(
                raw_audit["totals"]["active_candidates"],
                1,
            )
            self.assertEqual(
                raw_audit["totals"]["inactive_candidates"],
                2,
            )
            self.assertEqual(governed_audit["totals"]["groups"], 1)
            self.assertEqual(
                governed_audit["filtered_negative_groups"],
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

    def test_candidate_supervision_weights_cross_string_transitions(
        self,
    ) -> None:
        with self.temporary_directory() as temporary:
            manifest_path = Path(temporary) / "manifest.jsonl"
            write_manifest(
                manifest_path,
                [
                    TrackAnnotation(
                        track_id="cross-string",
                        group_id="cross-string",
                        audio="unused.wav",
                        split="train",
                        duration=1.0,
                        events=[
                            LabeledEvent(0.1, 0.2, 1, 5),
                            LabeledEvent(0.1, 0.2, 2, 10),
                            LabeledEvent(0.5, 0.6, 3, 14),
                        ],
                    )
                ],
            )
            dataset = OnsetPositionDataset(
                manifest_path,
                split="train",
                config=OnsetPositionConfig(
                    candidate_temporal_string_residual=True,
                ),
                candidate_ambiguity_weight=1.25,
                candidate_rare_pair_max_weight=1.5,
                candidate_cross_string_weight=2.0,
                candidate_cross_string_interval=2.0,
                candidate_max_supervision_weight=4.0,
            )

            self.assertEqual(
                dataset.candidate_supervision["cross_string_events"],
                3,
            )
            self.assertEqual(
                dataset.candidate_supervision["maximum_weight"],
                2.5,
            )
            self.assertEqual(
                dataset.candidate_temporal_contexts[0][69].shape,
                (6, 20),
            )
            self.assertEqual(
                [
                    dataset.candidate_target_weights[(0, index)]
                    for index in range(3)
                ],
                [2.5, 2.5, 2.5],
            )

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

        extended = candidate_temporal_context(
            [0.1, 0.5, 1.5],
            [[64], [64, 67], [69]],
            [[0.8], [0.6, 0.8], [0.5]],
            [[0.4], [0.2, 0.6], [0.1]],
            repeat_interval=0.75,
            maximum_interval=2.0,
            feature_count=12,
        )
        self.assertEqual(extended.shape, (3, 12))
        np.testing.assert_allclose(
            extended[1, 6:],
            np.asarray(
                [0.8, 0.6, 0.1, 3 / 48, 0.2, 0.1],
                dtype=np.float32,
            ),
        )
        extended_head = CandidateActivityHead(
            OnsetPositionConfig(),
            CandidateActivityConfig(
                hidden_size=8,
                dropout=0,
                temporal_feature_count=12,
            ),
        )
        self.assertEqual(
            tuple(
                extended_head(
                    torch.zeros((3, 384)),
                    torch.from_numpy(extended),
                ).shape
            ),
            (3, 2),
        )

    def test_candidate_query_weights_merge_same_pitch_events(self) -> None:
        weights = candidate_position_query_weights(
            torch.tensor([[64, 64, 67, -1, -1, -1]]),
            torch.tensor([[2.0, 4.0, 1.5, 1.0, 1.0, 1.0]]),
        )

        self.assertEqual(weights.tolist(), [3.0, 1.5])

    def test_candidate_position_loss_applies_sample_weights(self) -> None:
        logits = torch.tensor(
            [
                [0.0, -3.0, 3.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 3.0, -3.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        targets = torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        unweighted, _correct, _total = candidate_position_loss(
            logits,
            targets,
            objective="spectral_residual",
        )
        weighted, _correct, _total = candidate_position_loss(
            logits,
            targets,
            objective="spectral_residual",
            sample_weights=torch.tensor([4.0, 1.0]),
        )

        self.assertGreater(float(weighted), float(unweighted))

    def test_candidate_string_temporal_context_is_bidirectional(
        self,
    ) -> None:
        contexts = candidate_string_temporal_context(
            [0.1, 0.5, 1.4],
            [[64], [64, 67], [67]],
            [[0.8], [0.6, 0.9], [0.7]],
            [[0.4], [0.3, 0.5], [0.2]],
            repeat_interval=0.75,
            maximum_interval=2.0,
        )

        self.assertEqual(len(contexts), 3)
        self.assertEqual(contexts[1][64].shape, (6, 20))
        np.testing.assert_allclose(
            contexts[1][64][0, :16],
            np.asarray(
                [
                    0.2,
                    0.45,
                    0.2,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    3 / 24,
                    0.8,
                    0.7,
                    0.8,
                    0.0,
                    1 / 6,
                    1 / 6,
                    0.3 / 2,
                    0.6,
                ],
                dtype=np.float32,
            ),
        )
        self.assertTrue(
            np.all(contexts[1][64][:, 16:20] >= 0)
        )
        self.assertTrue(
            np.all(contexts[1][64][:, 16:20] <= 1)
        )

    def test_onset_position_research_bundle_is_bound_to_its_models(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_config = FeatureConfig()
            position_config = OnsetPositionConfig(
                convolution_channels=(4, 6, 8),
                hidden_size=16,
                dropout=0,
                use_candidate_confidence=True,
            )
            position_path = root / "position.pt"
            torch.save(
                {
                    "format_version": 1,
                    "model_state": OnsetPositionNet(
                        feature_config,
                        position_config,
                    ).state_dict(),
                    "model_config": position_config.to_dict(),
                    "feature_config": feature_config.to_dict(),
                    "metadata": {"name": "test-position"},
                },
                position_path,
            )
            position_hash = sha256_file(position_path)

            def write_activity(
                name: str,
                config: CandidateActivityConfig,
            ) -> Path:
                path = root / name
                torch.save(
                    {
                        "format_version": 1,
                        "model_state": CandidateActivityHead(
                            position_config,
                            config,
                        ).state_dict(),
                        "activity_config": config.to_dict(),
                        "position_config": position_config.to_dict(),
                        "feature_config": feature_config.to_dict(),
                        "metadata": {
                            "base_checkpoint": {
                                "name": position_path.name,
                                "sha256": position_hash,
                            }
                        },
                    },
                    path,
                )
                return path

            primary_path = write_activity(
                "primary.pt",
                CandidateActivityConfig(temporal_feature_count=6),
            )
            supplemental_path = write_activity(
                "supplemental.pt",
                CandidateActivityConfig(temporal_feature_count=12),
            )
            bundle_path = root / "research-bundle.pt"
            result = build_research_bundle(
                position_path,
                primary_path,
                supplemental_path,
                bundle_path,
                version="test-v1",
            )
            loaded = load_research_bundle(
                bundle_path,
                torch.device("cpu"),
            )

            self.assertEqual(result["kind"], BUNDLE_KIND)
            self.assertEqual(loaded.metadata["release_channel"], "research")
            self.assertEqual(
                loaded.primary_activity_head.config.temporal_feature_count,
                6,
            )
            self.assertEqual(
                loaded.supplemental_activity_head.config.temporal_feature_count,
                12,
            )
            self.assertEqual(
                loaded.inference_config["supplemental_assignment"][
                    "minimum_activity_probability"
                ],
                0.85,
            )
            self.assertEqual(
                loaded.position_config.candidate_position_objective,
                "binary",
            )

    def test_candidate_position_checkpoint_accepts_base_activity_heads(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_config = FeatureConfig()
            base_config = OnsetPositionConfig(
                convolution_channels=(4, 6, 8),
                hidden_size=16,
                dropout=0,
            )
            base_path = root / "base-position.pt"
            base_model = OnsetPositionNet(feature_config, base_config)
            torch.save(
                {
                    "format_version": 1,
                    "model_state": base_model.state_dict(),
                    "model_config": base_config.to_dict(),
                    "feature_config": feature_config.to_dict(),
                    "metadata": {"name": "base-position"},
                },
                base_path,
            )
            base_hash = sha256_file(base_path)

            candidate_config = OnsetPositionConfig(
                convolution_channels=(4, 6, 8),
                hidden_size=16,
                dropout=0,
                candidate_position_head=True,
                candidate_position_objective="spectral_residual",
                candidate_spectral_evidence=True,
            )
            candidate_model = OnsetPositionNet(
                feature_config,
                candidate_config,
            )
            candidate_model.load_state_dict(
                base_model.state_dict(),
                strict=False,
            )
            candidate_path = root / "candidate-position.pt"
            torch.save(
                {
                    "format_version": 1,
                    "model_state": candidate_model.state_dict(),
                    "model_config": candidate_config.to_dict(),
                    "feature_config": feature_config.to_dict(),
                    "metadata": {
                        "name": "candidate-position",
                        "base_checkpoint": {
                            "name": base_path.name,
                            "sha256": base_hash,
                        },
                    },
                },
                candidate_path,
            )

            def activity_checkpoint(name: str) -> Path:
                config = CandidateActivityConfig(
                    temporal_feature_count=6,
                )
                path = root / name
                torch.save(
                    {
                        "format_version": 1,
                        "model_state": CandidateActivityHead(
                            base_config,
                            config,
                        ).state_dict(),
                        "activity_config": config.to_dict(),
                        "position_config": base_config.to_dict(),
                        "feature_config": feature_config.to_dict(),
                        "metadata": {
                            "base_checkpoint": {
                                "name": base_path.name,
                                "sha256": base_hash,
                            }
                        },
                    },
                    path,
                )
                return path

            primary_path = activity_checkpoint("primary.pt")
            supplemental_path = activity_checkpoint("supplemental.pt")
            sequence_path = root / "candidate-sequence.pt"
            sequence_config = CandidateSequenceConfig(
                sequence_length=4,
                sequence_stride=2,
                token_size=8,
                hidden_size=6,
                output_hidden_size=7,
                dropout=0,
            )
            torch.save(
                {
                    "format_version": 1,
                    "model_state": CandidateSequenceStringModel(
                        sequence_config
                    ).state_dict(),
                    "sequence_config": sequence_config.to_dict(),
                    "position_config": candidate_config.to_dict(),
                    "feature_config": feature_config.to_dict(),
                    "metadata": {
                        "base_checkpoint": {
                            "name": candidate_path.name,
                            "sha256": sha256_file(candidate_path),
                        }
                    },
                },
                sequence_path,
            )
            sequence_activity_path = (
                root / "candidate-sequence-activity.pt"
            )
            torch.save(
                {
                    "format_version": 1,
                    "model_state": CandidateSequenceActivityModel(
                        sequence_config
                    ).state_dict(),
                    "sequence_config": sequence_config.to_dict(),
                    "position_config": candidate_config.to_dict(),
                    "feature_config": feature_config.to_dict(),
                    "metadata": {
                        "base_checkpoint": {
                            "name": candidate_path.name,
                            "sha256": sha256_file(candidate_path),
                        },
                        "string_sequence_checkpoint": {
                            "name": sequence_path.name,
                            "sha256": sha256_file(sequence_path),
                        },
                    },
                },
                sequence_activity_path,
            )
            bundle_path = root / "candidate-bundle.pt"

            build_research_bundle(
                candidate_path,
                primary_path,
                supplemental_path,
                bundle_path,
                version="candidate-v1",
                candidate_sequence_checkpoint_path=sequence_path,
                candidate_sequence_activity_checkpoint_path=(
                    sequence_activity_path
                ),
            )
            loaded = load_research_bundle(
                bundle_path,
                torch.device("cpu"),
            )

            self.assertTrue(
                loaded.position_config.candidate_position_head
            )
            self.assertEqual(
                loaded.inference_config["assignment"]["mode"],
                "candidate",
            )
            self.assertIsNotNone(loaded.candidate_sequence_model)
            self.assertIsNotNone(
                loaded.candidate_sequence_activity_model
            )
            self.assertEqual(
                loaded.inference_config["assignment"][
                    "candidate_decision"
                ],
                "hierarchical",
            )
            mismatched = torch.load(
                sequence_activity_path,
                map_location="cpu",
                weights_only=False,
            )
            mismatched["metadata"]["string_sequence_checkpoint"][
                "sha256"
            ] = "0" * 64
            torch.save(mismatched, sequence_activity_path)
            with self.assertRaisesRegex(
                ValueError,
                "different string sequence checkpoint",
            ):
                build_research_bundle(
                    candidate_path,
                    primary_path,
                    supplemental_path,
                    root / "invalid-bundle.pt",
                    version="candidate-invalid",
                    candidate_sequence_checkpoint_path=sequence_path,
                    candidate_sequence_activity_checkpoint_path=(
                        sequence_activity_path
                    ),
                )

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

    def test_candidate_activity_group_counts_follow_filtered_groups(
        self,
    ) -> None:
        annotations = [
            TrackAnnotation(
                track_id="guitarset",
                group_id="guitarset",
                audio="guitarset.wav",
                split="train",
                duration=1.0,
                events=[],
                provenance={"dataset": "GuitarSet"},
            ),
            TrackAnnotation(
                track_id="agpt",
                group_id="agpt",
                audio="agpt.wav",
                split="train",
                duration=1.0,
                events=[],
                provenance={"dataset": "AG-PT-set"},
            ),
        ]
        groups = [
            OnsetGroupIndex(annotation_index=0, event_indexes=()),
            OnsetGroupIndex(annotation_index=1, event_indexes=()),
            OnsetGroupIndex(annotation_index=1, event_indexes=()),
        ]

        self.assertEqual(
            candidate_group_counts(annotations, groups),
            {"AG-PT-set": 2, "GuitarSet": 1},
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

    def test_candidate_position_assignment_is_pitch_constrained(
        self,
    ) -> None:
        probabilities = np.full((2, 6), 0.01, dtype=np.float32)
        probabilities[0, 0] = 0.8
        probabilities[0, 2] = 0.9
        probabilities[1, 0] = 0.95
        probabilities[1, 1] = 0.85

        positions = assign_candidate_pitch_group(
            probabilities,
            [64, 59],
        )

        self.assertEqual(positions, [(2, 9), (1, 0)])
        self.assertTrue(
            all(
                OPEN_STRING_PITCHES[string] + fret == pitch
                for (string, fret), pitch in zip(
                    positions,
                    [64, 59],
                )
            )
        )

    def test_categorical_candidate_assignment_can_reject_and_uses_unique_strings(
        self,
    ) -> None:
        probabilities = np.full((3, 7), 0.001, dtype=np.float32)
        probabilities[0, 2] = 0.8
        probabilities[0, 3] = 0.7
        probabilities[1, 2] = 0.9
        probabilities[1, 4] = 0.2
        probabilities[2, 0] = 0.95

        positions = assign_candidate_pitch_group(
            probabilities,
            [64, 59, 67],
        )

        self.assertEqual(positions, [(2, 9), (1, 0), None])

    def test_hierarchical_candidate_assignment_filters_before_strings(
        self,
    ) -> None:
        activity = np.array([0.6, 0.4], dtype=np.float32)
        strings = np.full((2, 6), 0.01, dtype=np.float32)
        strings[0, 5] = 0.95
        strings[1, 1] = 0.95

        positions = assign_hierarchical_candidate_pitch_group(
            activity,
            strings,
            [64, 59],
            minimum_activity_probability=0.5,
        )

        self.assertEqual(positions, [(5, 24), None])

    def test_hierarchical_candidate_assignment_filters_string_margin(
        self,
    ) -> None:
        activity = np.array([0.8, 0.8], dtype=np.float32)
        strings = np.full((2, 6), 0.0, dtype=np.float32)
        strings[0, 4:6] = [0.51, 0.49]
        strings[1, 4:6] = [0.8, 0.2]

        positions = assign_hierarchical_candidate_pitch_group(
            activity,
            strings,
            [64, 64],
            minimum_activity_probability=0.5,
            minimum_string_margin=0.05,
        )

        self.assertEqual(positions, [None, (4, 19)])

    def test_hierarchical_candidate_assignment_can_mask_after_grouping(
        self,
    ) -> None:
        activity = np.array([0.8, 0.8], dtype=np.float32)
        strings = np.array(
            [
                [0.163, 0.163, 0.163, 0.163, 0.184, 0.164],
                [0.0925, 0.0925, 0.0925, 0.30, 0.33, 0.0925],
            ],
            dtype=np.float32,
        )

        baseline = assign_hierarchical_candidate_pitch_group(
            activity,
            strings,
            [64, 64],
            minimum_activity_probability=0.5,
            string_margin_assignment_order="post_assignment",
        )
        pre_assignment = assign_hierarchical_candidate_pitch_group(
            activity,
            strings,
            [64, 64],
            minimum_activity_probability=0.5,
            minimum_string_margin=0.025,
            string_margin_assignment_order="pre_assignment",
        )
        post_assignment = assign_hierarchical_candidate_pitch_group(
            activity,
            strings,
            [64, 64],
            minimum_activity_probability=0.5,
            minimum_string_margin=0.025,
            string_margin_assignment_order="post_assignment",
        )

        self.assertEqual(baseline, [(4, 19), (3, 14)])
        self.assertEqual(pre_assignment, [None, (4, 19)])
        self.assertEqual(post_assignment, [None, baseline[1]])

    def test_candidate_assignment_mode_uses_candidate_position_head(
        self,
    ) -> None:
        feature_config = FeatureConfig()
        model_config = OnsetPositionConfig(
            dropout=0,
            candidate_position_head=True,
        )
        model = OnsetPositionNet(feature_config, model_config).eval()
        base_logits = torch.full((1, 6, 26), -10.0)
        base_logits[..., 0] = 10.0
        candidate_logits = torch.full((2, 6), -5.0)
        candidate_logits[0, 2] = 4.0
        candidate_logits[1, 1] = 4.0
        events = [
            {
                "onset": 0.1,
                "offset": 0.3,
                "pitch": 64,
                "confidence": 1.0,
            },
            {
                "onset": 0.11,
                "offset": 0.3,
                "pitch": 59,
                "confidence": 1.0,
            },
        ]
        features = np.zeros((30, feature_config.n_bins), dtype=np.float32)

        with (
            patch.object(
                model,
                "encode",
                return_value=torch.zeros((1, 384)),
            ),
            patch.object(model, "classify", return_value=base_logits),
            patch.object(
                model,
                "classify_candidate_positions",
                return_value=candidate_logits,
            ),
        ):
            predictions = assign_detected_events(
                events,
                features,
                model,
                feature_config,
                model_config,
                torch.device("cpu"),
                mode="candidate",
                minimum_direct_probability=0.5,
            )

        self.assertEqual(
            [
                (
                    event.string,
                    event.fret,
                    OPEN_STRING_PITCHES[event.string - 1] + event.fret,
                )
                for event in predictions
            ],
            [(3, 9, 64), (2, 0, 59)],
        )

    def test_zero_initialized_sequence_keeps_candidate_assignment(
        self,
    ) -> None:
        feature_config = FeatureConfig()
        model_config = OnsetPositionConfig(
            dropout=0,
            candidate_position_head=True,
            candidate_position_objective="spectral_residual",
            candidate_spectral_evidence=True,
        )
        model = OnsetPositionNet(feature_config, model_config).eval()
        sequence_model = CandidateSequenceStringModel(
            CandidateSequenceConfig(dropout=0)
        ).eval()
        sequence_activity_model = CandidateSequenceActivityModel(
            CandidateSequenceConfig(dropout=0)
        ).eval()
        sequence_joint_model = CandidateSequenceJointModel(
            CandidateSequenceConfig(dropout=0)
        ).eval()
        sequence_confidence_model = CandidateSequenceConfidenceModel(
            CandidateSequenceConfig(dropout=0),
            identity_accept_probability=0.99,
        ).eval()
        base_logits = torch.full((1, 6, 26), -10.0)
        base_logits[..., 0] = 10.0
        candidate_logits = torch.full((2, 7), -5.0)
        candidate_logits[0, 3] = 4.0
        candidate_logits[1, 2] = 4.0
        events = [
            {
                "onset": 0.1,
                "offset": 0.3,
                "pitch": 64,
                "confidence": 0.9,
            },
            {
                "onset": 0.11,
                "offset": 0.3,
                "pitch": 59,
                "confidence": 0.8,
            },
        ]
        features = np.zeros((30, feature_config.n_bins), dtype=np.float32)

        with (
            patch.object(
                model,
                "encode",
                return_value=torch.zeros((1, 384)),
            ),
            patch.object(model, "classify", return_value=base_logits),
            patch.object(
                model,
                "classify_candidate_positions",
                return_value=candidate_logits,
            ),
            patch.object(
                model,
                "classify_candidate_activity",
                return_value=torch.ones(2),
            ),
        ):
            baseline = assign_detected_events(
                events,
                features,
                model,
                feature_config,
                model_config,
                torch.device("cpu"),
                mode="candidate",
            )
            refined = assign_detected_events(
                events,
                features,
                model,
                feature_config,
                model_config,
                torch.device("cpu"),
                mode="candidate",
                sequence_model=sequence_model,
            )
            hierarchical_baseline = assign_detected_events(
                events,
                features,
                model,
                feature_config,
                model_config,
                torch.device("cpu"),
                mode="candidate",
                candidate_decision="hierarchical",
                sequence_model=sequence_model,
            )
            activity_refined = assign_detected_events(
                events,
                features,
                model,
                feature_config,
                model_config,
                torch.device("cpu"),
                mode="candidate",
                candidate_decision="hierarchical",
                sequence_model=sequence_model,
                sequence_activity_model=sequence_activity_model,
            )
            joint_refined = assign_detected_events(
                events,
                features,
                model,
                feature_config,
                model_config,
                torch.device("cpu"),
                mode="candidate",
                candidate_decision="hierarchical",
                sequence_model=sequence_model,
                sequence_joint_model=sequence_joint_model,
            )
            confidence_refined = assign_detected_events(
                events,
                features,
                model,
                feature_config,
                model_config,
                torch.device("cpu"),
                mode="candidate",
                candidate_decision="hierarchical",
                sequence_model=sequence_model,
                sequence_confidence_model=sequence_confidence_model,
            )

        self.assertEqual(baseline, refined)
        self.assertEqual(hierarchical_baseline, activity_refined)
        self.assertEqual(hierarchical_baseline, joint_refined)
        self.assertEqual(hierarchical_baseline, confidence_refined)

    def test_sequence_activity_keeps_group_activity_logits_across_batches(
        self,
    ) -> None:
        feature_config = FeatureConfig()
        model_config = OnsetPositionConfig(
            dropout=0,
            candidate_position_head=True,
            candidate_position_objective="spectral_residual",
            candidate_spectral_evidence=True,
        )
        model = OnsetPositionNet(feature_config, model_config).eval()
        sequence_config = CandidateSequenceConfig(dropout=0)
        sequence_model = CandidateSequenceStringModel(
            sequence_config
        ).eval()
        sequence_activity_model = CandidateSequenceActivityModel(
            sequence_config
        ).eval()
        events = [
            {
                "onset": 0.1 + index * 0.1,
                "offset": 0.15 + index * 0.1,
                "pitch": 52,
                "confidence": 0.9,
            }
            for index in range(33)
        ]
        events.append(
            {
                "onset": events[-1]["onset"],
                "offset": events[-1]["offset"],
                "pitch": 55,
                "confidence": 0.8,
            }
        )
        features = np.zeros(
            (400, feature_config.n_bins),
            dtype=np.float32,
        )

        predictions = assign_detected_events(
            events,
            features,
            model,
            feature_config,
            model_config,
            torch.device("cpu"),
            mode="candidate",
            candidate_decision="hierarchical",
            sequence_model=sequence_model,
            sequence_activity_model=sequence_activity_model,
        )

        self.assertIsInstance(predictions, list)

    def test_categorical_candidate_assignment_honors_none_class(
        self,
    ) -> None:
        feature_config = FeatureConfig()
        model_config = OnsetPositionConfig(
            dropout=0,
            candidate_position_head=True,
            candidate_position_objective="categorical",
        )
        model = OnsetPositionNet(feature_config, model_config).eval()
        base_logits = torch.full((1, 6, 26), -10.0)
        base_logits[..., 0] = 10.0
        candidate_logits = torch.full((2, 7), -5.0)
        candidate_logits[0, 3] = 4.0
        candidate_logits[1, 0] = 4.0
        events = [
            {
                "onset": 0.1,
                "offset": 0.3,
                "pitch": 64,
                "confidence": 1.0,
            },
            {
                "onset": 0.11,
                "offset": 0.3,
                "pitch": 59,
                "confidence": 1.0,
            },
        ]
        features = np.zeros((30, feature_config.n_bins), dtype=np.float32)

        with (
            patch.object(
                model,
                "encode",
                return_value=torch.zeros((1, 384)),
            ),
            patch.object(model, "classify", return_value=base_logits),
            patch.object(
                model,
                "classify_candidate_positions",
                return_value=candidate_logits,
            ),
        ):
            predictions = assign_detected_events(
                events,
                features,
                model,
                feature_config,
                model_config,
                torch.device("cpu"),
                mode="candidate",
                minimum_direct_probability=0.0,
            )

        self.assertEqual(
            [
                (
                    event.string,
                    event.fret,
                    OPEN_STRING_PITCHES[event.string - 1] + event.fret,
                )
                for event in predictions
            ],
            [(3, 9, 64)],
        )

    def test_hierarchical_candidate_assignment_uses_total_activity(
        self,
    ) -> None:
        feature_config = FeatureConfig()
        model_config = OnsetPositionConfig(
            dropout=0,
            candidate_position_head=True,
            candidate_position_objective="categorical",
        )
        model = OnsetPositionNet(feature_config, model_config).eval()
        base_logits = torch.full((1, 6, 26), -10.0)
        base_logits[..., 0] = 10.0
        candidate_logits = torch.log(
            torch.tensor(
                [[0.45, 0.10, 0.10, 0.10, 0.10, 0.10, 0.05]]
            )
        )
        events = [
            {
                "onset": 0.1,
                "offset": 0.3,
                "pitch": 64,
                "confidence": 1.0,
            }
        ]
        features = np.zeros((30, feature_config.n_bins), dtype=np.float32)

        with (
            patch.object(
                model,
                "encode",
                return_value=torch.zeros((1, 384)),
            ),
            patch.object(model, "classify", return_value=base_logits),
            patch.object(
                model,
                "classify_candidate_positions",
                return_value=candidate_logits,
            ),
        ):
            joint = assign_detected_events(
                events,
                features,
                model,
                feature_config,
                model_config,
                torch.device("cpu"),
                mode="candidate",
                minimum_direct_probability=0.0,
            )
            hierarchical = assign_detected_events(
                events,
                features,
                model,
                feature_config,
                model_config,
                torch.device("cpu"),
                mode="candidate",
                minimum_direct_probability=0.0,
                candidate_decision="hierarchical",
                candidate_minimum_activity_probability=0.5,
            )

        self.assertEqual(joint, [])
        self.assertEqual(len(hierarchical), 1)
        self.assertEqual(
            OPEN_STRING_PITCHES[hierarchical[0].string - 1]
            + hierarchical[0].fret,
            64,
        )

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

        accepted_trace: dict[str, int] = {}
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
            stage_trace=accepted_trace,
        )
        rejected_trace: dict[str, int] = {}
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
            stage_trace=rejected_trace,
        )

        self.assertEqual(len(accepted), 1)
        self.assertEqual(rejected, [])
        self.assertEqual(accepted_trace["candidate_groups"], 1)
        self.assertEqual(accepted_trace["activity_accepted_groups"], 1)
        self.assertEqual(accepted_trace["position_accepted_events"], 1)
        self.assertEqual(accepted_trace["confidence_accepted_events"], 1)
        self.assertEqual(accepted_trace["confidence_rejected_events"], 0)
        self.assertEqual(rejected_trace["position_accepted_events"], 1)
        self.assertEqual(rejected_trace["confidence_accepted_events"], 0)
        self.assertEqual(rejected_trace["confidence_rejected_events"], 1)

    def test_onset_position_trace_records_activity_rejection(self) -> None:
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
            for parameter in activity_head.parameters():
                parameter.zero_()
            activity_head.network[-1].bias.fill_(-10)
        events = [
            {
                "onset": 0.1,
                "offset": 0.3,
                "pitch": 64,
                "confidence": 1.0,
            }
        ]
        trace: dict[str, int] = {}

        predictions = assign_detected_events(
            events,
            np.zeros((30, feature_config.n_bins), dtype=np.float32),
            position_model,
            feature_config,
            position_config,
            torch.device("cpu"),
            activity_head=activity_head,
            minimum_activity_probability=0.5,
            stage_trace=trace,
        )

        self.assertEqual(predictions, [])
        self.assertEqual(trace["candidate_events"], 1)
        self.assertEqual(trace["activity_rejected_groups"], 1)
        self.assertEqual(trace["activity_rejected_candidate_events"], 1)
        self.assertEqual(trace["position_accepted_events"], 0)
        self.assertEqual(trace["emitted_events"], 0)

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

    def test_pitch_fusion_keeps_missing_chord_tones(self) -> None:
        candidates = [
            {"onset": 1.02, "offset": 1.2, "pitch": 64},
            {"onset": 1.02, "offset": 1.2, "pitch": 67},
            {"onset": 1.2, "offset": 1.4, "pitch": 64},
        ]

        unmatched = unmatched_pitch_candidates(
            [{"onset": 1.0, "offset": 1.2, "pitch": 64}],
            candidates,
            match_tolerance=0.05,
        )

        self.assertEqual(unmatched, candidates[1:])

    def test_onset_fusion_merge_deduplicates_tablature_events(self) -> None:
        primary = [LabeledEvent(0.1, 0.3, 2, 5, confidence=0.8)]
        supplemental = [
            LabeledEvent(0.14, 0.4, 2, 5, confidence=0.9),
            LabeledEvent(0.14, 0.4, 3, 5, confidence=0.9),
        ]

        trace: dict[str, int] = {}
        merged = merge_predictions(
            primary,
            supplemental,
            onset_tolerance=0.05,
            stage_trace=trace,
        )

        self.assertEqual(len(merged), 2)
        self.assertEqual(
            [(event.string, event.fret) for event in merged],
            [(2, 5), (3, 5)],
        )
        self.assertEqual(trace["accepted_supplemental_events"], 1)
        self.assertEqual(trace["suppressed_duplicate_events"], 1)
        self.assertEqual(trace["emitted_events"], 2)

    def test_frontend_candidate_identity_merge_precedes_position_assignment(
        self,
    ) -> None:
        primary = [
            {
                "onset": 0.1,
                "offset": 0.3,
                "pitch": 64,
                "confidence": 0.7,
            }
        ]
        supplemental = [
            {
                "onset": 0.13,
                "offset": 0.4,
                "pitch": 64,
                "confidence": 0.9,
            },
            {
                "onset": 0.13,
                "offset": 0.4,
                "pitch": 67,
                "confidence": 0.8,
            },
        ]

        trace: dict[str, int] = {}
        merged = merge_candidate_identities(
            primary,
            supplemental,
            onset_tolerance=0.05,
            stage_trace=trace,
        )

        self.assertEqual(
            [
                (
                    event["onset"],
                    event["offset"],
                    event["pitch"],
                    event["confidence"],
                )
                for event in merged
            ],
            [
                (0.1, 0.4, 64, 0.9),
                (0.13, 0.4, 67, 0.8),
            ],
        )
        self.assertEqual(trace["suppressed_supplemental_duplicates"], 1)
        self.assertEqual(trace["merged_candidate_events"], 2)

    def test_stage_trace_aggregation_sums_nested_counts(self) -> None:
        aggregate = aggregate_stage_traces(
            [
                {
                    "stage_trace": {
                        "primary": {
                            "assignment": {
                                "candidate_events": 2,
                                "emitted_events": 1,
                            }
                        },
                        "final_emitted_events": 1,
                    }
                },
                {
                    "stage_trace": {
                        "primary": {
                            "assignment": {
                                "candidate_events": 3,
                                "emitted_events": 2,
                            }
                        },
                        "supplemental": {
                            "merge": {
                                "suppressed_duplicate_events": 1,
                            }
                        },
                        "final_emitted_events": 2,
                    }
                },
            ]
        )

        self.assertEqual(aggregate["track_count"], 2)
        self.assertEqual(
            aggregate["primary"]["assignment"]["candidate_events"],
            5,
        )
        self.assertEqual(aggregate["final_emitted_events"], 3)
        self.assertEqual(
            aggregate["supplemental"]["merge"][
                "suppressed_duplicate_events"
            ],
            1,
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

    def test_supplemental_activity_cli_requires_a_supplemental_frontend(
        self,
    ) -> None:
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parse_arguments(
                [
                    "--manifest",
                    "manifest.jsonl",
                    "--checkpoint",
                    "position.pt",
                    "--supplemental-candidate-activity-checkpoint",
                    "activity.pt",
                ]
            )

        parsed = parse_arguments(
            [
                "--manifest",
                "manifest.jsonl",
                "--checkpoint",
                "position.pt",
                "--supplemental-pitch-onset-threshold",
                "0.25",
                "--supplemental-candidate-activity-checkpoint",
                "activity.pt",
                "--supplemental-minimum-activity-probability",
                "0.7",
            ]
        )
        self.assertEqual(
            parsed.supplemental_candidate_activity_checkpoint,
            Path("activity.pt"),
        )
        self.assertEqual(
            parsed.supplemental_minimum_activity_probability,
            0.7,
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

    def test_hierarchical_candidate_cli_rejects_none_bias(self) -> None:
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parse_arguments(
                [
                    "--manifest",
                    "manifest.jsonl",
                    "--checkpoint",
                    "position.pt",
                    "--mode",
                    "candidate",
                    "--candidate-decision",
                    "hierarchical",
                    "--candidate-none-logit-bias",
                    "-0.5",
                ]
            )

        parsed = parse_arguments(
            [
                "--manifest",
                "manifest.jsonl",
                "--checkpoint",
                "position.pt",
                "--mode",
                "candidate",
                "--candidate-decision",
                "hierarchical",
                "--candidate-minimum-activity-probability",
                "0.7",
                "--candidate-minimum-string-margin",
                "0.1",
            ]
        )
        self.assertEqual(parsed.candidate_decision, "hierarchical")
        self.assertEqual(
            parsed.candidate_minimum_activity_probability,
            0.7,
        )
        self.assertEqual(parsed.candidate_minimum_string_margin, 0.1)

        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parse_arguments(
                [
                    "--manifest",
                    "manifest.jsonl",
                    "--checkpoint",
                    "position.pt",
                    "--mode",
                    "candidate",
                    "--candidate-minimum-string-margin",
                    "0.1",
                ]
            )

    def test_candidate_sequence_cli_requires_candidate_mode(self) -> None:
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parse_arguments(
                [
                    "--manifest",
                    "manifest.jsonl",
                    "--checkpoint",
                    "position.pt",
                    "--candidate-sequence-checkpoint",
                    "sequence.pt",
                ]
            )

        parsed = parse_arguments(
            [
                "--manifest",
                "manifest.jsonl",
                "--checkpoint",
                "position.pt",
                "--mode",
                "candidate",
                "--candidate-sequence-checkpoint",
                "sequence.pt",
            ]
        )
        self.assertEqual(
            parsed.candidate_sequence_checkpoint,
            Path("sequence.pt"),
        )
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parse_arguments(
                [
                    "--manifest",
                    "manifest.jsonl",
                    "--checkpoint",
                    "position.pt",
                    "--mode",
                    "candidate",
                    "--candidate-sequence-activity-checkpoint",
                    "activity-sequence.pt",
                ]
            )

        parsed_activity = parse_arguments(
            [
                "--manifest",
                "manifest.jsonl",
                "--checkpoint",
                "position.pt",
                "--mode",
                "candidate",
                "--candidate-decision",
                "hierarchical",
                "--candidate-sequence-checkpoint",
                "sequence.pt",
                "--candidate-sequence-activity-checkpoint",
                "activity-sequence.pt",
            ]
        )
        self.assertEqual(
            parsed_activity.candidate_sequence_activity_checkpoint,
            Path("activity-sequence.pt"),
        )
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parse_arguments(
                [
                    "--manifest",
                    "manifest.jsonl",
                    "--checkpoint",
                    "position.pt",
                    "--mode",
                    "candidate",
                    "--candidate-sequence-confidence-checkpoint",
                    "confidence-sequence.pt",
                ]
            )

        parsed_confidence = parse_arguments(
            [
                "--manifest",
                "manifest.jsonl",
                "--checkpoint",
                "position.pt",
                "--mode",
                "candidate",
                "--candidate-decision",
                "hierarchical",
                "--candidate-sequence-checkpoint",
                "sequence.pt",
                "--candidate-sequence-confidence-checkpoint",
                "confidence-sequence.pt",
            ]
        )
        self.assertEqual(
            parsed_confidence.candidate_sequence_confidence_checkpoint,
            Path("confidence-sequence.pt"),
        )

    def test_margin_calibration_cli_requires_frozen_candidate_path(
        self,
    ) -> None:
        base_arguments = [
            "--manifest",
            "manifest.jsonl",
            "--checkpoint",
            "position.pt",
            "--mode",
            "candidate",
            "--candidate-decision",
            "hierarchical",
            "--candidate-sequence-checkpoint",
            "sequence.pt",
            "--candidate-string-margin-calibration",
            "margin.json",
        ]

        parsed = parse_arguments(base_arguments)
        self.assertEqual(
            parsed.candidate_string_margin_calibration,
            Path("margin.json"),
        )

        incompatible_arguments = [
            [
                "--candidate-minimum-string-margin",
                "0.1",
            ],
            [
                "--candidate-minimum-activity-probability",
                "0.6",
            ],
            [
                "--candidate-sequence-confidence-checkpoint",
                "confidence.pt",
            ],
            [
                "--supplemental-pitch-onset-threshold",
                "0.2",
            ],
        ]
        for extra_arguments in incompatible_arguments:
            with (
                self.subTest(arguments=extra_arguments),
                patch("sys.stderr"),
                self.assertRaises(SystemExit),
            ):
                parse_arguments(base_arguments + extra_arguments)

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

    def test_beat_gold_requires_two_annotations_and_third_party_adjudication(
        self,
    ) -> None:
        with self.temporary_directory() as directory:
            root = Path(directory)
            audio_path = root / "source.wav"
            analysis_path = root / "analysis.json"
            soundfile.write(
                audio_path,
                np.zeros(22_050 * 12, dtype=np.float32),
                22_050,
                subtype="PCM_16",
            )
            analysis = {
                "time_signature": [4, 4],
                "pickup_beats": 0.25,
                "beats": [
                    round(0.25 + index * 0.5, 2)
                    for index in range(24)
                ],
            }
            analysis_path.write_text(
                json.dumps(analysis),
                encoding="utf-8",
            )
            paths = {
                alias: root / f"{alias}.json"
                for alias in ("annotator-a", "annotator-b", "adjudicator")
            }
            for alias, review_path in paths.items():
                create_beat_review(
                    audio_path=audio_path,
                    analysis_path=analysis_path,
                    review_path=review_path,
                    clip_id="steady-4-4-001",
                    group_id="source-song-001",
                    annotator_alias=alias,
                    source_start=4,
                    duration=4,
                    corpus_version="beat-pilot-v2-test",
                    pre_roll=2,
                    post_roll=2,
                    tags=["steady", "4-4"],
                )
            right = json.loads(paths["annotator-b"].read_text(encoding="utf-8"))
            for marker in right["beats"]:
                marker["at"] = round(float(marker["at"]) + 0.01, 6)
            paths["annotator-b"].write_text(
                json.dumps(right),
                encoding="utf-8",
            )
            submitted = {
                alias: submit_beat_review(review_path)
                for alias, review_path in paths.items()
            }
            agreement = compare_beat_reviews(
                submitted["annotator-a"],
                submitted["annotator-b"],
            )
            self.assertEqual(agreement["beat_metrics"]["f1"], 1)
            self.assertTrue(agreement["meter_agreement"])

            gold_path = root / "gold.json"
            gold = adjudicate_beat_reviews(
                left_path=paths["annotator-a"],
                right_path=paths["annotator-b"],
                resolved_path=paths["adjudicator"],
                output_path=gold_path,
                adjudicator_alias="adjudicator",
            )
            self.assertEqual(gold["verification_status"], "adjudicated_gold")
            self.assertTrue(gold["benchmark_eligible"])
            report = evaluate_beat_review(gold, analysis)
            self.assertEqual(report["reference_beats"], 8)
            self.assertEqual(report["beat_metrics"]["f1"], 1)
            self.assertEqual(report["downbeat_metrics"]["f1"], 1)
            self.assertTrue(report["pickup_correct"])
            self.assertFalse(
                evaluate_beat_review(
                    gold,
                    {**analysis, "pickup_beats": 0},
                )["pickup_correct"]
            )
            self.assertTrue(evaluate_beat_gate(report)["passed"])
            report["beat_metrics"]["p95_error_ms"] = 80
            self.assertFalse(evaluate_beat_gate(report)["passed"])

            gold["beats"][0]["at"] = 0.3
            with self.assertRaisesRegex(ValueError, "invalid digest"):
                validate_beat_review(gold, require_adjudicated=True)

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

    def test_model_agreement_requires_positions_from_every_model(
        self,
    ) -> None:
        measures = [
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
                                "string": 6,
                                "fret": 0,
                            }
                        ],
                    }
                ],
            }
        ]
        by_measure, aggregate = analyze_model_disagreements(
            measures,
            {
                "model-a": [
                    {
                        "note_id": "note-1",
                        "onset": 0.1,
                        "pitch": 40,
                        "string": 6,
                        "fret": 0,
                        "confidence": 0.95,
                    }
                ],
                "model-b": [
                    {
                        "note_id": "note-1",
                        "onset": 0.1,
                        "pitch": 40,
                        "confidence": 0.95,
                    }
                ],
            },
        )

        self.assertEqual(aggregate["auto_candidates"], 0)
        self.assertEqual(aggregate["missing_position_labels"], 1)
        self.assertEqual(
            by_measure[1]["event_decisions"][0]["reasons"],
            ["missing-position-label"],
        )

    def test_consensus_leftovers_are_one_project_disagreement(
        self,
    ) -> None:
        measures = [
            {
                "number": 1,
                "start": 0.0,
                "duration": 1.0,
                "beats": [],
            }
        ]
        event = {
            "onset": 0.2,
            "pitch": 40,
            "string": 6,
            "fret": 0,
            "confidence": 0.95,
        }
        by_measure, aggregate = analyze_model_disagreements(
            measures,
            {
                "model-a": [event],
                "model-b": [{**event, "onset": 0.21}],
            },
        )

        self.assertEqual(aggregate["unmatched_predictions"], 1)
        self.assertEqual(aggregate["presence_disagreements"], 0)
        self.assertEqual(aggregate["review_only_events"], 1)
        self.assertEqual(
            by_measure[1]["event_decisions"][0]["reasons"],
            ["unmatched-project-note"],
        )

    def test_leftover_note_identity_precedes_onset_clustering(
        self,
    ) -> None:
        measures = [
            {
                "number": 1,
                "start": 0.0,
                "duration": 1.0,
                "beats": [],
            }
        ]
        by_measure, aggregate = analyze_model_disagreements(
            measures,
            {
                "model-a": [
                    {
                        "note_id": "shared-extra",
                        "onset": 0.2,
                        "pitch": 40,
                        "string": 6,
                        "fret": 0,
                        "confidence": 0.95,
                    }
                ],
                "model-b": [
                    {
                        "note_id": "shared-extra",
                        "onset": 0.3,
                        "pitch": 40,
                        "string": 6,
                        "fret": 0,
                        "confidence": 0.95,
                    }
                ],
            },
        )

        self.assertEqual(aggregate["unmatched_predictions"], 1)
        self.assertEqual(aggregate["presence_disagreements"], 0)
        self.assertEqual(aggregate["timing_disagreements"], 1)
        self.assertEqual(
            by_measure[1]["event_decisions"][0]["note_id"],
            "shared-extra",
        )

    def test_review_queue_ranks_model_disagreement_without_auto_gold(
        self,
    ) -> None:
        with self.temporary_directory() as directory:
            root = Path(directory)
            project_path = root / "project.json"
            notes = [
                {
                    "id": "agreement",
                    "at": 0.1,
                    "duration": 0.2,
                    "string": 6,
                    "fret": 0,
                    "positionConfidence": 0.9,
                    "positionSource": "playable-optimizer-v2",
                },
                {
                    "id": "position-disagreement",
                    "at": 1.1,
                    "duration": 0.2,
                    "string": 5,
                    "fret": 2,
                    "positionConfidence": 0.9,
                    "positionSource": "playable-optimizer-v2",
                },
                {
                    "id": "presence-disagreement",
                    "at": 2.1,
                    "duration": 0.2,
                    "string": 4,
                    "fret": 2,
                    "positionConfidence": 0.9,
                    "positionSource": "playable-optimizer-v2",
                },
            ]
            project_path.write_text(
                json.dumps(
                    {
                        "id": "project-model-disagreement",
                        "source": {"kind": "upload"},
                        "tab": {
                            "measures": [
                                {
                                    "number": index + 1,
                                    "start": float(index),
                                    "duration": 1.0,
                                    "beats": [
                                        {
                                            "at": float(index),
                                            "notes": [note],
                                        }
                                    ],
                                }
                                for index, note in enumerate(notes)
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            model_predictions = {
                "model-a": [
                    {
                        "note_id": "agreement",
                        "onset": 0.1,
                        "pitch": 40,
                        "string": 6,
                        "fret": 0,
                        "confidence": 0.95,
                    },
                    {
                        "note_id": "position-disagreement",
                        "onset": 1.1,
                        "pitch": 47,
                        "string": 5,
                        "fret": 2,
                        "confidence": 0.95,
                    },
                    {
                        "note_id": "presence-disagreement",
                        "onset": 2.1,
                        "pitch": 52,
                        "string": 4,
                        "fret": 2,
                        "confidence": 0.95,
                    },
                ],
                "model-b": [
                    {
                        "note_id": "agreement",
                        "onset": 0.11,
                        "pitch": 40,
                        "string": 6,
                        "fret": 0,
                        "confidence": 0.9,
                    },
                    {
                        "note_id": "position-disagreement",
                        "onset": 1.11,
                        "pitch": 47,
                        "string": 6,
                        "fret": 7,
                        "confidence": 0.9,
                    },
                ],
            }
            report = build_review_queue(
                project_path,
                excerpt_measures=1,
                limit=3,
                model_predictions=model_predictions,
            )

        candidates = {
            candidate["measure_numbers"][0]: candidate
            for candidate in report["candidates"]
        }
        self.assertEqual(
            candidates[1]["candidate_disposition"],
            "auto-candidate",
        )
        self.assertEqual(
            candidates[2]["candidate_disposition"],
            "review-only",
        )
        self.assertEqual(
            candidates[2]["model_disagreement"][
                "position_disagreements"
            ],
            1,
        )
        self.assertEqual(
            candidates[3]["model_disagreement"][
                "presence_disagreements"
            ],
            1,
        )
        self.assertEqual(
            report["model_disagreement_summary"]["auto_candidates"],
            1,
        )
        self.assertFalse(
            report["promotion_policy"]["auto_candidate_is_gold"]
        )
        self.assertTrue(
            report["promotion_policy"][
                "gold_requires_approved_human_review"
            ]
        )

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

    def test_datasets_filter_included_track_ids_before_indexing(self) -> None:
        with self.temporary_directory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            write_manifest(
                manifest,
                [
                    TrackAnnotation(
                        track_id=track_id,
                        group_id=track_id,
                        audio=f"{track_id}.wav",
                        split="train",
                        duration=1.0,
                        events=[
                            LabeledEvent(
                                onset=0.1,
                                offset=0.2,
                                string=1,
                                fret=index,
                            )
                        ],
                    )
                    for index, track_id in enumerate(("first", "second"))
                ],
            )

            track_dataset = GuitarTabDataset(
                manifest,
                split="train",
                included_track_ids={"second"},
            )
            onset_dataset = OnsetPositionDataset(
                manifest,
                split="train",
                included_track_ids={"second"},
            )

            self.assertEqual(
                [item.track_id for item in track_dataset.annotations],
                ["second"],
            )
            self.assertEqual(
                [item.track_id for item in onset_dataset.annotations],
                ["second"],
            )
            self.assertEqual(len(onset_dataset.groups), 1)
            with self.assertRaisesRegex(ValueError, "cannot be empty"):
                GuitarTabDataset(
                    manifest,
                    split="train",
                    included_track_ids=set(),
                )
            with self.assertRaisesRegex(ValueError, "absent"):
                OnsetPositionDataset(
                    manifest,
                    split="train",
                    included_track_ids={"missing"},
                )

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

    def test_pitch_onset_head_aggregates_equivalent_string_positions(
        self,
    ) -> None:
        model = StringFretNet(
            ModelConfig(
                pitch_onset_head=True,
                pitch_onset_hidden_size=16,
            )
        )
        outputs = model(torch.rand(1, 8, 216))
        self.assertEqual(outputs["pitch_onset_logits"].shape, (1, 8, 49))

        fret_targets = torch.zeros(1, 8, 6, dtype=torch.long)
        onset_targets = torch.zeros(1, 8, 6)
        fret_targets[0, 3, 0] = 1
        fret_targets[0, 3, 1] = 6
        onset_targets[0, 3, 0] = 1
        onset_targets[0, 3, 1] = 1
        targets = build_pitch_onset_targets(
            fret_targets,
            onset_targets,
        )
        self.assertEqual(float(targets[0, 3, 64 - PITCH_MIN]), 1)
        self.assertEqual(int((targets > 0).sum()), 1)

        loss, values = transcription_loss(
            outputs,
            fret_targets,
            onset_targets,
            torch.ones(1, 8, dtype=torch.bool),
            pitch_onset_loss_weight=1,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(values["pitch_onset_loss"]), 0)

    def test_pitch_onset_head_only_freezes_every_other_parameter(
        self,
    ) -> None:
        model = StringFretNet(ModelConfig(pitch_onset_head=True))

        configure_trainable_parameters(
            model,
            pitch_onset_head_only=True,
        )

        trainable = {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(
            all(name.startswith("pitch_onset_head.") for name in trainable)
        )

    def test_pitch_onset_decoder_retains_local_pitch_peaks(self) -> None:
        loaded = LoadedModel(
            model=StringFretNet(ModelConfig(pitch_onset_head=True)),
            feature_config=FeatureConfig(),
            metadata={},
        )
        probabilities = np.zeros((10, 49), dtype=np.float32)
        probabilities[2, 24] = 0.8
        probabilities[3, 24] = 0.7
        probabilities[7, 24] = 0.9
        with patch(
            "stringtrace_ml.evaluate_onset_position."
            "predict_pitch_onset_probabilities",
            return_value=probabilities,
        ):
            events = direct_pitch_onset_candidates(
                np.zeros((10, 216), dtype=np.float32),
                loaded,
                torch.device("cpu"),
                onset_threshold=0.5,
                minimum_onset_gap=0.046,
            )

        self.assertEqual([event["pitch"] for event in events], [64, 64])
        self.assertAlmostEqual(float(events[0]["confidence"]), 0.8)
        self.assertAlmostEqual(float(events[1]["confidence"]), 0.9)

    def test_cqt_feature_augmentation_preserves_shape_and_range(self) -> None:
        np.random.seed(7)
        features = np.full((12, 24), 0.5, dtype=np.float32)
        augmented = augment_cqt_features(
            features,
            noise_std=0.03,
            frequency_mask_bins=4,
            contrast_range=0.2,
            spectral_tilt=0.1,
        )

        self.assertEqual(augmented.shape, features.shape)
        self.assertEqual(augmented.dtype, np.float32)
        self.assertGreaterEqual(float(augmented.min()), 0)
        self.assertLessEqual(float(augmented.max()), 1)
        self.assertTrue(np.any(augmented == 0))
        self.assertFalse(np.array_equal(augmented, features))

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

    def test_beat_metrics_report_accuracy_and_tail_error(self) -> None:
        metrics = evaluate_beats(
            [0.0, 0.5, 1.0, 1.5],
            [0.01, 0.48, 1.04, 1.72],
            tolerance=0.07,
        )

        self.assertEqual(metrics["true_positive"], 3)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["false_negative"], 1)
        self.assertAlmostEqual(metrics["f1"], 0.75)
        self.assertAlmostEqual(metrics["median_error_ms"], 20.0)
        self.assertGreater(metrics["p95_error_ms"], 35.0)

    def test_chord_metrics_separate_root_and_quality_accuracy(self) -> None:
        metrics = evaluate_chords(
            [
                ChordRegion(0.0, 2.0, "G"),
                ChordRegion(2.0, 4.0, "Em"),
            ],
            [
                ChordRegion(0.0, 2.0, "G7"),
                ChordRegion(2.0, 4.0, "Em"),
            ],
        )

        self.assertEqual(metrics["coverage"], 1.0)
        self.assertEqual(metrics["exact_accuracy"], 0.5)
        self.assertEqual(metrics["root_accuracy"], 1.0)

    def test_event_metrics_score_reviewed_techniques_only(self) -> None:
        reference = [
            LabeledEvent(0.1, 0.4, 2, 1, "bend"),
            LabeledEvent(0.6, 0.9, 2, 3, "unknown"),
        ]
        predicted = [
            LabeledEvent(0.1, 0.4, 2, 1, "bend"),
            LabeledEvent(0.6, 0.9, 2, 3, "slide"),
        ]

        metrics = evaluate_events(reference, predicted)

        self.assertEqual(metrics["technique"]["true_positive"], 1)
        self.assertEqual(metrics["technique"]["false_negative"], 0)

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

    def test_full_score_gate_requires_every_accuracy_layer(self) -> None:
        report = {
            "reference_events": 200,
            "beat_metrics": {"f1": 0.97, "median_error_ms": 12.0},
            "chord_metrics": {"exact_accuracy": 0.93},
            "metrics": {
                "onset": {"f1": 0.94},
                "pitch": {"f1": 0.92},
                "tablature": {"f1": 0.91},
                "repeated_tablature": {"recall": 0.92},
                "technique": {
                    "true_positive": 18,
                    "false_negative": 2,
                    "precision": 0.9,
                },
            },
        }

        passed = evaluate_score_gate(report)
        self.assertTrue(passed["passed"])

        report["metrics"]["pitch"]["f1"] = 0.7
        failed = evaluate_score_gate(report)
        self.assertFalse(failed["passed"])
        self.assertFalse(failed["checks"]["pitch_f1"])

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
            clipped = shifted_events(
                [TechniqueEvent(0.0, 0.2, 64, 1, "pick")],
                -0.02,
                1.0,
            )

            self.assertEqual(
                [(event.string, event.pitch) for event in shifted],
                [(1, 69), (6, 43)],
            )
            self.assertAlmostEqual(shifted[0].onset, 0.08)
            self.assertEqual(len(clipped), 1)
            self.assertEqual(clipped[0].onset, 0.0)
            self.assertAlmostEqual(clipped[0].offset, 0.18)
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

    def test_guitar_techs_p3_preserves_polyphonic_per_string_events(
        self,
    ) -> None:
        with self.temporary_directory() as directory:
            root = Path(directory)
            midi_directory = root / "P3_music" / "midi"
            midi_directory.mkdir(parents=True)
            for modality in ("directinput", "micamp"):
                audio_directory = (
                    root / "P3_music" / "audio" / modality
                )
                audio_directory.mkdir(parents=True)
                (audio_directory / f"{modality}_01.wav").touch()

            midi_path = midi_directory / "midi_01.mid"
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
            b_string = pretty_midi.Instrument(program=0, name="B")
            b_string.notes.append(
                pretty_midi.Note(
                    velocity=100,
                    pitch=71,
                    start=0.1,
                    end=0.35,
                )
            )
            midi.instruments.extend((high_e, b_string))
            midi.write(str(midi_path))

            recordings = discover_recordings(root, ("P3",))

            self.assertEqual(len(recordings), 2)
            self.assertEqual(
                [recording.audio_path.parent.name for recording in recordings],
                ["directinput", "micamp"],
            )
            self.assertTrue(
                all(
                    recording.content_type == "music"
                    and recording.source_label == "01"
                    and not recording.merge_gestures
                    for recording in recordings
                )
            )
            events = recording_events(recordings[0])
            self.assertEqual(
                [
                    (round(event.onset, 6), event.string, event.pitch)
                    for event in events
                ],
                [(0.1, 1, 69), (0.1, 2, 71)],
            )

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

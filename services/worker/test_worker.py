import unittest
from pathlib import Path
from unittest.mock import patch

from worker import (
    AudioFeature,
    NoteEvent,
    annotate_techniques,
    assign_fingering_group,
    collapse_harmonic_stacks,
    collapse_note_fragments,
    infer_chords,
    infer_technique,
    infer_technique_candidates,
    natural_harmonic_positions,
    select_harmonic_position,
    stabilize_beat_times,
    transcribe_guitar,
)


def note(
    start: float,
    end: float,
    fret: int,
    *,
    string: int = 3,
    velocity: int = 75,
) -> NoteEvent:
    return NoteEvent(
        start=start,
        end=end,
        pitch=55 + fret,
        velocity=velocity,
        string=string,
        fret=fret,
        technique="pick",
        confidence=0.8,
    )


class TechniqueInferenceTest(unittest.TestCase):
    def test_beat_stabilizer_reduces_frame_quantization_jitter(self) -> None:
        raw = [0.6, 1.1109, 1.6217, 2.1093, 2.5969]

        stabilized = stabilize_beat_times(raw)
        raw_intervals = [
            current - previous
            for previous, current in zip(raw, raw[1:])
        ]
        stabilized_intervals = [
            current - previous
            for previous, current in zip(stabilized, stabilized[1:])
        ]

        self.assertEqual(stabilized[0], raw[0])
        self.assertEqual(stabilized[-1], raw[-1])
        self.assertLess(
            max(stabilized_intervals) - min(stabilized_intervals),
            max(raw_intervals) - min(raw_intervals),
        )

    def test_group_fingering_maps_an_e_chord_without_string_collisions(self) -> None:
        positions, confidence = assign_fingering_group([40, 47, 52, 56, 59, 64])

        self.assertEqual(
            positions,
            [(6, 0), (5, 2), (4, 2), (3, 1), (2, 0), (1, 0)],
        )
        self.assertEqual(len({string for string, _fret in positions}), 6)
        self.assertGreaterEqual(confidence, 0.9)

    def test_harmonic_pitch_maps_to_the_touch_node(self) -> None:
        self.assertEqual(natural_harmonic_positions(59), [(6, 7)])
        self.assertEqual(
            select_harmonic_position(59, (2, 0)),
            (6, 7, "natural", None, 0.92),
        )
        self.assertEqual(
            select_harmonic_position(66, (4, 4)),
            (4, 4, "artificial", 16, 0.68),
        )

    def test_collapses_a_coincident_harmonic_overtone_stack(self) -> None:
        events = [
            NoteEvent(0.023, 1.38, 59, 76, 2, 0, "pick", 0.88),
            NoteEvent(0.012, 1.34, 71, 61, 3, 16, "pick", 0.81),
            NoteEvent(0.012, 1.32, 83, 66, 1, 19, "pick", 0.84),
        ]

        collapsed, forced = collapse_harmonic_stacks(events)

        self.assertEqual([event.pitch for event in collapsed], [59])
        self.assertEqual(forced, {0})
        annotated = annotate_techniques(
            collapsed,
            [AudioFeature(0.0, 632, 0.0, 0.03)],
            forced,
        )
        self.assertEqual(annotated[0].technique, "harmonic")
        self.assertEqual((annotated[0].string, annotated[0].fret), (6, 7))
        self.assertIn(
            "harmonic-overtone-stack",
            annotated[0].technique_evidence,
        )

    def test_drops_a_short_weak_fragment_before_the_same_retriggered_pitch(self) -> None:
        weak_fragment = NoteEvent(
            1.9621,
            2.0911,
            41,
            36,
            6,
            1,
            "pick",
            0.7,
        )
        main_event = NoteEvent(
            2.1027,
            2.7064,
            41,
            50,
            6,
            1,
            "pick",
            0.77,
        )

        collapsed = collapse_note_fragments([weak_fragment, main_event])

        self.assertEqual(len(collapsed), 1)
        self.assertEqual(collapsed[0].start, 2.1027)
        self.assertEqual(collapsed[0].end, 2.7064)

    def test_keeps_two_deliberate_same_pitch_attacks(self) -> None:
        first = NoteEvent(1.0, 1.14, 53, 70, 4, 3, "pick", 0.82)
        second = NoteEvent(1.15, 1.38, 53, 76, 4, 3, "pick", 0.86)

        collapsed = collapse_note_fragments([first, second])

        self.assertEqual(len(collapsed), 2)

    def test_single_pitch_does_not_fabricate_a_chord(self) -> None:
        notes = [
            NoteEvent(0.0, 1.0, 59, 76, 6, 7, "harmonic", 0.88),
        ]

        chords = infer_chords(notes, 120)

        self.assertEqual(chords[0]["chord"], "N.C.")

    def test_acoustic_features_distinguish_common_techniques(self) -> None:
        previous = note(0.0, 0.35, 2)
        cases = {
            "hammer-on": (
                note(0.36, 0.7, 4),
                AudioFeature(0.15, 1400, 0.05, 0.03),
            ),
            "pull-off": (
                note(0.36, 0.7, 0),
                AudioFeature(0.14, 1300, 0.04, 0.02),
            ),
            "slide": (
                note(0.36, 0.8, 7),
                AudioFeature(0.12, 1500, 0.06, 0.03),
            ),
            "harmonic": (
                note(0.5, 1.1, 12, velocity=70),
                AudioFeature(0.4, 2600, 0.05, 0.04),
            ),
            "slap": (
                note(0.5, 0.64, 3, velocity=110),
                AudioFeature(0.9, 3200, 0.22, 0.12),
            ),
            "body-tap": (
                note(0.5, 0.6, 3, velocity=82),
                AudioFeature(0.95, 1900, 0.35, 0.18),
            ),
            "tremolo": (
                note(0.12, 0.25, 3),
                AudioFeature(0.8, 1600, 0.08, 0.04),
            ),
            "pick": (
                note(0.6, 1.0, 5),
                AudioFeature(0.8, 1800, 0.08, 0.04),
            ),
        }
        for expected, (current, feature) in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(
                    infer_technique(previous, current, feature),
                    expected,
                )

    def test_candidates_include_ranked_confidence_and_evidence(self) -> None:
        previous = note(0.0, 0.35, 2)
        current = note(0.36, 0.7, 4)
        candidates = infer_technique_candidates(
            previous,
            current,
            AudioFeature(0.15, 1400, 0.05, 0.03),
        )

        self.assertEqual(candidates[0].technique, "hammer-on")
        self.assertGreater(candidates[0].confidence, candidates[1].confidence)
        self.assertIn("same-string", candidates[0].evidence)
        self.assertLessEqual(len(candidates), 3)

    def test_annotation_links_relational_technique_to_previous_note(self) -> None:
        events = [
            note(0.0, 0.35, 2),
            note(0.36, 0.7, 4),
        ]
        features = [
            AudioFeature(0.8, 1500, 0.06, 0.03),
            AudioFeature(0.15, 1400, 0.05, 0.03),
        ]

        annotated = annotate_techniques(events, features)

        self.assertEqual(annotated[1].technique, "hammer-on")
        self.assertEqual(annotated[1].related_note_index, 0)
        self.assertEqual(annotated[1].technique_source, "transition-heuristic-v2")
        self.assertGreater(annotated[1].technique_confidence, 0.8)
        self.assertEqual(
            annotated[1].technique_candidates[0].technique,
            "hammer-on",
        )

    def test_legato_evidence_can_remap_a_note_to_the_previous_string(self) -> None:
        previous = note(0.0, 0.35, 2, string=3)
        current = NoteEvent(
            start=0.36,
            end=0.7,
            pitch=59,
            velocity=70,
            string=2,
            fret=0,
            technique="pick",
            confidence=0.82,
            position_confidence=0.6,
        )

        annotated = annotate_techniques(
            [previous, current],
            [
                AudioFeature(0.8, 1500, 0.06, 0.03),
                AudioFeature(0.15, 1400, 0.05, 0.03),
            ],
        )

        self.assertEqual(annotated[1].technique, "hammer-on")
        self.assertEqual((annotated[1].string, annotated[1].fret), (3, 4))
        self.assertEqual(annotated[1].related_note_index, 0)
        self.assertEqual(annotated[1].position_source, "legato-optimizer-v1")

    def test_trained_backend_preserves_its_string_and_fret_positions(self) -> None:
        previous = note(0.0, 0.35, 2, string=3)
        current = NoteEvent(
            start=0.36,
            end=0.7,
            pitch=59,
            velocity=70,
            string=2,
            fret=0,
            technique="pick",
            confidence=0.82,
            position_confidence=0.8,
            position_source="string-fret-model-v1",
        )

        annotated = annotate_techniques(
            [previous, current],
            [
                AudioFeature(0.8, 1500, 0.06, 0.03),
                AudioFeature(0.15, 1400, 0.05, 0.03),
            ],
            preserve_positions=True,
        )

        self.assertEqual((annotated[1].string, annotated[1].fret), (2, 0))
        self.assertEqual(annotated[1].position_source, "string-fret-model-v1")

    def test_transcription_router_uses_a_configured_trained_checkpoint(self) -> None:
        expected_notes = [note(0.0, 0.2, 2)]
        expected_model = {
            "name": "stringtrace-string-fret-net",
            "version": "test",
            "kind": "trained",
        }
        with patch(
            "worker.transcribe_string_model",
            return_value=(expected_notes, expected_model),
        ):
            notes, dropped, model_info = transcribe_guitar(
                Path("guitar.wav"),
                Path("model.pt"),
            )

        self.assertEqual(notes, expected_notes)
        self.assertEqual(dropped, 0)
        self.assertEqual(model_info, expected_model)

    def test_transcription_router_supports_the_hybrid_candidate(self) -> None:
        expected_notes = [note(0.0, 0.2, 2)]
        expected_model = {
            "name": "basic-pitch-stringtrace-hybrid",
            "version": "test",
            "kind": "hybrid",
        }
        with patch(
            "worker.transcribe_hybrid_model",
            return_value=(expected_notes, 2, expected_model),
        ):
            notes, dropped, model_info = transcribe_guitar(
                Path("guitar.wav"),
                Path("model.pt"),
                model_mode="hybrid",
            )

        self.assertEqual(notes, expected_notes)
        self.assertEqual(dropped, 2)
        self.assertEqual(model_info, expected_model)

    def test_very_weak_large_transition_prefers_slide(self) -> None:
        previous = NoteEvent(0.0, 0.35, 53, 72, 4, 3, "pick", 0.84)
        current = NoteEvent(
            0.37,
            0.72,
            60,
            68,
            3,
            5,
            "pick",
            0.81,
            position_confidence=0.6,
        )

        annotated = annotate_techniques(
            [previous, current],
            [
                AudioFeature(0.8, 1500, 0.06, 0.03),
                AudioFeature(0.18, 1500, 0.06, 0.03),
            ],
        )

        self.assertEqual(annotated[1].technique, "slide")
        self.assertEqual((annotated[1].string, annotated[1].fret), (4, 10))
        self.assertEqual(annotated[1].related_note_index, 0)

    def test_long_onset_gap_is_not_labeled_as_legato(self) -> None:
        previous = NoteEvent(0.0, 0.9, 53, 72, 4, 3, "pick", 0.84)
        current = NoteEvent(0.95, 1.2, 60, 68, 4, 10, "pick", 0.81)

        candidates = infer_technique_candidates(
            previous,
            current,
            AudioFeature(0.18, 1500, 0.06, 0.03),
        )

        self.assertEqual(candidates[0].technique, "pick")
        self.assertFalse(
            any(
                candidate.technique in {"hammer-on", "pull-off", "slide"}
                for candidate in candidates
            )
        )

    def test_tremolo_uses_transition_source_without_a_slur_relation(self) -> None:
        events = [
            note(0.0, 0.2, 2),
            note(0.12, 0.25, 3),
        ]
        features = [
            AudioFeature(0.8, 1500, 0.06, 0.03),
            AudioFeature(0.8, 1600, 0.08, 0.04),
        ]

        annotated = annotate_techniques(events, features)

        self.assertEqual(annotated[1].technique, "tremolo")
        self.assertEqual(annotated[1].technique_source, "transition-heuristic-v2")
        self.assertIsNone(annotated[1].related_note_index)

    def test_annotation_remaps_natural_and_artificial_harmonics(self) -> None:
        natural = NoteEvent(
            start=0.5,
            end=1.1,
            pitch=59,
            velocity=70,
            string=2,
            fret=0,
            technique="pick",
            confidence=0.84,
            position_confidence=0.55,
        )
        artificial = NoteEvent(
            start=1.5,
            end=2.1,
            pitch=66,
            velocity=70,
            string=4,
            fret=4,
            technique="pick",
            confidence=0.82,
            position_confidence=0.55,
        )
        features = [
            AudioFeature(0.4, 2600, 0.05, 0.04),
            AudioFeature(0.4, 2800, 0.05, 0.04),
        ]

        annotated = annotate_techniques([natural, artificial], features)

        self.assertEqual(
            (
                annotated[0].technique,
                annotated[0].string,
                annotated[0].fret,
                annotated[0].harmonic_type,
            ),
            ("harmonic", 6, 7, "natural"),
        )
        self.assertEqual(
            (
                annotated[1].technique,
                annotated[1].string,
                annotated[1].fret,
                annotated[1].harmonic_type,
                annotated[1].harmonic_touch_fret,
            ),
            ("harmonic", 4, 4, "artificial", 16),
        )
        self.assertEqual(annotated[1].position_source, "harmonic-optimizer-v1")


if __name__ == "__main__":
    unittest.main()

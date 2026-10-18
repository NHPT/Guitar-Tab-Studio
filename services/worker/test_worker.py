import hashlib
import tempfile
import unittest
import zipfile
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from worker import (
    AudioFeature,
    NoteEvent,
    acquire_url,
    align_lyrics_to_melody,
    annotate_techniques,
    assign_fingering_group,
    collapse_harmonic_stacks,
    collapse_note_fragments,
    complete_vocal_melody_for_lyrics,
    emit_muscriptor_progress,
    fuse_transcription_candidates,
    infer_chords,
    infer_technique,
    infer_technique_candidates,
    main,
    natural_harmonic_positions,
    normalize_audio,
    parse_timed_lyric_text,
    python_module_command,
    prepare_mr_mt3_model,
    prepare_noise2fret_model,
    prepare_transcription_provider,
    prepare_yourmt3_model,
    refine_positions_with_noise2fret,
    regularize_beat_times,
    run,
    run_with_percent_progress,
    select_harmonic_position,
    select_quarter_note_grid,
    separate_audio,
    separate_guitar_parts,
    stabilize_beat_times,
    suppress_guitar_stem_leakage,
    suppress_vocal_leakage_in_guitar_part,
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


class RuntimeGuardTest(unittest.TestCase):
    def test_packaged_module_commands_reenter_the_frozen_worker(self) -> None:
        with (
            patch("worker.sys.frozen", True, create=True),
            patch("worker.sys.executable", "guitar-tab-worker.exe"),
        ):
            self.assertEqual(
                python_module_command("demucs.separate"),
                ["guitar-tab-worker.exe", "--internal-demucs"],
            )
            self.assertEqual(
                python_module_command("yt_dlp"),
                ["guitar-tab-worker.exe", "--internal-yt-dlp"],
            )

    def test_downloads_and_verifies_mr_mt3_checkpoint_atomically(
        self,
    ) -> None:
        root_parent = Path(".cache/tmp")
        root_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root_parent) as directory:
            root = Path(directory)
            model = b"mr-mt3-weights"
            source = root / "fixture.pth"
            source.write_bytes(model)
            with (
                patch.dict(
                    "os.environ",
                    {"GTS_MODEL_PROVIDER_ROOT": str(root / "providers")},
                ),
                patch(
                    "worker.MR_MT3_MODEL_URL",
                    source.resolve().as_uri(),
                ),
                patch(
                    "worker.MR_MT3_MODEL_SHA256",
                    hashlib.sha256(model).hexdigest(),
                ),
                patch("worker.MR_MT3_MODEL_BYTES", len(model)),
            ):
                path = prepare_mr_mt3_model()

            self.assertEqual(path.read_bytes(), model)
            self.assertIn(
                hashlib.sha256(model).hexdigest(),
                path.with_name("manifest.json").read_text(),
            )

    def test_downloads_and_verifies_noise2fret_checkpoint_atomically(
        self,
    ) -> None:
        root_parent = Path(".cache/tmp")
        root_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root_parent) as directory:
            root = Path(directory)
            model = b"noise2fret-weights"
            archive = root / "fixture.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr(
                    "GOAT_N2F_Laux/my_checkpoints/best_checkpoint.pth",
                    model,
                )
            with (
                patch.dict(
                    "os.environ",
                    {"GTS_MODEL_PROVIDER_ROOT": str(root / "providers")},
                ),
                patch(
                    "worker.NOISE2FRET_ARCHIVE_URL",
                    archive.resolve().as_uri(),
                ),
                patch(
                    "worker.NOISE2FRET_ARCHIVE_SHA256",
                    hashlib.sha256(archive.read_bytes()).hexdigest(),
                ),
                patch(
                    "worker.NOISE2FRET_ARCHIVE_BYTES",
                    archive.stat().st_size,
                ),
                patch(
                    "worker.NOISE2FRET_MODEL_SHA256",
                    hashlib.sha256(model).hexdigest(),
                ),
            ):
                path = prepare_noise2fret_model()

            self.assertEqual(path.read_bytes(), model)
            self.assertIn(
                hashlib.sha256(model).hexdigest(),
                path.with_name("manifest.json").read_text(),
            )

    def test_prepares_an_existing_local_muscriptor_weight(self) -> None:
        root_parent = Path(".cache/tmp")
        root_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root_parent) as directory:
            weight = Path(directory) / "model.safetensors"
            weight.write_bytes(b"weights")
            with (
                patch("worker.prepare_noise2fret_model"),
                patch("sys.stdout", new_callable=StringIO) as output,
            ):
                prepare_transcription_provider(
                    "muscriptor",
                    str(weight),
                    "small",
                )
            self.assertIn('"type": "gts-model-ready"', output.getvalue())
            self.assertIn('"model": "local"', output.getvalue())

    def test_prepares_mr_mt3_only_when_cuda_can_accelerate_it(self) -> None:
        root_parent = Path(".cache/tmp")
        root_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root_parent) as directory:
            weight = Path(directory) / "model.safetensors"
            weight.write_bytes(b"weights")
            with (
                patch("worker.prepare_noise2fret_model"),
                patch("worker.prepare_mr_mt3_model") as prepare_mr_mt3,
                patch(
                    "worker.hardware_probe",
                    return_value={
                        "accelerator": "mps",
                        "providers": {"mr-mt3": True},
                    },
                ),
                patch("sys.stdout", new_callable=StringIO),
            ):
                prepare_transcription_provider(
                    "muscriptor",
                    str(weight),
                    "medium",
                )
            prepare_mr_mt3.assert_not_called()

            with (
                patch("worker.prepare_noise2fret_model"),
                patch("worker.prepare_mr_mt3_model") as prepare_mr_mt3,
                patch(
                    "worker.hardware_probe",
                    return_value={
                        "accelerator": "cuda",
                        "providers": {"mr-mt3": True},
                    },
                ),
                patch("sys.stdout", new_callable=StringIO),
            ):
                prepare_transcription_provider(
                    "muscriptor",
                    str(weight),
                    "medium",
                )
            prepare_mr_mt3.assert_called_once_with()

            with (
                patch("worker.prepare_noise2fret_model"),
                patch("worker.prepare_mr_mt3_model") as prepare_mr_mt3,
                patch(
                    "worker.hardware_probe",
                    return_value={
                        "accelerator": "cuda",
                        "providers": {"mr-mt3": False},
                    },
                ),
                patch("sys.stdout", new_callable=StringIO),
            ):
                prepare_transcription_provider(
                    "muscriptor",
                    str(weight),
                    "large",
                )
            prepare_mr_mt3.assert_not_called()

    def test_prepares_remote_muscriptor_weights_before_activation(self) -> None:
        with (
            patch("huggingface_hub.get_token", return_value="hf_test"),
            patch("huggingface_hub.hf_hub_download") as download,
            patch("worker.prepare_noise2fret_model"),
            patch("sys.stdout", new_callable=StringIO) as output,
        ):
            prepare_transcription_provider("muscriptor", "small")

        self.assertEqual(
            [call.kwargs["filename"] for call in download.call_args_list],
            ["model.safetensors", "config.json"],
        )
        self.assertIn('"type": "gts-model-progress"', output.getvalue())
        self.assertIn('"type": "gts-model-ready"', output.getvalue())

    def test_prepares_multitrack_research_models_before_activation(self) -> None:
        with (
            patch("worker.prepare_yourmt3_model") as prepare_yourmt3,
            patch("worker.prepare_noise2fret_model") as prepare_noise2fret,
            patch("sys.stdout", new_callable=StringIO) as output,
        ):
            prepare_transcription_provider(
                "multitrack-research",
                None,
            )

        prepare_yourmt3.assert_called_once_with()
        prepare_noise2fret.assert_called_once_with()
        self.assertIn('"type": "gts-model-ready"', output.getvalue())

    def test_logs_in_in_browser_then_continues_model_download(self) -> None:
        device_info = {
            "device_code": "private",
            "user_code": "ABCD-EFGH",
            "verification_uri": "https://huggingface.co/device",
            "verification_uri_complete": (
                "https://huggingface.co/device?user_code=ABCD-EFGH"
            ),
            "interval": 1,
            "expires_in": 900,
        }
        token_response = {
            "access_token": "hf_oauth_test",
            "refresh_token": "refresh",
            "expires_in": 3600,
        }

        def complete_login(
            _device_info: dict[str, object],
            *,
            on_pending: object,
        ) -> dict[str, object]:
            assert callable(on_pending)
            on_pending()
            return token_response

        with (
            patch("huggingface_hub.get_token", return_value=None),
            patch("huggingface_hub.hf_hub_download"),
            patch(
                "huggingface_hub.utils._oauth_device.request_device_code",
                return_value=device_info,
            ),
            patch(
                "huggingface_hub.utils._oauth_device.poll_device_token",
                side_effect=complete_login,
            ) as poll,
            patch("huggingface_hub._login._save_oauth_token") as save,
            patch("worker.prepare_noise2fret_model"),
            patch("sys.stdout", new_callable=StringIO) as output,
        ):
            prepare_transcription_provider("muscriptor", "small")

        poll.assert_called_once()
        save.assert_called_once_with(token_response)
        self.assertEqual(
            output.getvalue().count('"type": "gts-auth-required"'),
            2,
        )
        self.assertIn("ABCD-EFGH", output.getvalue())
        self.assertIn('"type": "gts-model-ready"', output.getvalue())

    def test_acquire_only_stops_before_audio_analysis(self) -> None:
        root_parent = Path(".cache/tmp")
        root_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root_parent) as directory:
            root = Path(directory)
            source = root / "source.ncm"
            source.write_bytes(b"protected-media")
            with (
                patch(
                    "worker.acquire_url",
                    return_value=(source, {"title": "fixture"}),
                ),
                patch("worker.normalize_audio") as normalize,
                patch(
                    "sys.argv",
                    [
                        "guitar-tab-worker",
                        "--url",
                        "https://music.example/song/1",
                        "--output",
                        str(root),
                        "--acquire-only",
                    ],
                ),
            ):
                main()

            normalize.assert_not_called()

    def test_spatially_distinct_guitars_become_independent_tracks(self) -> None:
        import numpy as np
        import soundfile as sf

        root_parent = Path(".cache/tmp")
        root_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root_parent) as directory:
            root = Path(directory)
            sample_rate = 22050
            time = np.arange(sample_rate * 5) / sample_rate
            low = np.sin(2 * np.pi * 220 * time) * 0.42
            high = np.sin(2 * np.pi * 660 * time) * 0.36
            stereo = np.stack(
                [low + high * 0.12, low * 0.12 + high],
                axis=1,
            )
            source = root / "guitar.wav"
            sf.write(source, stereo, sample_rate)

            parts = separate_guitar_parts(source, root)

            self.assertEqual([part["id"] for part in parts], ["guitar-1", "guitar-2"])
            self.assertTrue(all(Path(part["path"]).is_file() for part in parts))
            self.assertTrue(
                all(
                    part["separation_method"]
                    == "fastica-stereo-unmix-v1"
                    for part in parts
                )
            )
            first, _ = sf.read(parts[0]["path"], always_2d=True)
            second, _ = sf.read(parts[1]["path"], always_2d=True)
            first_mono = np.mean(first, axis=1)
            second_mono = np.mean(second, axis=1)
            self.assertLess(
                abs(float(np.corrcoef(first_mono, second_mono)[0, 1])),
                0.05,
            )
            reconstruction = first + second
            self.assertGreater(
                10 * np.log10(
                    np.sum(np.square(stereo)) /
                    np.sum(np.square(stereo - reconstruction))
                ),
                50,
            )

    def test_guitar_leakage_suppression_removes_coherent_reference(self) -> None:
        import numpy as np
        import soundfile as sf

        root_parent = Path(".cache/tmp")
        root_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root_parent) as directory:
            root = Path(directory)
            sample_rate = 22050
            time = np.arange(sample_rate * 5) / sample_rate
            left_guitar = np.sin(2 * np.pi * 220 * time) * 0.38
            right_guitar = np.sin(2 * np.pi * 660 * time) * 0.34
            leakage = np.sin(2 * np.pi * 440 * time) * 0.24
            guitar = np.stack(
                [left_guitar + leakage, right_guitar + leakage],
                axis=1,
            )
            reference = np.stack([leakage, leakage], axis=1)
            guitar_path = root / "guitar.wav"
            reference_path = root / "vocals.wav"
            sf.write(guitar_path, guitar, sample_rate)
            sf.write(reference_path, reference, sample_rate)

            cleaned_path, report = suppress_guitar_stem_leakage(
                guitar_path,
                {"vocals": reference_path},
                root,
            )
            cleaned, _ = sf.read(cleaned_path, always_2d=True)

            before_projection = abs(
                float(np.dot(guitar[:, 0], leakage) / np.dot(leakage, leakage))
            )
            after_projection = abs(
                float(np.dot(cleaned[:, 0], leakage) / np.dot(leakage, leakage))
            )
            self.assertLess(after_projection, before_projection * 0.35)
            self.assertGreater(
                float(np.corrcoef(cleaned[:, 0], left_guitar)[0, 1]),
                0.97,
            )
            self.assertGreater(report["retained_rms_ratio"], 0.72)

    def test_post_separation_mask_removes_vocals_without_erasing_guitar(
        self,
    ) -> None:
        import numpy as np
        import soundfile as sf

        root_parent = Path(".cache/tmp")
        root_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root_parent) as directory:
            root = Path(directory)
            sample_rate = 22050
            time = np.arange(sample_rate * 5) / sample_rate
            guitar = (
                np.sin(2 * np.pi * 220 * time) * 0.32
                + np.sin(2 * np.pi * 660 * time) * 0.18
            )
            vocals = (
                np.sin(2 * np.pi * 440 * time) * 0.24
                + np.sin(2 * np.pi * 880 * time) * 0.1
            )
            contaminated = guitar + vocals * 0.35
            guitar_path = root / "guitar-1.wav"
            vocals_path = root / "vocals.wav"
            sf.write(
                guitar_path,
                np.stack([contaminated, contaminated], axis=1),
                sample_rate,
            )
            sf.write(
                vocals_path,
                np.stack([vocals, vocals], axis=1),
                sample_rate,
            )

            cleaned_path, report = suppress_vocal_leakage_in_guitar_part(
                guitar_path,
                vocals_path,
                root,
            )
            cleaned, _ = sf.read(cleaned_path, always_2d=True)
            before_projection = abs(
                float(np.dot(contaminated, vocals) / np.dot(vocals, vocals))
            )
            after_projection = abs(
                float(np.dot(cleaned[:, 0], vocals) / np.dot(vocals, vocals))
            )

            self.assertLess(after_projection, before_projection * 0.45)
            self.assertGreater(
                float(np.corrcoef(cleaned[:, 0], guitar)[0, 1]),
                0.96,
            )
            self.assertGreater(report["vocal_similarity_reduction"], 0.25)
            self.assertGreater(report["retained_rms_ratio"], 0.8)

    def test_url_acquisition_enables_node_for_youtube_extraction(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def create_download(_arguments: list[str]) -> None:
                (root / "source.m4a").write_bytes(b"audio")

            with (
                patch("worker.ffmpeg_command", return_value="/bin/ffmpeg"),
                patch("worker.shutil.which", return_value="/usr/bin/node"),
                patch("worker.run", side_effect=create_download) as command,
            ):
                source, metadata = acquire_url(
                    "https://www.youtube.com/watch?v=example",
                    root,
                )

            arguments = command.call_args.args[0]
            self.assertEqual(source, root / "source.m4a")
            self.assertEqual(metadata["title"], "source")
            self.assertIn("--no-config", arguments)
            self.assertIn("--no-cache-dir", arguments)
            self.assertEqual(
                arguments[arguments.index("--format") + 1],
                "bestaudio[acodec!=none]/best[acodec!=none]/bestaudio/best",
            )
            self.assertEqual(
                arguments[
                    arguments.index("--js-runtimes") :
                    arguments.index("--js-runtimes") + 2
                ],
                ["--js-runtimes", "node"],
            )

    def test_normalize_rejects_insufficient_space_and_removes_partial_output(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.m4a"
            normalized = root / "normalized.wav"
            source.write_bytes(b"valid-source")
            normalized.write_bytes(b"partial")

            with (
                patch("worker.ffmpeg_command", return_value="/bin/ffmpeg"),
                patch(
                    "worker.shutil.disk_usage",
                    return_value=SimpleNamespace(free=512 * 1024 * 1024),
                ),
                patch("worker.run") as command,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "可用磁盘空间不足",
                ):
                    normalize_audio(source, root)

            command.assert_not_called()
            self.assertFalse(normalized.exists())

    def test_external_command_maps_enospc_exit_code(self) -> None:
        with patch(
            "worker.subprocess.run",
            return_value=SimpleNamespace(returncode=228),
        ):
            with self.assertRaisesRegex(RuntimeError, "可用磁盘空间不足"):
                run(["ffmpeg", "normalized.wav"])

    def test_demucs_maps_enospc_traceback_to_a_storage_error(self) -> None:
        process = SimpleNamespace(
            stderr=StringIO(
                "OSError: [Errno 28] No space left on device\n"
            ),
            wait=lambda: 1,
        )
        with (
            patch("worker.subprocess.Popen", return_value=process),
            patch("sys.stderr"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "可用磁盘空间不足",
            ):
                run_with_percent_progress(
                    ["python", "-m", "demucs.separate"],
                    start=22,
                    end=50,
                    status="separating",
                    stage_label="正在分离人声与乐器",
                )

    def test_packaged_demucs_reports_a_missing_runtime_component(self) -> None:
        process = SimpleNamespace(
            stderr=StringIO(
                "ModuleNotFoundError: No module named 'demucs.audio'\n"
            ),
            wait=lambda: 1,
        )
        with (
            patch("worker.subprocess.Popen", return_value=process),
            patch("sys.stderr"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "音轨分离组件缺失或损坏",
            ):
                run_with_percent_progress(
                    ["guitar-tab-worker.exe", "--internal-demucs"],
                    start=22,
                    end=50,
                    status="separating",
                    stage_label="正在分离人声与乐器",
                )

    def test_separation_rechecks_space_after_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "normalized.wav"
            source.write_bytes(b"normalized")
            with (
                patch(
                    "worker.shutil.disk_usage",
                    return_value=SimpleNamespace(free=1024 * 1024 * 1024),
                ),
                patch("worker.run_with_percent_progress") as command,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "无法安全完成六轨音频分离",
                ):
                    separate_audio(source, root)

            command.assert_not_called()

    def test_external_command_surfaces_sanitized_extractor_error(self) -> None:
        with patch(
            "worker.subprocess.run",
            return_value=SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="WARNING: retrying\nERROR: Sign in to confirm you are not a bot\n",
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Sign in to confirm you are not a bot",
            ):
                run(["yt-dlp", "https://example.invalid/video"])


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

    def test_beat_regularizer_preserves_tempo_motion_while_reducing_jitter(
        self,
    ) -> None:
        import numpy as np

        intervals = np.linspace(0.71, 0.76, 80)
        intervals[35:39] += np.asarray([0.04, 0.06, -0.05, -0.03])
        raw = np.concatenate(([0.5], 0.5 + np.cumsum(intervals)))

        regularized = regularize_beat_times(raw)
        smoothed_intervals = np.diff(regularized)

        self.assertAlmostEqual(regularized[0], raw[0], places=8)
        self.assertAlmostEqual(regularized[-1], raw[-1], places=8)
        self.assertLess(
            float(np.max(smoothed_intervals) - np.min(smoothed_intervals)),
            float(np.max(intervals) - np.min(intervals)),
        )
        self.assertLess(
            float(np.max(np.abs(np.asarray(regularized) - raw))),
            0.05,
        )
        self.assertGreater(
            float(
                np.mean(smoothed_intervals[-10:])
                - np.mean(smoothed_intervals[:10])
            ),
            0.025,
        )

    def test_tempo_octave_uses_the_slower_grid_only_with_accent_evidence(
        self,
    ) -> None:
        import numpy as np

        fast_frames = list(range(10, 330, 10))
        slow_frames = fast_frames[::2]
        accented = np.zeros(340)
        accented[fast_frames] = 7
        accented[slow_frames[::2]] = 10
        accented[slow_frames[1::2]] = 4

        tempo, frames = select_quarter_note_grid(
            accented,
            161.5,
            fast_frames,
            80.75,
            slow_frames,
        )

        self.assertEqual(tempo, 80.75)
        self.assertEqual(frames, slow_frames)

        flat = np.zeros(340)
        flat[fast_frames] = 7
        tempo, frames = select_quarter_note_grid(
            flat,
            161.5,
            fast_frames,
            80.75,
            slow_frames,
        )
        self.assertEqual(tempo, 161.5)
        self.assertEqual(frames, fast_frames)

    def test_lyrics_follow_nearby_vocal_note_attacks(self) -> None:
        aligned = align_lyrics_to_melody(
            [
                {"at": 1.0, "text": "take me home"},
                {"at": 3.0, "text": "country roads"},
            ],
            [
                {"start": 1.1},
                {"start": 1.6},
                {"start": 2.2},
                {"start": 3.1},
                {"start": 3.8},
            ],
        )

        self.assertEqual(
            aligned,
            [
                {"at": 1.1, "text": "take"},
                {"at": 1.6, "text": "me"},
                {"at": 2.2, "text": "home"},
                {"at": 3.1, "text": "country"},
                {"at": 3.8, "text": "roads"},
            ],
        )

    def test_parses_embedded_qq_music_lrc_without_credit_lines(self) -> None:
        self.assertEqual(
            parse_timed_lyric_text(
                "[00:01.20]作词：Example\n"
                "[00:04.50]Country roads\n"
                "[00:07.00]Take me home\n"
            ),
            [
                {"at": 4.5, "text": "Country roads"},
                {"at": 7.0, "text": "Take me home"},
            ],
        )

    def test_lyrics_receive_missing_numbered_note_anchors(self) -> None:
        melody = [
            {"start": 1.0, "end": 1.5, "pitch": 60, "confidence": 0.9},
            {"start": 2.0, "end": 2.4, "pitch": 62, "confidence": 0.8},
        ]
        lyrics = [
            {"at": 1.0, "text": "take"},
            {"at": 1.65, "text": "me"},
            {"at": 2.0, "text": "home"},
        ]

        completed = complete_vocal_melody_for_lyrics(melody, lyrics)

        self.assertEqual(len(completed), 3)
        inserted = next(note for note in completed if note["start"] == 1.65)
        self.assertEqual(inserted["pitch"], 60)
        self.assertLess(inserted["confidence"], melody[0]["confidence"])

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

    def test_transcription_router_supports_muscriptor_provider(self) -> None:
        expected_notes = [note(0.0, 0.2, 2)]
        muscriptor_model = {
            "name": "muscriptor",
            "version": "0.3.0",
            "kind": "trained",
        }
        with (
            patch(
                "worker.transcribe_muscriptor",
                return_value=(expected_notes, 1, muscriptor_model),
            ),
            patch(
                "worker.transcribe_basic_pitch",
                return_value=([], 0),
            ),
            patch(
                "worker.release_torch_accelerator_cache",
            ) as release_cache,
            patch("worker.emit_progress") as progress,
        ):
            notes, dropped, model_info = transcribe_guitar(
                Path("guitar.wav"),
                transcription_provider="muscriptor",
                transcription_model="medium",
            )

        self.assertEqual(notes, expected_notes)
        self.assertEqual(dropped, 1)
        self.assertEqual(
            model_info,
            {
                "name": "muscriptor-basic-pitch",
                "version": "muscriptor-0.3.0+basic-pitch-0.4",
                "kind": "trained",
            },
        )
        progress.assert_called_once_with(
            81,
            "transcribing",
            "正在运行 Basic Pitch 复核",
        )
        release_cache.assert_called_once_with()

    def test_muscriptor_chunk_progress_maps_into_transcription_range(self) -> None:
        with patch("worker.emit_progress") as progress:
            emit_muscriptor_progress(0, 2)
            emit_muscriptor_progress(1, 2)
            emit_muscriptor_progress(2, 2)
            emit_muscriptor_progress(3, 2)
            emit_muscriptor_progress(1, 0)

        self.assertEqual(
            [entry.args for entry in progress.call_args_list],
            [
                (
                    65.0,
                    "transcribing",
                    "正在识别吉他音符（0/2 个片段）",
                ),
                (
                    72.5,
                    "transcribing",
                    "正在识别吉他音符（1/2 个片段）",
                ),
                (
                    80.0,
                    "transcribing",
                    "正在识别吉他音符（2/2 个片段）",
                ),
                (
                    80.0,
                    "transcribing",
                    "正在识别吉他音符（2/2 个片段）",
                ),
            ],
        )

    def test_transcription_ensemble_merges_consensus_and_keeps_strong_residuals(
        self,
    ) -> None:
        primary = [note(1.0, 1.4, 2)]
        corroborating = note(1.03, 1.5, 2)
        corroborating.confidence = 0.9
        residual = note(2.0, 2.3, 4)
        residual.confidence = 0.9
        weak = note(3.0, 3.2, 5)
        weak.confidence = 0.7

        fused = fuse_transcription_candidates(
            primary,
            [corroborating, residual, weak],
        )

        self.assertEqual(len(fused), 2)
        self.assertEqual(fused[0].position_source, "multi-model-consensus-v1")
        self.assertAlmostEqual(fused[0].start, 1.009, places=4)
        self.assertEqual(
            fused[1].position_source,
            "basic-pitch-residual-v1",
        )

    def test_high_accuracy_pipeline_uses_mr_mt3_when_configured(self) -> None:
        primary = [note(0.0, 0.2, 2)]
        verifier = note(0.02, 0.22, 2)
        verifier.confidence = 0.9
        with (
            patch(
                "worker.transcribe_muscriptor",
                return_value=(
                    primary,
                    0,
                    {
                        "name": "muscriptor",
                        "version": "0.3.0",
                        "kind": "trained",
                    },
                ),
            ),
            patch(
                "worker.transcribe_basic_pitch",
                return_value=([], 0),
            ),
            patch(
                "worker.transcribe_mr_mt3",
                return_value=([verifier], 0),
            ),
            patch("worker.release_torch_accelerator_cache"),
            patch("worker.emit_progress") as progress,
        ):
            notes, dropped, model_info = transcribe_guitar(
                Path("guitar.wav"),
                transcription_provider="muscriptor",
                transcription_model="medium",
                verification_model=Path("mr-mt3.pth"),
            )

        self.assertEqual(len(notes), 1)
        self.assertEqual(dropped, 0)
        self.assertEqual(
            model_info["name"],
            "muscriptor-mr-mt3-basic-pitch",
        )
        self.assertIn("mr-mt3-0.2.0", model_info["version"])
        self.assertEqual(
            [entry.args for entry in progress.call_args_list],
            [
                (81, "transcribing", "正在运行 Basic Pitch 复核"),
                (81.5, "transcribing", "正在运行 MR-MT3 复核"),
            ],
        )

    def test_noise2fret_refines_only_pitch_consistent_positions(self) -> None:
        first = NoteEvent(0.1, 0.4, 48, 90, 6, 8, "pick", 0.82)
        second = NoteEvent(0.1, 0.4, 52, 90, 5, 7, "pick", 0.82)
        with patch(
            "stringtrace_ml.noise2fret.predict_position_priors",
            return_value={0: [[(5, 3, 48), (4, 2, 52)], []]},
        ):
            refined, changed = refine_positions_with_noise2fret(
                Path("guitar.wav"),
                [first, second],
                Path("noise2fret.pth"),
            )

        self.assertEqual(changed, 2)
        self.assertEqual(
            [(event.string, event.fret) for event in refined],
            [(5, 3), (4, 2)],
        )
        self.assertTrue(
            all(
                event.position_source == "noise2fret-consensus-v1"
                for event in refined
            )
        )

    def test_muscriptor_failure_falls_back_to_basic_pitch(self) -> None:
        expected_notes = [note(0.0, 0.2, 2)]
        with (
            patch(
                "worker.transcribe_muscriptor",
                side_effect=ModuleNotFoundError("muscriptor"),
            ),
            patch(
                "worker.transcribe_basic_pitch",
                return_value=(expected_notes, 2),
            ),
        ):
            notes, dropped, model_info = transcribe_guitar(
                Path("guitar.wav"),
                transcription_provider="muscriptor",
                transcription_model="small",
            )

        self.assertEqual(notes, expected_notes)
        self.assertEqual(dropped, 2)
        self.assertEqual(model_info["kind"], "baseline")
        self.assertEqual(
            model_info["fallback_reason"],
            "MuScriptor 运行组件未安装",
        )

    def test_multitrack_research_pipeline_uses_yourmt3(self) -> None:
        expected_notes = [note(0.0, 0.2, 2)]
        with (
            patch(
                "worker.transcribe_yourmt3",
                return_value=(expected_notes, 1),
            ),
            patch(
                "worker.transcribe_basic_pitch",
                return_value=([], 0),
            ),
        ):
            notes, dropped, model_info = transcribe_guitar(
                Path("guitar.wav"),
                transcription_provider="multitrack-research",
                transcription_model="yourmt3.ckpt",
            )

        self.assertEqual(notes, expected_notes)
        self.assertEqual(dropped, 1)
        self.assertEqual(
            model_info["name"],
            "yourmt3-basic-pitch",
        )
        self.assertEqual(model_info["kind"], "research")

    def test_transcription_router_supports_explicit_research_bundle(
        self,
    ) -> None:
        expected_notes = [note(0.0, 0.2, 2)]
        expected_model = {
            "name": "stringtrace-onset-position-research",
            "version": "test",
            "kind": "research",
        }
        with patch(
            "worker.transcribe_onset_position_research_model",
            return_value=(expected_notes, 3, expected_model),
        ):
            notes, dropped, model_info = transcribe_guitar(
                Path("guitar.wav"),
                Path("research-bundle.pt"),
                model_mode="onset-position-research",
            )

        self.assertEqual(notes, expected_notes)
        self.assertEqual(dropped, 3)
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

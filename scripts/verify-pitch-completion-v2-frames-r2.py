#!/usr/bin/env python3
"""Resource-bounded full model replay for the frozen frame audit."""
# ruff: noqa: E402
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import resource
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services/worker"))
from stringtrace_ml.download_guitar_techs_polyphonic import digests, write_json
from stringtrace_ml.pitch_completion_v2_model import V2Dataset, V2Model
from stringtrace_ml.pitch_completion_v2_sounding import verify_sounding_contract

BASE = ROOT / "apps/api/data/training"
RUN = BASE / "pitch-completion-v2-frames-v1"
SOURCE = BASE / "pitch-completion-v2-sounding-v1-r2"
PAIR = BASE / "pitch-completion-v2-fit-transfer-resource-r1"
ARMS = ("budget", "budget_p1")
AMENDMENT = ROOT / "pitch-completion-v2-frame-verification-r2-amendment.md"
ORIGINAL = ROOT / "scripts/verify-pitch-completion-v2-frames.py"


def load_original():
    spec = importlib.util.spec_from_file_location("frame_verifier_v1", ORIGINAL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return digests(path)["sha256"]


def freeze_r2():
    destination = RUN / "verification-r2-plan.json"
    if destination.exists():
        plan = read(destination)
        for path, expected in plan["artifact_sha256"].items():
            assert sha(ROOT / path) == expected, path
        return plan
    paths = [
        AMENDMENT,
        Path(__file__).resolve(),
        RUN / "plan.json",
        RUN / "summary.json",
        RUN / "inference.json",
        RUN / "budget.npz",
        RUN / "budget_p1.npz",
        RUN / "verification-interrupted.log",
        RUN / "verification.log",
        ORIGINAL,
    ]
    plan = {
        "kind": "fixed-weight-frame-replay-r2",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "reason": "split model replay from post-inference checks after two SIGKILL terminations",
        "feature_cache_mib": 64,
        "audio_cache_mib": 64,
        "algorithm_changed": False,
        "weights_changed": False,
        "scores_changed": False,
        "artifact_sha256": {str(path.relative_to(ROOT)): sha(path) for path in paths},
    }
    write_json(destination, plan)
    return plan


def main():
    assert not (RUN / "verification.json").exists(), "Preserve completed verification"
    original = load_original()
    plan = read(RUN / "plan.json")
    inference = read(RUN / "inference.json")
    r2 = freeze_r2()
    original.frozen(plan)
    for name, expected in inference["output_sha256"].items():
        assert sha(RUN / name) == expected
    _, recipe, all_tracks, examples, _ = verify_sounding_contract(SOURCE)
    allowed = {examples[i]["track_id"] for i in plan["ids"]}
    replay_dir = RUN / "independent-replay-r2"
    assert not replay_dir.exists()
    replay_dir.mkdir()
    dataset = V2Dataset(
        [track for track in all_tracks if track["track_id"] in allowed],
        examples,
        recipe,
        replay_dir,
        [],
    )
    dataset.cache_budget = r2["feature_cache_mib"] * 1024**2
    dataset.audio_budget = r2["audio_cache_mib"] * 1024**2
    times = dataset.times - recipe["pre_onset_seconds"]
    assert np.array_equal(times, np.asarray(read(RUN / "summary.json")["times"]))
    expected = {}
    for arm in ARMS:
        with np.load(RUN / f"{arm}.npz", allow_pickle=False) as source:
            assert np.array_equal(source["ids"], np.asarray(plan["ids"]))
            assert np.array_equal(source["times"], times)
            expected[arm] = {key: np.asarray(source[key]) for key in
                             ("pitch", "onset", "raw", "cardinality")}
    torch.set_num_threads(plan["threads"])
    torch.set_num_interop_threads(1)
    torch.manual_seed(plan["seed"])
    torch.use_deterministic_algorithms(True)
    models = {}
    for arm in ARMS:
        checkpoint = torch.load(PAIR / f"{arm}-recovery.pt", map_location="cpu", weights_only=True)
        assert checkpoint["epoch"] == 40 and checkpoint["optimizer_steps"] == 5120
        assert checkpoint["numeric_recipe"] == recipe and not checkpoint["production_admitted"]
        model = V2Model(recipe, reduced=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        models[arm] = model.eval()
    tracks = {track["track_id"]: track for track in all_tracks}
    replay_max = {f"{arm}/{key}": 0. for arm in ARMS
                  for key in ("pitch", "onset", "raw", "cardinality")}
    flips = {key: 0 for key in replay_max}
    complete_checked = 0
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(plan["ids"]), plan["batch_size"]):
            assert time.monotonic() - started < plan["maximum_seconds"]
            assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < plan["maximum_rss_gib"] * 1024**3
            assert shutil.disk_usage(RUN).free > plan["free_space_reserve_bytes"]
            ids = plan["ids"][start:start + plan["batch_size"]]
            batch = dataset.batch(ids, "cpu")
            for row, index in enumerate(ids):
                track = tracks[examples[index]["track_id"]]
                if track["label_policy"] != "complete":
                    continue
                labels, attacks, mask = original.manual_labels(
                    track, examples[index]["onset"], times)
                assert np.array_equal(labels, batch["dense_pitch"][row].numpy().astype(bool))
                assert np.array_equal(attacks, batch["dense_onset"][row].numpy().astype(bool))
                assert np.array_equal(mask, batch["pitch_mask"][row].numpy())
                assert np.array_equal(mask, batch["onset_mask"][row].numpy())
                complete_checked += 1
            for arm, model in models.items():
                output = model(batch["features"], batch["candidate_mask"])
                replay = {
                    "pitch": output["pitch"].sigmoid().numpy(),
                    "onset": output["onset"].sigmoid().numpy(),
                    "raw": output["completion"].sigmoid().numpy(),
                    "cardinality": output["cardinality"].softmax(-1).numpy(),
                }
                for key, actual in replay.items():
                    frozen = expected[arm][key][start:start + len(ids)]
                    assert np.isfinite(actual).all()
                    error = float(np.max(np.abs(actual - frozen)))
                    name = f"{arm}/{key}"
                    replay_max[name] = max(replay_max[name], error)
                    flips[name] += int(np.count_nonzero((actual >= .5) != (frozen >= .5)))
                    assert error <= plan["replay_tolerance"]
            if start % 80 == 0 or start + len(ids) == len(plan["ids"]):
                print(json.dumps({"replayed_groups": start + len(ids),
                                  "elapsed_seconds": round(time.monotonic() - started, 2)}),
                      flush=True)
    assert complete_checked == 1257
    verified_tracks = len(dataset.verified)
    original.frozen(plan)
    for path, expected_hash in r2["artifact_sha256"].items():
        assert sha(ROOT / path) == expected_hash
    result = {
        "passed": True,
        "verification_revision": "r2-resource-bounded-split",
        "full_model_replay_groups": len(plan["ids"]),
        "arm_records": len(plan["ids"]) * len(ARMS),
        "complete_full_frame_labels_checked": complete_checked,
        "replay_max_error": replay_max,
        "replay_threshold_flips": flips,
        "audio_tracks_verified": verified_tracks,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
        "summary_sha256": sha(RUN / "summary.json"),
        "plan_sha256": sha(RUN / "plan.json"),
        "r2_plan_sha256": sha(RUN / "verification-r2-plan.json"),
        "post_inference_checks": "delivery-verification.json",
    }
    write_json(RUN / "verification.json", result)
    diagnostic = replay_dir / "diagnostic-features"
    assert not list(diagnostic.iterdir())
    diagnostic.rmdir()
    replay_dir.rmdir()
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

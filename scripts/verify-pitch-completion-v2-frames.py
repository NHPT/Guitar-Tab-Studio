#!/usr/bin/env python3
"""Independent feature/inference replay, event labels, ranking and series checks."""
# ruff: noqa: E402
from collections import defaultdict
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
MUTED = BASE / "pitch-completion-v2-muted-v1"
ARMS = ("budget", "budget_p1")


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def frozen(plan):
    for path, expected in plan["artifact_sha256"].items():
        assert digests(ROOT / path)["sha256"] == expected, path


def outside(event, track):
    return any((v["start"] <= event["onset"] < v["end"]) if isinstance(v, dict)
               else v[0] <= event["onset"] < v[1] for v in track.get("excluded_ranges", []))


def manual_labels(track, onset, times):
    """Independent full per-frame pitch activity/onset labels and negative mask."""
    absolute = onset + times
    valid = (absolute >= 0) & (absolute < track["duration"])
    for item in track.get("excluded_ranges", []):
        start, end = (item["start"], item["end"]) if isinstance(item, dict) else item
        valid &= ~((absolute >= start) & (absolute < end))
    mask = np.repeat(valid[:, None], 49, axis=1)
    for start, end in track.get("unknown_string_intervals", []):
        mask[(absolute >= start) & (absolute < end)] = False
    labels, onset_labels = np.zeros_like(mask), np.zeros_like(mask)
    halfstep = (times[1] - times[0]) / 2 + 1e-8
    for e in track["events"]:
        p = e["sounding_pitch"]
        if p is None or not e.get("positive_label_valid", True) or outside(e, track) or not 40 <= p <= 88:
            continue
        labels[:, p - 40] |= (absolute >= e["onset"]) & (absolute < e["offset"]) & valid
        onset_labels[:, p - 40] |= (np.abs(absolute - e["onset"]) <= halfstep) & valid
    return labels, onset_labels, mask


def independently_score(pitch, attack, times):
    local = [j for j, t in enumerate(times) if 0 <= t <= .16]
    pre = [j for j, t in enumerate(times) if -.16 <= t <= -.02]
    early = [j for j, t in enumerate(times) if 0 <= t <= .08]
    on = [j for j, t in enumerate(times) if -.05 <= t <= .05]
    return {"local_mean": np.mean(pitch[local], axis=0),
            "nearest": pitch[min(range(len(times)), key=lambda j: abs(times[j]))],
            "attack_mean": np.mean(pitch[early], axis=0),
            "attack_joint": np.sqrt(np.mean(pitch[early], axis=0) * np.max(attack[on], axis=0)),
            "positive_rise": np.maximum(0, np.minimum(1, np.max(pitch[early], axis=0) -
                                                     np.mean(pitch[pre], axis=0)))}


def independent_rank(values, example, pitch):
    permitted = [p for p in range(40, 89) if p not in example["candidate_pitches"]
                 and (p == pitch or p not in example["target_pitches"])]
    return sorted(permitted, key=lambda p: (-float(values[p - 40]), p)).index(pitch) + 1


def main():
    assert not (RUN / "verification.json").exists(), "Preserve verification"
    plan, summary, inference = (read(RUN / name) for name in ("plan.json", "summary.json", "inference.json"))
    frozen(plan)
    for name, expected in {**summary["output_sha256"], **inference["output_sha256"]}.items():
        assert digests(RUN / name)["sha256"] == expected
    _, recipe, all_tracks, examples, _ = verify_sounding_contract(SOURCE)
    tracks = {t["track_id"]: t for t in all_tracks}
    allowed = {examples[i]["track_id"] for i in plan["ids"]}
    torch.set_num_threads(plan["threads"])
    torch.set_num_interop_threads(1)
    torch.manual_seed(plan["seed"])
    torch.use_deterministic_algorithms(True)
    replay_dir = RUN / "independent-replay"
    assert not replay_dir.exists()
    replay_dir.mkdir()
    dataset = V2Dataset([t for t in all_tracks if t["track_id"] in allowed], examples, recipe, replay_dir, [])
    times = dataset.times - recipe["pre_onset_seconds"]
    assert np.array_equal(times, np.asarray(summary["times"]))
    data = {a: np.load(RUN / f"{a}.npz", allow_pickle=False) for a in ARMS}
    models = {}
    for arm in ARMS:
        assert np.array_equal(data[arm]["ids"], np.asarray(plan["ids"]))
        assert np.array_equal(data[arm]["times"], times)
        checkpoint = torch.load(PAIR / f"{arm}-recovery.pt", map_location="cpu", weights_only=True)
        assert checkpoint["epoch"] == 40 and checkpoint["optimizer_steps"] == 5120
        assert checkpoint["numeric_recipe"] == recipe and not checkpoint["production_admitted"]
        model = V2Model(recipe, reduced=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        models[arm] = model.eval()
    started, replay_max, flips = time.monotonic(), defaultdict(float), defaultdict(int)
    complete_checked = 0
    with torch.inference_mode():
        for start in range(0, len(plan["ids"]), plan["batch_size"]):
            assert time.monotonic() - started < plan["maximum_seconds"]
            assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < plan["maximum_rss_gib"] * 1024**3
            assert shutil.disk_usage(RUN).free > plan["free_space_reserve_bytes"]
            ids = plan["ids"][start:start + plan["batch_size"]]
            batch = dataset.batch(ids, "cpu")
            for j, i in enumerate(ids):
                track = tracks[examples[i]["track_id"]]
                if track["label_policy"] == "complete":
                    labels, attacks, mask = manual_labels(track, examples[i]["onset"], times)
                    assert np.array_equal(labels, batch["dense_pitch"][j].numpy().astype(bool))
                    assert np.array_equal(attacks, batch["dense_onset"][j].numpy().astype(bool))
                    assert np.array_equal(mask, batch["pitch_mask"][j].numpy())
                    assert np.array_equal(mask, batch["onset_mask"][j].numpy())
                    complete_checked += 1
            for arm, model in models.items():
                out = model(batch["features"], batch["candidate_mask"])
                replay = {"pitch": out["pitch"].sigmoid(), "onset": out["onset"].sigmoid(),
                          "raw": out["completion"].sigmoid(), "cardinality": out["cardinality"].softmax(-1)}
                for key, value in replay.items():
                    expected = data[arm][key][start:start + len(ids)]
                    actual = value.numpy()
                    assert np.isfinite(actual).all()
                    error = float(np.abs(actual - expected).max())
                    replay_max[f"{arm}/{key}"] = max(replay_max[f"{arm}/{key}"], error)
                    flips[f"{arm}/{key}"] += int(((actual >= .5) != (expected >= .5)).sum())
                    assert error <= plan["replay_tolerance"]
            if start % 80 == 0:
                print(json.dumps({"replayed_groups": start + len(ids),
                                  "elapsed_seconds": round(time.monotonic() - started, 2)}), flush=True)
    assert complete_checked == 1257
    position = {i: j for j, i in enumerate(plan["ids"])}
    scores = {(a, i): independently_score(d["pitch"][j], d["onset"][j], times)
              for a, d in data.items() for i, j in position.items()}
    target_rows = rows(RUN / "targets.jsonl")
    assert len(target_rows) == 2 * (635 + 150)
    for r in target_rows:
        for policy, value in r["policies"].items():
            s = scores[(r["arm"], r["index"])][policy]
            assert abs(float(s[r["pitch"] - 40]) - value["score"]) < 1e-7
            assert independent_rank(s, examples[r["index"]], r["pitch"]) == value["rank"]
            assert bool(s[r["pitch"] - 40] >= .5) == value["above_threshold"]
    for subset, arms in summary["ranking"].items():
        for arm, techniques in arms.items():
            for tech, policies in techniques.items():
                selected = [r for r in target_rows if r["arm"] == arm and r["subset"] == subset
                            and (tech == "all" or r["technique"] == tech)]
                for policy, metric in policies.items():
                    assert metric["targets"] == len(selected)
                    assert metric["top1"] == sum(r["policies"][policy]["rank"] == 1 for r in selected)
                    assert abs(metric["mrr"] - np.mean([1 / r["policies"][policy]["rank"]
                                                       for r in selected])) < 1e-12
                    assert metric["above_threshold"] == sum(r["policies"][policy]["above_threshold"] for r in selected)
                    assert metric["top1_above_threshold"] == sum(r["policies"][policy]["rank"] == 1
                               and r["policies"][policy]["above_threshold"] for r in selected)
    acoustic = {a["index"]: a for a in rows(MUTED / "acoustic.jsonl")}
    fixed_pairs = {(p["arm"], p["index"]): p for p in rows(MUTED / "pairs.jsonl")}
    series = rows(RUN / "series.jsonl")
    assert len(series) == 300
    for s in series:
        i, arm = s["index"], s["arm"]
        p, c, j = s["pitch"] - 40, s["competitor"] - 40, position[i]
        example, a = examples[i], acoustic[i]
        assert s["competitor"] == fixed_pairs[(arm, i)]["competitor"]
        assert s["technique"] == a["technique"] and s["label_policy"] == a["label_policy"]
        other = [q for q in range(49) if q + 40 not in example["candidate_pitches"]
                 and q + 40 not in example["target_pitches"]]
        for key, tensor, at in (("target_pitch", "pitch", p), ("competitor_pitch", "pitch", c),
                                ("target_onset", "onset", p), ("competitor_onset", "onset", c)):
            assert np.array_equal(np.asarray(s[key]), data[arm][tensor][j, :, at])
        assert np.array_equal(s["max_other_pitch"], data[arm]["pitch"][j][:, other].max(1))
        assert s["frame_rank"] == [independent_rank(v, example, p + 40) for v in data[arm]["pitch"][j]]
        for name, alignment in s["aligned"].items():
            delta = 0 if name == "group" else a["reference_group_delta_seconds"]
            if name == "attack":
                assert a["temporal"]["attack_reliable"]
                delta += a["temporal"]["attack_proxy_seconds"]
            assert delta == alignment["delta_seconds"]
            values = {}
            for label, bounds in (("pre", [-.16, -.02]), ("early", [0, .08]), ("late", [.16, .32])):
                ids = [k for k, t in enumerate(times) if bounds[0] <= t - delta <= bounds[1]]
                assert ids == alignment["frames"][label]
                values[label] = data[arm]["pitch"][j, ids].mean(0)
                assert float(values[label][p]) == alignment["target"][label]
                assert float(values[label][c]) == alignment["competitor"][label]
            assert float(values["early"][p] - values["pre"][p]) == alignment["target_rise"]
            assert float(values["early"][c] - values["pre"][c]) == alignment["competitor_rise"]
            assert float(values["early"][p] - values["early"][c]) == alignment["early_margin"]
            assert independent_rank(values["early"], example, p + 40) == alignment["early_rank"]
    for arm, techniques in summary["temporal"].items():
        for tech, alignments in techniques.items():
            selected = [s for s in series if s["arm"] == arm and (tech == "all" or s["technique"] == tech)]
            for alignment, metric in alignments.items():
                values = [s["aligned"][alignment] for s in selected if alignment in s["aligned"]]
                assert metric["targets"] == len(values)
                for key in ("target_rise", "competitor_rise", "early_margin"):
                    v = [a[key] for a in values]
                    for stat, actual in (("median", np.median(v)), ("mean", np.mean(v)),
                                         ("p10", np.quantile(v, .1)), ("p90", np.quantile(v, .9))):
                        assert abs(metric[key][stat] - actual) < 1e-12
                for key in ("target", "competitor"):
                    for w in ("pre", "early", "late"):
                        v = [a[key][w] for a in values]
                        assert abs(metric[key][w]["median"] - np.median(v)) < 1e-12
                        assert abs(metric[key][w]["mean"] - np.mean(v)) < 1e-12
                assert metric["target_rises"] == sum(a["target_rise"] > 0 for a in values)
                assert metric["target_rise_exceeds_competitor"] == sum(
                    a["target_rise"] > a["competitor_rise"] for a in values)
                assert metric["early_beats_fixed_competitor"] == sum(a["early_margin"] > 0 for a in values)
                assert metric["early_top1"] == sum(a["early_rank"] == 1 for a in values)
    control_rows = rows(RUN / "controls.jsonl")
    assert len(control_rows) == 1257 * 2 * 5
    totals = defaultdict(lambda: dict(tp=0, fp=0, fn=0, tn=0))
    local = [j for j, t in enumerate(times) if 0 <= t <= .16]
    for r in control_rows:
        i, arm = r["index"], r["arm"]
        example, track = examples[i], tracks[examples[i]["track_id"]]
        labels, _, mask = manual_labels(track, example["onset"], times)
        active, allowed = labels[local].any(0), mask[local].all(0)
        allowed[np.asarray(example["candidate_pitches"], int) - 40] = False
        assigned = np.zeros(49, bool)
        assigned[np.asarray(example["target_pitches"], int) - 40] = True
        predicted = scores[(arm, i)][r["policy"]] >= .5
        for label, truth in (("activity", active), ("assigned", assigned)):
            pairs = [(bool(predicted[p]), bool(truth[p])) for p in range(49) if allowed[p]]
            counts = {key: pairs.count(pair) for key, pair in
                      (("tp", (True, True)), ("fp", (True, False)),
                       ("fn", (False, True)), ("tn", (False, False)))}
            assert counts == r["counts"][label]
            for key, value in counts.items():
                totals[(r["subset"], arm, r["policy"], label)][key] += value
    for (subset, arm, policy, label), counts in totals.items():
        assert counts == summary["controls"][subset][arm][policy][label]
    frozen(plan)
    (replay_dir / "diagnostic-features").rmdir()
    replay_dir.rmdir()
    result = {"passed": True, "full_model_replay_groups": len(plan["ids"]), "arm_records": 2910,
              "complete_full_frame_labels_checked": complete_checked,
              "score_target_records": len(target_rows), "series_records": len(series),
              "control_records": len(control_rows), "replay_max_error": replay_max,
              "replay_threshold_flips": flips, "audio_tracks_verified": len(dataset.verified),
              "elapsed_seconds": time.monotonic() - started,
              "summary_sha256": digests(RUN / "summary.json")["sha256"],
              "plan_sha256": digests(RUN / "plan.json")["sha256"]}
    write_json(RUN / "verification.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

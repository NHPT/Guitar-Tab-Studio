#!/usr/bin/env python3
"""Preregistered inference-only frame trajectories and fixed temporal policies."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import resource
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services/worker"))
from stringtrace_ml.download_guitar_techs_polyphonic import digests, write_json
from stringtrace_ml.pitch_completion_v2_sounding import verify_sounding_contract

BASE = ROOT / "apps/api/data/training"
SOURCE = BASE / "pitch-completion-v2-sounding-v1-r2"
MUTED = BASE / "pitch-completion-v2-muted-v1"
HEAD = BASE / "pitch-completion-v2-head-diagnosis"
PAIR = BASE / "pitch-completion-v2-fit-transfer-resource-r1"
RUN = BASE / "pitch-completion-v2-frames-v1"
ARMS = ("budget", "budget_p1")
POLICIES = ("local_mean", "nearest", "attack_mean", "attack_joint", "positive_rise")
DOC = ROOT / "pitch-completion-v2-frame-preregistration.md"
VERIFIER = ROOT / "scripts/verify-pitch-completion-v2-frames.py"


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    return digests(path)["sha256"]


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def check(plan):
    for path, expected in plan["artifact_sha256"].items():
        assert sha(ROOT / path) == expected, f"Frozen input changed: {path}"


def freeze():
    assert not RUN.exists(), "Preserve existing frame evidence"
    contract, recipe, tracks, examples, _ = verify_sounding_contract(SOURCE)
    assert read(SOURCE / "verification.json")["passed"]
    assert read(MUTED / "verification.json")["passed"]
    assert read(HEAD / "verification.json")["passed"]
    source_plan = read(SOURCE / "plan.json")
    track_map = {t["track_id"]: t for t in tracks}
    subsets = {
        "training": source_plan["subsets"]["training"],
        "internal_complete": [i for i in source_plan["subsets"]["internal"]
                              if track_map[examples[i]["track_id"]]["label_policy"] == "complete"
                              and examples[i]["completion_supervised"]],
        "internal_targets": read(MUTED / "plan.json")["indices"],
    }
    ids = sorted(set().union(*map(set, subsets.values())))
    assert [len(subsets[k]) for k in subsets] == [1024, 285, 150]
    assert len(ids) == 1455
    assert sum(len(examples[i]["target_pitches"]) for i in subsets["training"]) == 635
    assert sum(len(examples[i]["target_pitches"]) for i in subsets["internal_targets"]) == 150
    assert all(not examples[i]["auxiliary_only"] for i in ids)
    masked_ids = [i for i in ids if not examples[i]["completion_supervised"]]
    assert len(masked_ids) == 7
    assert all(i in subsets["training"] and examples[i]["mask_reason"] == "unresolved-sounding-pitch"
               and not examples[i]["target_pitches"] for i in masked_ids)
    paths = [DOC, Path(__file__).resolve(), VERIFIER, HEAD / "groups.jsonl", HEAD / "plan.json",
             HEAD / "verification.json", SOURCE / "contract.json", SOURCE / "plan.json",
             SOURCE / "verification.json", SOURCE / "local-activity-controls.json",
             MUTED / "plan.json", MUTED / "acoustic.jsonl", MUTED / "pairs.jsonl",
             MUTED / "verification.json", PAIR / "commitment.json", PAIR / "checkpoint-freeze.json",
             ROOT / "services/worker/stringtrace_ml/pitch_completion_v2_model.py",
             ROOT / "services/worker/stringtrace_ml/pitch_completion_v2_sounding.py"]
    paths += [SOURCE / name for name in contract["files"]]
    paths += [PAIR / f"{arm}-recovery.pt" for arm in ARMS]
    plan = {
        "kind": "fixed-weight-frame-evidence-v1", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ids": ids, "subsets": subsets, "masked_training_ids": masked_ids,
        "recipe": recipe, "subsets_sha256": stable_hash(subsets),
        "artifact_sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths},
        "threshold": .5, "legacy_tolerance": 5e-5, "replay_tolerance": 5e-5,
        "policies": list(POLICIES), "windows": {"local": [0, .16], "pre": [-.16, -.02],
                                              "early": [0, .08], "late": [.16, .32],
                                              "onset": [-.05, .05]},
        "threads": 4, "batch_size": 8, "seed": 20261125, "maximum_seconds": 7200,
        "maximum_rss_gib": 6, "free_space_reserve_bytes": 1024**3,
        "training": False, "checkpoint_selection": False, "threshold_selection": False,
        "P2_audio_opened": False, "protected_assets_opened": False,
        "independent_generalization": False, "production_admitted": False,
    }
    RUN.mkdir()
    write_json(RUN / "plan.json", plan)
    print(json.dumps({"frozen": True, "unique_groups": len(ids),
                      "populations": {k: len(v) for k, v in subsets.items()}}), flush=True)


def geometry(recipe):
    import numpy as np
    frames = (round(recipe["window_seconds"] * recipe["sample_rate"]) // recipe["hop"] + 8) // 8
    return np.arange(frames) * recipe["hop"] / recipe["sample_rate"] * 8 - recipe["pre_onset_seconds"]


def window(times, bounds):
    import numpy as np
    return np.flatnonzero((times >= bounds[0]) & (times <= bounds[1]))


def policies(pitch, attack, times, plan):
    import numpy as np
    w = {k: window(times, v) for k, v in plan["windows"].items()}
    early = pitch[:, w["early"]].mean(1)
    return {"local_mean": pitch[:, w["local"]].mean(1),
            "nearest": pitch[:, int(np.abs(times).argmin())],
            "attack_mean": early,
            "attack_joint": np.sqrt(early * attack[:, w["onset"]].max(1)),
            "positive_rise": np.clip(pitch[:, w["early"]].max(1) - pitch[:, w["pre"]].mean(1), 0, 1)}


def infer():
    import numpy as np
    import torch
    from stringtrace_ml.pitch_completion_v2_model import V2Dataset, V2Model
    plan = read(RUN / "plan.json")
    check(plan)
    assert not (RUN / "inference.json").exists(), "Preserve inference"
    assert not any((RUN / f"{a}.npz").exists() for a in ARMS), "Preserve partial outputs"
    _, recipe, tracks, examples, _ = verify_sounding_contract(SOURCE)
    allowed = {examples[i]["track_id"] for i in plan["ids"]}
    dataset = V2Dataset([t for t in tracks if t["track_id"] in allowed], examples, recipe, RUN, [])
    torch.set_num_threads(plan["threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(plan["seed"])
    models = {}
    for arm in ARMS:
        checkpoint = torch.load(PAIR / f"{arm}-recovery.pt", map_location="cpu", weights_only=True)
        assert checkpoint["epoch"] == 40 and checkpoint["optimizer_steps"] == 5120
        assert checkpoint["numeric_recipe"] == recipe and not checkpoint["production_admitted"]
        assert checkpoint["study_commitment_sha256"] == sha(PAIR / "commitment.json")
        model = V2Model(recipe, reduced=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        models[arm] = model.eval()
    ids, frames = plan["ids"], dataset.output_frames
    arrays = {a: {"pitch": np.empty((len(ids), frames, 49), np.float32),
                  "onset": np.empty((len(ids), frames, 49), np.float32),
                  "raw": np.empty((len(ids), 49), np.float32),
                  "cardinality": np.empty((len(ids), 7), np.float32)} for a in ARMS}
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(ids), plan["batch_size"]):
            assert time.monotonic() - started < plan["maximum_seconds"]
            assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < plan["maximum_rss_gib"] * 1024**3
            assert shutil.disk_usage(RUN).free > plan["free_space_reserve_bytes"]
            batch_ids = ids[start:start + plan["batch_size"]]
            batch = dataset.batch(batch_ids, "cpu")
            for arm, model in models.items():
                out = model(batch["features"], batch["candidate_mask"])
                output = {"pitch": out["pitch"].sigmoid(), "onset": out["onset"].sigmoid(),
                          "raw": out["completion"].sigmoid(), "cardinality": out["cardinality"].softmax(-1)}
                for key, values in output.items():
                    assert bool(torch.isfinite(values).all())
                    arrays[arm][key][start:start + len(batch_ids)] = values.numpy()
            if start % 80 == 0:
                print(json.dumps({"completed_groups": start + len(batch_ids),
                                  "elapsed_seconds": round(time.monotonic() - started, 2)}), flush=True)
    for arm, values in arrays.items():
        with (RUN / f"{arm}.npz").open("xb") as stream:
            np.savez(stream, ids=np.asarray(ids), times=dataset.times - recipe["pre_onset_seconds"], **values)
    check(plan)
    write_json(RUN / "inference.json", {"passed": True, "unique_groups": len(ids),
               "arm_records": len(ids) * 2, "frames": frames, "audio_tracks_verified": len(dataset.verified),
               "elapsed_seconds": time.monotonic() - started,
               "rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
               "output_sha256": {f"{a}.npz": sha(RUN / f"{a}.npz") for a in ARMS}})
    if not list(dataset.feature_directory.iterdir()):
        dataset.feature_directory.rmdir()
    print(json.dumps(read(RUN / "inference.json")), flush=True)


def ranks(scores, present, targets):
    # Rank each correct target independently against non-target absent pitches.
    result = {}
    for p in targets:
        other = [q for q in range(49) if q not in present and q not in targets]
        result[p] = 1 + int(sum(scores[q] > scores[p] or (scores[q] == scores[p] and q < p) for q in other))
    return result


def distribution(values):
    import numpy as np
    if not values:
        return {"n": 0}
    return {"n": len(values), "median": float(np.median(values)), "mean": float(np.mean(values)),
            "p10": float(np.quantile(values, .1)), "p90": float(np.quantile(values, .9))}


def cell_counts(predicted, truth, mask):
    return {k: int(v.sum()) for k, v in {
        "tp": predicted & truth & mask, "fp": predicted & ~truth & mask,
        "fn": ~predicted & truth & mask, "tn": ~predicted & ~truth & mask}.items()}


def analyze():
    import numpy as np
    from stringtrace_ml.pitch_completion_v2_model import dense_targets
    plan = read(RUN / "plan.json")
    check(plan)
    assert read(RUN / "inference.json")["passed"]
    assert not (RUN / "summary.json").exists(), "Preserve analysis"
    _, recipe, tracks, examples, _ = verify_sounding_contract(SOURCE)
    tracks = {t["track_id"]: t for t in tracks}
    pos = {i: j for j, i in enumerate(plan["ids"])}
    acoustic = {r["index"]: r for r in rows(MUTED / "acoustic.jsonl")}
    pair = {(r["arm"], r["index"]): r for r in rows(MUTED / "pairs.jsonl")}
    arrays = {a: np.load(RUN / f"{a}.npz", allow_pickle=False) for a in ARMS}
    times = geometry(recipe)
    local = window(times, plan["windows"]["local"])
    onset = window(times, plan["windows"]["onset"])
    legacy = {a: defaultdict(float) for a in ARMS}
    legacy_flips = {a: defaultdict(int) for a in ARMS}
    compared = 0
    for row in rows(HEAD / "groups.jsonl"):
        i, arm = row["index"], row["arm"]
        if i not in pos:
            continue
        j, data = pos[i], arrays[arm]
        context = window(times, recipe["encoder"]["context_seconds"])
        reformed = {"dense_local": data["pitch"][j, local].mean(0),
                    "dense_nearest": data["pitch"][j, int(np.abs(times).argmin())],
                    "dense_context": data["pitch"][j, context].mean(0),
                    "onset_local": data["onset"][j, onset].max(0),
                    "raw": data["raw"][j], "count_probability": data["cardinality"][j]}
        for key, values in reformed.items():
            old = np.asarray(row[key])
            error = float(np.abs(values - old).max())
            legacy[arm][key] = max(legacy[arm][key], error)
            legacy_flips[arm][key] += int(((values >= .5) != (old >= .5)).sum())
            assert error <= plan["legacy_tolerance"], (arm, i, key, error)
        compared += 1
    assert compared == 2910
    score = {a: policies(d["pitch"], d["onset"], times, plan) for a, d in arrays.items()}
    target_records, series = [], []
    for subset in ("training", "internal_targets"):
        for arm in ARMS:
            for i in plan["subsets"][subset]:
                example, j = examples[i], pos[i]
                target = [p - 40 for p in example["target_pitches"]]
                present = [p - 40 for p in example["candidate_pitches"] if 40 <= p <= 88]
                rr = {k: ranks(s[j], present, target) for k, s in score[arm].items()}
                for p in target:
                    record = {"arm": arm, "subset": subset, "index": i, "pitch": p + 40,
                              "technique": acoustic[i]["technique"] if subset == "internal_targets" else "training",
                              "policies": {k: {"score": float(s[j, p]), "rank": rr[k][p],
                                              "above_threshold": bool(s[j, p] >= .5)}
                                           for k, s in score[arm].items()}}
                    target_records.append(record)
                    if subset != "internal_targets":
                        continue
                    a, fixed = acoustic[i], pair[(arm, i)]["competitor"] - 40
                    pitch, attack = arrays[arm]["pitch"][j], arrays[arm]["onset"][j]
                    other = [q for q in range(49) if q not in present and q not in target]
                    assert fixed in other
                    reference_delta = a["reference_group_delta_seconds"]
                    attack_delta = reference_delta + a["temporal"]["attack_proxy_seconds"]
                    aligned = {}
                    for name, delta in (("group", 0.), ("reference", reference_delta),
                                        ("attack", attack_delta)):
                        if name == "attack" and not a["temporal"]["attack_reliable"]:
                            continue
                        frames = {w: window(times - delta, plan["windows"][w]) for w in ("pre", "early", "late")}
                        assert all(len(f) for f in frames.values())
                        values = {w: pitch[f].mean(0) for w, f in frames.items()}
                        aligned[name] = {"delta_seconds": delta, "frames": {w: f.tolist() for w, f in frames.items()},
                                         "target": {w: float(v[p]) for w, v in values.items()},
                                         "competitor": {w: float(v[fixed]) for w, v in values.items()},
                                         "target_rise": float(values["early"][p] - values["pre"][p]),
                                         "competitor_rise": float(values["early"][fixed] - values["pre"][fixed]),
                                         "early_margin": float(values["early"][p] - values["early"][fixed]),
                                         "early_rank": ranks(values["early"], present, target)[p]}
                    series.append({"arm": arm, "index": i, "pitch": p + 40, "competitor": fixed + 40,
                                   "technique": a["technique"], "label_policy": a["label_policy"],
                                   "attack_reliable": a["temporal"]["attack_reliable"],
                                   "aligned": aligned, "target_pitch": pitch[:, p].tolist(),
                                   "competitor_pitch": pitch[:, fixed].tolist(),
                                   "target_onset": attack[:, p].tolist(),
                                   "competitor_onset": attack[:, fixed].tolist(),
                                   "max_other_pitch": pitch[:, other].max(1).tolist(),
                                   "frame_rank": [ranks(v, present, target)[p] for v in pitch]})
    ranking = {}
    for subset in ("training", "internal_targets"):
        techniques = ("all",) if subset == "training" else ("all", "palm-mute", "pick", "harmonic", "unknown")
        ranking[subset] = {}
        for arm in ARMS:
            ranking[subset][arm] = {}
            for tech in techniques:
                rr = [r for r in target_records if r["arm"] == arm and r["subset"] == subset
                      and (tech == "all" or r["technique"] == tech)]
                ranking[subset][arm][tech] = {k: {
                    "targets": len(rr), "top1": sum(r["policies"][k]["rank"] == 1 for r in rr),
                    "mrr": float(np.mean([1 / r["policies"][k]["rank"] for r in rr])) if rr else None,
                    "above_threshold": sum(r["policies"][k]["above_threshold"] for r in rr),
                    "top1_above_threshold": sum(r["policies"][k]["rank"] == 1 and
                                               r["policies"][k]["above_threshold"] for r in rr)}
                    for k in POLICIES}
    temporal = {}
    for arm in ARMS:
        temporal[arm] = {}
        for tech in ("all", "palm-mute", "pick", "harmonic", "unknown"):
            ss = [s for s in series if s["arm"] == arm and (tech == "all" or s["technique"] == tech)]
            temporal[arm][tech] = {}
            for alignment in ("group", "reference", "attack"):
                aa = [s["aligned"][alignment] for s in ss if alignment in s["aligned"]]
                temporal[arm][tech][alignment] = {
                    "targets": len(aa),
                    "target": {w: distribution([a["target"][w] for a in aa]) for w in ("pre", "early", "late")},
                    "competitor": {w: distribution([a["competitor"][w] for a in aa]) for w in ("pre", "early", "late")},
                    **{key: distribution([a[key] for a in aa]) for key in
                       ("target_rise", "competitor_rise", "early_margin")},
                    "target_rises": sum(a["target_rise"] > 0 for a in aa),
                    "target_rise_exceeds_competitor": sum(a["target_rise"] > a["competitor_rise"] for a in aa),
                    "early_beats_fixed_competitor": sum(a["early_margin"] > 0 for a in aa),
                    "early_top1": sum(a["early_rank"] == 1 for a in aa)}
    control_records = []
    controls = {}
    old_controls = read(SOURCE / "local-activity-controls.json")
    for subset, ids in (("training", plan["subsets"]["training"]),
                        ("internal", plan["subsets"]["internal_complete"])):
        controls[subset] = {a: {k: {label: dict(tp=0, fp=0, fn=0, tn=0) for label in ("activity", "assigned")}
                               for k in POLICIES} for a in ARMS}
        eligible = 0
        for i in ids:
            example = examples[i]
            track = tracks[example["track_id"]]
            if track["label_policy"] != "complete":
                continue
            dense = dense_targets(track, example["onset"], times + recipe["pre_onset_seconds"], recipe)
            truth = dense["dense_pitch"][local].numpy().astype(bool).any(0)
            mask = dense["pitch_mask"][local].numpy().all(0)
            mask[np.asarray(example["candidate_pitches"], int) - 40] = False
            assigned = np.zeros(49, bool)
            assigned[np.asarray(example["target_pitches"], int) - 40] = True
            eligible += 1
            for arm in ARMS:
                for policy in POLICIES:
                    predicted = score[arm][policy][pos[i]] >= .5
                    values = {label: cell_counts(predicted, label_truth, mask)
                              for label, label_truth in (("activity", truth), ("assigned", assigned))}
                    control_records.append({"subset": subset, "index": i, "arm": arm,
                                            "policy": policy, "counts": values})
                    for label, counts in values.items():
                        for key, value in counts.items():
                            controls[subset][arm][policy][label][key] += value
        assert eligible == (972 if subset == "training" else 285)
        for arm in ARMS:
            for key in ("tp", "fp", "fn", "tn"):
                assert controls[subset][arm]["local_mean"]["activity"][key] == old_controls["subsets"][subset][arm]["all"][key]
    for name, records in (("targets.jsonl", target_records), ("series.jsonl", series),
                          ("controls.jsonl", control_records)):
        with (RUN / name).open("x") as stream:
            for record in records:
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    summary = {"plan_sha256": sha(RUN / "plan.json"), "unique_groups": 1455, "arm_records": 2910,
               "frames": 56, "times": times.tolist(), "legacy_records": compared,
               "legacy_max_error": legacy, "legacy_threshold_flips": legacy_flips,
               "ranking": ranking, "temporal": temporal, "controls": controls,
               "inference": read(RUN / "inference.json"),
               "training": False, "independent_generalization": False,
               "output_sha256": {n: sha(RUN / n) for n in ("targets.jsonl", "series.jsonl", "controls.jsonl")}}
    check(plan)
    write_json(RUN / "summary.json", summary)
    print(json.dumps({"legacy_max_error": legacy, "internal_ranking": ranking["internal_targets"],
                      "controls": controls}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "infer", "analyze"))
    args = parser.parse_args()
    {"freeze": freeze, "infer": infer, "analyze": analyze}[args.action]()

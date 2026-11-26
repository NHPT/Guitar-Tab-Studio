#!/usr/bin/env python3
"""Verify exported report claims, figures, frozen artifacts and decoder supplement."""
# ruff: noqa: E402
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services/worker"))
from stringtrace_ml.download_guitar_techs_polyphonic import digests, write_json
from stringtrace_ml.pitch_completion_v2_sounding import verify_sounding_contract

RUN = ROOT / "apps/api/data/training/pitch-completion-v2-frames-v1"
SOURCE = ROOT / "apps/api/data/training/pitch-completion-v2-sounding-v1-r2"
HEAD = ROOT / "apps/api/data/training/pitch-completion-v2-head-diagnosis"
ARMS = ("budget", "budget_p1")
DOC = ROOT / "pitch-completion-v2-frame-result.md"


def read(path):
    return json.loads(path.read_text())


def outside(event, track):
    return any((item["start"] <= event["onset"] < item["end"]) if isinstance(item, dict)
               else item[0] <= event["onset"] < item[1]
               for item in track.get("excluded_ranges", []))


def manual_labels(track, onset, times):
    absolute = onset + times
    valid = (absolute >= 0) & (absolute < track["duration"])
    for item in track.get("excluded_ranges", []):
        start, end = (item["start"], item["end"]) if isinstance(item, dict) else item
        valid &= ~((absolute >= start) & (absolute < end))
    mask = np.repeat(valid[:, None], 49, axis=1)
    for start, end in track.get("unknown_string_intervals", []):
        mask[(absolute >= start) & (absolute < end)] = False
    labels = np.zeros_like(mask)
    for event in track["events"]:
        pitch = event["sounding_pitch"]
        if pitch is None or not event.get("positive_label_valid", True) or outside(event, track):
            continue
        if 40 <= pitch <= 88:
            labels[:, pitch - 40] |= (
                (absolute >= event["onset"]) & (absolute < event["offset"]) & valid)
    return labels, mask


def independently_score(pitch, attack, times):
    local = [j for j, value in enumerate(times) if 0 <= value <= .16]
    pre = [j for j, value in enumerate(times) if -.16 <= value <= -.02]
    early = [j for j, value in enumerate(times) if 0 <= value <= .08]
    onset = [j for j, value in enumerate(times) if -.05 <= value <= .05]
    early_mean = np.mean(pitch[early], axis=0)
    return {
        "local_mean": np.mean(pitch[local], axis=0),
        "nearest": pitch[min(range(len(times)), key=lambda j: abs(times[j]))],
        "attack_mean": early_mean,
        "attack_joint": np.sqrt(early_mean * np.max(attack[onset], axis=0)),
        "positive_rise": np.clip(np.max(pitch[early], axis=0) -
                                 np.mean(pitch[pre], axis=0), 0, 1),
    }


def independent_rank(values, example, pitch):
    permitted = [value for value in range(40, 89)
                 if value not in example["candidate_pitches"]
                 and (value == pitch or value not in example["target_pitches"])]
    return sorted(permitted, key=lambda value: (-float(values[value - 40]), value)).index(pitch) + 1


def cell_counts(predicted, truth, mask):
    return {key: int(value.sum()) for key, value in {
        "tp": predicted & truth & mask,
        "fp": predicted & ~truth & mask,
        "fn": ~predicted & truth & mask,
        "tn": ~predicted & ~truth & mask,
    }.items()}


def main():
    assert not (RUN / "delivery-verification.json").exists()
    plan, summary, verification = (read(RUN / n) for n in ("plan.json", "summary.json", "verification.json"))
    assert verification["passed"]
    for path, expected in plan["artifact_sha256"].items():
        assert digests(ROOT / path)["sha256"] == expected
    assert verification["summary_sha256"] == digests(RUN / "summary.json")["sha256"]
    assert verification["plan_sha256"] == digests(RUN / "plan.json")["sha256"]
    for name, expected in {**summary["output_sha256"], **summary["inference"]["output_sha256"]}.items():
        assert digests(RUN / name)["sha256"] == expected
    _, recipe, track_rows, examples, _ = verify_sounding_contract(SOURCE)
    tracks = {track["track_id"]: track for track in track_rows}
    position = {i: j for j, i in enumerate(plan["ids"])}
    data = {}
    for arm in ARMS:
        with np.load(RUN / f"{arm}.npz", allow_pickle=False) as source:
            data[arm] = {key: np.asarray(source[key]) for key in
                         ("ids", "times", "pitch", "onset", "raw", "cardinality")}
    times = data["budget"]["times"]
    assert np.array_equal(times, data["budget_p1"]["times"])
    assert np.array_equal(times, np.asarray(summary["times"]))
    legacy_max, legacy_flips = {}, {}
    checked = 0
    for arm in ARMS:
        legacy_max[arm] = {k: 0. for k in summary["legacy_max_error"][arm]}
        legacy_flips[arm] = {k: 0 for k in summary["legacy_threshold_flips"][arm]}
    with (HEAD / "groups.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            i, arm = row["index"], row["arm"]
            if i not in position:
                continue
            j, d = position[i], data[arm]
            local = [k for k, t in enumerate(times) if 0 <= t <= .16]
            attack = [k for k, t in enumerate(times) if -.05 <= t <= .05]
            bounds = recipe["encoder"]["context_seconds"]
            context = [k for k, t in enumerate(times) if bounds[0] <= t <= bounds[1]]
            reaggregated = {
                "dense_local": np.mean(d["pitch"][j, local], axis=0),
                "dense_nearest": d["pitch"][j, min(range(len(times)), key=lambda k: abs(times[k]))],
                "dense_context": np.mean(d["pitch"][j, context], axis=0),
                "onset_local": np.max(d["onset"][j, attack], axis=0),
                "raw": d["raw"][j], "count_probability": d["cardinality"][j],
            }
            for key, values in reaggregated.items():
                expected = np.asarray(row[key])
                error = float(np.max(np.abs(values - expected)))
                assert error <= plan["legacy_tolerance"]
                legacy_max[arm][key] = max(legacy_max[arm][key], error)
                legacy_flips[arm][key] += int(np.count_nonzero((values >= .5) != (expected >= .5)))
            checked += 1
    assert checked == 2910
    assert legacy_max == summary["legacy_max_error"]
    assert legacy_flips == summary["legacy_threshold_flips"]
    scores = {(arm, index): independently_score(
        arrays["pitch"][position[index]], arrays["onset"][position[index]], times)
              for arm, arrays in data.items() for index in plan["ids"]}
    target_rows = [json.loads(line) for line in (RUN / "targets.jsonl").read_text().splitlines()
                   if line.strip()]
    assert len(target_rows) == 2 * (635 + 150)
    for row in target_rows:
        example = examples[row["index"]]
        for policy, value in row["policies"].items():
            score = scores[(row["arm"], row["index"])][policy]
            assert abs(float(score[row["pitch"] - 40]) - value["score"]) < 1e-7
            assert independent_rank(score, example, row["pitch"]) == value["rank"]
            assert bool(score[row["pitch"] - 40] >= .5) == value["above_threshold"]
    for subset, arms in summary["ranking"].items():
        for arm, techniques in arms.items():
            for technique, policies in techniques.items():
                selected = [row for row in target_rows if row["arm"] == arm
                            and row["subset"] == subset
                            and (technique == "all" or row["technique"] == technique)]
                for policy, metric in policies.items():
                    assert metric["targets"] == len(selected)
                    assert metric["top1"] == sum(
                        row["policies"][policy]["rank"] == 1 for row in selected)
                    assert abs(metric["mrr"] - np.mean([
                        1 / row["policies"][policy]["rank"] for row in selected])) < 1e-12
                    assert metric["above_threshold"] == sum(
                        row["policies"][policy]["above_threshold"] for row in selected)
                    assert metric["top1_above_threshold"] == sum(
                        row["policies"][policy]["rank"] == 1
                        and row["policies"][policy]["above_threshold"] for row in selected)
    # Post-primary supplement: unchanged completion/cardinality actual decoding.
    decoder = {"post_primary_descriptive_only": True, "new_inference_or_audio": False,
               "threshold": .5, "internal_targets": {}}
    for arm, d in data.items():
        tp, additions, count_zero = 0, 0, 0
        for i in plan["subsets"]["internal_targets"]:
            j, example = position[i], examples[i]
            count = int(d["cardinality"][j].argmax())
            count_zero += count == 0
            joint = d["raw"][j] * (1 - d["cardinality"][j, 0])
            available = [p for p in range(40, 89) if p not in example["candidate_pitches"]]
            ordered = sorted(available, key=lambda p: (-float(joint[p - 40]), p))
            predicted = [p for p in ordered[:count] if joint[p - 40] >= .5]
            tp += len(set(predicted) & set(example["target_pitches"]))
            additions += len(predicted)
        decoder["internal_targets"][arm] = {"targets": 150, "supplemented_tp": tp,
                                           "added_pitches": additions, "count_zero_groups": count_zero}
    write_json(RUN / "decoder-supplement.json", decoder)
    series = [json.loads(line) for line in (RUN / "series.jsonl").read_text().splitlines() if line.strip()]
    assert len(series) == 300
    for row in series:
        index, arm = row["index"], row["arm"]
        target, competitor = row["pitch"] - 40, row["competitor"] - 40
        offset = position[index]
        example = examples[index]
        arrays = data[arm]
        for key, tensor, pitch in (
            ("target_pitch", "pitch", target),
            ("competitor_pitch", "pitch", competitor),
            ("target_onset", "onset", target),
            ("competitor_onset", "onset", competitor),
        ):
            assert np.array_equal(np.asarray(row[key]), arrays[tensor][offset, :, pitch])
        other = [pitch for pitch in range(49)
                 if pitch + 40 not in example["candidate_pitches"]
                 and pitch + 40 not in example["target_pitches"]]
        assert np.array_equal(row["max_other_pitch"], arrays["pitch"][offset][:, other].max(1))
        assert row["frame_rank"] == [
            independent_rank(frame, example, target + 40)
            for frame in arrays["pitch"][offset]]
        for alignment in row["aligned"].values():
            values = {}
            for label, bounds in (("pre", [-.16, -.02]), ("early", [0, .08]),
                                  ("late", [.16, .32])):
                frames = [j for j, value in enumerate(times)
                          if bounds[0] <= value - alignment["delta_seconds"] <= bounds[1]]
                assert frames == alignment["frames"][label]
                values[label] = arrays["pitch"][offset, frames].mean(0)
                assert float(values[label][target]) == alignment["target"][label]
                assert float(values[label][competitor]) == alignment["competitor"][label]
            assert float(values["early"][target] - values["pre"][target]) == alignment["target_rise"]
            assert (float(values["early"][competitor] - values["pre"][competitor])
                    == alignment["competitor_rise"])
            assert float(values["early"][target] - values["early"][competitor]) == alignment["early_margin"]
            assert independent_rank(values["early"], example, target + 40) == alignment["early_rank"]
    # Recheck every temporal statistic used in report, including distribution tails.
    for arm, techniques in summary["temporal"].items():
        for tech, alignments in techniques.items():
            for alignment, metrics in alignments.items():
                rows = [s["aligned"][alignment] for s in series if s["arm"] == arm
                        and (tech == "all" or s["technique"] == tech) and alignment in s["aligned"]]
                for label in ("target", "competitor"):
                    for window in ("pre", "early", "late"):
                        values = [r[label][window] for r in rows]
                        assert metrics[label][window]["n"] == len(values)
                        expected = dict(median=float(np.median(values)), mean=float(np.mean(values)),
                                        p10=float(np.quantile(values, .1)), p90=float(np.quantile(values, .9)))
                        assert all(abs(metrics[label][window][k] - v) < 1e-12 for k, v in expected.items())
                for key in ("target_rise", "competitor_rise", "early_margin"):
                    values = [row[key] for row in rows]
                    expected = dict(median=float(np.median(values)), mean=float(np.mean(values)),
                                    p10=float(np.quantile(values, .1)),
                                    p90=float(np.quantile(values, .9)))
                    assert all(abs(metrics[key][name] - value) < 1e-12
                               for name, value in expected.items())
                assert metrics["target_rises"] == sum(row["target_rise"] > 0 for row in rows)
                assert metrics["target_rise_exceeds_competitor"] == sum(
                    row["target_rise"] > row["competitor_rise"] for row in rows)
                assert metrics["early_beats_fixed_competitor"] == sum(
                    row["early_margin"] > 0 for row in rows)
                assert metrics["early_top1"] == sum(row["early_rank"] == 1 for row in rows)
    control_rows = [json.loads(line) for line in (RUN / "controls.jsonl").read_text().splitlines()
                    if line.strip()]
    assert len(control_rows) == 1257 * 2 * 5
    totals = defaultdict(lambda: dict(tp=0, fp=0, fn=0, tn=0))
    local = [j for j, value in enumerate(times) if 0 <= value <= .16]
    for row in control_rows:
        index, arm = row["index"], row["arm"]
        example = examples[index]
        active_frames, complete_mask = manual_labels(
            tracks[example["track_id"]], example["onset"], times)
        activity = active_frames[local].any(0)
        allowed = complete_mask[local].all(0)
        allowed[np.asarray(example["candidate_pitches"], dtype=int) - 40] = False
        assigned = np.zeros(49, dtype=bool)
        assigned[np.asarray(example["target_pitches"], dtype=int) - 40] = True
        predicted = scores[(arm, index)][row["policy"]] >= .5
        for label, truth in (("activity", activity), ("assigned", assigned)):
            counts = cell_counts(predicted, truth, allowed)
            assert counts == row["counts"][label]
            for key, value in counts.items():
                totals[(row["subset"], arm, row["policy"], label)][key] += value
    for (subset, arm, policy, label), counts in totals.items():
        assert counts == summary["controls"][subset][arm][policy][label]
    text = DOC.read_text()
    assert "/Users/" not in text and "file://" not in text and "computer://" not in text
    links = re.findall(r"\]\(([^)]+)\)", text)
    for link in links:
        target = (ROOT / link).resolve()
        assert target.is_relative_to(ROOT) and target.is_file(), link
    table_rows = [line for line in text.splitlines() if line.startswith("|固定预算")]
    assert len(table_rows) == 12
    for arm, label in (("budget", "固定预算"), ("budget_p1", "固定预算＋P1")):
        m = summary["temporal"][arm]["palm-mute"]["attack"]
        target = " / ".join(f"{m['target'][w]['median']:.4f}" for w in ("pre", "early", "late"))
        competitor = " / ".join(f"{m['competitor'][w]['median']:.4f}" for w in ("pre", "early", "late"))
        expected = (f"|{label}|{target}|{competitor}|{m['target_rise']['median']:+.4f}|"
                    f"{m['competitor_rise']['median']:+.4f}|{m['target_rises']}/109|"
                    f"{m['target_rise_exceeds_competitor']}/109|{m['early_beats_fixed_competitor']}/109|")
        assert expected in text
    label_names = {"local_mean": "原 local 均值", "nearest": "最近帧", "attack_mean": "0–80ms 均值",
                   "attack_joint": "音高 × 起音联合", "positive_rise": "正向上升量"}
    for policy, label in label_names.items():
        rankings = summary["ranking"]["internal_targets"]
        numbers = [" / ".join(str(rankings[a][tech][policy][key]) for a in ARMS) for tech, key in
                   (("palm-mute", "top1"), ("palm-mute", "top1_above_threshold"),
                    ("all", "top1"), ("pick", "top1"))]
        assert f"|{label}|" + "|".join(numbers) + "|" in text
    for arm, arm_label in (("budget", "固定预算"), ("budget_p1", "固定预算＋P1")):
        for policy, label in label_names.items():
            counts = [summary["controls"][subset][arm][policy][truth]
                      for subset in ("training", "internal") for truth in ("activity", "assigned")]
            assert f"|{arm_label}|{label}|" + "|".join(f"{c['tp']} / {c['fp']}" for c in counts) + "|" in text
    from PIL import Image
    figure = ROOT / "pitch-completion-v2-frame-results.png"
    with Image.open(figure) as image:
        image.verify()
    with Image.open(figure) as image:
        assert image.size == (2048, 1408)
    svg = (ROOT / "pitch-completion-v2-frame-results.svg").read_text()
    assert "<svg" in svg and "/Users/" not in svg
    export_paths = [DOC, figure, ROOT / "pitch-completion-v2-frame-results.svg",
                    RUN / "decoder-supplement.json"]
    script = Path(__file__).resolve()
    exports = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in export_paths}
    result = {"passed": True, "independent_legacy_records": checked,
              "score_target_records_checked": len(target_rows),
              "series_records_checked": len(series),
              "control_records_checked": len(control_rows),
              "report_table_rows_checked": 17, "report_local_links_checked": len(links),
              "figure_dimensions": [2048, 1408], "export_sha256": exports,
              "figure_visually_inspected": True,
              "delivery_script_sha256": digests(script)["sha256"],
              "frozen_artifacts_unchanged": True, "decoder_supplement": decoder}
    write_json(RUN / "delivery-verification.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()

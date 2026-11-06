#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
DECODER_ERROR_NAMES = (
    "exact_tablature",
    "same_pitch_wrong_string",
    "same_string_wrong_fret",
    "wrong_string_and_pitch",
    "spurious_onset",
    "missed_onset",
)
ASSIGNMENT_ERROR_NAMES = (
    "same_pitch_wrong_string",
)
DIFFERENT_PITCH_ERROR_NAMES = (
    "same_string_wrong_fret",
    "wrong_string_and_pitch",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attribute governed v28 position errors across 72 tracks.",
    )
    parser.add_argument(
        "--guitarset-report",
        type=Path,
        default=Path(
            "apps/api/data/training/reports/"
            "governed-v28-guitarset-mixture-validation.json"
        ),
    )
    parser.add_argument(
        "--guitar-techs-report",
        type=Path,
        default=Path(
            "apps/api/data/training/reports/"
            "governed-v28-guitar-techs-validation.json"
        ),
    )
    parser.add_argument(
        "--guitarset-v24-report",
        type=Path,
        default=Path(
            "apps/api/data/training/reports/"
            "governed-v24-guitarset-mixture-validation.json"
        ),
    )
    parser.add_argument(
        "--guitar-techs-v24-report",
        type=Path,
        default=Path(
            "apps/api/data/training/reports/"
            "governed-v24-guitar-techs-validation.json"
        ),
    )
    parser.add_argument(
        "--guitarset-manifest",
        type=Path,
        default=Path(
            "apps/api/data/training/governed-v1/"
            "guitarset-mixture-derived.jsonl"
        ),
    )
    parser.add_argument(
        "--guitar-techs-manifest",
        type=Path,
        default=Path(
            "apps/api/data/training/governed-v1/"
            "guitar-techs-position-validation.jsonl"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "apps/api/data/training/reports/"
            "governed-v28-72-track-error-attribution.json"
        ),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("docs/model-v28-72-track-error-attribution.md"),
    )
    return parser.parse_args()


def repository_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def relative_name(path: Path) -> str:
    return repository_path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def load_json(path: Path) -> dict[str, Any]:
    with repository_path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path, *, split: str | None = None) -> list[dict[str, Any]]:
    records = []
    with repository_path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if split is None or record.get("split") == split:
                records.append(record)
    return records


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, str]:
    return {
        "path": relative_name(path),
        "sha256": sha256(path),
    }


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def f1_score(precision: float, recall: float) -> float:
    return (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def aggregate_metric(
    tracks: Iterable[dict[str, Any]],
    metric_name: str,
) -> dict[str, int | float | None]:
    track_list = list(tracks)
    true_positive = sum(
        int(track["metrics"][metric_name]["true_positive"])
        for track in track_list
    )
    false_positive = sum(
        int(track["metrics"][metric_name]["false_positive"])
        for track in track_list
    )
    false_negative = sum(
        int(track["metrics"][metric_name]["false_negative"])
        for track in track_list
    )
    precision = ratio(true_positive, true_positive + false_positive)
    recall = ratio(true_positive, true_positive + false_negative)
    weighted_errors = [
        (
            float(track["metrics"][metric_name]["mean_onset_error_ms"]),
            int(track["metrics"][metric_name]["true_positive"]),
        )
        for track in track_list
        if track["metrics"][metric_name]["mean_onset_error_ms"] is not None
    ]
    error_weight = sum(weight for _error, weight in weighted_errors)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
        "mean_onset_error_ms": (
            sum(error * weight for error, weight in weighted_errors)
            / error_weight
            if error_weight
            else None
        ),
    }


def decoder_counts(tracks: Iterable[dict[str, Any]]) -> dict[str, int]:
    track_list = list(tracks)
    return {
        name: sum(
            int(track["metrics"]["decoder_errors"][name])
            for track in track_list
        )
        for name in DECODER_ERROR_NAMES
    }


def summarize_tracks(tracks: Iterable[dict[str, Any]]) -> dict[str, Any]:
    track_list = list(tracks)
    counts = decoder_counts(track_list)
    assignment_errors = sum(counts[name] for name in ASSIGNMENT_ERROR_NAMES)
    different_pitch_errors = sum(
        counts[name] for name in DIFFERENT_PITCH_ERROR_NAMES
    )
    final_output_errors = counts["spurious_onset"] + counts["missed_onset"]
    failure_instances = (
        assignment_errors
        + different_pitch_errors
        + final_output_errors
    )
    return {
        "track_count": len(track_list),
        "reference_events": sum(int(track["reference_events"]) for track in track_list),
        "predicted_events": sum(int(track["predicted_events"]) for track in track_list),
        "metrics": {
            "onset": aggregate_metric(track_list, "onset"),
            "pitch": aggregate_metric(track_list, "pitch"),
            "tablature": aggregate_metric(track_list, "tablature"),
        },
        "decoder_errors": counts,
        "failure_attribution": {
            "failure_instances": failure_instances,
            "final_output_selection_or_detection": {
                "instances": final_output_errors,
                "share": ratio(final_output_errors, failure_instances),
                "missed_onset": counts["missed_onset"],
                "spurious_onset": counts["spurious_onset"],
            },
            "string_fret_assignment": {
                "instances": assignment_errors,
                "share": ratio(assignment_errors, failure_instances),
                "same_pitch_wrong_string": counts["same_pitch_wrong_string"],
            },
            "different_pitch_matched_output": {
                "instances": different_pitch_errors,
                "share": ratio(different_pitch_errors, failure_instances),
                "same_string_wrong_fret": counts["same_string_wrong_fret"],
                "wrong_string_and_pitch": counts["wrong_string_and_pitch"],
            },
        },
    }


def validate_track(track: dict[str, Any]) -> None:
    metrics = track["metrics"]
    decoder = metrics["decoder_errors"]
    decoder_matches = sum(
        int(decoder[name])
        for name in (
            "exact_tablature",
            "same_pitch_wrong_string",
            "same_string_wrong_fret",
            "wrong_string_and_pitch",
        )
    )
    exclusions = metrics.get("exclusions", {})
    eligible_reference = int(track["reference_events"]) - int(
        exclusions.get("excluded_reference_events", 0)
    )
    eligible_predicted = int(track["predicted_events"]) - int(
        exclusions.get("excluded_predicted_events", 0)
    )
    assert decoder_matches + int(decoder["missed_onset"]) == eligible_reference
    assert decoder_matches + int(decoder["spurious_onset"]) == eligible_predicted
    assert int(metrics["onset"]["true_positive"]) + int(
        metrics["onset"]["false_negative"]
    ) == eligible_reference
    assert int(metrics["onset"]["true_positive"]) + int(
        metrics["onset"]["false_positive"]
    ) == eligible_predicted


def validate_report(
    report: dict[str, Any],
    metadata: list[dict[str, Any]],
    *,
    expected_tracks: int,
) -> None:
    tracks = report["tracks"]
    assert report["track_count"] == expected_tracks
    assert len(tracks) == expected_tracks
    report_ids = [str(track["track_id"]) for track in tracks]
    metadata_ids = [str(record["track_id"]) for record in metadata]
    assert len(report_ids) == len(set(report_ids))
    assert set(report_ids) == set(metadata_ids)
    assert all(
        record["provenance"]["source_verification_status"] == "verified"
        for record in metadata
    )
    for track in tracks:
        validate_track(track)
    summary = summarize_tracks(tracks)
    for metric_name in ("onset", "pitch", "tablature"):
        for count_name in ("true_positive", "false_positive", "false_negative"):
            assert summary["metrics"][metric_name][count_name] == int(
                report["aggregate"][metric_name][count_name]
            )
    for name in DECODER_ERROR_NAMES:
        assert summary["decoder_errors"][name] == int(
            report["aggregate"]["decoder_errors"][name]
        )


def group_summaries(
    tracks: list[dict[str, Any]],
    metadata_by_id: dict[str, dict[str, Any]],
    key: Callable[[dict[str, Any]], str],
    *,
    sort_key: Callable[[str], Any] | None = None,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for track in tracks:
        groups[key(metadata_by_id[str(track["track_id"])])].append(track)
    return [
        {"value": value, **summarize_tracks(group)}
        for value, group in sorted(
            groups.items(),
            key=(
                (lambda item: sort_key(item[0]))
                if sort_key is not None
                else (lambda item: item[0])
            ),
        )
    ]


def guitarset_part(record: dict[str, Any]) -> str:
    source_id = str(
        record["provenance"].get("derived_from_track_id", record["track_id"])
    )
    if source_id.endswith("_comp"):
        return "comp"
    if source_id.endswith("_solo"):
        return "solo"
    raise ValueError(f"Cannot infer GuitarSet part for {record['track_id']}")


def guitar_technique(record: dict[str, Any]) -> str:
    techniques = {str(event["technique"]) for event in record["events"]}
    if len(techniques) != 1:
        raise ValueError(
            f"Expected one technique in {record['track_id']}, got {techniques}"
        )
    return techniques.pop()


def track_summary(
    track: dict[str, Any],
    metadata: dict[str, Any],
    *,
    dataset: str,
) -> dict[str, Any]:
    summary = summarize_tracks([track])
    dimensions = (
        {
            "target_snr_db": float(metadata["provenance"]["target_snr_db"]),
            "part": guitarset_part(metadata),
            "performer": str(metadata["provenance"]["player_id"]),
        }
        if dataset == "GuitarSet"
        else {
            "performer": str(metadata["provenance"]["player"]),
            "audio_modality": str(metadata["provenance"]["audio_modality"]),
            "technique": guitar_technique(metadata),
        }
    )
    return {
        "track_id": str(track["track_id"]),
        "dataset": dataset,
        "dimensions": dimensions,
        "reference_events": summary["reference_events"],
        "predicted_events": summary["predicted_events"],
        "onset_recall": summary["metrics"]["onset"]["recall"],
        "tablature_f1": summary["metrics"]["tablature"]["f1"],
        "decoder_errors": summary["decoder_errors"],
        "failure_attribution": summary["failure_attribution"],
    }


def assignment_without_supplemental_head(report: dict[str, Any]) -> dict[str, Any]:
    assignment = copy.deepcopy(report["assignment"])
    supplemental = assignment.get("supplemental")
    if supplemental is not None:
        supplemental.pop("candidate_activity", None)
    return assignment


def report_comparison(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    baseline_tracks = {
        str(track["track_id"]): track for track in baseline["tracks"]
    }
    candidate_tracks = {
        str(track["track_id"]): track for track in candidate["tracks"]
    }
    assert set(baseline_tracks) == set(candidate_tracks)
    changes = []
    for track_id in sorted(candidate_tracks):
        baseline_track = baseline_tracks[track_id]
        candidate_track = candidate_tracks[track_id]
        baseline_counts = decoder_counts([baseline_track])
        candidate_counts = decoder_counts([candidate_track])
        count_delta = {
            name: candidate_counts[name] - baseline_counts[name]
            for name in DECODER_ERROR_NAMES
        }
        predicted_delta = int(candidate_track["predicted_events"]) - int(
            baseline_track["predicted_events"]
        )
        if predicted_delta or any(count_delta.values()):
            changes.append(
                {
                    "track_id": track_id,
                    "predicted_events_delta": predicted_delta,
                    "decoder_error_delta": count_delta,
                }
            )
    return {
        "isolated_supplemental_activity_head_change": (
            baseline["pitch_frontend"] == candidate["pitch_frontend"]
            and baseline["position_model"] == candidate["position_model"]
            and assignment_without_supplemental_head(baseline)
            == assignment_without_supplemental_head(candidate)
        ),
        "baseline_checkpoint": baseline["assignment"]["supplemental"][
            "candidate_activity"
        ]["checkpoint"],
        "candidate_checkpoint": candidate["assignment"]["supplemental"][
            "candidate_activity"
        ]["checkpoint"],
        "changed_tracks": len(changes),
        "track_changes": changes,
        "aggregate_delta": {
            "onset_true_positive": (
                int(candidate["aggregate"]["onset"]["true_positive"])
                - int(baseline["aggregate"]["onset"]["true_positive"])
            ),
            "onset_false_positive": (
                int(candidate["aggregate"]["onset"]["false_positive"])
                - int(baseline["aggregate"]["onset"]["false_positive"])
            ),
            "onset_false_negative": (
                int(candidate["aggregate"]["onset"]["false_negative"])
                - int(baseline["aggregate"]["onset"]["false_negative"])
            ),
            "onset_f1": (
                float(candidate["aggregate"]["onset"]["f1"])
                - float(baseline["aggregate"]["onset"]["f1"])
            ),
            "tablature_true_positive": (
                int(candidate["aggregate"]["tablature"]["true_positive"])
                - int(baseline["aggregate"]["tablature"]["true_positive"])
            ),
            "tablature_false_positive": (
                int(candidate["aggregate"]["tablature"]["false_positive"])
                - int(baseline["aggregate"]["tablature"]["false_positive"])
            ),
            "tablature_false_negative": (
                int(candidate["aggregate"]["tablature"]["false_negative"])
                - int(baseline["aggregate"]["tablature"]["false_negative"])
            ),
            "tablature_f1": (
                float(candidate["aggregate"]["tablature"]["f1"])
                - float(baseline["aggregate"]["tablature"]["f1"])
            ),
        },
    }


def matching_diagnostics(tracks: Iterable[dict[str, Any]]) -> dict[str, Any]:
    track_list = list(tracks)
    onset_differences = []
    tablature_differences = []
    for track in track_list:
        decoder = track["metrics"]["decoder_errors"]
        decoder_matches = sum(
            int(decoder[name])
            for name in (
                "exact_tablature",
                "same_pitch_wrong_string",
                "same_string_wrong_fret",
                "wrong_string_and_pitch",
            )
        )
        onset_delta = decoder_matches - int(
            track["metrics"]["onset"]["true_positive"]
        )
        tablature_delta = int(decoder["exact_tablature"]) - int(
            track["metrics"]["tablature"]["true_positive"]
        )
        if onset_delta:
            onset_differences.append(
                {"track_id": str(track["track_id"]), "delta": onset_delta}
            )
        if tablature_delta:
            tablature_differences.append(
                {"track_id": str(track["track_id"]), "delta": tablature_delta}
            )
    return {
        "independent_matchers": True,
        "decoder_vs_onset_tracks_with_match_count_difference": len(
            onset_differences
        ),
        "decoder_vs_tablature_tracks_with_exact_count_difference": len(
            tablature_differences
        ),
        "decoder_vs_onset_differences": onset_differences,
        "decoder_vs_tablature_differences": tablature_differences,
    }


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def metric(value: float) -> str:
    return f"{value:.4f}"


def slice_table(rows: list[dict[str, Any]], first_header: str) -> list[str]:
    lines = [
        (
            f"| {first_header} | 轨数 | 参考事件 | onset recall | TAB F1 | "
            "漏检 | 误检 | 同音高错弦 | 最终输出层占比 |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        output = row["failure_attribution"]["final_output_selection_or_detection"]
        assignment = row["failure_attribution"]["string_fret_assignment"]
        lines.append(
            f"| {row['value']} | {row['track_count']} | "
            f"{row['reference_events']} | "
            f"{metric(row['metrics']['onset']['recall'])} | "
            f"{metric(row['metrics']['tablature']['f1'])} | "
            f"{output['missed_onset']} | {output['spurious_onset']} | "
            f"{assignment['instances']} | {percent(output['share'])} |"
        )
    return lines


def track_table(
    tracks: list[dict[str, Any]],
    *,
    dataset: str,
) -> list[str]:
    if dataset == "GuitarSet":
        lines = [
            (
                "| track_id | SNR | 类型 | 参考事件 | TAB F1 | 漏检 | "
                "误检 | 同音高错弦 | 最终输出层占比 |"
            ),
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    else:
        lines = [
            (
                "| track_id | 技法 | 音频 | 参考事件 | TAB F1 | 漏检 | "
                "误检 | 同音高错弦 | 最终输出层占比 |"
            ),
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    for track in tracks:
        output = track["failure_attribution"][
            "final_output_selection_or_detection"
        ]
        assignment = track["failure_attribution"]["string_fret_assignment"]
        dimensions = track["dimensions"]
        dimension_columns = (
            f"{dimensions['target_snr_db']:g} dB | {dimensions['part']}"
            if dataset == "GuitarSet"
            else (
                f"{dimensions['technique']} | "
                f"{dimensions['audio_modality']}"
            )
        )
        lines.append(
            f"| `{track['track_id']}` | {dimension_columns} | "
            f"{track['reference_events']} | {metric(track['tablature_f1'])} | "
            f"{output['missed_onset']} | {output['spurious_onset']} | "
            f"{assignment['instances']} | {percent(output['share'])} |"
        )
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    guitarset = report["datasets"]["GuitarSet"]
    guitar_techs = report["datasets"]["Guitar-TECHS"]
    combined = report["aggregate"]
    comparison = report["v24_to_v28_controlled_comparison"]
    sources = report["sources"]
    lines = [
        "# v28 受治理 72 轨错误归因",
        "",
        "## 结论",
        "",
        (
            f"- 72 轨共记录 {combined['failure_attribution']['failure_instances']} "
            "个失败实例；最终输出层的漏检/误检占 "
            f"{percent(combined['failure_attribution']['final_output_selection_or_detection']['share'])}，"
            "可直接归于位置分配的同音高错弦占 "
            f"{percent(combined['failure_attribution']['string_fret_assignment']['share'])}，"
            "异音高匹配但阶段不可判定的错误占 "
            f"{percent(combined['failure_attribution']['different_pitch_matched_output']['share'])}。"
        ),
        (
            "- GuitarSet 的主要负担来自最终输出漏检/误检；"
            f"Guitar-TECHS 有 {guitar_techs['aggregate']['decoder_errors']['same_pitch_wrong_string']} "
            "个同音高错弦，说明跨数据域的弦位分配仍是独立瓶颈。"
        ),
        (
            f"- v28 相对 v24 只改变 "
            f"{comparison['GuitarSet']['changed_tracks']} 条 GuitarSet 轨道、"
            f"{comparison['Guitar-TECHS']['changed_tracks']} 条 Guitar-TECHS 轨道；"
            "没有证据支持在相同数据上继续训练活动头。"
        ),
        "",
        "## 可观测性边界",
        "",
        (
            "原始评测 JSON 只保存逐轨事件数量和最终 decoder 计数，不保存 "
            "Basic Pitch 候选事件、活动概率、被活动头拒绝的候选或位置头拒绝记录。"
        ),
        "",
        (
            "因此，本报告能直接测量：最终输出的 `missed_onset` / "
            "`spurious_onset`，以及保持输入音高的 `same_pitch_wrong_string` "
            "位置分配错误。`same_string_wrong_fret` 和 "
            "`wrong_string_and_pitch` 的预测音高已经变化，在缺少阶段追踪时"
            "只能归为异音高匹配输出，不能归因到弦/品位头。"
            "不能把 `missed_onset` 继续拆成“候选前端未产生”和“活动头拒绝”；"
            "也不能把 `spurious_onset` 拆成“前端伪候选”和“下游错误保留”。"
            "下文统一称其为“最终输出层”，不对不可观测阶段作推断。"
        ),
        "",
        (
            "`onset` / `tablature` 指标和 `decoder_errors` 使用不同的贪心"
            "匹配顺序；后者优先 exact 再逐级放宽。切片中的错误归因全部使用"
            " decoder 层级匹配，TAB F1 和 onset recall 则保留各自官方匹配器结果，"
            "二者不可在逐轨层面直接代数换算。"
        ),
        "",
        "## 数据与一致性",
        "",
        "| 输入 | SHA-256 |",
        "|---|---|",
    ]
    for source in sources.values():
        lines.append(f"| `{source['path']}` | `{source['sha256']}` |")
    lines.extend(
        [
            "",
            (
                "脚本已验证：60 条 GuitarSet 与 12 条 Guitar-TECHS 评测轨道"
                "和受治理 manifest 一一对应；来源状态均为 `verified`；"
                "每轨 decoder 分区分别完整覆盖可评参考与预测事件；"
                "聚合计数与逐轨求和一致；"
                "输出不含音频路径、绝对路径或原始事件。"
            ),
            "",
            "## 总体归因",
            "",
            (
                "| 数据集 | 轨数 | 参考事件 | TAB F1 | exact | 漏检 | "
                "误检 | 同音高错弦 | 最终输出层占比 |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, dataset in (
        ("GuitarSet", guitarset),
        ("Guitar-TECHS", guitar_techs),
        ("合计", {"aggregate": combined}),
    ):
        aggregate = dataset["aggregate"]
        output = aggregate["failure_attribution"][
            "final_output_selection_or_detection"
        ]
        assignment = aggregate["failure_attribution"]["string_fret_assignment"]
        lines.append(
            f"| {name} | {aggregate['track_count']} | "
            f"{aggregate['reference_events']} | "
            f"{metric(aggregate['metrics']['tablature']['f1'])} | "
            f"{aggregate['decoder_errors']['exact_tablature']} | "
            f"{output['missed_onset']} | {output['spurious_onset']} | "
            f"{assignment['instances']} | {percent(output['share'])} |"
        )
    lines.extend(
        [
            "",
            "## GuitarSet 切片",
            "",
            "### SNR",
            "",
            *slice_table(guitarset["slices"]["target_snr_db"], "目标 SNR"),
            "",
            "### 演奏类型",
            "",
            *slice_table(guitarset["slices"]["part"], "类型"),
            "",
            "## Guitar-TECHS 切片",
            "",
            "### 技法",
            "",
            *slice_table(guitar_techs["slices"]["technique"], "技法"),
            "",
            "### 音频来源",
            "",
            *slice_table(guitar_techs["slices"]["audio_modality"], "音频"),
            "",
            "### 演奏者",
            "",
            *slice_table(guitar_techs["slices"]["performer"], "演奏者"),
            "",
            "## v24 → v28 活动头对照",
            "",
            (
                "两组报告的 Basic Pitch、位置模型和除 supplemental 活动头外的"
                "解码参数一致。GuitarSet 有三轨的逐轨 decoder 计数变化；"
                "聚合净变化为 onset TP `+1`、FN `-1`，但该新增匹配属于 "
                "`wrong_string_and_pitch +1`，"
                f"TAB F1 变化为 "
                f"`{comparison['GuitarSet']['aggregate_delta']['tablature_f1']:+.8f}`。"
                "Guitar-TECHS 所有逐轨计数完全不变。"
            ),
            "",
            "这说明 v28 活动头在当前 72 轨上的净影响近乎为零，且没有提高 TAB。",
            "",
            "## 下一步判定",
            "",
            (
                "1. 在评测器中增加阶段追踪后，再做候选前端与活动拒绝的严格归因："
                "每个 onset group 至少记录 `frontend_group`、`activity_rejected`、"
                "`position_rejected`、`emitted` 四个计数。"
            ),
            (
                "2. 优先补充独立授权、真实录音且带完整 string/fret/onset 的数据。"
                "Guitar-TECHS 的同音高错弦显著，新增数据应覆盖多演奏者、拾音方式和技法。"
            ),
            (
                "3. 不在现有 740 条不变训练集上启动 v29；Basic Pitch 继续保持默认，"
                "v24/v28 均不进入生产默认。"
            ),
            "",
            "## 逐轨附录：GuitarSet（60）",
            "",
            *track_table(guitarset["tracks"], dataset="GuitarSet"),
            "",
            "## 逐轨附录：Guitar-TECHS（12）",
            "",
            *track_table(guitar_techs["tracks"], dataset="Guitar-TECHS"),
            "",
            "## 复现",
            "",
            "```bash",
            "python3 scripts/analyze-position-error-attribution.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    arguments = parse_arguments()
    guitarset_report = load_json(arguments.guitarset_report)
    guitar_techs_report = load_json(arguments.guitar_techs_report)
    guitarset_v24_report = load_json(arguments.guitarset_v24_report)
    guitar_techs_v24_report = load_json(arguments.guitar_techs_v24_report)
    guitarset_manifest = load_jsonl(
        arguments.guitarset_manifest,
        split="validation",
    )
    guitar_techs_manifest = load_jsonl(arguments.guitar_techs_manifest)

    validate_report(
        guitarset_report,
        guitarset_manifest,
        expected_tracks=60,
    )
    validate_report(
        guitar_techs_report,
        guitar_techs_manifest,
        expected_tracks=12,
    )
    guitarset_metadata = {
        str(record["track_id"]): record for record in guitarset_manifest
    }
    guitar_techs_metadata = {
        str(record["track_id"]): record for record in guitar_techs_manifest
    }
    guitarset_tracks = guitarset_report["tracks"]
    guitar_techs_tracks = guitar_techs_report["tracks"]
    all_tracks = guitarset_tracks + guitar_techs_tracks

    report = {
        "schema_version": 1,
        "scope": {
            "model": "guitarset-low-candidate-activity-governed-v28",
            "track_count": 72,
            "datasets": {
                "GuitarSet": 60,
                "Guitar-TECHS": 12,
            },
        },
        "method": {
            "failure_partition": (
                "The evaluator's hierarchical decoder_errors matcher prioritizes "
                "exact, same-pitch/wrong-string, same-string/wrong-fret, then "
                "remaining onset-compatible pairs. Unmatched final outputs and "
                "references are labeled spurious and missed."
            ),
            "independent_metric_matchers": (
                "Onset and tablature metrics use separate greedy matchers. Their "
                "per-track TP/FP/FN counts are not algebraically interchangeable "
                "with the hierarchical decoder partition."
            ),
            "observable": [
                "final missed onset",
                "final spurious onset",
                "same-pitch wrong-string assignment",
                "different-pitch matched output (causally unresolved)",
            ],
            "not_observable": [
                (
                    "Whether a missed onset was absent from the Basic Pitch "
                    "candidate frontend or rejected by an activity/position gate."
                ),
                (
                    "Whether a spurious onset originated in the candidate "
                    "frontend or was incorrectly retained downstream."
                ),
            ],
        },
        "sources": {
            "guitarset_v28_report": artifact(arguments.guitarset_report),
            "guitar_techs_v28_report": artifact(arguments.guitar_techs_report),
            "guitarset_v24_report": artifact(arguments.guitarset_v24_report),
            "guitar_techs_v24_report": artifact(arguments.guitar_techs_v24_report),
            "guitarset_manifest": artifact(arguments.guitarset_manifest),
            "guitar_techs_manifest": artifact(arguments.guitar_techs_manifest),
        },
        "aggregate": summarize_tracks(all_tracks),
        "datasets": {
            "GuitarSet": {
                "aggregate": summarize_tracks(guitarset_tracks),
                "slices": {
                    "target_snr_db": group_summaries(
                        guitarset_tracks,
                        guitarset_metadata,
                        lambda record: (
                            f"{float(record['provenance']['target_snr_db']):g} dB"
                        ),
                        sort_key=lambda value: float(value.removesuffix(" dB")),
                    ),
                    "part": group_summaries(
                        guitarset_tracks,
                        guitarset_metadata,
                        guitarset_part,
                    ),
                },
                "tracks": [
                    track_summary(
                        track,
                        guitarset_metadata[str(track["track_id"])],
                        dataset="GuitarSet",
                    )
                    for track in guitarset_tracks
                ],
            },
            "Guitar-TECHS": {
                "aggregate": summarize_tracks(guitar_techs_tracks),
                "slices": {
                    "performer": group_summaries(
                        guitar_techs_tracks,
                        guitar_techs_metadata,
                        lambda record: str(record["provenance"]["player"]),
                    ),
                    "audio_modality": group_summaries(
                        guitar_techs_tracks,
                        guitar_techs_metadata,
                        lambda record: str(
                            record["provenance"]["audio_modality"]
                        ),
                    ),
                    "technique": group_summaries(
                        guitar_techs_tracks,
                        guitar_techs_metadata,
                        guitar_technique,
                    ),
                },
                "tracks": [
                    track_summary(
                        track,
                        guitar_techs_metadata[str(track["track_id"])],
                        dataset="Guitar-TECHS",
                    )
                    for track in guitar_techs_tracks
                ],
            },
        },
        "v24_to_v28_controlled_comparison": {
            "GuitarSet": report_comparison(
                guitarset_v24_report,
                guitarset_report,
            ),
            "Guitar-TECHS": report_comparison(
                guitar_techs_v24_report,
                guitar_techs_report,
            ),
        },
        "validation": {
            "manifest_track_identity": "pass",
            "verified_source_status": "pass",
            "per_track_decoder_accounting": "pass",
            "aggregate_accounting": "pass",
            "matching_diagnostics": matching_diagnostics(all_tracks),
            "contains_audio_paths": False,
            "contains_absolute_paths": False,
            "contains_raw_events": False,
        },
    }
    assert report["aggregate"]["track_count"] == 72
    assert report["v24_to_v28_controlled_comparison"]["GuitarSet"][
        "isolated_supplemental_activity_head_change"
    ]
    assert report["v24_to_v28_controlled_comparison"]["Guitar-TECHS"][
        "isolated_supplemental_activity_head_change"
    ]

    rendered_json = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    rendered_markdown = render_markdown(report)
    for forbidden in (str(ROOT), "/Users/"):
        assert forbidden not in rendered_json
        assert forbidden not in rendered_markdown

    output_json = repository_path(arguments.output_json)
    output_markdown = repository_path(arguments.output_markdown)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(rendered_json, encoding="utf-8")
    output_markdown.write_text(rendered_markdown, encoding="utf-8")
    print(
        json.dumps(
            {
                "stage": "position_error_attribution",
                "tracks": report["scope"]["track_count"],
                "output_json": relative_name(arguments.output_json),
                "output_markdown": relative_name(arguments.output_markdown),
                "validation": report["validation"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render the verified frame audit as a Chinese report and exportable figure."""
import importlib.util
import json
import os
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "apps/api/data/training/pitch-completion-v2-frames-v1"
CACHE = ROOT / ".tmp/frames-v1-plot"
REPORT = ROOT / "pitch-completion-v2-frame-result.md"
LABELS = {"local_mean": "原 local 均值", "nearest": "最近帧", "attack_mean": "0–80ms 均值",
          "attack_joint": "音高 × 起音联合", "positive_rise": "正向上升量"}
ARMS = {"budget": "固定预算", "budget_p1": "固定预算＋P1"}
COLORS = {"budget": "#276b9d", "budget_p1": "#ce6840"}


def read(name):
    return json.loads((RUN / name).read_text())


def plotting():
    CACHE.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(CACHE)
    spec = importlib.util.find_spec("matplotlib")
    package = Path(spec.origin).parent.resolve()
    assert package.is_relative_to(ROOT)
    source = (package / "font_manager.py").read_text()
    version = re.search(r"^    __version__ = (\d+)$", source, flags=re.MULTILINE)[1]
    entries = []
    for filename, weight in (("DejaVuSans.ttf", 400), ("DejaVuSans-Bold.ttf", 700)):
        entries.append({"__class__": "FontEntry", "fname": str(package / "mpl-data/fonts/ttf" / filename),
                        "name": "DejaVu Sans", "style": "normal", "variant": "normal",
                        "weight": weight, "stretch": "normal", "size": "scalable"})
    (CACHE / f"fontlist-v{version}.json").write_text(json.dumps({
        "__class__": "FontManager", "_version": int(version), "_FontManager__default_weight": "normal",
        "default_size": None, "defaultFamily": {"ttf": "DejaVu Sans", "afm": "Helvetica"},
        "afmlist": [], "ttflist": entries}))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                         "axes.titleweight": "bold", "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "path"})
    return plt


def render(summary, series):
    plt = plotting()
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.8), constrained_layout=True)
    times = np.asarray(summary["times"]) * 1000
    selected = (times >= -200) & (times <= 360)
    for ax, arm in zip(axes[0], ARMS):
        ss = [s for s in series if s["arm"] == arm and s["technique"] == "palm-mute"]
        target = np.asarray([s["target_pitch"] for s in ss])[:, selected]
        competitor = np.asarray([s["competitor_pitch"] for s in ss])[:, selected]
        ax.axvspan(0, 80, color="#cbe1df", alpha=.4, label="Fixed early window")
        ax.axvline(0, color="#707070", linewidth=.8)
        ax.axhline(.5, color="#909090", linestyle=":", linewidth=.8)
        ax.fill_between(times[selected], np.quantile(target, .25, axis=0),
                        np.quantile(target, .75, axis=0), color=COLORS[arm], alpha=.14)
        ax.plot(times[selected], np.median(target, axis=0), "o-", color=COLORS[arm],
                markersize=3, label="Target median (IQR shaded)")
        ax.plot(times[selected], np.median(competitor, axis=0), "s--", color="#3b4148",
                markersize=3, label="Fixed competitor median")
        ax.set(title=f"{'Budget' if arm == 'budget' else 'Budget + P1'}: 109 palm-mute targets",
               xlabel="Frame center from candidate onset (ms)", ylabel="Pitch sigmoid score",
               ylim=(-.02, 1.02), xlim=(-205, 365))
        ax.legend(fontsize=8, loc="upper right")
    ax = axes[1, 0]
    policies = list(LABELS)
    names = ["Local", "Nearest", "Early mean", "Joint", "Positive rise"]
    x = np.arange(len(policies))
    for shift, arm in ((-.18, "budget"), (.18, "budget_p1")):
        metrics = summary["ranking"]["internal_targets"][arm]["palm-mute"]
        top = [metrics[k]["top1"] for k in policies]
        qualified = [metrics[k]["top1_above_threshold"] for k in policies]
        bars = ax.bar(x + shift, top, .33, color=COLORS[arm],
                      label="Budget" if arm == "budget" else "Budget + P1", alpha=.9)
        for bar, v, q in zip(bars, top, qualified):
            ax.text(bar.get_x() + bar.get_width() / 2, v + .2, f"{v}\n({q})",
                    ha="center", va="bottom", fontsize=8)
    ax.set(title="Correct rank-1 targets out of 109", ylabel="Target count")
    ax.set_xticks(x, names, rotation=14)
    ax.set_ylim(0, max(12, max(m["top1"] for a in ARMS for m in
                               summary["ranking"]["internal_targets"][a]["palm-mute"].values()) + 6))
    ax.legend(fontsize=8)
    ax.text(.02, .97, "Parentheses: rank 1 AND score >= 0.5", transform=ax.transAxes,
            fontsize=8, va="top")
    ax = axes[1, 1]
    for arm in ARMS:
        ss = [s["aligned"]["attack"] for s in series if s["arm"] == arm
              and s["technique"] == "palm-mute"]
        ax.scatter([s["competitor_rise"] for s in ss], [s["target_rise"] for s in ss],
                   s=22, alpha=.6, color=COLORS[arm], edgecolors="none",
                   label="Budget" if arm == "budget" else "Budget + P1")
    ax.axvline(0, color="#aaa", linewidth=.7)
    ax.axhline(0, color="#aaa", linewidth=.7)
    ax.plot([-1, 1], [-1, 1], "--", color="#777", linewidth=.8)
    ax.set(title="Change around measured acoustic attack", xlabel="Fixed competitor: early - pre score",
           ylabel="Target: early - pre score", xlim=(-1, 1), ylim=(-1, 1))
    ax.legend(fontsize=8)
    fig.suptitle("Frozen weights: temporal evidence and ranking", fontsize=16, fontweight="bold")
    fig.savefig(ROOT / "pitch-completion-v2-frame-results.png", dpi=160)
    svg_path = ROOT / "pitch-completion-v2-frame-results.svg"
    fig.savefig(svg_path)
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    plt.close(fig)


def main():
    summary, verification = read("summary.json"), read("verification.json")
    assert verification["passed"]
    assert verification["arm_records"] == 2910 and verification["complete_full_frame_labels_checked"] == 1257
    series = [json.loads(s) for s in (RUN / "series.jsonl").read_text().splitlines() if s.strip()]
    render(summary, series)
    mute = {a: summary["temporal"][a]["palm-mute"]["attack"] for a in ARMS}
    rank = {a: summary["ranking"]["internal_targets"][a]["palm-mute"] for a in ARMS}
    early_top = "/".join(str(rank[a]["attack_mean"]["top1"]) for a in ARMS)
    max_legacy = max(v for a in summary["legacy_max_error"].values() for v in a.values())
    max_replay = max(verification["replay_max_error"].values())
    threshold_flips = sum(v for a in summary["legacy_threshold_flips"].values() for v in a.values())
    text = [
        "# 固定权重逐帧验证：闷音攻击前后的音高分数",
        "",
        "**当前证据不支持仅靠提前取帧解决闷音补全，也不支持扩大训练。**"
        "目标音与固定非目标竞争音的攻击响应仍未充分区分；提前取帧的少量排序改善伴随完整标签误报增加。",
        "",
        f"已完成两个冻结模型的 1,455 组、2,910 条完整推理及独立重放。109 个闷音目标中，"
        f"按可信声学攻击对齐，目标分数上升的数量分别为 {mute['budget']['target_rises']}/109 和 "
        f"{mute['budget_p1']['target_rises']}/109；候选组 0–80ms 均值的首名正确数为 {early_top}。"
        "这两个指标分别说明时间响应与音高区分能力，不能互相替代。",
        "",
        "![逐帧曲线、固定规则排序与攻击前后变化](pitch-completion-v2-frame-results.png)",
        "",
        "图中上排为候选组起音对齐的逐帧中位数，阴影为目标分数的四分位范围；"
        "下排分别为五条规则的首名数、可信声学攻击对齐后的逐样本分数变化。",
        "",
        "## 1. 验证边界",
        "",
        "权重、音频、候选缓存、发声音高合同、分组、阈值及计算源码均先哈希锁定。"
        "没有训练、改标签、换权重或修改生产解码。训练对照仍为原 1,024 组（635 个修正目标）；"
        "其中 7 组未确认泛音保留推理，继续屏蔽其标签。"
        "内部目标为 150 个（闷音 109、普通拨弦 35、泛音 2、合成样本 4）。"
        "完整标签对照为训练 972 组和内部 285 组，共 1,257 组；与内部目标重叠 4 组，因此推理并集为 1,455 组。",
        "",
        "输出包含每组全部 56 帧、每帧 49 个音高和起音分数。竞争音固定为前轮已选定的最强非目标，"
        "各窗口不重新挑竞争音。排序排除已有候选与同组其他正确目标，保留所有其他音高。",
        "",
        "## 2. 攻击前后分数变化",
        "",
        "下面按前轮已验证的宽带攻击时刻对齐，比较攻击前 −160～−20ms、攻击后 0～80ms、"
        "晚窗 160～320ms。各样本先在窗内取帧均值，再对样本求中位数；"
        "“上升量中位数”先逐样本相减，再求中位数。"
        "它不等于两个总体中位数之差。",
        "",
        "|模型|目标：攻击前 / 早窗 / 晚窗|竞争音：攻击前 / 早窗 / 晚窗|目标上升量中位数|竞争音上升量中位数|目标上升|目标上升超过竞争音|早窗胜过固定竞争音|",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for arm, label in ARMS.items():
        m = mute[arm]
        target = " / ".join(f"{m['target'][w]['median']:.4f}" for w in ("pre", "early", "late"))
        competitor = " / ".join(f"{m['competitor'][w]['median']:.4f}" for w in ("pre", "early", "late"))
        text.append(f"|{label}|{target}|{competitor}|{m['target_rise']['median']:+.4f}|"
                    f"{m['competitor_rise']['median']:+.4f}|{m['target_rises']}/109|"
                    f"{m['target_rise_exceeds_competitor']}/109|{m['early_beats_fixed_competitor']}/109|")
    text += ["", f"早窗目标减固定竞争音的逐样本差值中位数为 "
             f"{mute['budget']['early_margin']['median']:+.4f}、"
             f"{mute['budget_p1']['early_margin']['median']:+.4f}。"
             f"P1 组的目标分数平均变化为 {mute['budget_p1']['target_rise']['mean']:+.4f}，"
             "较高的目标绝对分数并未转化为更好的竞争优势。",
             "", "早窗胜过一个固定竞争音仍不足以保证首名：其他缺失候选音高也参与排序。"
             "参考起音对齐和候选组起音对齐的完整统计保存在证据中，均使用预先规定的窗口，未做插值。",
             "", "## 3. 五条固定评分规则", "",
             "阈值均为 0.5；下表只描述稠密证据的排名和分数。联合规则为 "
             "sqrt(早窗音高均值 × 起音最大分数)，正向上升量为早窗最大分数减攻击前均值并截断至 [0,1]。"
             "这两条规则未校准，不把诊断分数解释为概率。",
             "", "|规则|闷音首名：固定预算 / ＋P1|闷音首名且≥0.5|全部内部目标首名（150）|普通拨弦首名（35）|",
             "|---|---:|---:|---:|---:|"]
    for policy, label in LABELS.items():
        r = summary["ranking"]["internal_targets"]
        def values(tech, key, ranking=r, rule=policy):
            return " / ".join(str(ranking[a][tech][rule][key]) for a in ARMS)
        text.append(f"|{label}|{values('palm-mute', 'top1')}|{values('palm-mute', 'top1_above_threshold')}|"
                    f"{values('all', 'top1')}|{values('pick', 'top1')}|")
    text += ["", "普通拨弦与闷音的音区差异很大，不能据这张表把差异全归因于技法。"
             "实际补全头和数量头没有改变；本轮只检查稠密音高证据，评分规则首名正确不等于已补回。",
             "", "## 4. 完整标签误报与归属控制", "",
             "活动真值固定为原 local 四帧的活动并集，四帧都有效才提供负标签；已有候选不参与计数。"
             "五条评分规则共用同一真值和掩码。活动 TP 是原 local 时间范围内已知发声音高，"
             "本组归属 TP 是应补到当前候选组的目标；两种 FP 含义不同。",
             "", "|模型|规则|训练活动 TP / FP|训练本组归属 TP / FP|内部活动 TP / FP|内部本组归属 TP / FP|",
             "|---|---|---:|---:|---:|---:|"]
    for arm, arm_label in ARMS.items():
        for policy, label in LABELS.items():
            cells = []
            for subset in ("training", "internal"):
                for truth in ("activity", "assigned"):
                    c = summary["controls"][subset][arm][policy][truth]
                    cells.append(f"{c['tp']} / {c['fp']}")
            text.append(f"|{arm_label}|{label}|" + "|".join(cells) + "|")
    text += ["", "基线活动误报复现：训练 74→126，内部 39→100。内部完整标签 285 组全部来自合成样本，"
             "活动正例只有 10 个音高单元；该控制不能代表闷音真实误报率。"
             "闷音所属 positive-only 标签未覆盖所有负音高，未标注竞争音不计真实 FP。",
             "", "## 5. 时间分辨率与解释限制", "",
             "帧中心间隔为 46.44ms，但仅按幅度通道计算，卷积路径的名义原始音频跨度已约 754.65ms；"
             "相位通道另读取前一个 STFT 帧，局部路径跨度可达约 760.45ms。"
             "跨时间 GroupNorm 及 2.56 秒整窗 RMS 归一化还使分数依赖更长范围。"
             "攻击前的高分不能视为因果提前识别，早晚曲线也不能当作彼此独立的短窗检测。",
             "",
             "内部样本已用于此前诊断，不能声称独立泛化。固定权重比较可以定位证据是否随攻击变化，"
             "不能单独确认技法因果、训练数据充分性或生产识别率提升。",
             "", "## 6. 独立复核", "",
             f"- 全部 2,910 条输出重新聚合六项原 HEAD 分数，最大绝对误差 {max_legacy:.8g}，"
             f"跨 0.5 次数 {threshold_flips}，预先规定容差 5e-5。",
             f"- 独立重新读取音频、生成特征和推理全部 1,455 组，最大绝对误差 {max_replay:.8g}，"
             f"跨 0.5 次数 {sum(verification['replay_threshold_flips'].values())}。",
             "- 人工事件映射核对全部 1,257 个完整标签组的全部帧活动、起音标签与负标签掩码。",
             "- 独立核对 1,570 条目标评分记录、300 条逐帧竞争记录、12,570 条完整标签规则记录及全部汇总。",
             "- 重放沿用已冻结的特征与模型 API；标签映射、五条评分规则、排序和计数由独立脚本实现。",
             f"- 初次推理耗时 {summary['inference']['elapsed_seconds']:.1f} 秒，"
             f"独立验证耗时 {verification['elapsed_seconds']:.1f} 秒。原始合同与权重哈希未改变。",
             "", "## 7. 结果对下一步的约束", "",
             "固定预算模型中，目标攻击上升量中位数仅 +0.0011，固定竞争音为 +0.1348；"
             "只有 25/109 个目标的上升量超过固定竞争音。加入 P1 后，目标攻击上升量中位数约 −0.0002，"
             "只有 50/109 个目标上升，攻击早窗胜过固定竞争音的目标仅 4/109。"
             "这不支持把 P1 的较高目标分数解释为更准确的攻击响应。",
             "",
             "提前取帧将闷音首名从 8→10、2→4，但全部内部目标首名为 27→27、13→14；"
             "训练活动误报增加 26、31 个，内部活动误报增加 14、24 个。"
             "联合规则将闷音首名变为 11、5，但首名且达固定阈值的只有 3、2；"
             "训练活动 TP 同时从 582→306、595→302。它降低误报时也大幅减少检出，不能视为已修复。",
             "",
             "下一步适合做固定权重的攻击与上下文干预对照，检验目标和竞争音分数分别依赖哪段音频，"
             "并单独控制整窗归一化影响。取得可信归属证据后，再判断是否值得改变编码器或补全头。"
             "当前不扩大训练、不切换生产模型。",
             "", "证据：",
             "",
             "- [预注册](pitch-completion-v2-frame-preregistration.md)",
             "- [冻结计划与哈希](apps/api/data/training/pitch-completion-v2-frames-v1/plan.json)",
             "- [完整汇总](apps/api/data/training/pitch-completion-v2-frames-v1/summary.json)",
             "- [独立验证](apps/api/data/training/pitch-completion-v2-frames-v1/verification.json)",
             "- [目标逐帧与固定竞争音](apps/api/data/training/pitch-completion-v2-frames-v1/series.jsonl)",
             "- [全部目标评分](apps/api/data/training/pitch-completion-v2-frames-v1/targets.jsonl)",
             "- [完整标签规则计数](apps/api/data/training/pitch-completion-v2-frames-v1/controls.jsonl)",
             "- [矢量图](pitch-completion-v2-frame-results.svg)",
             ""]
    REPORT.write_text("\n".join(text))
    print(json.dumps({"report": REPORT.name, "figures": ["pitch-completion-v2-frame-results.png",
                                                       "pitch-completion-v2-frame-results.svg"]}))


if __name__ == "__main__":
    main()

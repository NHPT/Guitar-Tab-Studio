# v28 受治理 72 轨错误归因

## 结论

- 72 轨共记录 6193 个失败实例；最终输出层的漏检/误检占 74.1%，可直接归于位置分配的同音高错弦占 13.2%，异音高匹配但阶段不可判定的错误占 12.8%。
- GuitarSet 的主要负担来自最终输出漏检/误检；Guitar-TECHS 有 322 个同音高错弦，说明跨数据域的弦位分配仍是独立瓶颈。
- v28 相对 v24 只改变 3 条 GuitarSet 轨道、0 条 Guitar-TECHS 轨道；没有证据支持在相同数据上继续训练活动头。

## 可观测性边界

重建后的评测 JSON 已保存逐轨和聚合阶段计数：候选事件/组、活动门控接受/拒绝、位置阶段有输出/整组无输出、置信度拒绝、合并去重和最终输出。

这些计数能直接定位候选流量在哪一阶段减少，但仍不保存候选事件身份、活动概率或候选与参考事件的逐事件匹配。因此仍不能把某个 `missed_onset` 严格指定为“前端未产生”“活动拒绝”或“位置拒绝”，也不能把某个 `spurious_onset` 反向指定到其来源阶段。`same_pitch_wrong_string` 仍是可直接归因的位置分配错误；`same_string_wrong_fret` 和 `wrong_string_and_pitch` 继续归为异音高匹配输出，不作超出证据的因果判断。

`onset` / `tablature` 指标和 `decoder_errors` 使用不同的贪心匹配顺序；后者优先 exact 再逐级放宽。切片中的错误归因全部使用 decoder 层级匹配，TAB F1 和 onset recall 则保留各自官方匹配器结果，二者不可在逐轨层面直接代数换算。

## 数据与一致性

| 输入 | SHA-256 |
|---|---|
| `apps/api/data/training/reports/governed-v28-guitarset-mixture-validation.json` | `935cf9223dac61a6c4e785e9dfe218864a74fcec1c46bdc3b94c59bd51a9362a` |
| `apps/api/data/training/reports/governed-v28-guitar-techs-validation.json` | `d674358550e5165b6f5f8448fe52b50600e0a45db0bda0efa4a0712cbcc79409` |
| `apps/api/data/training/reports/governed-v24-guitarset-mixture-validation.json` | `d38f900f31aa407951065bb2a4c0105288ccf09b30de70c02055acc5981ddee6` |
| `apps/api/data/training/reports/governed-v24-guitar-techs-validation.json` | `38dac7222e0318c081181e82e04cf2ea829accdc885f8209592e826f23db8e5a` |
| `apps/api/data/training/governed-v1/guitarset-mixture-derived.jsonl` | `a90668d480af274c579891990e7192f48b6b604914be33d0d85078f5e648fbda` |
| `apps/api/data/training/governed-v1/guitar-techs-position-validation.jsonl` | `6f796466be2cae19783c177b4a821389a1f1fee3a7d1346e98dd1c4761540a4b` |

脚本已验证：60 条 GuitarSet 与 12 条 Guitar-TECHS 评测轨道和受治理 manifest 一一对应；来源状态均为 `verified`；每轨 decoder 分区分别完整覆盖可评参考与预测事件；阶段计数守恒且与逐轨求和一致；输出不含音频路径、绝对路径或原始事件。

## 阶段追踪证据

| 路径 | 候选事件 | 候选组 | 活动拒绝组 | 位置整组拒绝 | 位置输出 |
|---|---:|---:|---:|---:|---:|
| 主前端 | 22666 | 12207 | 3580 | 2696 | 10668 |
| 补充前端 | 35548 | 18213 | 13046 | 1053 | 7083 |

主路径输出 10668 个事件。补充路径输出 7083 个事件，但合并时 6898 个与主路径重复，仅新增 185 个，最终输出 10853 个事件。当前 `minimum_output_confidence=0`，所以置信度阶段没有拒绝事件。

Guitar-TECHS 主路径有 2208 组通过活动门控，其中 1570 组在位置阶段没有任何输出。该比例明显高于 GuitarSet，与跨录音域的位置模型失配一致，但不是对单个漏检的因果证明。

## 总体归因

| 数据集 | 轨数 | 参考事件 | TAB F1 | exact | 漏检 | 误检 | 同音高错弦 | 最终输出层占比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GuitarSet | 60 | 10253 | 0.6799 | 6943 | 2036 | 1955 | 495 | 75.8% |
| Guitar-TECHS | 12 | 1090 | 0.2880 | 255 | 502 | 93 | 322 | 64.1% |
| 合计 | 72 | 11343 | 0.6486 | 7198 | 2538 | 2048 | 817 | 74.1% |

## GuitarSet 切片

### SNR

| 目标 SNR | 轨数 | 参考事件 | onset recall | TAB F1 | 漏检 | 误检 | 同音高错弦 | 最终输出层占比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| -6 dB | 18 | 2363 | 0.7431 | 0.6499 | 609 | 376 | 120 | 77.0% |
| -3 dB | 15 | 3131 | 0.8141 | 0.6893 | 580 | 417 | 206 | 68.9% |
| 0 dB | 9 | 1183 | 0.7084 | 0.5848 | 346 | 237 | 49 | 76.7% |
| 3 dB | 7 | 1221 | 0.8600 | 0.6589 | 171 | 552 | 44 | 85.8% |
| 6 dB | 11 | 2355 | 0.8594 | 0.7536 | 330 | 373 | 76 | 75.0% |

### 演奏类型

| 类型 | 轨数 | 参考事件 | onset recall | TAB F1 | 漏检 | 误检 | 同音高错弦 | 最终输出层占比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| comp | 30 | 7669 | 0.7864 | 0.6523 | 1637 | 1641 | 321 | 76.1% |
| solo | 30 | 2584 | 0.8460 | 0.7629 | 399 | 314 | 174 | 74.3% |

## Guitar-TECHS 切片

### 技法

| 技法 | 轨数 | 参考事件 | onset recall | TAB F1 | 漏检 | 误检 | 同音高错弦 | 最终输出层占比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bend | 2 | 58 | 0.5345 | 0.4576 | 27 | 29 | 4 | 93.3% |
| harmonic | 2 | 50 | 0.3200 | 0.1176 | 34 | 2 | 10 | 75.0% |
| palm-mute | 2 | 268 | 0.6381 | 0.3465 | 97 | 17 | 92 | 55.3% |
| pick | 2 | 256 | 0.6289 | 0.3303 | 95 | 19 | 89 | 56.2% |
| pinch-harmonic | 2 | 238 | 0.3529 | 0.1877 | 154 | 19 | 43 | 76.9% |
| vibrato | 2 | 220 | 0.5682 | 0.2330 | 95 | 7 | 84 | 54.8% |

### 音频来源

| 音频 | 轨数 | 参考事件 | onset recall | TAB F1 | 漏检 | 误检 | 同音高错弦 | 最终输出层占比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| directinput | 6 | 545 | 0.5505 | 0.2864 | 245 | 42 | 167 | 62.4% |
| micamp | 6 | 545 | 0.5284 | 0.2896 | 257 | 51 | 155 | 65.8% |

### 演奏者

| 演奏者 | 轨数 | 参考事件 | onset recall | TAB F1 | 漏检 | 误检 | 同音高错弦 | 最终输出层占比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P2 | 12 | 1090 | 0.5394 | 0.2880 | 502 | 93 | 322 | 64.1% |

## v24 → v28 活动头对照

两组报告的 Basic Pitch、位置模型和除 supplemental 活动头外的解码参数一致。GuitarSet 有三轨的逐轨 decoder 计数变化；聚合净变化为 onset TP `+1`、FN `-1`，但该新增匹配属于 `wrong_string_and_pitch +1`，TAB F1 变化为 `-0.00003329`。Guitar-TECHS 所有逐轨计数完全不变。

阶段追踪显示，v28 相对 v24 在 72 轨上少接受 92 个补充活动组、少产生 49 个位置输出，但去重后独有补充事件净变化为 +1。这说明 v28 的额外筛选主要移除了原本就会被主路径去重的事件，对最终输出和 TAB 没有实质改善。

## 下一步判定

1. 阶段计数已实现。若要把最终漏检严格分解到前端、活动门控和位置阶段，下一版需要为每个候选保留稳定身份并与参考事件逐阶段匹配；不能用当前聚合计数反推单个错误。
2. 优先补充独立授权、真实录音且带完整 string/fret/onset 的数据。Guitar-TECHS 的同音高错弦显著，新增数据应覆盖多演奏者、拾音方式和技法。
3. 不在现有 740 条不变训练集上启动 v29；Basic Pitch 继续保持默认，v24/v28 均不进入生产默认。

## 逐轨附录：GuitarSet（60）

| track_id | SNR | 类型 | 参考事件 | TAB F1 | 漏检 | 误检 | 同音高错弦 | 最终输出层占比 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `04_BN1-129-Eb_comp-mixture-demucs-guitar` | 6 dB | comp | 127 | 0.7619 | 3 | 22 | 1 | 55.6% |
| `04_BN1-129-Eb_solo-mixture-demucs-guitar` | -6 dB | solo | 42 | 0.8395 | 6 | 3 | 0 | 81.8% |
| `04_BN1-147-Gb_comp-mixture-demucs-guitar` | 6 dB | comp | 169 | 0.8190 | 26 | 3 | 0 | 67.4% |
| `04_BN1-147-Gb_solo-mixture-demucs-guitar` | 0 dB | solo | 62 | 0.5714 | 14 | 9 | 13 | 62.2% |
| `04_BN2-131-B_comp-mixture-demucs-guitar` | -3 dB | comp | 206 | 0.6765 | 23 | 19 | 15 | 48.3% |
| `04_BN2-131-B_solo-mixture-demucs-guitar` | -3 dB | solo | 67 | 0.7770 | 9 | 14 | 3 | 85.2% |
| `04_BN2-166-Ab_comp-mixture-demucs-guitar` | 0 dB | comp | 219 | 0.6784 | 52 | 12 | 9 | 66.7% |
| `04_BN2-166-Ab_solo-mixture-demucs-guitar` | -3 dB | solo | 48 | 0.7872 | 7 | 5 | 2 | 75.0% |
| `04_BN3-119-G_comp-mixture-demucs-guitar` | -3 dB | comp | 172 | 0.7588 | 10 | 35 | 9 | 67.2% |
| `04_BN3-119-G_solo-mixture-demucs-guitar` | -6 dB | solo | 69 | 0.4789 | 12 | 16 | 21 | 54.9% |
| `04_BN3-154-E_comp-mixture-demucs-guitar` | 0 dB | comp | 163 | 0.1920 | 80 | 77 | 2 | 75.1% |
| `04_BN3-154-E_solo-mixture-demucs-guitar` | 0 dB | solo | 84 | 0.7886 | 10 | 17 | 4 | 84.4% |
| `04_Funk1-114-Ab_comp-mixture-demucs-guitar` | 0 dB | comp | 232 | 0.7732 | 39 | 38 | 2 | 84.6% |
| `04_Funk1-114-Ab_solo-mixture-demucs-guitar` | 0 dB | solo | 102 | 0.7303 | 33 | 7 | 1 | 90.9% |
| `04_Funk1-97-C_comp-mixture-demucs-guitar` | 0 dB | comp | 99 | 0.5969 | 22 | 15 | 15 | 64.9% |
| `04_Funk1-97-C_solo-mixture-demucs-guitar` | -3 dB | solo | 111 | 0.6321 | 20 | 10 | 23 | 55.6% |
| `04_Funk2-108-Eb_comp-mixture-demucs-guitar` | -3 dB | comp | 329 | 0.6555 | 85 | 22 | 20 | 68.6% |
| `04_Funk2-108-Eb_solo-mixture-demucs-guitar` | 6 dB | solo | 119 | 0.8151 | 12 | 12 | 9 | 70.6% |
| `04_Funk2-119-G_comp-mixture-demucs-guitar` | -3 dB | comp | 352 | 0.7315 | 67 | 52 | 7 | 78.3% |
| `04_Funk2-119-G_solo-mixture-demucs-guitar` | 3 dB | solo | 124 | 0.7718 | 22 | 15 | 8 | 80.4% |
| `04_Funk3-112-C#_comp-mixture-demucs-guitar` | 6 dB | comp | 395 | 0.6639 | 93 | 29 | 25 | 66.7% |
| `04_Funk3-112-C#_solo-mixture-demucs-guitar` | -3 dB | solo | 90 | 0.7586 | 15 | 9 | 8 | 72.7% |
| `04_Funk3-98-A_comp-mixture-demucs-guitar` | -3 dB | comp | 404 | 0.6925 | 75 | 41 | 21 | 65.5% |
| `04_Funk3-98-A_solo-mixture-demucs-guitar` | -6 dB | solo | 115 | 0.7860 | 17 | 16 | 4 | 80.5% |
| `04_Jazz1-130-D_comp-mixture-demucs-guitar` | -6 dB | comp | 206 | 0.5948 | 76 | 7 | 5 | 74.8% |
| `04_Jazz1-130-D_solo-mixture-demucs-guitar` | -6 dB | solo | 55 | 0.7925 | 10 | 6 | 0 | 84.2% |
| `04_Jazz1-200-B_comp-mixture-demucs-guitar` | 0 dB | comp | 151 | 0.2574 | 85 | 55 | 0 | 81.9% |
| `04_Jazz1-200-B_solo-mixture-demucs-guitar` | 6 dB | solo | 65 | 0.6929 | 12 | 9 | 9 | 70.0% |
| `04_Jazz2-110-Bb_comp-mixture-demucs-guitar` | 3 dB | comp | 133 | 0.7747 | 22 | 9 | 8 | 70.5% |
| `04_Jazz2-110-Bb_solo-mixture-demucs-guitar` | -6 dB | solo | 78 | 0.8153 | 11 | 12 | 0 | 88.5% |
| `04_Jazz2-187-F#_comp-mixture-demucs-guitar` | -3 dB | comp | 171 | 0.4595 | 49 | 16 | 28 | 56.0% |
| `04_Jazz2-187-F#_solo-mixture-demucs-guitar` | -6 dB | solo | 66 | 0.7778 | 11 | 5 | 6 | 72.7% |
| `04_Jazz3-137-Eb_comp-mixture-demucs-guitar` | -6 dB | comp | 229 | 0.7306 | 47 | 27 | 2 | 77.1% |
| `04_Jazz3-137-Eb_solo-mixture-demucs-guitar` | -6 dB | solo | 79 | 0.6923 | 15 | 13 | 3 | 73.7% |
| `04_Jazz3-150-C_comp-mixture-demucs-guitar` | 3 dB | comp | 180 | 0.7221 | 32 | 21 | 6 | 70.7% |
| `04_Jazz3-150-C_solo-mixture-demucs-guitar` | -6 dB | solo | 88 | 0.8187 | 13 | 8 | 3 | 80.8% |
| `04_Rock1-130-A_comp-mixture-demucs-guitar` | -3 dB | comp | 190 | 0.3977 | 44 | 16 | 59 | 44.1% |
| `04_Rock1-130-A_solo-mixture-demucs-guitar` | -6 dB | solo | 94 | 0.6864 | 28 | 9 | 0 | 82.2% |
| `04_Rock1-90-C#_comp-mixture-demucs-guitar` | 6 dB | comp | 238 | 0.8214 | 29 | 40 | 0 | 88.5% |
| `04_Rock1-90-C#_solo-mixture-demucs-guitar` | -3 dB | solo | 74 | 0.8105 | 6 | 11 | 3 | 73.9% |
| `04_Rock2-142-D_comp-mixture-demucs-guitar` | 6 dB | comp | 353 | 0.7660 | 45 | 44 | 7 | 70.1% |
| `04_Rock2-142-D_solo-mixture-demucs-guitar` | -3 dB | solo | 72 | 0.8028 | 9 | 7 | 3 | 72.7% |
| `04_Rock2-85-F_comp-mixture-demucs-guitar` | 3 dB | comp | 126 | 0.6275 | 7 | 112 | 2 | 94.4% |
| `04_Rock2-85-F_solo-mixture-demucs-guitar` | 3 dB | solo | 111 | 0.8532 | 13 | 9 | 1 | 81.5% |
| `04_Rock3-117-Bb_comp-mixture-demucs-guitar` | 6 dB | comp | 509 | 0.7606 | 65 | 83 | 17 | 74.7% |
| `04_Rock3-117-Bb_solo-mixture-demucs-guitar` | -6 dB | solo | 107 | 0.7451 | 18 | 8 | 6 | 66.7% |
| `04_Rock3-148-C_comp-mixture-demucs-guitar` | -6 dB | comp | 374 | 0.6364 | 132 | 22 | 7 | 79.8% |
| `04_Rock3-148-C_solo-mixture-demucs-guitar` | 0 dB | solo | 71 | 0.7971 | 11 | 7 | 3 | 78.3% |
| `04_SS1-100-C#_comp-mixture-demucs-guitar` | 6 dB | comp | 162 | 0.6078 | 24 | 108 | 2 | 90.4% |
| `04_SS1-100-C#_solo-mixture-demucs-guitar` | -6 dB | solo | 71 | 0.7832 | 8 | 9 | 3 | 70.8% |
| `04_SS1-68-E_comp-mixture-demucs-guitar` | -3 dB | comp | 733 | 0.7233 | 146 | 151 | 2 | 84.4% |
| `04_SS1-68-E_solo-mixture-demucs-guitar` | -3 dB | solo | 112 | 0.8532 | 15 | 9 | 3 | 85.7% |
| `04_SS2-107-Ab_comp-mixture-demucs-guitar` | -6 dB | comp | 232 | 0.5051 | 73 | 104 | 11 | 83.9% |
| `04_SS2-107-Ab_solo-mixture-demucs-guitar` | -6 dB | solo | 67 | 0.8788 | 7 | 5 | 2 | 85.7% |
| `04_SS2-88-F_comp-mixture-demucs-guitar` | 3 dB | comp | 185 | 0.5597 | 21 | 187 | 5 | 93.7% |
| `04_SS2-88-F_solo-mixture-demucs-guitar` | 6 dB | solo | 83 | 0.9123 | 3 | 8 | 1 | 84.6% |
| `04_SS3-84-Bb_comp-mixture-demucs-guitar` | 3 dB | comp | 362 | 0.5938 | 54 | 199 | 14 | 83.5% |
| `04_SS3-84-Bb_solo-mixture-demucs-guitar` | 6 dB | solo | 135 | 0.8240 | 18 | 15 | 5 | 82.5% |
| `04_SS3-98-C_comp-mixture-demucs-guitar` | -6 dB | comp | 268 | 0.4200 | 111 | 75 | 20 | 78.2% |
| `04_SS3-98-C_solo-mixture-demucs-guitar` | -6 dB | solo | 123 | 0.6084 | 14 | 31 | 27 | 60.8% |

## 逐轨附录：Guitar-TECHS（12）

| track_id | 技法 | 音频 | 参考事件 | TAB F1 | 漏检 | 误检 | 同音高错弦 | 最终输出层占比 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `guitar-techs-P2-pick-directinput` | pick | directinput | 128 | 0.3271 | 50 | 8 | 43 | 57.4% |
| `guitar-techs-P2-pick-micamp` | pick | micamp | 128 | 0.3333 | 45 | 11 | 46 | 54.9% |
| `guitar-techs-P2-bend-directinput` | bend | directinput | 29 | 0.4615 | 15 | 9 | 2 | 92.3% |
| `guitar-techs-P2-bend-micamp` | bend | micamp | 29 | 0.4545 | 12 | 20 | 2 | 94.1% |
| `guitar-techs-P2-harmonic-directinput` | harmonic | directinput | 25 | 0.1176 | 17 | 1 | 5 | 75.0% |
| `guitar-techs-P2-harmonic-micamp` | harmonic | micamp | 25 | 0.1176 | 17 | 1 | 5 | 75.0% |
| `guitar-techs-P2-palm-mute-directinput` | palm-mute | directinput | 134 | 0.3391 | 46 | 8 | 49 | 52.4% |
| `guitar-techs-P2-palm-mute-micamp` | palm-mute | micamp | 134 | 0.3540 | 51 | 9 | 43 | 58.3% |
| `guitar-techs-P2-pinch-harmonic-directinput` | pinch-harmonic | directinput | 119 | 0.2022 | 72 | 12 | 24 | 74.3% |
| `guitar-techs-P2-pinch-harmonic-micamp` | pinch-harmonic | micamp | 119 | 0.1718 | 82 | 7 | 19 | 79.5% |
| `guitar-techs-P2-vibrato-directinput` | vibrato | directinput | 110 | 0.2346 | 45 | 4 | 44 | 52.7% |
| `guitar-techs-P2-vibrato-micamp` | vibrato | micamp | 110 | 0.2312 | 50 | 3 | 40 | 57.0% |

## 复现

```bash
python3 scripts/analyze-position-error-attribution.py
```

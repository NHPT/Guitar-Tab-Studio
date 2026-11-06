# v24 与受治理 v28 模型对比报告

日期：2026-11-06

## 0. 结论

**v28 不晋级。** 它补齐了 v24 缺失的训练 manifest、来源计数和许可台账
身份，research 与 production 两种 checkpoint 血缘审计均通过，但端到端
准确度没有实质提升：

- GuitarSet mixture TAB F1 由 `0.679886` 降至 `0.679853`。
- GuitarSet 重复 TAB 召回保持 `0.690274`，没有增加。
- Guitar-TECHS TAB F1 与重复 TAB 召回完全不变，仍为 `0.287973` 和
  `0.166667`。

v28 仅作为受治理训练流程的研究证据保留，不替换 v24 research bundle，
不改变默认 Basic Pitch Worker 后端，不触发 Commit、Tag 或 Release。

## 1. 目标与边界

本轮实验只回答一个问题：在保持 v12 指板位置模型、v18 主 activity
head、候选前端与所有解码阈值不变的前提下，使用经过许可台账约束的
GuitarSet、Guitar-TECHS 和符号合成数据继续微调 v24 补充 activity head，
能否稳定提高跨域 TAB 准确度与重复音召回。

本轮未访问 GuitarSet player 05 旧测试集、密封测试集或待复核的
Residual Onset v2 数据。Guitar-TECHS 验证集仅用于最终同口径评测，不参与
候选模型训练。

## 2. 数据治理

| 用途 | 清单 | 轨道 | SHA-256 | 审计 |
| --- | --- | ---: | --- | --- |
| 正式训练/选择 | `formal-training-manifest.jsonl` | 740 | `cb8b456d4ca8736249d3494543a8019d459463d420fb939983865db194707a26` | 通过 |
| GuitarSet mixture 评测 | `guitarset-mixture-derived.jsonl` | 300，其中 validation 60 | `a90668d480af274c579891990e7192f48b6b604914be33d0d85078f5e648fbda` | 通过 |
| Guitar-TECHS 评测 | `guitar-techs-position-validation.jsonl` | 12 | `6f796466be2cae19783c177b4a821389a1f1fee3a7d1346e98dd1c4761540a4b` | 通过 |

训练清单由 620 条训练记录和 120 条 GuitarSet 验证记录组成：

- GuitarSet：480 train，120 validation。
- Guitar-TECHS：12 train，仅启用 positive-only 稀疏标注策略。
- 原创符号合成：128 train。

三份清单均无绝对路径、无 test 记录、无 group 级 split 泄漏、无许可
违规或审计警告。许可台账版本为 `2026-11-06`，SHA-256 为
`7b95bd51db8c1f75c8be49efd297f393bf95327c0d933324ee119f18b5c6a583`。

后续数据获取阶段将 AG-PT-set 的官方 CC BY 4.0 记录完成独立核验，当前
许可台账已升为 `2026-11-06.2`。v28 使用的 740 条训练清单仍保持上表
SHA-256，不包含 AG-PT-set；但 checkpoint 内嵌的是历史台账身份，不能把本节
当作当前发布审计。当前 research 复核的唯一违规类型为
`stale_source_ledger_sha256` 和 `stale_source_ledger_version`；来源登记、
权利和 manifest 哈希格式仍通过。v28 已因性能门槛失败，继续只保留为历史
研究证据。

## 3. 固定实验条件

| 部件 | 固定值 |
| --- | --- |
| 指板位置模型 | v12，`3e1e3507627b15c7ec1863022e6ae2ad8da5717a73cab7b6352d90e0d108c5dd` |
| 主 activity head | v18，`2ee80b85a228a7f675f464adec82097b48437087ad4c70778653aa07c566be58` |
| v24 补充 head | `931946fcf507f089f556b8e4c469ae4c41240c9dd3262b988815ce9e37f886e7` |
| 主候选前端 | onset `0.35`、frame `0.25`、最短音符 `70 ms` |
| 主解码阈值 | direct `0.50`、activity `0.20`、repeat `0.44`、repeat direct `0.45` |
| 补充候选前端 | onset `0.25`、frame `0.20`、最短音符 `70 ms` |
| 补充解码阈值 | direct `0.70`、activity `0.85`、repeat `0.44` |
| v28 训练 | v24 初始化，2 epochs，batch `128`，LR `1e-5`，CPU，按 activity F1 选优 |

v24 不含可机器验证的训练 manifest 与许可台账身份，因此当前 checkpoint
治理审计结果为失败，唯一违规项是 `missing_training_data_identity`。v28
训练器会在 checkpoint 中写入脱敏清单名称、SHA-256、来源计数和当前许可
台账身份，不写入本机路径。

## 4. v24 受治理基线

| 验证域 | 轨道 | Onset F1 | Pitch F1 | TAB F1 | 重复 TAB 召回 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GuitarSet mixture-derived | 60 | 0.804544 | 0.728359 | 0.679886 | 0.690274 |
| Guitar-TECHS | 12 | 0.664032 | 0.651609 | 0.287973 | 0.166667 |

GuitarSet 结果与原 v24 冻结报告完全一致，证明 manifest 治理重建没有改变
样本、音频或评测口径。60 首中有 29 首达到单曲 TAB F1 `0.75`，中位数为
`0.745098`。Guitar-TECHS 中没有轨道达到 `0.75`，显示出明显的跨来源
泛化缺口。

## 5. v28 训练与对比

v28 从 v24 初始化，仅微调补充 activity head。训练生成了 128 条此前缺失
的合成特征，实际使用 97,844 个训练候选组和 24,417 个验证候选组；
Guitar-TECHS positive-only 策略过滤了 5,983 个未标注负候选组。第 2 个
epoch 以 activity F1 `0.882811` 入选。该数值不能直接与 v24 的
`0.879169` 比较，因为 v28 使用的是 120 轨 source/stem 联合验证集，而
v24 训练时只使用了 60 轨 mixture-stem 验证集。

v28 checkpoint 大小约 104 KB，SHA-256 为
`d52692521f4e3f9bda4f9cc9ceb37b3797ccb7b621b3eb492dd650b704cca603`。

### 5.1 GuitarSet mixture-derived

| 指标 | v24 | v28 | 差值 |
| --- | ---: | ---: | ---: |
| Onset F1 | 0.804544 | 0.804602 | +0.000059 |
| Pitch F1 | 0.728359 | 0.728323 | -0.000036 |
| TAB F1 | 0.679886 | 0.679853 | -0.000033 |
| 重复 TAB F1 | 0.506937 | 0.506906 | -0.000032 |
| 重复 TAB 召回 | 0.690274 | 0.690274 | 0.000000 |
| 单曲 TAB F1 ≥ 0.75 | 29/60 | 29/60 | 0 |

v28 多匹配了 1 个 onset，但没有增加正确 TAB 事件，并增加了 1 个错误 TAB
事件。不同 SNR 分层的变化在 `-0.000262` 到 `+0.000215` 之间，幅度极小
且方向不一致，不能视为稳定提升。

### 5.2 Guitar-TECHS

| 指标 | v24 | v28 | 差值 |
| --- | ---: | ---: | ---: |
| Onset F1 | 0.664032 | 0.664032 | 0.000000 |
| Pitch F1 | 0.651609 | 0.651609 | 0.000000 |
| TAB F1 | 0.287973 | 0.287973 | 0.000000 |
| 重复 TAB F1 | 0.002911 | 0.002911 | 0.000000 |
| 重复 TAB 召回 | 0.166667 | 0.166667 | 0.000000 |
| 单曲 TAB F1 ≥ 0.75 | 0/12 | 0/12 | 0 |

跨来源验证没有任何事件级变化。这表明新增合成数据和 P1
Guitar-TECHS 训练样本没有转化为 P2 Guitar-TECHS 的端到端泛化收益。

## 6. 生产门禁

| 条件 | 结果 | 说明 |
| --- | --- | --- |
| 100 首以上未参与设计的多来源密封集 | 未通过 | 本轮未访问密封集 |
| TAB F1 ≥ 0.75 | 未通过 | GuitarSet `0.679853`，Guitar-TECHS `0.287973` |
| 相对 Basic Pitch 提升 ≥ 0.15 | 未评测 | 本轮目标是与 v24 同口径比较 |
| 重复音召回 ≥ 0.85 | 未通过 | GuitarSet `0.690274`，Guitar-TECHS `0.166667` |
| 无 split 泄漏和非法弦品 | 通过 | 严格 manifest 审计无违规 |
| 音频与 TAB 联合错误复核 | 未完成 | 本轮只做自动验证 |
| checkpoint 来源与发布权利 | 通过 | research/production 血缘审计均通过 |

门禁为合取条件，因此最终结果是 **拒绝生产晋级**。血缘审计中的
`production` 通过只表示来源已登记、许可允许模型发布且未嵌入本机路径，
不代表性能门禁或密封评测通过。

## 7. 工程验证

- Core：45 项通过。
- API：105 项通过。
- Desktop：54 项通过。
- Worker/StringTrace：182 项通过。
- Promotion gate、training runner、sealed gate：11 项通过。
- Python lint、Web lint、TypeScript 构建和全量应用构建通过。
- 10 首节拍结构回归全部通过，倍率错误为 0。
- 完整 Playwright E2E：131 项通过，17 项按设备或场景设计跳过。

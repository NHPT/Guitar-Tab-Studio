# v29 受治理单变量实验预注册

## 状态

- 预注册日期：2026-11-06
- 状态：预注册完成，尚未开始训练
- 产品默认：保持 `basic-pitch / auto / built-in`
- 候选用途：research-only；未通过全部门槛不得生成生产 bundle

## 假设与单变量

v29 检验一个假设：在 v24 补充活动头的固定架构、初始化、前端、阈值、
训练 seed 和选择规则不变时，加入经权利核验且与评测组隔离的 AG-PT-set
真实单音训练数据，能改善独立跨域活动筛选，同时不损害 GuitarSet TAB 和
重复音表现。

唯一允许改变的模型输入是训练 manifest 及其预注册采样计划：

- v28 训练快照：740 轨，不含当前受治理 AG-PT-set；
- v29 训练快照：1,093 轨，其中 train 973 轨，新增 AG-PT-set train 353 轨；
- GuitarSet validation、位置模型、v24 初始化、模型架构、优化器参数、
  解码阈值和 seed 均冻结。

## 固定输入

| 输入 | SHA-256 |
|---|---|
| `governed-v1/formal-training-manifest-v2.jsonl` | `630d08ede28ba7c220793715738444dab22ca53e3cde3c2311bbc6b028879916` |
| `governed-v1/guitarset-mixture-combined.jsonl` | `7646fcb2e321b48e85c8e4b4f0c2ed278cc05f0b7267cc97c4daa2e673443e2d` |
| `governed-v1/agpt-position-validation.jsonl` | `83abea78a2786131bdf3776def93d80c4141cb825afadf44a7908c925d67d708` |
| `governed-v1/guitar-techs-position-validation.jsonl` | `6f796466be2cae19783c177b4a821389a1f1fee3a7d1346e98dd1c4761540a4b` |
| `checkpoints/guitarset-onset-position-mixture-v12.pt` | `3e1e3507627b15c7ec1863022e6ae2ad8da5717a73cab7b6352d90e0d108c5dd` |
| `checkpoints/guitarset-low-candidate-activity-v24.pt` | `931946fcf507f089f556b8e4c469ae4c41240c9dd3262b988815ce9e37f886e7` |

AG-PT-set test split 不得读取或用于训练、阈值选择、模型选择和最终评测。

## 采样计划

训练单元是 Basic Pitch candidate group，不是轨道。AG-PT-set 的期望 batch
占比上限固定为 10%，避免单音技法轨道按轨道数主导训练。缓存完成后只允许
按以下公式计算一次权重：

```text
agpt_weight = min(
  1,
  (0.10 / 0.90) * (non_agpt_eligible_groups / agpt_eligible_groups)
)
```

其中 eligible group 是按固定低阈值前端生成，并在 positive-only 规则后实际
进入训练的数据。AG-PT-set 和 Guitar-TECHS 均固定为 positive-only；其他
数据集权重为 1。训练采用带放回的确定性 WeightedRandomSampler，样本数等于
eligible group 总数，seed 为 `20260923`。

固定候选缓存完成后得到：

| 数据集 | eligible groups | 采样权重 |
|---|---:|---:|
| GuitarSet | 94,780 | 1 |
| Guitar-TECHS | 1,121 | 1 |
| stringtrace-symbolic-synthetic | 1,943 | 1 |
| AG-PT-set | 16,786 | `0.6476561155460239` |

总计 114,630 个 eligible groups；positive-only 共过滤 20,601 个无参考匹配
候选组。代入公式后 AG-PT-set 的期望采样占比为精确的 10%。这些值在训练
启动前冻结，不得再改变。

## 固定训练配置

| 参数 | 值 |
|---|---|
| 初始化 | v24 supplemental activity head |
| 冻结编码器 | onset-position v12 |
| epoch | 2 |
| batch size | 128 |
| learning rate | `1e-5` |
| weight decay | `1e-4` |
| activity loss weight | `0.25` |
| repeat loss weight | `1.0` |
| optimizer | AdamW |
| scheduler | CosineAnnealingLR，`eta_min=5e-7` |
| hidden size / dropout | 64 / 0.1 |
| temporal features | 6 |
| repeat interval | 0.75s |
| selection metric | GuitarSet validation activity F1 |
| seed | `20260923` |
| device | Apple Metal / MPS |

训练候选前端固定为 Basic Pitch 0.4：

- onset threshold `0.25`
- frame threshold `0.20`
- minimum note length `70ms`

## 固定解码

主路径保持 v18：

- Basic Pitch `0.35 / 0.25 / 70ms`
- activity probability `0.20`
- repeat probability `0.44`
- direct probability `0.50`
- repeat direct probability `0.45`

补充路径仅替换活动头 checkpoint：

- Basic Pitch `0.25 / 0.20 / 70ms`
- activity probability `0.85`
- direct probability `0.70`
- 不启用 repeat 降阈值

不得根据 AG-PT validation、Guitar-TECHS P2 或最终 72 轨结果调阈值。

## 评测顺序与门槛

1. GuitarSet mixture validation 用于训练 epoch 选择。
2. checkpoint 冻结后，依次评测 60 轨 GuitarSet、12 轨 Guitar-TECHS P2、
   72 轨 AG-PT held-out performer。
3. 三组独立评测全部完成后一次性判定，不做追加训练或阈值搜索。

相对 v24 固定基线，v29 必须同时满足：

| 门槛 | 判定 |
|---|---|
| GuitarSet TAB | TAB F1 不下降 |
| GuitarSet 重复音 | repeated tablature recall 不下降 |
| Guitar-TECHS 跨域 | TAB F1 严格提高，onset F1 不下降 |
| AG-PT 单音活动 | onset F1 严格提高，TAB F1 不下降 |
| 数据治理 | checkpoint 内嵌输入哈希、来源统计、采样计划和完整训练配置 |

任一门槛失败即拒绝 v29，并停止在当前活动头架构上继续训练。即使相对门槛
全部通过，仍需满足生产硬门槛 `TAB F1 >= 0.75`、重复音召回 `>= 0.85`
及既有发布治理要求，才允许讨论生产晋级。

## 固定输出

- checkpoint：`checkpoints/guitarset-low-candidate-activity-governed-v29.pt`
- 训练日志：`reports/governed-v29-training.jsonl`
- GuitarSet 报告：`reports/governed-v29-guitarset-mixture-validation.json`
- Guitar-TECHS 报告：`reports/governed-v29-guitar-techs-validation.json`
- AG-PT 基线：`reports/governed-v24-agpt-validation.json`
- AG-PT 候选：`reports/governed-v29-agpt-validation.json`
- 最终判定：`docs/model-v29-governed-result.md`

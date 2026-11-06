# v29 受治理实验结果

## 结论

v29 **拒绝晋级**，保持 `basic-pitch / auto / built-in` 为生产默认，不生成生产
bundle，也不继续在当前活动头架构上训练 v30。

加入受治理 AG-PT-set 训练数据后，训练内 GuitarSet validation activity F1
由 v28 的 `0.882811` 提高到 `0.883085`，但独立评测没有转化为最终输出收益：

- GuitarSet TAB F1 下降 `0.00010178`，未通过“不下降”门槛；
- Guitar-TECHS TAB F1 完全不变，未通过“严格提高”门槛；
- AG-PT onset F1 完全不变，未通过“严格提高”门槛。

GuitarSet 重复音召回提高 `0.00016943`，Guitar-TECHS onset 和 AG-PT TAB
均未下降，但不能抵消上述三个失败门槛。

## 冻结产物

| 产物 | SHA-256 |
|---|---|
| `checkpoints/guitarset-low-candidate-activity-governed-v29.pt` | `93ec0dfa3f25203508fb0b542f13e20a5fd36050091af47fcd49a51663b8f683` |
| `reports/governed-v29-training.jsonl` | `e228986652ca90e84fe9ef12c0a92d71e330e1b4b183a22531ec2c2ca647f9b1` |
| `reports/governed-v24-guitarset-mixture-validation.json` | `d38f900f31aa407951065bb2a4c0105288ccf09b30de70c02055acc5981ddee6` |
| `reports/governed-v29-guitarset-mixture-validation.json` | `a190960c0a061379780a20f3ef4f1a7031b338eea3297619f7b3bb024dfebc34` |
| `reports/governed-v24-guitar-techs-validation.json` | `38dac7222e0318c081181e82e04cf2ea829accdc885f8209592e826f23db8e5a` |
| `reports/governed-v29-guitar-techs-validation.json` | `4dc36b5c2d11461217ee2c0b741b309ca9efc6420263cc353c52a17c17f325ec` |
| `reports/governed-v24-agpt-validation.json` | `0c6278377b16e876c6fed40190fb059c946c61890c95fd1712be615f051002f2` |
| `reports/governed-v29-agpt-validation.json` | `1a4c89a9d51c34ce6f4dba961653c4d07059c2b520b06f2b189c551a0a2e3f07` |

训练 checkpoint 选择 epoch 1。内嵌元数据已复核包含固定输入哈希、来源统计、
positive-only 数据集、完整采样计划、完整训练配置、v24 初始化身份和 v12
基础模型身份。AG-PT-set 的期望采样占比为精确的 10%，AG-PT-set test split
未被读取。

## 独立评测

| 数据集 | 指标 | v24 | v29 | 差值 |
|---|---|---:|---:|---:|
| GuitarSet 60 轨 | onset F1 | 0.80454367 | 0.80450318 | -0.00004049 |
| GuitarSet 60 轨 | TAB F1 | 0.67988641 | 0.67978463 | -0.00010178 |
| GuitarSet 60 轨 | 重复音召回 | 0.69027448 | 0.69044392 | +0.00016943 |
| Guitar-TECHS 12 轨 | onset F1 | 0.66403162 | 0.66403162 | 0 |
| Guitar-TECHS 12 轨 | TAB F1 | 0.28797290 | 0.28797290 | 0 |
| AG-PT 72 轨 | onset F1 | 0.84356895 | 0.84356895 | 0 |
| AG-PT 72 轨 | TAB F1 | 0.37166860 | 0.37166860 | 0 |

三组评测的轨道集合、Basic Pitch 主/补充前端、v12 位置模型、v18 主活动头、
阈值和解码模式均一致，唯一替换项是补充活动头 checkpoint。

## 门槛判定

| 门槛 | 结果 | 判定 |
|---|---|---|
| GuitarSet TAB 不下降 | `-0.00010178` | 失败 |
| GuitarSet 重复音召回不下降 | `+0.00016943` | 通过 |
| Guitar-TECHS TAB 严格提高 | `0` | 失败 |
| Guitar-TECHS onset 不下降 | `0` | 通过 |
| AG-PT onset 严格提高 | `0` | 失败 |
| AG-PT TAB 不下降 | `0` | 通过 |
| checkpoint 数据治理 | 哈希、来源、采样和配置完整 | 通过 |

相对门槛未全部通过。生产硬门槛也未达到：GuitarSet TAB F1 为
`0.679785 < 0.75`，重复音召回为 `0.690444 < 0.85`。

## 阶段证据

v29 相对 v24 的补充路径活动接受组变化如下：

| 数据集 | 接受组变化 | 独有补充事件变化 | 最终事件变化 |
|---|---:|---:|---:|
| GuitarSet | +19 | +6 | +6 |
| Guitar-TECHS | +9 | 0 | 0 |
| AG-PT | +21 | 0 | 0 |

Guitar-TECHS 和 AG-PT 中新增接受组没有形成任何新增独有输出，说明它们最终被
位置阶段拒绝或与主路径重复。GuitarSet 仅新增 6 个独有事件，但 TAB 与 onset
指标反而轻微下降。现有活动头即使吸收更多真实单音训练数据，也没有改善当前
主路径、位置模型和去重策略共同决定的最终结果。

## 后续判定

1. 停止继续训练当前活动头架构，不启动 v30 同类实验。
2. Basic Pitch 继续作为生产默认；v24 保持 research bundle，v29 仅保留为
   可审计研究 checkpoint。
3. 下一轮模型工作转向位置模型和逐候选身份追踪，优先解释 Guitar-TECHS
   同音高错弦及活动接受后的位置整组拒绝；新实验需重新预注册。

# 真实吉他弦品数据源权利与接入报告

## 执行摘要

截至 2026-11-06，本轮找到并核验了一份可立即用于训练的独立真实录音来源：
AG-PT-set。其官方 Zenodo 记录将数据集标为 CC BY 4.0，Creative Commons
条款允许在署名条件下为任何目的复制、再分发和改编，包括商业用途。
[cite:1][cite:3] 数据集论文和官方记录均说明其包含精确 onset 标注的真实木吉他
单音技法录音。[cite:1][cite:2]

AG-PT-set 已解除隔离并接入新的受治理训练快照。接入结果为 353 条 train
记录、4 个演奏者组和 16,908 个有效的 0–24 品 string/fret/onset 事件。
原 validation 与 test 共 143 条记录未进入训练，108 个超出位置约束的事件被
审计过滤。另有 72 条 validation 记录和 3,690 个事件被导出为独立
evaluation-only 清单，test 仍锁定。新的 `formal-training-manifest-v2.jsonl`
共 1,093 条记录和 127,620 个事件，来源审计、路径检查、test 排除和组隔离
均通过。

内容最匹配当前缺口的 GOAT 不能用于产品训练。它提供真实电吉他 DI、完整
tablature、对齐 MIDI 和技法信息，但官方数据记录是 CC BY-NC 4.0、文件受限，
并明确拒绝任何商业用途，包括内部测试；项目仓库也只提供辅助代码和样例，
要求通过受限记录申请数据。[cite:4][cite:5][cite:6][cite:7] 因此未请求访问、
未下载、未登记为训练源。

本轮没有找到新的、可商用且提供真实多音乐句 string/fret/onset 全标注的公开
来源。`[INSUFFICIENT DATA]` AG-PT-set 能补充真实音色、演奏者和精确 onset，
但其单音技法结构不能替代独立的完整乐句 TAB 数据。

## 核验方法

核验同时要求满足以下条件：

1. 官方数据集或发布机构页面明确给出许可证。
2. 数据内容包含真实音频，以及可确定 onset、string 和 fret 的标注。
3. 数据文件可获得，且本地资产能通过发布方哈希或逐文件哈希检查。
4. 训练、模型发布和潜在商业用途不受 NonCommercial、NoDerivatives 或
   “evaluation only”限制。
5. 演奏者或录音组能在 train、validation、test 之间隔离。

许可证名称本身不被当作充分证据。数据说明、论文、许可证正文和本地原始
元数据分别用于核对内容、用途权利和实际可转换性。

## 来源判定

| 数据集 | 真实标注内容 | 权利与可得性 | 判定 | 置信度 |
|---|---|---|---|---|
| AG-PT-set | 木吉他单音技法；毫秒 onset；本地 CSV 含 pitch、string 和开放弦音高 | Zenodo 开放下载，CC BY 4.0[cite:1][cite:3] | 允许训练、评测与模型发布；已接入 v2 | `[CONFIRMED]` |
| GOAT | 5.9 小时独立 DI，完整 string/fret/技法 TAB 和对齐 MIDI[cite:5] | 文件受限，CC BY-NC 4.0；官方明确禁止商业及内部测试[cite:4][cite:7] | 拒绝产品训练，不请求访问 | `[CONFIRMED]` |
| IDMT-SMT-GUITAR | 真实木/电吉他，含单音、多音、技法和 XML 参数标注 | 官方仅允许 evaluation，CC BY-NC-ND 4.0[cite:8] | 保持 evaluation-only | `[CONFIRMED]` |
| GuitarSet | 真实木吉他乐句，含时间对齐 string/fret、和弦、节拍和风格 | Zenodo 开放下载，CC BY 4.0[cite:9] | 已有主要训练与验证源 | `[CONFIRMED]` |
| Guitar-TECHS | 三位演奏者、多录音链路、逐弦同步 MIDI 和技法/乐句内容 | Zenodo 开放下载，CC BY 4.0[cite:10] | 已有辅助训练与独立评测源 | `[CONFIRMED]` |

GOAT 论文对当时主要吉他音频数据集的综述也显示，带 tablature 弦品标注的
真实音频来源主要集中在 GuitarSet、IDMT、Guitar-TECHS 和 GOAT；
GAPS、FrançoisLeduc 与 EGDB 仅列为 MIDI 标注，不满足完整弦品监督。
[cite:5] 2026 年检索未发现能同时补齐内容和商业权利的新来源，因此这个
结论记为 `[MAJORITY]`，不是对所有未公开或需单独签约数据的穷尽证明。

## AG-PT-set 接入证据

官方记录说明数据集包含 15 小时 55 分钟真实木吉他录音，其中 10 小时
4 分钟、8 类技法和 32,592 个音符已标注，onset 精确到毫秒。
[cite:1][cite:2] 本地原始 `note_labels.csv` 进一步确认已标注音符具有：

- `onset_label_seconds`
- `pitch_midi`
- `string_number`
- `string_openpitch_midi`
- `expressive_technique_id`

本地六根开放弦映射为 MIDI `[64, 59, 55, 50, 45, 40]`。导入器将频率形式
的自然泛音标签归一为 MIDI，按发布方 `files.csv` 的 SHA-256 逐文件校验，
并排除一份已知隔离文件。位置转换使用
`fret = pitch - open_string_pitch`，只保留 0–24 品。

| 产物 | 记录/事件 | SHA-256 |
|---|---:|---|
| `source_ledger.json`（版本 `2026-11-06.2`） | 5 个登记来源 | `de5b817d884422a15bb918b094caafbef856f4b612fb67cbbd9328ef90d2610a` |
| `ag-pt-set/technique-manifest.jsonl` | 496 轨 / 24,132 事件 | `ebab1714b34af7de16f50e57f6da3390b4223ad545458e78029d88bfe874f376` |
| `governed-v1/agpt-position-train.jsonl` | 353 轨 / 16,908 事件 | `0f2f45c4e8a3a485bb1255449551a63ecb7d14d4d09ac80b980fe0cbc34a8a8b` |
| `governed-v1/agpt-position-validation.jsonl` | 72 轨 / 3,690 事件 | `83abea78a2786131bdf3776def93d80c4141cb825afadf44a7908c925d67d708` |
| `governed-v1/formal-training-manifest-v2.jsonl` | 1,093 轨 / 127,620 事件 | `630d08ede28ba7c220793715738444dab22ca53e3cde3c2311bbc6b028879916` |

v2 正式快照包含 973 条 train 和 120 条 GuitarSet validation 记录。
相对 v28 使用的 740 条快照，新增 353 条 AG-PT train 记录；v28 原 manifest
已恢复并保持原 SHA-256
`cb8b456d4ca8736249d3494543a8019d459463d420fb939983865db194707a26`。

## 限制与风险

AG-PT-set 的新增监督集中在 harmonic、palm-mute 和两种 pick 位置映射后的
`pick` 类。其录音为单音技法序列，不包含真实完整乐句中的和弦、复调、快速
换弦和伴奏泄漏。因此，它适合降低候选活动和单音弦位的域偏差，但不能直接
证明 GuitarSet mixture 或 Guitar-TECHS 跨域 TAB 指标会提高。

官方 Zenodo 描述写“6 位演奏者、7 把吉他”，论文正文写“7 位演奏者、6 件
乐器”。[cite:1][cite:2] 本地训练接入不依赖这两个总量，而以原始
`player_id` 和 `guitar_id` 为准；当前 train split 实际包含 4 个演奏者组。
这个发布资料差异保留为 `[DISPUTED]`，不自行改写。

CC BY 4.0 允许商业使用和改编，但仍要求署名、许可证链接和修改说明，且不
保证已经处理肖像权、隐私权或其他第三方权利。[cite:3] 模型及数据卡发布时
仍需保留来源、作者、许可证和派生说明。

## 结论与执行边界

AG-PT-set 已满足“独立权利证据、真实音频、完整可转换弦品 onset、文件可得”
的接入门槛，v2 清单可以作为下一次候选训练的输入。当前不启动 v29，因为
尚未定义针对单音数据比例、positive-only 策略和跨域门槛的预注册实验。

GOAT、IDMT 和其他非商业或仅评测来源不得进入产品训练。下一轮数据获取应
优先争取一份明确允许商业训练和模型发布的真实电吉他多音乐句 TAB 数据，
并要求至少包含多演奏者、DI/amp/mic 多录音链路、逐弦 onset/offset、fret、
调弦和技法标注。

## Sources

[cite:1] AG-PT-set official Zenodo record: https://zenodo.org/records/10159492

[cite:2] Stefani, Giudici, Turchet, “On the Importance of Temporally Precise
Onset Annotations for Real-Time Music Information Retrieval”:
https://doi.org/10.1145/3678299.3678325

[cite:3] Creative Commons Attribution 4.0 deed:
https://creativecommons.org/licenses/by/4.0/

[cite:4] GOAT official dataset record:
https://zenodo.org/records/15690894

[cite:5] Loth et al., “GOAT: A Large Dataset of Paired Guitar Audio Recordings
and Tablatures”: https://arxiv.org/html/2509.22655

[cite:6] GOAT dataset repository:
https://github.com/JackJamesLoth/GOAT-Dataset

[cite:7] Creative Commons Attribution-NonCommercial 4.0 deed:
https://creativecommons.org/licenses/by-nc/4.0/

[cite:8] Fraunhofer IDMT, IDMT-SMT-Guitar official dataset page:
https://www.idmt.fraunhofer.de/en/publications/datasets/guitar.html

[cite:9] GuitarSet official Zenodo record:
https://zenodo.org/records/3371780

[cite:10] Guitar-TECHS official Zenodo record:
https://zenodo.org/records/14963133

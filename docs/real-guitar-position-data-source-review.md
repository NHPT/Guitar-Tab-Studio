# 真实吉他弦品数据源权利与接入报告

## 执行摘要

截至 2026-11-20，本轮核验并接入了两份可用于训练的独立真实录音来源：
AG-PT-set，以及此前尚未导入的 Guitar-TECHS P3 music。两者的官方 Zenodo
记录均标为 CC BY 4.0，允许在署名条件下为任何目的复制、再分发和改编，包括
商业用途。[cite:1][cite:3][cite:10]

AG-PT-set 已解除隔离并接入新的受治理训练快照。接入结果为 353 条 train
记录、4 个演奏者组和 16,908 个有效的 0–24 品 string/fret/onset 事件。
原 validation 与 test 共 143 条记录未进入训练，108 个超出位置约束的事件被
审计过滤。另有 72 条 validation 记录和 3,690 个事件被导出为独立
evaluation-only 清单，test 仍锁定。新的 `formal-training-manifest-v2.jsonl`
共 1,093 条记录和 127,620 个事件，来源审计、路径检查、test 排除和组隔离
均通过。

Guitar-TECHS P3 music 已完成独立候选接入，并在候选审计通过后进入新的
受治理 v3 数据快照，但尚未启动模型训练。官方 `P3_music.zip` 含 12 首完整
原创乐曲片段、逐弦 MIDI，以及 direct input 和 mic+amp 两条音频链路。
[cite:10] 本地核验得到 1,948 个原始 MIDI 音符，其中 1,941 个可转换为
标准调弦 0–24 品位置；双链路清单共 24 轨、3,882 个有效事件。38 个复调
onset 组全部按弦保留，未使用单音技法数据的连通片段合并逻辑。

内容最匹配当前缺口的 GOAT 不能用于产品训练。它提供真实电吉他 DI、完整
tablature、对齐 MIDI 和技法信息，但官方数据记录是 CC BY-NC 4.0、文件受限，
并明确拒绝任何商业用途，包括内部测试；项目仓库也只提供辅助代码和样例，
要求通过受限记录申请数据。[cite:4][cite:5][cite:6][cite:7] 因此未请求访问、
未下载、未登记为训练源。

P3 修正了上一版“没有新的、可商用真实多音乐句来源”的结论。它能补充独立
硬件、演奏者和真实乐句结构，但仅有一名 P3 演奏者；因此不能按曲目切分后
宣称获得新的 performer-independent validation。P3 全部保持为一个训练候选
组，P2 继续作为独立验证来源。

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
| Guitar-TECHS | 三位演奏者、多录音链路、逐弦同步 MIDI 和技法/完整乐句内容 | Zenodo 开放下载，CC BY 4.0[cite:10] | P1/P2 已用于辅助训练与独立评测；P3 music 已进入 v3 快照并完成预注册，尚未训练 | `[CONFIRMED]` |

GOAT 论文对当时主要吉他音频数据集的综述也显示，带 tablature 弦品标注的
真实音频来源主要集中在 GuitarSet、IDMT、Guitar-TECHS 和 GOAT；
GAPS、FrançoisLeduc 与 EGDB 仅列为 MIDI 标注，不满足完整弦品监督。
[cite:5] P3 是 Guitar-TECHS 官方记录内此前未接入的分区，不是 2026 年新发布
的数据集。公开来源范围仍记为 `[MAJORITY]`，不是对所有未公开或需单独签约
数据的穷尽证明。

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

## Guitar-TECHS P3 music 接入证据

官方 Zenodo 记录提供 `P3_music.zip`，发布大小为 129,505,089 bytes、MD5 为
`071ba80aecf00f4a31fbd167b3f22198`，许可证为 CC BY 4.0。[cite:10] 本地下载
同时计算 SHA-256，并在每次 P3 导入前重新校验大小与 MD5。归档内含 12 份
逐弦 MIDI、12 份 direct input WAV 和 12 份 mic+amp WAV；视频音轨不进入
训练候选。

导入器对 P1/P2 单音技法数据继续使用 `connected-midi-gesture-v1`，但对 P3
使用 `preserve-per-string-midi-notes-v1`。负对齐偏移造成的音频起点事件不再
丢弃，而是裁剪到 `0s`。24 轨估计偏移范围为 -10ms 至 +5ms，中位数 -5ms。

| 产物 | 记录/事件 | SHA-256 |
|---|---:|---|
| `P3_music.zip` | 129,505,089 bytes | `033489e22600751fb5a1633e7d856b901c6782e0486fa02135e830780d9dbfe2` |
| `p3-music-technique-manifest-v1.jsonl` | 24 轨 / 3,896 事件 | `ffce9f7fead6dd141ceed077abb863cba6a68304a15dc9915a9ad4975b561d49` |
| `guitar-techs-p3-music-position-train-candidate-v1.jsonl` | 24 轨 / 3,882 事件 | `f991cc2475ae65d558df9bfc4603b4a22ea3b1c10bbc9c6745b2704765cb22c8` |
| `guitar-techs-p3-music-position-candidate-v1-audit.json` | 14 项门禁全部通过 | `266bef7f7879198595b934ce38ec52320eb16c61f70dae61929ebb50adcaff39` |
| P3 candidate cache report | 24 文件 / 6,685 候选 / 丢弃 157 | `f167413ca2a25c6d233fddf90979bc459d3b3c14f5e9d31acedd71c0d84860eb` |
| P3 sequence coverage audit | 3,416 组 / 250 次双链路转换 | `6c12bd0fdeb5776bed997a4d3efd8ba90bdf825d78b9127f25d393ec2fe5a728` |
| P3 frozen string-v1 audit | 1,136 个 active reference | `48d6afde3ac03e8c7b83809dbfac19e07fb67776143039b563281b38ec01f732` |
| `formal-training-manifest-v3.jsonl` | 1,117 轨 / 131,502 事件 | `cb7cc83e0a887f3413a58fc0bbd8e4258db17a94936d8472f6cb3952afc6504d` |

原始 MIDI 覆盖六根弦，弦事件数为 184、434、643、245、186、256；有效品位
覆盖 0–21 和 24 品。38 个 onset 包含复调，最大同时 3 音。7 个发布标签对应
`-4` 或 25–27 品，超出当前标准调弦 0–24 品约束；转换器不猜测其正确弦位，
而是将双链路中的 14 个副本写入过滤审计。

P3 的 24 轨全部使用 `group_id=guitar-techs-P3`。按 12 首曲目拆分验证集只能
隔离曲目，不能隔离演奏者和硬件；因此本轮不做这种切分。候选清单与
`formal-training-manifest-v2.jsonl` 的 track overlap 为 0，后者仍保持
1,093 轨和原 SHA-256
`630d08ede28ba7c220793715738444dab22ca53e3cde3c2311bbc6b028879916`。

在候选审计通过后，v3 由 v2 的四个来源加 P3 组合而成，共 997 条 train、
120 条 validation、144 个组和 131,502 个事件。组合报告、来源审计和跨源
隔离报告的 SHA-256 分别为
`6d3b2731e1699df887eabf947b309e665cd196ba93955dfb395a4a76320e4cb0`、
`ed171d832719ddfac766aabe3e45161e0961f0d0eb819812a052dcdebebad770`
和 `59dbc6a2d5f60ff88dc52d4980389f2be78e13128fe54067e5cc84ab103e2d17`。
来源审计通过；v3 与 AG-PT validation、Guitar-TECHS P2 validation 在音频
哈希、组、演奏者和 track 四个维度均为零重叠。

P3 候选缓存包含 6,685 个 Basic Pitch 候选和 3,416 个有序组。冻结的
string-v1 在其中接受 1,136 个 reference-active 候选，正确弦 659、错误弦
477，准确率 `0.580106`；重复弦正确数为 `246 / 431`。双链路共观察到 250
次两秒内同音高跨弦转换，对应 125 次独立演奏转换。

## 限制与风险

AG-PT-set 的新增监督集中在 harmonic、palm-mute 和两种 pick 位置映射后的
`pick` 类。其录音为单音技法序列，不包含真实完整乐句中的和弦、复调、快速
换弦和伴奏泄漏。因此，它适合降低候选活动和单音弦位的域偏差，但不能直接
证明 GuitarSet mixture 或 Guitar-TECHS 跨域 TAB 指标会提高。

P3 虽然是完整乐句，但仍只有一名演奏者，且 direct input 与 mic+amp 是同一
次演奏的相关视图。3,882 个候选事件不是 3,882 个独立演奏实例，而是 1,941
个有效标签的双链路观测。训练采样必须防止它以重复标签数量压过 GuitarSet
与多演奏者 AG-PT-set，模型收益也必须继续在锁定的 GuitarSet、AG-PT-set 和
P2 Guitar-TECHS 验证源上判断。

官方 Zenodo 描述写“6 位演奏者、7 把吉他”，论文正文写“7 位演奏者、6 件
乐器”。[cite:1][cite:2] 本地训练接入不依赖这两个总量，而以原始
`player_id` 和 `guitar_id` 为准；当前 train split 实际包含 4 个演奏者组。
这个发布资料差异保留为 `[DISPUTED]`，不自行改写。

CC BY 4.0 允许商业使用和改编，但仍要求署名、许可证链接和修改说明，且不
保证已经处理肖像权、隐私权或其他第三方权利。[cite:3] 模型及数据卡发布时
仍需保留来源、作者、许可证和派生说明。

## 结论与执行边界

AG-PT-set 已进入 v2 正式快照。Guitar-TECHS P3 music 也满足“独立权利证据、
真实音频、逐弦 onset/offset、文件可得、资产哈希可验证”的数据接入门槛。
它未修改 v2，而是作为第五个输入进入独立的 v3 快照。v3 是受治理的数据
产物，不等于模型训练或生产发布授权。

GOAT、IDMT 和其他非商业或仅评测来源不得进入产品训练。下一轮数据获取应
优先争取一份明确允许商业训练和模型发布的真实电吉他多音乐句 TAB 数据，
并要求至少包含多演奏者、DI/amp/mic 多录音链路、逐弦 onset/offset、fret、
调弦和技法标注。

`candidate-sequence-string-v3-preregistration.md` 已冻结，SHA-256 为
`e6594381f5bc45a99afa92f29a775f1362e60d6bb4ef5c61c5863f6497f0173d`。
它将 P3 固定为 20% 采样概率质量，不把双链路视为独立演奏者，并锁定三轮
训练预算及 GuitarSet、AG-PT、Guitar-TECHS P2 的顺序门禁。下一步只能先
实现并验证该固定训练合同；实现回归测试通过前不得启动训练，也不得读取
新的 governed/cross-source 结果来调整方案。

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

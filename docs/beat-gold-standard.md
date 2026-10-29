# 节拍层金标准

## 目标与边界

节拍层金标准优先解决：

- 拍点、强拍、小节线、拍号和弱起判断；
- 稳定速度、渐快渐慢和 rubato 的逐拍 Tempo Map；
- 音符起音映射到八分、十六分、三连音等谱面位置；
- 后续和弦窗口、重复拨弦和技法时间窗的共同时间基准。

它能显著提升谱面节奏与下游时间对齐，但不会直接修正音高、弦品或技法分类。
这些层必须在节拍层稳定后使用各自的金标准继续评测。

## 语料规模

最终目标为 300–500 个 15–30 秒片段，并保留 8–12 首完整歌曲作为长期回归。
同一录音的全部片段只能进入同一个数据分区，禁止按片段随机拆分造成泄漏。

| 阶段 | 规模 | 用途 |
| --- | ---: | --- |
| Pilot | 48 片段 | 校验标注规范、工具和评测阈值 |
| Expansion | 200 片段 | 覆盖主要速度、拍号和演奏纹理 |
| Gate | 400 片段以上 | 锁定 validation/test，作为生产门禁 |

建议分区为 train 50%、validation 20%、test 30%。`test` 在裁决后冻结，算法调参
不得读取其逐片段错误。

## 覆盖配额

- 拍号：4/4 约 55%，3/4 约 8%，6/8 与 12/8 约 15%，奇数或混合拍号
  约 7%，自由速度或 rubato 约 15%。
- 速度：小于 70、70–100、100–140、大于 140 BPM 四档均不少于 15%。
- 节奏：弱起、切分、shuffle/swing、三连音、十六分、停顿、渐快渐慢均不少于
  30 个片段。
- 纹理：原声扫弦、指弹、电吉他清音、失真、强鼓点、无鼓、完整混音和分离吉他
  音轨均需覆盖。
- 质量：现场版、录音室版、串音、压缩失真和低信噪比素材均需保留独立标签。

## 标注协议

每个片段记录 `at`、拍序号、小节号、强拍标记、拍号和弱起拍数。流程固定为：

1. 系统先给出完整录音的速度、主拍号、首个强拍、弱起和变拍候选。
2. 标注者试听开头、中段和结尾三个代表位置；结果正确时直接确认提交，发现偏差时
   才展开拍号、强拍和变拍纠正。强拍相位无法可靠确认时必须标记为“不确定”，不得
   强行选择。
3. 每提交一首录音的全曲结构，即开放该录音的片段时间复核，无需等待其他录音。
4. 两名标注者独立创建并提交同一片段，不共享对方结果。
5. 系统生成 Beat/Downbeat F1、时间误差、拍号和弱起一致性报告。
6. 第三名裁决者监听原始片段并提交最终时间轴。
7. 只有第三方裁决结果标记为 `adjudicated_gold`，才允许进入评测。

全曲结构复核不要求逐拍校正整首歌。候选拍网格只用于定位首强拍和变拍点；
逐拍时间误差仍在第二阶段的短片段中校正。相位标记为“不确定”的录音可以继续
进入 Beat 时间复核，但在相位裁决前不得进入 Downbeat 门禁。当前自动候选没有
量化置信度，人工确认仍是进入金标准流程的必要门禁。

使用 `npm run beat:gold --` 执行：

```bash
npm run beat:gold -- init \
  --audio data/beat-sources/song.wav \
  --analysis data/beat-candidates/song-analysis.json \
  --review data/beat-reviews/song-001-a.json \
  --clip-id song-001 \
  --group-id song-recording \
  --annotator-alias reviewer-a \
  --start 30 \
  --duration 24 \
  --pre-roll 4 \
  --post-roll 4 \
  --corpus-version beat-pilot-v2 \
  --tag syncopated

npm run beat:gold -- submit \
  --review data/beat-reviews/song-001-a.json

npm run beat:gold -- compare \
  --left data/beat-reviews/song-001-a.json \
  --right data/beat-reviews/song-001-b.json \
  --output data/beat-reports/song-001-agreement.json

npm run beat:gold -- adjudicate \
  --left data/beat-reviews/song-001-a.json \
  --right data/beat-reviews/song-001-b.json \
  --resolved data/beat-reviews/song-001-c.json \
  --output data/beat-gold/song-001.gold.json \
  --adjudicator-alias reviewer-c

npm run beat:gold -- evaluate \
  --review data/beat-gold/song-001.gold.json \
  --analysis data/beat-candidates/song-analysis.json \
  --output data/beat-reports/song-001-evaluation.json

npm run beat:gold -- gate \
  --report data/beat-reports/song-001-evaluation.json
```

## 门禁

- Beat F1 不低于 0.95；
- Beat 中位误差不高于 20ms，P95 不高于 50ms；
- Downbeat F1 不低于 0.95；
- 拍号准确率 100%；
- 弱起拍数必须完全一致；
- 任何未裁决、哈希变化或同一人重复标注的片段不得进入门禁。

## 首批曲库

公版曲目只代表作品本身可能已进入公版，具体录音仍可能受保护。应使用自行演奏、
自行渲染或明确 CC0/兼容授权的录音。

| 曲目 | 重点 | 素材要求 |
| --- | --- | --- |
| 原创基础扫弦 80/96/120 BPM | 稳定 4/4、弱起 | 项目自行录制或生成 |
| 原创 6/8 指弹 | 复合拍、琶音 | 项目自行录制或生成 |
| 原创 swing/shuffle 练习 | 三连音底层 | 项目自行录制或生成 |
| Amazing Grace | 3/4 与自由速度 | 自行演奏或授权录音 |
| Greensleeves | 3/4、弱起 | 自行演奏或授权录音 |
| Spanish Romance | 3/4、连续琶音 | 自行演奏或授权录音 |
| Bach Bourrée in E minor | 2/2、复调起音 | 自行演奏或授权录音 |
| Scarborough Fair | 3/4 或 6/8 版本差异 | 自行演奏或授权录音 |
| 唐朝《国际歌》 | 4/4、前奏、速度漂移 | 已有本地候选，继续长期审核 |

以下录音仅作为用户本地评测源，不进入仓库或安装包：

| 优先级 | 曲目 | 覆盖点 | 建议片段 |
| ---: | --- | --- | ---: |
| 1 | Metallica – Nothing Else Matters | 6/8、琶音、弱起 | 4 |
| 2 | Eagles – Hotel California | 4/4、复合琶音、长前奏 | 4 |
| 3 | Kansas – Dust in the Wind | 稳定指弹、十六分 | 3 |
| 4 | The Beatles – Blackbird | 切分、拍内位移 | 3 |
| 5 | Pink Floyd – Wish You Were Here | 弱起、扫弦与单音切换 | 3 |
| 6 | Eric Clapton – Tears in Heaven | 指弹、切分、rubato | 3 |
| 7 | Dire Straits – Sultans of Swing | laid-back、连续单音 | 3 |
| 8 | Led Zeppelin – Stairway to Heaven | 速度与织体变化 | 4 |
| 9 | Jimi Hendrix – Little Wing | rubato、装饰音密集 | 3 |
| 10 | John Mayer – Neon | 打板、切分、复杂指弹 | 3 |
| 11 | Metallica – Master of Puppets | 高速十六分、混合小节 | 4 |
| 12 | Metallica – Enter Sandman | palm mute、稳定强拍 | 3 |
| 13 | Nirvana – Smells Like Teen Spirit | 强弱变化、失真扫弦 | 3 |
| 14 | Oasis – Wonderwall | 持续切分扫弦 | 3 |
| 15 | Tracy Chapman – Fast Car | 稳定指弹、重复型 | 3 |
| 16 | Sting – Shape of My Heart | 切分琶音 | 3 |
| 17 | 许巍《蓝莲花》 | 中文摇滚、稳定 4/4 | 3 |
| 18 | Beyond《海阔天空》 | 前奏、段落速度稳定性 | 3 |
| 19 | Beyond《真的爱你》 | 分解和弦与扫弦 | 3 |
| 20 | 朴树《平凡之路》 | 重复分解型、完整混音 | 3 |

Pilot 不需要一次收集全部素材。优先准备表中前 10 首的本地无损文件或最高质量
原始文件，每首先截取 4–5 个互不重叠片段，形成 48 片段候选池。

### Pilot 准备脚本

本地受保护媒体必须先通过桌面端已有的官方授权链路解码为标准音频，并放入 Git
忽略目录。脚本不读取平台账号、不执行解码，也不会把源音频、片段或审核结果写入
仓库。当前 Pilot v2 使用 10 个录音、每段 24 秒评测窗，并在前后各附加 4 秒
只听上下文，共 48 段：

- train：24 段；
- validation：10 段；
- test：14 段；
- 同一录音只属于一个分区。

48 段是第二阶段候选池，不再要求首轮全部复核。每首录音确认全曲结构后，即可
进入该录音的核心片段复核。当前 10 首录音默认选择 24 段：

- 节奏稳定且重复度高的录音默认保留 2 段；
- 有明显段落、织体或速度变化的录音默认保留 3 段；
- rubato、混合拍号或变速录音可保留 3–5 段；
- 其余片段仅在低置信度、结构变化或双人审核分歧时补充。

相位不确定或确认存在变拍时，自动增加至最多 5 段。已经提交或勾选检查项的
片段始终保留在核心队列中，避免隐藏正在进行的审核。

现有 48 段清单与草稿必须保留，不得通过 `--force` 重建；缩减核心队列只改变
第二阶段的任务选择，不覆盖候选音频或既有审核记录。“全部候选”可查看当前已确认
录音的全部片段；未确认录音保持锁定。

片段页支持主动应用同一审核者已提交的全曲结构，更新拍号、弱起和连续拍位，
保持已有拍点时间，并重置强拍和拍号检查。应用后需试听并保存；相位不确定、来源
不匹配或片段内部存在变拍时，禁止应用。该操作不会自动写入或覆盖草稿。

准备命令：

```bash
npm run beat:pilot -- \
  --source-dir .cache/beat-gold/sources
```

脚本输出到 `.cache/beat-gold/pilot-v2/`，包括逐录音节拍候选、48 个 PCM
审核片段、指定审核者的 draft 文件和 `pilot-manifest.json`。先生成
`reviewer-a`，再使用同一源目录独立初始化 `reviewer-b`：

```bash
npm run beat:pilot -- \
  --source-dir .cache/beat-gold/sources \
  --annotator-alias reviewer-b
```

不得使用 `--force` 覆盖已经开始人工复核的队列。manifest 固定记录
`corpus_version`、语料摘要、音频边界和评测窗，审核文件不得修改这些字段。

前后上下文只用于判断完整小节和裁剪相位，不进入 Beat/Downbeat 指标；评分只读取
`evaluation_start <= at < evaluation_end` 的拍点。拍点始终保留全曲小节号与拍位，
不得把裁剪后的第一个拍点重新编号为强拍。Pilot v1 已归档，不得继续标注。

这些候选仅用于开始人工标注，状态固定为 `draft`、`benchmark_eligible: false`；
完成双人独立标注和第三方裁决前，不得作为评测金标准或用于宣称准确率。

### 本地审核后台

真实人工复核使用冻结快照，避免日常开发热更新影响正在进行的审核：

```bash
npm run beat:review:freeze
npm run beat:review:preview
```

`freeze` 会构建并复制当前 API/Web 制品，但不会复制或修改真实 Pilot。`preview`
使用独立运行目录和固定端口，继续只读同一份 Pilot 源数据；除非审核员主动保存或
提交，否则启动、刷新和页面导航不会改写审核文件。

访问地址：

```text
http://127.0.0.1:5191/admin/login
```

API 固定使用 `http://127.0.0.1:8791`。访问 `localhost:5191` 会自动跳转到
`127.0.0.1`，确保管理会话与 API 处于同一站点。冻结预览只监听本机回环地址，
可与 `5173/8787` 的日常开发环境并行运行。

当前本地预览 owner 账号为 `admin`，开发期密码为 `Admin@123456`。删除隔离运行
目录后，首次访问会重新进入 owner 初始化。账号状态保存在
`.cache/beat-gold/review-preview/runtime/admin/state.json`，审计日志保存在同目录
的 `audit.jsonl`；密码只保存 scrypt 盐值和摘要，不保存明文。会话只保存在 API
进程内存中，重启预览后需要重新登录。

后台使用 HttpOnly、SameSite 会话，并提供 owner、reviewer、adjudicator、
model-maintainer 和 viewer 五类角色。管理 API 统一位于 `/api/admin/*`，不与
普通用户接口混用。

后台与普通用户工作台隔离，不显示在用户导航中。仪表盘展示语料版本、队列数量、
分区、审核进度和生产门禁；侧栏按以下层级组织：

1. 语料与权利、金标准版本；
2. 节拍、起音/拨弦、音高、弦品、和弦、技法；
3. 差异与裁决、评测与门禁、模型与实验；
4. 用户与审计、系统与备份。

当前仅开放节拍层。后续层按“节拍 → 起音/拨弦 → 音高 → 弦品 → 和弦 → 技法”
锁定依赖，前置层未通过时禁止写入。

节拍复核地址：

```text
http://127.0.0.1:5191/admin/reviews/beat
```

页面只读取 manifest 白名单内的本地 Pilot 文件，支持上下文试听、评测窗标识、
不完整小节提示、全曲拍位调整、拍点增删与 10ms 微调、拍号、弱起、完整性检查、
草稿保存和独立提交。上下文拍点禁止拖动和增删。提交后文件锁定，后台差异比较与
第三方裁决沿用 `beat:gold` 的可移植 JSON schema 和规范摘要算法。

灰色上下文中的候选拍点仅用于判断连续拍位和小节边界，不参与指标或训练导出，
也不要求人工修正；候选时间明显错误时，以音频和亮色评测窗内的拍点为准。时间轴
徽标始终显示当前拍位，强拍显示 `1`；小节号只在悬停提示和拍点摘要中显示。

不同审核者使用互相隔离的目录。owner 可切换队列；reviewer 账号只能进入绑定的
审核者队列。人工双审时应在“用户与审计”中创建两个独立账号，并分别绑定
`reviewer-a` 与 `reviewer-b`：

```text
http://127.0.0.1:5191/admin/reviews/beat?reviewer=reviewer-a
```

每次提交同时记录管理账号 ID 和用户名。A、B 使用同一管理账号提交时，不满足
独立审核条件；裁决者也必须使用不同于 A、B 的第三个账号。最后一个有效 owner
不能被停用，所有账号创建、登录、修改和审核操作均写入审计日志。

### 后台闭环

1. `reviewer-a` 与 `reviewer-b` 分别完成 4 项检查并独立提交；任一方未提交时，
   “差异与裁决”只显示等待状态。
2. 第三方裁决员查看 Beat/Downbeat F1、时间误差、拍号、弱起和逐拍差异，生成
   `adjudicated_gold`。裁决结果写入 `gold/`，不覆盖 A/B 原始提交。
3. “评测与门禁”使用裁决金标准计算覆盖率和质量阈值；冻结报告会保存模型版本、
   语料摘要、指标、门禁结果与报告摘要，供后续回归比较。
4. 48 段全部裁决后，“金标准版本”才能冻结不可变目录，并生成 `.tar.gz` 与
   SHA-256。相同版本标识不能覆盖已有版本。
5. “模型与实验”只把已裁决的 `train` 分区转换为训练数据，禁止读取
   `validation/test` 作为训练样本，并执行跨分区录音泄漏检查。当前训练门槛为
   100 个 train 金标准；不足时任务只记录为 `blocked`，达到门槛后才进入
   `queued`。
6. “系统与备份”归档 manifest、分析、双人草稿、金标准、评测和版本数据，同时
   保存归档 SHA-256。恢复操作只解压到 `.restores/` 做清单与语料摘要校验，不
   覆盖当前 Pilot。

后续层页面已经固定 schema、字段和依赖状态，但在前置层完成前保持只读锁定。
依赖顺序为“节拍 → 起音/拨弦 → 音高 → 弦品 → 和弦 → 技法”。

生产环境初始化 owner 时必须配置 `GTS_ADMIN_BOOTSTRAP_TOKEN`，并在初始化请求中
提供对应令牌。只有本地开发、桌面本地运行，或生产环境显式配置绝对路径
`GTS_BEAT_GOLD_ROOT` 时，API 才会发现 Pilot 数据。

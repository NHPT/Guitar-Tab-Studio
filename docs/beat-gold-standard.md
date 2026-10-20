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

1. 两名标注者独立创建并提交同一片段，不共享对方结果。
2. 系统生成 Beat/Downbeat F1、时间误差、拍号和弱起一致性报告。
3. 第三名裁决者监听原始片段并提交最终时间轴。
4. 只有第三方裁决结果标记为 `adjudicated_gold`，才允许进入评测。

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
原始文件，每首先截取 2–4 个互不重叠片段，即可开始第一轮 48 片段标注。

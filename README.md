# Guitar Tab Studio

HackAll 旗下，面向吉他、贝斯等弦乐学习者的 AI 扒谱与分轨练习工作台。

## 当前版本

`v0.1.0` 提供：

- 本地音频/视频上传与异步分析任务。
- 哔哩哔哩、抖音和网易云音乐公开链接识别与来源策略；
  支持从应用分享文案中提取 URL，并即时高亮识别出的平台。
- 默认单行横向流水谱，可切换完整曲谱；包含和弦名称、指法图、拨/扫节奏、
  连续六线谱、节奏连梁、奏法标记、自然/人工泛音和歌词行。
- 播放时以固定中心播放线连续滚动曲谱；点选和弦、节奏或单音后才打开对应的
  局部校谱器。
- 提供动态波形、动感波光、心电图、频谱脉冲、呼吸光带、鼓点弹球和
  反应堆控制杆 7 种实时播放可视化；左侧保留已播放历史，当前位置显示
  音频特征，右侧不预绘。
- 实验性推断扫弦、琶音、轮扫、击弦、勾弦、滑音、泛音、拍弦、切音、
  闷音、打板和轮指。Worker v8 会输出独立技法置信度、前三候选、声学证据及
  击勾滑关联音符；技法结果尚不是训练模型结论，必须人工校对。
- 默认启用智能变调夹建议，也可关闭或手动选择 `0–12` 品。
- 内置 10 套可持久化界面配色。
- 吉他、贝斯和鼓声部切换；未生成的声部明确显示为空，不复用其他声部伪装。
- 仅显示检测到的有效音轨，并提供静音、独奏和音量控制；当前模型支持人声、
  吉他、贝斯、鼓组、钢琴/键盘及其他乐器聚合轨。
- 速度调整、小节循环、真实分轨同步播放和 PDF 打印。
- 项目持久化与可恢复的曲谱 URL。
- 独立 Python 推理 Worker，实现 Demucs 分轨、Basic Pitch 转录、
  指板映射、基础和弦及连奏技法推断。
- 社区协作控制面，支持隐藏校准、盲审共识、贡献撤回、数据版本、带认证租约的
  受控训练实验、模型门禁、灰度与运营指标；人工复核异步运行，不阻塞其他开发。

真实 Worker 不可用或平台下载失败时，导入任务直接失败，不会生成示例结果。

## 本地启动

要求 Node.js 20 或更高版本。

```bash
npm install
npm run dev
```

- Web：<http://localhost:5173>
- 社区协作：<http://localhost:5173/community>
- API：<http://127.0.0.1:8787>
- 健康检查：<http://127.0.0.1:8787/health>

正式 Web 域名规划为 `guitar.hackall.cn`，由宿主机 Nginx 同域转发 `/api` 与
`/media`，社区入口使用 `/community`。

## 推理 Worker

当前项目已在 `.venv` 中配置 Apple Silicon Python 3.10、Demucs、Basic Pitch
和 yt-dlp，并通过 `ffmpeg-static` 提供项目内 FFmpeg。API 会自动探测该环境。

```bash
./.venv/bin/guitar-tab-worker \
  --input /path/to/song.mp3 \
  --output /path/to/result
```

API 会将上传文件或允许处理的公开链接交给 Worker，并读取生成的
`analysis.json`、压缩分轨、波形、和弦和 TAB 音符。

训练后的六弦品位模型必须通过独立测试集门禁后才可显式启用：

```bash
STRINGTRACE_TAB_MODEL=/absolute/path/to/string-fret.pt \
./.venv/bin/guitar-tab-worker \
  --input /path/to/song.mp3 \
  --output /path/to/result
```

未配置模型时继续使用 Basic Pitch 基线，并在 `analysis.json` 的
`transcription_model` 中明确标记。数据导入、训练和评估流程见
[模型训练与数据规范](docs/model-training.md)。
当前稳定训练候选 v5 在 GuitarSet player-05 遗留基准上的 TAB F1 为
`0.7367`、重复音召回为 `0.7187`，仍未通过生产门禁，因此不会自动替换
默认 Basic Pitch 后端。player-05 已参与多轮门禁和误差分析，不能替代未来
不少于 100 首的全新密封盲测集。

长期数据建设采用社区微任务与受控训练：普通用户可复核起音、漏音、误报和时间
边界，弦品与技法按任务类型校准后解锁；社区实验只能使用批准配方和隔离执行器。
人工复核作为异步数据工作流，不阻塞产品、桌面、离线推理和其他数据上的模型开发。
任何社区贡献都不能直接改写金标准、读取隐藏测试或发布生产模型；当前单人阶段由
项目所有者在自动门禁后完成最终批准。完整规划见
[社区标注与开放训练路线图](docs/community-ml-roadmap.md)。

开发环境首次创建的社区账号自动成为本地所有者。正式部署必须设置
`GTS_COMMUNITY_BOOTSTRAP_KEY`，由首个所有者注册时提供；运行状态写入
`apps/api/data/community/`，不会提交到版本库。账号访问密钥只在客户端保存，
可从社区页右上角复制并用于其他浏览器登录。

外部训练执行器使用独立机器凭据，不复用社区账号。服务环境通过
`GTS_TRAINING_RUNNER_CREDENTIALS` 配置 JSON 映射，例如
`{"gpu-01":"至少 24 字符的随机密钥"}`；`GTS_TRAINING_LEASE_SECONDS`
可将租约设为 30 到 3600 秒，默认 300 秒。执行器以
`Authorization: Runner <id>.<secret>` 调用 claim、heartbeat 和 result 接口。
过期任务自动重新排队，结果仅接受当前租约持有者回传；模型必须先上传至受保护的
制品目录并由 API 流式计算 SHA-256。任务体不包含 shell、宿主机路径、社区身份或
任意代码。

仓库内 runner 只执行版本化白名单配方。macOS 开发环境使用 `sandbox-exec`
禁用网络并限制写目录；生产 GPU runner 仍应迁移到 rootless Podman。启动示例：

```bash
GTS_RUNNER_ID=gpu-01 \
GTS_RUNNER_SECRET='与 API 配置一致的随机密钥' \
npm run training:runner
```

生产 runner 可使用 [Containerfile](services/training_runner/Containerfile)
构建固定镜像，并按镜像摘要同时配置 API 的 `GTS_TRAINING_IMAGE` 与 runner 的
`GTS_RUNNER_IMAGE`。运行时使用：

```bash
GTS_RUNNER_IMAGE='registry.example/gts-training@sha256:<digest>' \
GTS_RUNNER_GPU_DEVICE='nvidia.com/gpu=all' \
npm run training:runner -- --isolation podman
```

Podman 后端固定启用 rootless user namespace、只读根文件系统、无网络、无
capabilities、`no-new-privileges`、PID/CPU/内存限制和独立读写卷。当前未在本机
构建该镜像，避免未经确认写入用户级 Podman 镜像存储。

模型的谱系、复现、公开验证和稳健性结果由独立门禁服务写入，维护者不能手工将这些
检查改为通过。API 使用 `GTS_PROMOTION_GATE_CREDENTIALS` 配置独立机器密钥，门禁
进程使用 `GTS_GATE_ID` 和 `GTS_GATE_SECRET` 启动：

```bash
GTS_GATE_ID=promotion-gate-01 \
GTS_GATE_SECRET='与 API 配置一致的随机密钥' \
npm run promotion:gate
```

未来密封评测服务使用单独的 `GTS_SEALED_GATE_CREDENTIALS` 和
`Authorization: SealedGate <id>.<secret>`，公开门禁密钥无权写入密封评测结果。
仓库已提供 `npm run sealed:gate` 执行入口：它要求显式提供至少 100 条全新
`test` 记录、冻结清单/基线哈希和隔离缓存，只向主 API 返回通过/失败与证据哈希，
完整指标保留在密封环境。密封数据和凭据未部署前，任何候选都无法满足生产晋级条件。

当前真实配方包括合成数据隔离 smoke、GuitarSet 弦品起音网络和社区冻结清单审计；
未完成执行规范的 Activity Head 显示为待部署且不能提交。内置数据只有在本机存在
且启动时计算出清单哈希后才会出现在控制台。

处理在线链接时：

```bash
./.venv/bin/guitar-tab-worker \
  --url "https://www.bilibili.com/video/..." \
  --output /path/to/result
```

Worker 只处理公开且用户有权使用的非 DRM 媒体。会员可播放或歌曲未标 VIP
都不代表平台提供可导出的媒体流。受保护内容不得绕过登录、付费或加密控制，
应由用户上传其合法持有的 MP3、FLAC、APE、WAV、M4A 等标准文件。QQ 音乐
和汽水音乐不提供在线导入。QMC、MFLAC、MGG、NCM、KGM、VPR 等专有加密
格式不会解密；若 MFLAC/MGG 文件内容本身实际是标准 FLAC/OGG/MP3，则通过
文件签名检测后仍可正常处理。

## 验证

```bash
npm run test
npm run test:runner
npm run test:gate
npm run test:sealed-gate
npm run build
npm run test:e2e
.venv/bin/python -m unittest discover -s services/worker -p 'test_*.py'
.venv/bin/python -m py_compile services/worker/worker.py services/worker/stringtrace_ml/*.py
```

## 文档

- [产品范围](docs/product-scope.md)
- [产品与平台路线图](docs/product-roadmap.md)
- [社区标注与开放训练路线图](docs/community-ml-roadmap.md)
- [系统架构](docs/architecture.md)
- [吉他技法精准识别实施计划](docs/technique-recognition-roadmap.md)
- [模型训练与数据规范](docs/model-training.md)

## License

本项目采用 [Apache License 2.0](LICENSE) 开源。

# Guitar Tab Studio

面向吉他、贝斯等弦乐学习者的 AI 扒谱与分轨练习工作台。

## 当前版本

`v0.1.0` 提供：

- 本地音频/视频上传与异步分析任务。
- 图片或 PDF 吉他谱导入：PDF 在客户端逐页解析，服务端识别六线谱、小节、
  弦品、和弦与歌词，并生成可编辑、可合成播放的项目；低置信度符号保留待校对。
- 支持哔哩哔哩、抖音、网易云音乐、QQ 音乐、YouTube、Vimeo、SoundCloud、
  Bandcamp、Jamendo、Audius、Internet Archive 和 Wikimedia Commons
  的公开链接；可从分享文案中提取 URL 并即时识别来源。
- 默认单行横向流水谱，可切换完整曲谱；包含和弦名称、指法图、拨/扫节奏、
  连续六线谱、节奏连梁、奏法标记、自然/人工泛音和歌词行。
- 播放时以固定中心播放线连续滚动曲谱；点选和弦、节奏或单音后才打开对应的
  局部校谱器。
- 提供动态波形、动感波光、心电图、频谱脉冲、呼吸光带、鼓点弹球和
  反应堆控制杆 7 种实时播放可视化；左侧保留已播放历史，当前位置显示
  音频特征，右侧不预绘。
- 实验性推断扫弦、琶音、轮扫、击弦、勾弦、滑音、泛音、拍弦、切音、
  闷音、打板和轮指。Worker v9 会输出独立技法置信度、前三候选、声学证据及
  击勾滑关联音符；技法结果尚不是训练模型结论，必须人工校对。
- 默认启用智能变调夹建议，也可关闭或手动选择 `0–12` 品。
- 内置 10 套可持久化界面配色。
- 吉他、贝斯和鼓声部切换；未生成的声部明确显示为空，不复用其他声部伪装。
- 仅显示检测到的有效音轨，并提供静音、独奏和音量控制；当前模型支持人声、
  吉他、贝斯、鼓组、钢琴/键盘及其他乐器聚合轨。
- 速度调整、小节循环、真实分轨同步播放和 PDF 打印；预备节与鼓点分别支持
  音色、音量设置并持久化。
- 项目持久化与可恢复的曲谱 URL。
- 内置 John Denver 的 `Take Me Home, Country Roads` 作为默认项目，包含
  自动生成的 132 小节 TAB，以及人声、双吉他、贝斯、鼓组、键盘和其他乐器
  7 条可独立静音、独奏和调节音量的分轨。
- 不可变项目修订、显式恢复记录，以及逐音保持绝对音高的原始谱 / 易弹版双层编辑。
- `@gts/core` 共享应用核心，统一 Web/API 的项目类型、媒体策略、版本化
  `analysis.json` 解析以及推理和仓储接口；旧项目读取时迁移到 schema v1。
- Electron 桌面壳层复用同一工作台，提供 SQLite、最近项目、拖放、备份恢复、
  诊断导出、本地/云端推理选择和经哈希校验的模型包回滚。
- 默认启用磁盘自动清理；运行中任务、项目分轨、修订、训练数据和模型包受保护，
  管理员可调整阈值、保留时间、手动执行或关闭自动清理。
- 独立 Python 推理 Worker，实现 Demucs 分轨、Basic Pitch 基线、
  可选 MuScriptor 高精度转录、指板映射、基础和弦及连奏技法推断。
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

正式 Web 域名由部署环境配置，宿主机 Nginx 同域转发 `/api` 与 `/media`，
社区入口使用 `/community`。

## 存储自动清理

API 默认在可用空间低于 `5 GB` 时清理到 `8 GB`，并保留最近 24 小时的失败任务
和上传临时文件。服务启动、每 30 分钟、任务结束及新分析空间不足时都会执行检查。
完成项目只回收原始下载副本、标准化 WAV、分析 JSON 和未使用的分轨中间文件；
项目引用的压缩分轨、项目状态、修订、训练数据、社区制品和模型包不会自动删除。

桌面版可在“运行模型与存储”面板中直接配置。服务端部署需设置
`GTS_RUNTIME_ADMIN_TOKEN`，随后使用 Bearer 令牌调用：

```text
GET  /api/runtime/storage-cleanup
PUT  /api/runtime/storage-cleanup
POST /api/runtime/storage-cleanup/run
```

配置持久化到应用数据目录的 `state/storage-cleanup.json`，自动清理默认启用，
管理员关闭后仍可手动执行一次安全清理。

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

桌面版可在“运行模型与存储 > 转录方案”中选择基础、高精度融合、吉他专用
或多轨研究方案。应用根据
系统内存、CPU 和 MPS/CUDA 能力自动推荐 Small、Medium 或 Large，用户可直接
接受推荐，也可手动固定档位。官方权重不会随安装包分发：用户必须先确认
CC BY-NC 4.0 非商用许可并在 Hugging Face 完成授权。

### Hugging Face OAuth 配置

> 这一步面向开发者、自托管者和发行方。正式发行包应内置发行方创建的公共
> Client ID。普通用户不应自行创建 OAuth 应用，但首次下载时仍需登录
> Hugging Face、取得对应 gated 模型的访问资格并确认模型许可。

打开 [Hugging Face OAuth 应用创建页](https://huggingface.co/settings/applications/new)
并登录账号，然后按下表填写。该桌面客户端使用 Authorization Code + PKCE，
因此必须创建 **Public OAuth App**，不使用 Client Secret。

| Hugging Face 字段 | 推荐填写值 | 说明 |
| --- | --- | --- |
| Application Name | `Guitar Tab Studio` | 会显示在用户授权页面，可按发行渠道增加后缀，例如 `Guitar Tab Studio Preview`。 |
| Homepage URL | `https://github.com/NHPT/Guitar-Tab-Studio` | 应填写用户能够访问的项目主页或产品官网。 |
| Logo URL | 留空 | 可选。只有已有公开 HTTPS 图片地址时才填写，本地文件路径无效。 |
| Scopes | 保留 `openid`、`profile`，增加 `gated-repos` | 只授予登录和读取已获准公开门控仓库所需权限；其他 scope 保持未选中，尤其不要启用 `write-repos` 或 `manage-repos`。 |
| Redirect URLs | `http://127.0.0.1/oauth/callback` | 必须逐字一致，不要添加端口、尾部斜杠、查询参数，也不要改成 `localhost`。 |

Hugging Face 允许回环地址在运行时使用随机端口。应用注册时填写上面的无端口
地址，实际授权时桌面程序会临时监听类似
`http://127.0.0.1:49152/oauth/callback` 的地址。只有端口可以变化，主机名和
路径必须与注册值一致。

创建完成后：

1. 点击 **Create application**。进入新应用的详情/设置页，确认没有
   **Client Secret**；如果存在，使用该页的删除 Secret 操作将其删除。
   Hugging Face 将没有 Secret 的 OAuth 应用视为 Public App。
2. 在同一详情页复制 **Client ID**。不要复制 Client Secret、Access Token
   或浏览器回调中的 Authorization Code。之后如需找回，依次进入
   **头像 > Settings > Connected Apps > Developer Applications**，再打开
   `Guitar Tab Studio`。
3. 打开“运行模型与存储”，保持“本地”，将“转录方案”切换为“高精度融合”。
4. 点击“创建 OAuth 应用”也可重新打开官方创建页；将 Client ID 粘贴到
   “OAuth 客户端 ID”，然后点击“保存 OAuth 配置”。
5. 点击程序中的“官方授权页”，在当前模型仓库接受许可或申请 gated access；
   OAuth 登录不能代替这一步。
6. 返回程序，确认 MuScriptor 的 CC BY-NC 4.0 非商用许可，点击“下载并启用”。
7. 桌面程序会打开系统浏览器。登录并同意授权后，Authorization Code 会自动
   回传到本机临时端口并换取令牌，无需手动复制；成功页面会尝试自动关闭。

也可在启动桌面程序前设置 `GTS_HF_OAUTH_CLIENT_ID`。Client ID 是公开的应用
标识，不是密码；Client Secret 不得写入桌面程序、仓库或安装包。

常见问题：

- `redirect_uri mismatch`：检查 Redirect URI 是否严格为
  `http://127.0.0.1/oauth/callback`。
- `invalid_client`：检查是否复制了 Client ID，并确认应用为 Public/无 Secret；
  如果现有应用无法删除 Secret，请重新创建公共应用。
- 下载返回 `401` 或 `403`：确认当前登录账号已在程序“官方授权页”对应的模型
  仓库接受许可或获得访问资格，并允许 `gated-repos` scope。
- 浏览器授权后没有返回：保持桌面程序运行，并检查防火墙是否阻止
  `127.0.0.1` 回环连接。

官方参考：[Sign in with Hugging Face](https://huggingface.co/docs/hub/oauth)。

点击“下载并启用”时，
桌面端会先检查并下载权重到应用管理的模型缓存，只有校验成功后才切换转录策略。
也可导入已下载的 `.safetensors` 权重。OAuth 令牌由 Electron `safeStorage`
加密后保存在应用私有数据目录；系统加密能力不可用时仅保存在当前进程内存。
高精度模式使用 MuScriptor 主转录，同时运行 Basic Pitch 做同音起点复核和
高置信度漏音补充；CUDA 环境下的 Medium/Large 档位再加入 MIT 许可的
MR-MT3 作为独立事件候选，最后由 Noise2Fret 对音高一致的候选执行弦品优化。
MR-MT3 当前不支持 MPS 加速，因此 Apple 芯片组合不会启用其 CPU 回退路径。
16 GB 统一内存设备自动选择 Small，避免连续运行多模型时发生大量换页；仍可
手动选择 Medium/Large。任何辅助模型失败都保留上一步的有效结果。

吉他专用方案使用 StringTrace 起音/弦品联合研究模型和可演奏性指法约束，
需要活动模型包中的 `transcription-model`；开发模式可使用仓库内 v24 研究
bundle。多轨研究方案使用 YourMT3+、Basic Pitch 和 Noise2Fret，首次启用时
下载并校验固定版本的 YourMT3+ 权重。两者均属于研究方案，不进入商业发行。

云端推理默认不启用。用户可在桌面运行时面板点击“云端 · 配置”，填写 HTTPS
API 地址并通过 `/health` 检查后切换；配置会保存在本地，也可继续通过
`GTS_CLOUD_API_ROOT` 预置。

MuScriptor 代码是 MIT 许可，官方权重是非商用许可。商业发行使用
`GTS_COMMERCIAL_BUILD=1` 或
`GTS_TRANSCRIPTION_PROVIDER_ALLOWLIST=basic-pitch`，会从桌面白名单和冻结
Worker 中排除 MuScriptor；业务代码、项目格式和 API/Worker 契约无需修改。
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

社区控制台已将晋级状态接入真实推理任务。模型维护者可配置影子采样、灰度比例、
错误预算和最小观测数。影子候选在非公开目录运行，用户始终获得基线结果；灰度或
生产制品失败时任务自动改用当前或上一稳定模型，仍不可用时回到内置基线。达到最小
观测数后，超预算灰度会被自动拒绝，超预算生产模型会自动退役并恢复上一冠军。
这些能力不依赖 Podman 镜像、生产 GPU 集群或全新密封数据集，因此相关基础设施
未就绪不会阻塞产品和控制面继续开发。

生产环境可同时配置 `GTS_DEPLOYMENT_ALERT_URL` 与
`GTS_DEPLOYMENT_ALERT_SECRET`。灰度自动拒绝或生产自动回滚时，API 会发送
不含用户内容的 HMAC-SHA256 签名 HTTPS 告警。

当前真实配方包括合成数据隔离 smoke、GuitarSet 弦品起音网络和社区冻结清单审计；
未完成执行规范的 Activity Head 显示为待部署且不能提交。内置数据只有在本机存在
且启动时计算出清单哈希后才会出现在控制台。

处理在线链接时：

```bash
./.venv/bin/guitar-tab-worker \
  --url "https://www.youtube.com/watch?v=..." \
  --output /path/to/result
```

Worker 会尝试读取各平台提取器能够直接取得的标准媒体流。QQ 音乐的歌曲详情
链接已接入；是否可取得音频取决于具体曲目、登录状态、地区限制和平台接口。
需要登录但未使用 DRM 的来源，可通过 `GTS_MEDIA_COOKIES_FILE` 指向用户导出的
Netscape `cookies.txt`；网络出口可通过 `GTS_MEDIA_PROXY` 配置。

平台官方导出的 MP3、FLAC、APE、WAV、M4A 等标准文件可直接上传。QMC、
MFLAC、MGG、NCM、KGM、VPR 等专有容器，以及可能受保护的 QQ 音乐和网易云
音乐链接，会先要求用户确认拥有版权、有效许可或其他合法处理权，并接受临时处理
免责声明。确认后，API 使用内置 Rust/WASM 算法在 API 运行设备上按块解码，再继续
执行分轨和曲谱生成。桌面本地模式下 API 运行在用户电脑，浏览器模式下 API 运行
在网站后端服务器。在线链接仅由现有提取器先下载到 API 临时目录，若得到的是专有
容器，再进入同一解码流程。

当前本地解码器支持 NCM、QMC1、带内嵌密钥的 QMC2/MFLAC/MGG、KWM、
KGM/VPR，以及导入匹配密钥库后的 KWM v2 和 KGM v5。临时标准媒体权限为
`0600`、最大 2 GB，不提供下载、不写入项目媒体清单，并在任务完成、失败或取消后
删除。若专有扩展名文件的内容本身已经是标准音频，则按文件签名直接进入普通流程，
不触发解码。

部分新版 QMC2/MFLAC/MGG 使用 MusicEx 页脚，不在文件中保存内容密钥。macOS
桌面版会在用户确认后读取本机 QQ 音乐的当前登录态，使用文件中的官方媒体 MID 和
加密文件名调用 QQ 音乐 `music.vkey.GetEVkey` 接口。只有当前账号获准访问该曲目
和音质时才会返回 ekey，随后仍由本机解码。登录凭据和 ekey 只存在于单次任务内存，
不写入项目、日志或配置文件；系统不调用第三方解码或密钥服务。由于登录信息位于
另一个应用的沙箱容器，macOS 可能要求先在“隐私与安全性 → 完全磁盘访问权限”中
启用 Guitar Tab Studio；发生权限拒绝时，导入窗口会提供直达设置页的按钮。

浏览器与桌面版的媒体能力不同：

| 能力 | 浏览器 | macOS 桌面版 | Windows / Linux 桌面版 |
| --- | --- | --- | --- |
| 文件选择 | 浏览器标准文件选择器；不能直接跳转到音乐客户端沙箱目录 | 系统原生窗口，提供 QQ 音乐、网易云音乐、MOO、酷我、酷狗下载目录快捷入口 | 系统原生窗口及对应平台目录快捷入口 |
| 解码执行位置 | 网站 API 服务器 | 用户电脑上的本地 API | 用户电脑上的本地 API |
| 内嵌密钥 QMC2、NCM、KWM、KGM/VPR | 支持，由 API 服务器处理 | 支持，本机处理 | 支持，本机处理 |
| 无内嵌密钥的 MusicEx QMC2 | 不支持读取用户电脑上的 QQ 音乐登录态 | 支持；取得完全磁盘访问权限后，使用当前 QQ 音乐登录态向官方接口验证授权并取得临时 ekey | 暂不支持自动提取 QQ 音乐登录态 |
| 分离密钥 KWM v2 / KGM v5 | 不接收用户设备密钥库 | 支持手动导入本机密钥库；仅保留于本次运行内存 | 支持手动导入本机密钥库；仅保留于本次运行内存 |
| 解码后媒体 | 不提供下载，任务结束即删除 | 不提供下载，任务结束即删除 | 不提供下载，任务结束即删除 |

若 QMC2 文件的 MusicEx 页脚缺失或损坏，工具不会猜测媒体 MID；请在原客户端确认
下载完成后重新选择文件。

QQ MusicEx 是唯一通过当前客户端登录态向平台官方接口逐文件取得临时 ekey 的
流程。NCM 的内容密钥位于文件容器内部，旧版 KWM、KGM/VPR 和 MOO/QMC1 使用
文件内信息或固定算法，不需要读取平台账号。

KWM v2 与 KGM v5 使用客户端已经保存的分离密钥，不套用 QQ 音乐接口。桌面导入
窗口提供“导入酷我密钥库”和“导入酷狗密钥库”：KWM v2 支持酷我 Android/iOS
MMKV，按资源 ID 和音质 ID 匹配；KGM v5 支持酷狗 Android MMKV 及桌面端加密
SQLite，按音频哈希匹配。解析出的映射只保留在当前桌面 API 进程内存，退出即清空。
酷狗 SQLite 解析时只创建权限为 `0600` 的临时副本，并在查询后立即删除。移动端
数据库的导出受系统沙箱限制，Android 通常需要 root、iOS 通常需要越狱；应用不会
尝试绕过设备权限，也不会将密钥库上传到远程服务。

解码核心使用 `@clamber_l/crypto 0.1.12`（MIT / Apache-2.0），NCM 容器解析
使用项目内的流式实现；所有运行时代码随 API 和 Electron 桌面包本地分发，无需
配置第三方 URL 或 Token。完整设计见
[系统架构](docs/architecture.md#local-authorized-media-decoder)。

在线获取失败时，界面会区分登录/验证、网络限制、链接格式、地区不可用、
无可解码格式和 DRM，不再统一显示成版权提示。

## 验证

```bash
npm run test
npm run test:runner
npm run test:gate
npm run test:sealed-gate
npm run build
npm run test:e2e
npm run desktop:smoke
npm run package:dir -w @gts/desktop
.venv/bin/python -m unittest discover -s services/worker -p 'test_*.py'
.venv/bin/python -m py_compile services/worker/worker.py services/worker/stringtrace_ml/*.py
```

## 文档

- [产品范围](docs/product-scope.md)
- [产品与平台路线图](docs/product-roadmap.md)
- [社区标注与开放训练路线图](docs/community-ml-roadmap.md)
- [系统架构](docs/architecture.md)
- [桌面分发说明](docs/desktop-distribution.md)
- [隐私与数据治理](docs/privacy-and-data-governance.md)
- [内置初学示例与来源说明](docs/builtin-demo.md)
- [吉他技法精准识别实施计划](docs/technique-recognition-roadmap.md)
- [模型训练与数据规范](docs/model-training.md)
- [第三方组件声明](THIRD_PARTY_NOTICES.md)

## License

本项目采用 [Apache License 2.0](LICENSE) 开源。

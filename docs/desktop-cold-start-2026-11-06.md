# 桌面端冷启动优化与 packaged smoke

## 结论

默认 `basic-pitch / auto / built-in` 路径不再在本地 API 启动前同步执行
Worker `--hardware-probe`。全新隔离数据目录下，packaged app 的首个完整窗口
在 `9.012s` 可访问；完整 `runtime-status` 在 `32.240s` 返回 ready。

首次主动查询转录状态仍会执行硬件探针，因此首窗之后还有 `23.228s` 的运行时
状态等待，但该等待已移出首窗阻塞路径。此前安装 smoke 仅能观察到 renderer
在 20 多秒后可自动化访问，本轮首窗已明显提前。

## 改动

`DesktopTranscriptionManager.workerConfiguration()` 先读取有效偏好。若有效
provider 为 Basic Pitch，直接返回内置 Worker 配置；只有研究 provider 才构造
完整 `status()` 并触发设备与 provider 探测。

新增回归测试证明默认 Basic Pitch 路径不会读取 Worker 命令，也就不会启动同步
硬件探针。研究模型手动选择、下载、授权失败回退和研究 bundle 配置测试保持
通过。

## 构建与产物

| 项目 | 结果 |
|---|---|
| 应用版本 | `0.1.0-preview.16` |
| 全量应用 build | 通过 |
| 用户版 bundle 审计 | 通过 |
| packaged app 目录 | 通过 |
| macOS arm64 ZIP | 通过，解压校验无错误 |
| macOS DMG | 本轮未重建；离线环境缺少 builder DMG 辅助工具 |
| Developer ID / notarization | 未执行 |

本轮 ZIP：

| 文件 | SHA-256 |
|---|---|
| `Guitar-Tab-Studio-0.1.0-preview.16-mac-arm64.zip` | `74b2b84c57e76274ed43a61621de042d051a69315965373416354a72d5296e45` |
| packaged `guitar-tab-worker` | `017869fc45c9930019cb43c1a0184c7b5771cac8aca8a3d0df83f642ce3415c9` |

electron-builder 使用项目依赖中已有的 Electron 44.4.5 完成 app 目录和 ZIP。
DMG 阶段尝试联网获取缺失辅助工具并超时；现有 DMG 时间戳早于本轮改动，不能
作为本轮验证产物，也不得发布。

## 冷启动测量

测量使用从新 ZIP 解压的隔离安装副本，并设置：

- 全新 `HOME` 和应用数据目录；
- 商业发行 provider 策略；
- 独立 CDP 端口；
- 未设置 smoke 或 research provider 开关。

计时起点为应用主进程创建前。首窗完成条件为 `desktop.html` 加载完成且正文
非空；runtime ready 通过 renderer bridge 实际调用 `runtime-status`。

| 阶段 | 耗时 |
|---|---:|
| 首个完整窗口 | 9.012s |
| 完整 runtime ready | 32.240s |
| 首窗后的硬件探针与状态查询 | 23.228s |

## Runtime Smoke

| 检查 | 结果 |
|---|---|
| 页面标题 | `Guitar Tab Studio` |
| 页面尺寸 | 2048 x 1084 |
| 本地 API | ready |
| SQLite 存储 | ready |
| Worker | ready |
| 上传音频 | false |
| 暴露 provider | 仅 `basic-pitch` |
| 默认偏好 | `basic-pitch / auto` |
| 权重来源 | `built-in` |
| 转录状态 | ready |
| 设备 | `darwin / arm64 / mps / Apple Metal` |

截图保存在
`apps/desktop/data/packaged-cold-start-2026-11-06/smoke.png`，SHA-256 为
`0d492d706053cbe7f7b59357abff77de43a439d445c08887bfb1a737241c5b04`。
工作区内容非空，主要控制区、侧栏和曲谱区域均正常显示，没有观察到控件重叠。

## 回归

| 套件 | 结果 |
|---|---:|
| Core | 45 passed |
| API | 106 passed |
| Desktop | 55 passed |
| Worker | 195 passed |
| `model-providers` 定向测试 | 12 passed |
| Promotion gate | 4 passed |
| Sealed gate | 2 passed |
| Training runner | 5 passed |
| Web lint | passed |
| Python lint | passed |
| 全量应用 build | passed |

## 剩余事项

1. 若还要继续缩短“模型 ready”时间，应将 `status()` 的硬件探针改为异步缓存，
   但不能用未探测的 fallback 结果覆盖研究 provider 可用性。
2. 正式发布前仍需在网络可用环境重建 DMG，并完成 Developer ID 签名、公证、
   stapling、Gatekeeper 和更新清单验证。

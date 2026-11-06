# macOS 安装包与真实启动验证

## 结论

`0.1.0-preview.16` 的 arm64 DMG 与 ZIP 已从干净的 Web、API、Electron
main 和 PyInstaller Worker 构建产生。DMG 校验、挂载、隔离安装和 packaged
runtime 启动均通过；本地 API、SQLite 和 Worker 均报告 ready，商业发行
策略下只暴露 Basic Pitch，默认偏好为 `basic-pitch / auto`，权重来源为
`built-in`。

当前产物只有 ad-hoc 签名。机器上没有 Apple Developer ID 身份，本轮未执行
公证和 Gatekeeper 分发验证，因此产物仅用于本地安装 smoke，不是可公开发布
的 macOS 安装包。

## 环境

| 项目 | 值 |
|---|---|
| macOS | `15.7.4 (24G517)` |
| 架构 | `arm64` |
| Node.js | `24.14.0` |
| npm | `11.9.0` |
| Python | `3.10.21` |
| Electron | `44.4.5` |
| electron-builder | `26.15.3` |
| 应用版本 | `0.1.0-preview.16` |

构建缓存和临时目录均限制在项目 `.cache` 内。构建时剩余空间最低约
3.0 GiB，没有写入或读取兄弟项目。

## 侧车验证

PyInstaller 侧车从当前 Worker 源码重新构建，manifest 包含 6,316 个审计
文件：

| 文件 | 大小 | SHA-256 |
|---|---:|---|
| `guitar-tab-worker` | 58,654,240 bytes | `017869fc45c9930019cb43c1a0184c7b5771cac8aca8a3d0df83f642ce3415c9` |
| `ffmpeg` | 45,568,216 bytes | `a90e3db6a3fd35f6074b013f948b1aa45b31c6375489d39e572bea3f18336584` |
| `sidecar-manifest.json` | - | `7cd2658203d66f8f95d2c813b9144811248cf5570e1466b76d0eb16b19e11eca` |

带 packaged runtime 实际使用的 `FFMPEG_BINARY` 环境执行
`--runtime-self-test` 和 `--hardware-probe` 后得到：

- Basic Pitch `0.4.0`
- Demucs `4.1.0`
- SoundFile `0.14.0`
- Torch `2.14.0`
- Apple Metal / MPS
- `basic-pitch` 可用

构建脚本还通过一秒 MP3 生成、FFmpeg 解码和 Worker `--media-probe`
检查。

## 安装包

| 产物 | 字节 | SHA-256 |
|---|---:|---|
| `Guitar-Tab-Studio-0.1.0-preview.16-mac-arm64.dmg` | 626,293,262 | `9654925cb98c7d6946ca5f3891d2504cefbe73bc7af41092242785054e8b78b4` |
| `Guitar-Tab-Studio-0.1.0-preview.16-mac-arm64.zip` | 630,680,013 | `9c9e42caae9fcfe196f32409a0f946314d042cfc6d2901e4a62a881799fcbe47` |
| `latest-mac.yml` | - | `4f0c0ae629d25c1ec6a1312b0eb95ba191fa9a8da0f22706f54848a3987b9c0d` |

`verify:user-only` 已确认桌面 renderer 和 API bundle 不含管理员工作区路由
或会话标记。DMG 的 GPT/HFS 校验全部通过；镜像随后以只读方式挂载，应用从
镜像复制到项目内隔离的 Applications 目录，再从该安装副本启动。

## Packaged Runtime

启动时没有设置 `GTS_DESKTOP_SMOKE` 或
`GTS_ENABLE_RESEARCH_MODELS`，并显式采用商业发行策略。CDP 页面标题为
`Guitar Tab Studio`，renderer 来自安装副本的
`Contents/Resources/web/desktop.html`，页面有非空、可交互的完整工作区。

`runtime-status` IPC 返回：

| 检查 | 结果 |
|---|---|
| `mode` | `desktop-local` |
| 本地 API | ready |
| SQLite 存储 | ready |
| Worker | ready |
| 上传音频 | false |
| 暴露 provider | 仅 `basic-pitch` |
| 默认偏好 | `basic-pitch / auto` |
| 权重来源 | `built-in` |
| 转录状态 | ready |
| 设备 | `darwin / arm64 / mps / Apple Metal` |

应用实际连接 GitHub preview 更新通道，确认当前
`0.1.0-preview.16` 已是最新版本，没有可安装更新；更新检查未导致进程退出。

## 回归结果

| 套件 | 结果 |
|---|---:|
| Core | 45 passed |
| API | 105 passed |
| Desktop | 54 passed |
| Worker ML | 100 passed |
| Python lint | passed |
| 用户版 bundle 审计 | passed |

## 剩余风险

1. 应用仅有 linker 生成的 ad-hoc 签名，`TeamIdentifier` 为空；正式发布前必须
   使用 Developer ID Application 签名，并完成 Apple notarization 和
   stapling。
2. 首次冷启动中，打包 Worker 的硬件探针和大型运行时载入使 renderer 约
   20 多秒后才可自动化访问。该观测不是基准测试，但应在发布门槛中增加
   “首屏可交互时间”测量，必要时将硬件探针移出首屏阻塞路径。
3. 本轮验证了“当前无更新”的真实通道检查和 updater 单元测试，没有可用的
   更高版本工件，未执行下载、暂停、恢复和重启安装的线上端到端流程。

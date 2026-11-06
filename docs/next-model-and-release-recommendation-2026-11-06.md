# 下一模型与桌面发布建议

## 当前判定

Basic Pitch 继续作为产品默认后端。v28 不晋级：72 轨评测没有 TAB 收益，
当前许可台账下又只保留历史台账身份。现有 DMG/ZIP 已通过本地安装和真实
packaged runtime smoke，但只有 ad-hoc 签名，不能公开发布。

新的受治理数据快照已经发生实质变化，因此后续可以设计 v29，而不是在 v28
的 740 条不变数据上继续训练。本轮没有启动 v29。

## 已就绪输入

| 输入 | 状态 |
|---|---|
| 72 轨错误归因 | 已完成；最终输出漏检/误检占失败实例 74.1% |
| AG-PT train | 353 轨、16,908 事件、4 个演奏者组 |
| AG-PT validation | 72 轨、3,690 事件、1 个独立演奏者组 |
| 正式训练快照 v2 | 1,093 轨、127,620 事件、0 test、0 组泄漏 |
| GuitarSet / Guitar-TECHS 评测 | 保持既有 60 / 12 轨 |
| 产品默认 | `basic-pitch / auto / built-in` |

## 推荐顺序

1. 先为 onset-position evaluator 增加阶段追踪。每轨至少保存
   `frontend_groups`、`activity_rejected`、`position_rejected`、
   `emitted_primary`、`emitted_supplemental` 和 `merged_output` 计数，
   这样下一轮才能区分 Basic Pitch 漏候选与活动头误拒绝。
2. 预注册 v29 单变量实验。仅改变训练数据为
   `formal-training-manifest-v2.jsonl`，保留 v24 初始化、位置模型、候选
   前端、解码阈值、seed 和选择规则；显式记录 AG-PT 的数据集采样权重，
   避免 353 条单音轨按记录数主导 batch。
3. GuitarSet validation 继续承担模型选择。AG-PT validation 和
   Guitar-TECHS P2 只用于冻结配置后的独立评测，不反向调阈值；现有 test
   split 保持锁定。
4. 只有同时通过 GuitarSet TAB/重复音无回退、Guitar-TECHS 跨域改进和
   AG-PT 单音活动改进，才生成新的 governed checkpoint。否则停止活动头路线，
   转向独立多音乐句真实 TAB 数据获取。
5. 桌面发布与模型研究分离推进。先优化约 20 多秒的冷启动观测，再配置
   Developer ID、notarization 和 stapling；不因模型实验改变 Basic Pitch
   默认。

## 不应执行

- 不把 GOAT 的 CC BY-NC 数据用于商业产品训练或内部测试。
- 不把 IDMT 的 CC BY-NC-ND 数据用于训练。
- 不重新发布或修改 v28 checkpoint 以伪装成当前台账身份。
- 不在没有阶段追踪和预注册采样策略前直接训练 v29。
- 不把当前 ad-hoc 签名 DMG 当作正式 macOS 发布物。

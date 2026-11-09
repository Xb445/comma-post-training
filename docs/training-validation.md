# 原生框架验证记录

最新的新环境适配框架、142 项测试、空间 Student/DAgger/TD3、固定恢复 suite 和
浏览器报告验收见 [新环境适配验收](acceptance/adaptation-20261109/README.md)。
该记录明确保留未通过性能准入的短预算结果；只验收工程链路。

本轮可视化功能的 126 项回归、浏览器离线播放/移动布局和策略 trace 验证见
[可视化验收](acceptance/visualization-20261109/README.md)。

新增的官方基座仿真后训练链路记录见
[2026-11-09 工程验收](acceptance/cloud-official-sim-20261109/README.md)：115 项测试、
30 秒四路录制、合成 ONNX 缓存/残差训练、实时 DAgger/TD3 及重载。
以下保留 2026-11-05 的原生模型历史结果；两次记录都不是官方权重下的驾驶效果验收。

日期：2026-11-05。环境为 Python 3.12、Torch 2.15.0+cpu、无 GPU；全部使用本项目合成数据，
没有下载官方模型，也没有使用用户真实数据。此记录用于检查实现与教程，不能作为驾驶效果实验。

## 工程检查

- 75 项 pytest 全部通过，包含既有文本/驾驶工具和新原生训练集成测试。
- Ruff 规则、格式、Git whitespace 检查通过。
- wheel / sdist 构建通过；检查包含原生模块、配置和文档，未打包上游源码、原始数据或输出。
- 五个上游 checkout 与 gitlink/锁一致，递归依赖无修改；44 个源码锚点核验通过。
- 可选的 `commavq/gpt2m` 模型资产仍未取得，源码检查没有把它报告为模型就绪。
- README、docs、experiments 内的本地链接存在。

## 实际 CLI 链路

按 [训练说明](training-framework.md) 的命令顺序，使用 `configs/training/` 模板，
离线三种训练各 12 step，闭环训练各 24 环境 step：

1. make-demo → validate-dataset。
2. train policy → evaluate-checkpoint test。
3. train compressor → cache-latents。
4. train world_model → evaluate-checkpoint test → rollout-world 三帧自回归。
5. train-closed-loop DAgger。
6. train-closed-loop reference TD3 → evaluate-closed-loop test。
7. train-closed-loop learned latent TD3 → evaluate-closed-loop test。

另执行 learned latent DAgger，共 15 个核心 CLI 调用均成功；另外检查了两条 CPU 设备覆盖命令。
数据为六个独立 synthetic route，各 24 帧，train/val/test 各两段，
图像 16×32、5 Hz。离线配置 seed=42、history=4，压缩器 history=1。
Flow hidden=64、两层 Transformer、4 heads、latent 为 4×4×8，anchors 为 +2/+3。

| 验证目标 | 初始 | 12 step 后 |
| --- | ---: | ---: |
| 策略 BC 动作 MSE | 0.0612898 | 0.00680743 |
| 压缩器 RGB 重建 MSE | 0.0400353 | 0.0187131 |
| Flow MSE + 0.1 × Plan MSE | 12.2157 | 2.09968 |

这些分数来自独立 val route；test 另用 evaluate-checkpoint 执行。各目标的单位与权重不同，
不能跨行比较质量。合成图像和动作十分简单，损失降低只证明参数实际更新及链路有效。

闭环 24 step、warmup=8、batch=8，共 17 次 learner update；TD3 policy_delay=2，
8 次 Actor 更新。每种环境用两个固定验证种子，结果如下（回报越大越好）：

| 环境/算法 | 初始 val 平均回报 | 最终 val 平均回报 |
| --- | ---: | ---: |
| reference / DAgger | -29.6443 | -56.3666 |
| reference / Residual TD3 | -29.6443 | -23.5542 |
| learned latent / DAgger | -0.00192674 | -2.89451 |
| learned latent / Residual TD3 | -0.00192674 | -0.00276493 |

DAgger 与 learned TD3 在本次极短预算内退化；代理专家 Plan Head 尚未学好也是待排查因素。
不同环境奖励定义不同，数值不能跨环境比较。当前结果不支持“闭环性能已提高”的结论。
这里验证的是训练确实执行、base/CNN 冻结正确、动作影响下一观测、回报和失败标志可追踪，
以及保存后能在独立 test 种子上推理。正式研究还需要校验模型 fidelity、专家、奖励、场景划分，
增加预算、多种子、真实扰动和失败统计。

本云工作区保留 `outputs/framework-verification-final-20261105/` 中的逐命令 JSON 与汇总；
检查点位于 `outputs/` 下的 `policy/`、`compressor/`、`world/`、`dagger/`、`td3/`、
`latent-dagger/`、`latent-td3/`。
这些都是忽略的运行产物，不随 Git 源码分发；重新运行请选择新目录。

## 关键语义的测试

覆盖 route 泄漏、数据单位/坐标、NaN 标签 mask、时间间断、历史动作移位、按域采样，
以及 NPZ/图片索引接入失败不留下半成品。CPU 策略和 Flow 都验证了连续训练与分段续训
逐参数完全相等，配置改变与旧 step 覆盖被拒绝。

压缩器 decoder 适配保持 encoder 指纹不变；latent 缓存不修改源元数据。
Plan Head 对 noisy target 和目标运动条件的改变保持输出不变；交互环境的运动条件来自
动作而非记录中的下一步位姿。TD3 验证 next base、双 Critic、真实终止/截断、延迟 Actor、
target 软更新及最终指令/实际动作区分。另验证可显式在 CPU 上读取标记为 GPU 训练的检查点。

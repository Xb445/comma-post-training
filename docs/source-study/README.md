# openpilot 源码研究手册

本手册按上传规划的“车端系统 → 世界模型 → 策略后训练 → 部署接口”组织，并以实际克隆的
固定版本逐项定位。先读系统接口，再追踪训练过程，最后设计模型导出与回放对齐实验。

| 顺序 | 文档 | 需要掌握的问题 | 对应实验 |
| --- | --- | --- | --- |
| 0 | [固定源码地图](00_source_map.md) | 本地文件、关键符号与上游准确行号在哪里？ | 源码校验 |
| 1 | [系统架构](01_system_architecture.md) | 图像、状态与控制消息如何跨进程流动？ | `01_openpilot_replay` |
| 2 | [modeld](02_modeld_analysis.md) | 预处理、状态缓存、网络输出如何变成物理动作？ | `02_model_inference` |
| 3 | [世界模型](03_world_model_architecture.md) | 离散视频模型、连续潜空间模型和真实动力学如何连接？ | `03_video_compression` |
| 4 | [Rectified Flow](04_rectified_flow.md) | 如何组织训练样本、条件、mask、损失和采样？ | `04_world_model` |
| 5 | [驾驶策略训练](05_policy_training.md) | BC/KD、上游 DAgger 与异步 Near On-Policy 如何区分和实现？ | `05_policy_distillation` |
| 6 | [残差 RL](06_residual_rl.md) | 冻结 base 后，如何构造有界 residual 与 TD3 更新？ | `06_residual_rl` |
| 7 | [部署连接](07_vehicle_deployment.md) | checkpoint 如何验证成能被 modeld 使用的运行产物？ | 导出/回放/受控硬件阶段 |

实现顺序和逐项验收位于 [实施路线](08_implementation_roadmap.md)。
既有 [研究分析](../openpilot-research.md)、[任务表](../tasks/openpilot.md) 和
[数据合同](../driving-contracts.md) 继续适用；本手册补充源码级结构和实验路径。

## 阅读时的证据边界

- **本地源码事实**：来自 `third_party/` 的实际文件；源码地图静态解析函数/类及文件 SHA256。
- **规划/论文描述**：例如 2B/4B 网络层数、压缩器训练损失、五步 TD3，先作为待核对的报告信息。
  不能由一份推理 wrapper 推导出完整官方训练实现。本次网络仍限制 Hugging Face/博客资产访问。
- **本项目的实现方案**：RF 训练器、TD3、Student 部署 adapter 等尚需开发，本文用数据结构、
  公式、模块边界和验收条件说明；不会把规划中的文件当成已经存在的程序。

当前可运行的是源码校验、阅读地图生成及父项目的 CPU 数据/评测工具。主仓库与 openpilot
六个固定依赖已经克隆；`commavq/gpt2m` 模型仓库和其他模型载荷尚未取得。
本次未安装/编译上游 runtime，未执行真实 Replay、GPU 训练或车辆通信。

## 四个版本边界

1. 本次车端源码快照：openpilot `493ae5a…`。
2. openpilot 构建时使用其内嵌的 opendbc/panda/msgq/tinygrad 固定版本。
3. 社区 distill 快照：`215f33d…`，它依赖 openpilot `084747c…` 及匹配的 ONNX Teacher。
4. 世界模型与 VAE 的资产 revision 尚待解析，不能使用 `main` 代替一次已固定的实验来源。

完整 SHA 见 [源码锁](../../third_party/sources.lock.json) 和
[模型/数据资产锁](../../configs/driving/upstream.lock.json)。源码快照是可阅读的版本集合，
不是可以直接把五个仓库的包互相导入的共享环境。

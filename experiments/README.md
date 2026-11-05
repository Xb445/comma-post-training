# 实验路径索引

这些目录保存协议和实现入口，尚未执行的实验明确写明前提与验收。每次运行结果存放到
`results/<experiment>/<run_id>/` 或项目已有的 `outputs/`，不将大权重和原始路测数据提交到源码仓库。

自有数据的可运行原生基线已新增，命令和实现入口统一见
[训练框架](../docs/training-framework.md) 与 [数据合同](../docs/training-data.md)。
本目录保留更大范围的上游复现/对照协议，不把合成测试视为真实实验完成。

| 目录 | 目标 |
| --- | --- |
| [01_openpilot_replay](01_openpilot_replay/README.md) | 系统消息流、更新频率和实际回放 |
| [02_model_inference](02_model_inference/README.md) | Teacher/Student 与流式接口对齐 |
| [03_video_compression](03_video_compression/README.md) | 离散压缩与连续 VAE 的小样本研究 |
| [04_world_model](04_world_model/README.md) | 世界模型服务、动作响应和短 rollout |
| [05_policy_distillation](05_policy_distillation/README.md) | 上游多任务复现、BC/KD 和 DAgger |
| [06_residual_rl](06_residual_rl/README.md) | 原生有界残差 TD3 基线与后续真实闭环对照 |

总体依赖、产物合同和运行状态见 [实施路线](../docs/source-study/08_implementation_roadmap.md)。

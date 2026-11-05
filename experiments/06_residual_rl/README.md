# 06 Residual TD3 研究实验

原生 one-step TD3 训练器位于 `src/comma_post_training/simulator/`，包含双 Critic、冻结 base、
有界残差、终止/截断处理和 target 更新。可运行命令及合成/learned 环境配置见
[训练框架](../../docs/training-framework.md)。配套
[残差 RL 设计](../../docs/source-study/06_residual_rl.md) 保留完整真实实验的验收要求。

前提：固定的 BC/KD base、已验证的闭环环境、共同扰动集和完整失败统计。

实现顺序：有界 residual 与冻结检查 → replay/n-step → twin critic → target smoothing/延迟更新 →
轻量已知动力学数值验证 → 与 DAgger 相同 world-model 环境的真实对照。

目标交付：全量配置和 resume 状态、base/encoder/reward/world-model 版本、交互/更新预算、每 seed
的 reward/跟踪/jerk/失败指标，以及模型偏差与动作约束消融。不能只依据训练 reward 上升声称改善。
轻量动力学与真实 world-model 的实验结果分别保存并标注。

当前原生训练保存 learner/replay 与推理导出，但没有仿真器状态快照，不支持在线精确续训；
n-step、多种子真实驾驶对照、完整失败指标仍需在接入实际环境后实现和测量。

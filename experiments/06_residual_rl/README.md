# 06 Residual TD3 研究实验

当前没有 TD3 训练器。配套 [残差 RL 设计](../../docs/source-study/06_residual_rl.md)
给出拟新增模块、物理单位、target、终止语义和验证路径，不提供指向不存在程序的训练命令。

前提：固定的 BC/KD base、已验证的闭环环境、共同扰动集和完整失败统计。

实现顺序：有界 residual 与冻结检查 → replay/n-step → twin critic → target smoothing/延迟更新 →
轻量已知动力学数值验证 → 与 DAgger 相同 world-model 环境的真实对照。

交付：全量配置和 resume 状态、base/encoder/reward/world-model 版本、交互/更新预算、每 seed
的 reward/跟踪/jerk/失败指标，以及模型偏差与动作约束消融。不能只依据训练 reward 上升声称改善。
轻量动力学与真实 world-model 的实验结果分别保存并标注。

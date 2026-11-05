# 06 残差强化学习：将研究设计落实为 TD3

已克隆的 distill `rl/train.py` 用 MSE 更新策略头，没有双 Critic、Bellman target 或 TD3 replay。
本章定义后续独立实现边界。上传规划中关于 0.11.2 五步回报和残差结构的描述，仍需与可访问的
原始报告及权重对照；这里不宣称已取得官方完整 TD3 代码。

## Actor、动作和冻结边界

定义冻结基础策略 `a_base=π_base(o)`，训练一个小 residual：

\[
\delta a=s\odot\tanh(f_\theta(h,a_{base},x)),\qquad
a=\operatorname{clip}(a_{base}+\delta a,a_{min},a_{max}).
\]

`h` 为固定视觉/时序特征，`x` 是部署时可用的车辆状态。`a` 为 `[a_lat,a_long]`，单位 m/s²；
`s` 是逐轴尺度，也具有加速度单位，不能把无量纲 tanh 范围当作任何车型适用的约束。

冻结 encoder/base 后，feature cache 才有稳定含义。用“更新前后参数一致”与梯度检查证明冻结，
不要只记录 optimizer 参数列表。Actor 不能读取真实部署时不可用的未来 plan；若 Critic 或 reward
使用额外信息，需要单独声明与对照。

## 最小数据结构

| Replay 字段 | 作用 |
| --- | --- |
| `obs/features, base_action, full_action` | 明确 Critic 使用最终动作还是残差动作，本方案推荐最终动作 |
| `reward, next_obs/features, next_base_action` | 形成与下一状态一致的 TD target |
| `terminated, truncated, bootstrap_valid` | 区分真实终止、时间限制与下一观测是否有效 |
| `episode_id, t, dt` | 防止跨 episode 拼 n-step；固定物理时间 |
| base/encoder/world-model/reward revision | 证明缓存与目标版本一致 |
| behavior policy version、seed、扰动、约束标志 | 复现采样分布与诊断失败 |

如果修改冻结 encoder、预处理或动作约定，应清理/迁移旧 replay，不能继续把旧特征当成新模型输入。

## TD3 更新路径

1. **采样**：使用受控探索噪声执行最终动作，记录真实应用到 Env 的动作、约束与失败。
2. **目标动作**：在 next state 上计算 next base 与 target residual，加 target smoothing，
   按同一物理单位/范围裁剪。
3. **Critic**：双网络独立拟合 target，目标使用两者较小值。
4. **延迟 Actor**：按配置降低 Actor 更新频率，用 Q1 的动作梯度训练 residual。
5. **Target 网络**：按配置做软更新，保存完整训练状态。

单步示意：

\[
y=r+\gamma b\min(Q'_1(s',a'),Q'_2(s',a')),
\]

其中 `b` 由真正的终止与可 bootstrap 状态决定。五步变体把它推广为折扣 reward 和实际可用
转移的累积，不能跨 episode；终止后的奖励/target 不继续拼接。是否采用 n=5 是独立的实验配置，
不是“TD3”名称自动规定的参数。

## Reward 与动作平滑

可先研究版本化的参考加速度跟踪与控制平滑目标。例如误差项包括轨迹/速度或实际达到加速度
与固定参考的偏差；参考源的产生时刻和坐标系必须明确，不能在采取动作后重新选择更容易的目标。
世界模型的参考存在偏差，模型误差不能用训练 reward 本身消除。

Actor 正则化可研究 residual 大小或 `||a_t-a_(t-1)||²/dt²`，后者对应加速度变化率即 jerk 的平方。
它不是加速度二阶差分。平滑惩罚不得掩盖必要控制或改变评测分母，所有权重、单位和时间周期
写入配置并做消融。

## 拟新增模块与单独验收

| 模块 | 首个验收 |
| --- | --- |
| ResidualActor / action fusion | 零 residual 恢复 base；逐轴尺度与裁剪正确；base 参数不变 |
| TwinCritic | 两个独立参数集；目标取 min，Actor 可接收到动作梯度 |
| Replay / n-step | 真实 terminal 不 bootstrap；时间截断按协议；不跨 episode |
| TD3 trainer | target smoothing、延迟更新和软更新次数可复核；resume 可重复 |
| World-model adapter | 与 DAgger 共用 reset、物理时间、动作单位和失败标准 |

这些模块尚未实现，不提供指向不存在训练器的运行命令。先用小型已知动力学任务验证算法，再
接入同一个 world-model benchmark；轻量动力学结果单独报告，不能替代 comma 世界模型结果。

## 与监督/DAgger 的公平比较

固定数据划分、Student/base checkpoint、扰动、种子、总交互步数和训练预算。分别报告：
开环误差、闭环恢复率、越界/碰撞/控制超限/超时比例、jerk、推理延迟及模型失效。
不能只比较 reward 均值，也不能把失败 episode 从成功率分母中移除。
对应 [06_residual_rl](../../experiments/06_residual_rl/README.md)。

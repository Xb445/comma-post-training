# 05 驾驶策略：监督蒸馏、DAgger 与 Near On-Policy

> 实现状态更新：已新增原生 CNN/GRU 的 BC/KD、DAgger 与完整离线续训；上游多任务 Student 仍需独立复现。入口见 [训练框架](../training-framework.md)。

研究入口是 `third_party/openpilot.distill/`，属于社区研究实现。
先复现它的真实数据/模型接口，再设计受控消融；不能把这套 Student 当成车端官方模型的同一架构。
源码定位见 [地图](00_source_map.md)。

## 从定位数据到训练样本

`localizer/` 提供定位处理入口；`supervised/dataset.py` 进一步构造训练样本：

1. 读取 `frame_info.safetensors`、`localizer.safetensors` 和相机视频索引。
2. 依据设备/传感器选择相机内参，按标定与增强变换做视角 warp。
3. 解码目标历史帧，转换到 packed YUV，形成窄/广角两路历史输入。
4. `_load_targets` 对定位状态做插值、坐标变换和平滑，构造未来 plan。
5. 根据横纵向 `action_t` 从 plan 提取曲率/加速度，形成 `[a_lat,a_long]`。
6. 送入固定 Teacher，得到需要蒸馏的输出/特征；Student 与定位、Teacher 共同监督。

上游训练器依赖 Python 3.12、固定 Torch 版本、CUDA、PyNvVideoCodec 和 ORT GPU。
Teacher/定位数据/相机处理三个版本必须一致。默认 segment hash 划分不能证明真实 route 不泄漏；
父项目的 manifest validator 用于在接入前补上这个检查。

## Student 的实际结构

| 部件 | 当前默认/接口 | 代码 |
| --- | --- | --- |
| 视觉骨干 | `convnext_pico`，`pretrained=True` | `VisionConfig` / `Vision` |
| 每路输入 | 两帧 packed YUV，共 12 通道，空间 128×256 | `channels_per_input` / dataset |
| 双路组合 | 沿通道拼接进入视觉骨干，归一化由 mean/std 定义 | `Vision.forward` |
| 空间特征 | stride=32，对默认空间得到 4×8 网格；实际 feature dim 由 timm 模型确定 | `grid_size` / `output_dim` |
| 时序历史 | `history_len=9`；条件窗口 `desire_window_len=25`；`temporal_len=33` | `SupercomboConfig` |
| 时序网络 | 因果 Transformer；加入时序/空间位置和 desire/traffic/action_t 编码 | `TemporalPolicy` |
| 辅助分支 | point_policy、image_policy、多头输出 | `pretraining_policies` / `Hydra` |
| 闭环更新分支 | `on_policy_temporal` | `rl_policies` |

这些是社区代码默认值，不是对官方 FastViT/4B 模型内部结构的核实。`pretrained=True` 还可能需要
视觉骨干权重，应记录其来源和 hash；不能把“Student 新实例”自动称为所有参数随机初始化。

`TemporalPolicy` 将时空特征展平后送入因果 Transformer，再按时间取输出。
`InputQueues` 负责 streaming 输入的历史缓存；`TorchSupercomboForInference` 和
`ORTSupercomboForInference` 分别适配可训练 Student 与 ONNX Teacher。
Teacher 的原始输出宽度与学生 head 宽度不一定就是动作维数，例如 action head 输出 4 个值，
训练/actor 的相应消费路径取前两个动作分量。解释余下分量时要继续核对 parser 与具体模型定义。

## 监督损失与对照组

上游 `supervised/train.py:loss_fn` 不是单一 action MSE：

- 定位/真实目标：图像、wide_from_device_euler、pose、plan、action。
- Teacher 目标：多种模型输出与 hidden_state；存在不同 loss 权重。

建议先保留原始 loss 完成上游复现 E01，再用相同 Student 做 action-only 受控对照：

\[
L=\lambda_{BC}\|a-a_{expert}\|^2+\lambda_{KD}\|a-a_{teacher}\|^2.
\]

本项目已实现带有效 mask、标签 detach 的 `action_distillation_loss`，并登记 BC、KD、BC+KD
三组矩阵。它尚未接到上游数据训练 runner；也不等同完整多任务 loss。
辅助分支、初始化、样本预算、route 划分、种子与动作延迟应保持一致，再比较方法变量。

## 世界模型中的实际训练循环

```text
监督 Student checkpoint + 可选 on-policy head checkpoint
 → RuntimeClient 查询世界模型服务
 → Episode 编码真实双路历史 / future anchor
 → actor.reset → Env + rollout
 → 取访问状态的特征、条件和世界模型 plan
 → plan + action_t → 参考 [a_lat,a_long]
 → 冻结共享模型，只训练 on_policy_temporal 的 action MSE
 → 保存该 head，进入下一轮
```

`SupercomboActor.act` 维护前一帧和 InputQueues，将预测横向加速度按
`a_lat / max(speed,1)^2` 转为 Env 接收的曲率。这里的低速约定不同于 modeld 的 `max(2.5,v)`；
导出/部署时必须显式对齐，不能仅因变量同叫 curvature 就认为数值行为一致。

`rl/train.py` 是同步的 rollout → 更新循环。它更新当前新样本，没有完整的长期 DAgger 数据
聚合器、异步 actor 集群或 TD3 replay/critic。因此 README 的 “DAgger-style” 应按实际代码理解。
若做严格数据聚合消融，需另外版本化 `D_k`、采样比例、遗忘策略和标注来源。

世界模型 plan 是生成的参考，不是无噪声专家。每条新状态要用对应状态的有效监督，不能把原始
真实轨迹的动作直接贴到偏离后的生成观测上。

## Near On-Policy 的后续实现路径

上传规划描述的异步 Near On-Policy 可作为研究扩展，当前小型脚本并未实现该集群：

| 模块（拟新增） | 必须记录/约束 |
| --- | --- |
| Rollout worker | behavior checkpoint hash、环境/数据 seed、horizon、扰动与有效状态 |
| 样本队列 | 生成时间、behavior version、标签来源、episode/route、采样优先级 |
| Trainer | 当前版本、允许的 policy staleness、样本预算、过期样本处理 |
| Checkpoint 发布器 | 完整写入后原子切换，worker 只加载验证成功的版本 |
| 评测器 | 固定 checkpoint 与 held-out 路段；不追随训练中的 latest |

先用单 worker 验证来源与状态一致性，再扩展吞吐量。改变 worker 数量时同时报告实际交互步数、
训练更新数和 checkpoint 延迟，避免把更多数据误认为算法本身带来的收益。

执行入口、未满足前提和检查表见
[05_policy_distillation](../../experiments/05_policy_distillation/README.md) 及
[既有任务表](../tasks/openpilot.md)。当前尚未进行真实 Student/GPU 后训练。

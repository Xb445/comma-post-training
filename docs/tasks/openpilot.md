# 驾驶策略后训练任务表

状态：完成 / 待开始 / 外部前提未满足。每个任务以独立可审查改动交付，先完成依赖，再扩大数据和计算量。

## 当前优先主线：官方基座冻结适配

用户已确定先冻结官方驾驶模型，只训练附加策略/残差头，世界模型适配延期。
实现及实际验收步骤见 [Teacher 优先指南](../official-teacher-first.md) 和
[P0–P4 仿真后训练手册](../official-sim-post-training.md)。

| ID | 本阶段任务 | 当前状态 |
| --- | --- | --- |
| O01 | 固定官方 driving ONNX 权重与哈希 | 用户本地已下载并校验；云端仍无权重 |
| O02 | 官方图推理、状态队列、输出解析 | 本地已报告 60 帧 v2 CUDA/reset/profile 验收；主仓库保留换行 metadata 与 DLL 兼容修复 |
| O03 | 自有数据标定与 Teacher 特征/动作缓存 | 本地已报告官方 Teacher 的 500 个有效仿真标签；真实相机标定仍待验证 |
| O04 | 冻结官方 base，监督训练有界 residual | 本地已报告 100 步 SFT 和 val MAE 改善；真实驾驶数据待测 |
| O05 | 官方基座与适配后策略的独立 route 对照 | 逐轴/逐 route MAE、jerk 和 Gate 2 完成；真实实验待执行 |
| O06 | 官方世界模型适配及闭环训练 | P4 延期，待资源就绪 |
| O07 | 仿真 episode 录制 | 双 pinhole 道路 backend、专家/实际动作、标定、5 Hz、最终观测与图表完成；MetaDrive/CARLA 待接入 |
| O08 | 实时官方基座 DAgger / TD3 | 本地短预算官方基座运行与重载完成；两算法 val 回报均退化，待 trace 诊断和多 seed 实验 |
| O09 | 可视化与评估报告 | HTML 双相机/轨迹/动作回放、训练曲线、逐 route 对照、PNG/SVG/CSV/GIF 与可选真实评估 trace |

## 自有数据原生训练路径

新的 [训练框架](../training-framework.md) 不依赖官方模型下载即可开展自有数据训练，
格式见 [episode 合同](../training-data.md)。当前完成并用 CPU 合成数据验收：

| 任务 | 当前交付 | 下一步真实实验 |
| --- | --- | --- |
| N01 | 版本化 episode、NPZ 导入、NPY mmap、图片索引、route 隔离与域采样 | 适配自己的原始日志，核实时间/标定/标签 |
| N02 | 多视角 CNN/GRU 的 BC/KD、warm start、独立评估、完整离线续训 | 固定数据集上的 BC/KD/混合对照 |
| N03 | 连续 AE、decoder 适配、冻结 encoder 缓存、指纹兼容检查 | 真实图像重建质量与容量实验 |
| N04 | 条件 latent Flow、Plan Head、自回归预测 | 实际数据和动作条件的 fidelity 评估 |
| N05 | 合成环境和动作条件 learned latent 环境、DAgger、残差 TD3 | 可信仿真器 adapter、专家/奖励校验及扰动基准 |

上述完成表示工程链路可运行，不改变下表中官方资产、原样上游复现和真实实验的验收状态。

## 上游复现与真实实验

2026-11-05 已补充 [固定版本源码集合](../../third_party/README.md) 与
[源码研究手册](../source-study/README.md)。这完成了源码阅读底座，下面的真实模型、数据与 GPU
训练验收状态不因克隆源码而自动改变。系统/世界模型/部署的扩展路径见
[实施路线](../source-study/08_implementation_roadmap.md)。

| ID | 任务与交付 | 依赖 | 验收条件 | 当前状态 |
| --- | --- | --- | --- | --- |
| T00 | 核查上游与研究路线 | 用户提供的参考内容 | 固定 distill commit，标出教师版本、额外数据文件和 DAgger/TD3 差别 | 完成 |
| T01 | 下载并固定资产 | 网络允许相关域名 | Teacher、comma1M revision、VAE/world model revision 与 SHA256 可追溯；首批仅 2 段 | 外部前提未满足：网络/资产 |
| T02 | route 级清单与拆分 | T01 | 每段有可信 route_id；相邻 segment 不跨 train/val/test；重复与缺失文件检查通过 | 校验器完成；真实清单待数据 |
| T03 | 解码、定位与动作对齐 | T01、T02、GPU | 验证 HEVC 解码、frame_info 字段、时间单调、标定坐标变换；画出参考动作与轨迹并做抽样人工检查 | 待开始 |
| T04 | 固定版本 Teacher 推理和标签缓存 | T03、Teacher、CUDA/ORT | 同一输入的输出可重复；缓存含教师 hash、预处理版本、action_t、采样频率、route_id；无状态跨 segment 泄漏 | 待开始 |
| T05 | 上游 Student 单步/小批复现 | T04、上游独立 GPU 环境 | `supervised.train` 单步有限 loss/梯度，保存再加载输出一致；记录解码时间和峰值显存 | 外部前提未满足：GPU/资产 |
| T06 | 50–100 段 pilot 扩展 | T05 | 依据实测存储/显存预算下载；固定测试集；过拟合小样本后再跑较大训练集 | 待开始 |
| E00 | 评测合同及 action-only BC/KD 损失 | CPU | 合成解析值验证 MAE/ADE/FDE/jerk；屏蔽 padding；标签无梯度；CLI 非零失败码有效 | 完成 |
| E01 | 上游多任务原样基线 | T05 | 保留上游 loss，记录全部 loss 项、模型和数据版本；新 checkpoint 在独立验证集测量 | 待开始 |
| E02 | BC / KD / BC+KD 对照 | T06、E01、E00 | 同架构/初始化/预算/辅助损失，3 个种子；report 开环指标和逐 route 结果 | 待开始 |
| W01 | 世界模型服务兼容性 | T01、兼容 GPU | VAE 编解码+单次 world-model 推理+reset 通过；记录 GPU、dtype/kernel、峰值显存和延迟 | 外部前提未满足：GPU/资产 |
| W02 | 闭环 rollout 与扰动基准 | W01、E02 | 同样 seed/初始状态/动作延迟；验证动作确实改变观测；固定扰动集及超时/越界/失败统计 | 待开始 |
| A01 | DAgger-style 更新复现 | W02 | 新访问状态有对应标签；数据聚合可追踪轮次/策略；与继续 SFT 的等预算对照比较恢复率 | 待开始 |
| R01 | Residual TD3 研究实现 | W02、A01 作为对照 | 冻结 base；有界残差；双 Critic、target 平滑、延迟更新、terminal/truncation 单测和稳定训练 | 待开始 |
| E03 | 泛化与消融报告 | E02、A01、R01 | 测试集不参与选择；报告各 seed/置信区间、失败比例、数据/交互/计算预算和世界模型误差 | 待开始 |

## 上游复现的下一组任务：T01–T05

1. 允许实际需要的下载目的地，先取得数据/模型的不可变 revision 与完整性信息，再下载最小 pilot。
2. 检查每段 `fcamera.hevc`、`frame_info.safetensors`、`localizer.safetensors`；有 `ecamera.hevc` 时验证
   额外摄像头标定。缺失 frame_info 时先复现其生成流程，不能用空文件或不明时间戳替代。
3. 取得真实的 route 分组映射。上游 loader 用 segment hash 切 val；这不证明相邻片段按 route 隔离。
   若公开段 ID 无 route 映射，明确记录限制并解决分组来源，不能编造 route_id。
4. 按 [GPU 复现说明](../gpu-reproduction.md) 验证硬件与上游独立环境，再做 1-step 和 tiny-overfit。
5. 上游真实闭环研究在 T05 验收后开展；本地原生合成框架可先用于算法与接口开发。

## 监督实验的固定项

登记表位于 `configs/driving/experiments.toml`。它是实验矩阵，尚未连接训练 runner。
当前新增的 loss helper 只处理两维动作；不能用它替代上游多头损失后声称“原样复现”。
E02 应显式接入目标/教师缓存、数据 mask 和统一动作单位，并增加 seed/配置/模型来源记录。
上游当前以进程 rank 设种子，不能把连续执行三次默认命令当作三个不同种子实验。

BC 与 KD 各自的标签是否有噪声应分别诊断；统一评测网格和动作延迟。保持辅助损失一致，
或在所有 action-only 对照组中一起关闭，不同时更换视觉骨干、数据划分和 RL 方法。

## 闭环与残差学习的验收细节

- 世界模型历史帧、未来锚定帧、专家查询和训练数据必须遵循同一划分，记录 privileged conditioning；
  不把测试未来帧加入训练缓存。标注来源及世界模型偏差单独报告。
- A01 是数据聚合与监督更新；R01 才包含奖励、replay buffer 与 Actor-Critic 更新。
- TD3 Critic 对“最终动作”或“残差动作”的定义必须前后一致，target action 包含正确的 next base action。
- 使用秒定义采样周期、平滑项与奖励；终止和时间截断分别处理，避免错误 bootstrap。
- 不预填恢复率、碰撞率或延迟。CPU 合成 fixture、开环指标与真实闭环实验分别报告。

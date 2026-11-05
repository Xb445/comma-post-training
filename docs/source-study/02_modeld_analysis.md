# 02 modeld：视觉、时序缓存与驾驶动作

核心文件：`third_party/openpilot/openpilot/selfdrive/modeld/modeld.py`。
`ModelState`、`get_action_from_model`、`helpers.load_oob`、`Parser` 和消息填充函数的实际行号
见 [源码地图](00_source_map.md)。本章对应当前编译运行产物的调用接口，不推断未下载权重的内部层数。

## 一帧输入的调用路径

```text
VisionIpcClient.recv
 → 图像 frame_id / timestamp 对齐
 → 相机内参与 extrinsicsCalibration → get_warp_matrix
 → desire_pulse / traffic_convention / action_t
 → ModelState.run
     pack/copy inputs → run_warp → run_model → copy outputs
 → output_slices → Parser.parse_outputs
 → get_action_from_model
 → fill_model_msg / fill_pose_msg → modelV2 等 service
```

`main` 选择窄/广角流，等待图像，并使用时钟与 frame 元数据检查同步。`deviceType/sensor`
决定相机内参。`traffic_convention`、车道变换 desire 和动作执行时间也会影响策略，不能只送 RGB。

## 输入结构与内存组织

| 数据 | 代码入口 | 说明 |
| --- | --- | --- |
| 双路图像 | `vision_input_names=('img','big_img')` | 从 VisionBuf 读取带 stride 的 NV12 数据 |
| 几何变换 | `tfm` / `run_warp` | 每路 3×3 变换，产出模型要求的图像布局 |
| 行为意图 | `prev_desire`、desire 上升沿 | 转为 pulse；不是每一步都重复同一离散标签 |
| 交通习惯 | `traffic_convention` | 左/右行约定的条件向量 |
| 动作时刻 | `action_t` | 横纵向延迟、帧延迟与采样周期补偿 |
| 循环状态 | `state_pairs` / `input_queues` | 根据 artifact 中的 input/output specs 动态发现 |

`pack_inputs` 将 host 数组按对齐偏移放入一个上传 buffer，减少设备传输次数；图像经过独立
warp 运行产物。具体图像/状态维度以 `input_specs/output_specs` 为准，不能把社区 Student 的
固定尺寸直接套到该 runtime 上。

## 时序状态不是每次重建的历史图片列表

`__init__` 查找：输入名 `name` 对应的输出 `next_name` 是否存在。
找到后，给相关输出安排可回写的状态 buffer；`run_model` 会沿这组输入/输出更新历史状态。
`warmup` 使用 dummy 帧后清零状态。独立 route/重放 episode 之间需要显式 reset，以免历史串扰。

需区分三种缓存：

1. **车端运行状态**：modeld 编译 artifact 的 `name → next_name` buffer。
2. **社区 Student 队列**：distill 的 `InputQueues`，保存视觉特征/条件历史。
3. **生成模型 KV cache**：视频 Transformer 的注意力历史，服务于 latent/token 生成。

即使都叫“历史”，它们的维度、生命周期和更新频率也不同。

## 网络输出到物理动作

`output_slices` 来自模型元数据，Parser 将展平输出恢复为 plan、pose、车道线、概率等结构。
`plan` 使用 33 个非均匀未来点，每点 15 个量：位置、速度、加速度、姿态和角速度。

`get_action_from_model` 有两条路径：

- 未提供直接 action：按 `action_t` 从 plan 计算目标纵向加速度与曲率。
- 提供直接 action：前两维分别按横向/纵向加速度解释，再做
  `desiredCurvature = a_lat / max(2.5, v_ego)^2`。

随后应用停止判定、纵向平滑、横向低速处理。非常低速时保留上一曲率，避免无意义放大。
最终的 `ModelDataV2.Action` 是：`desiredCurvature`（1/m）、`desiredAcceleration`（m/s²）、
`shouldStop`（bool）。它不是原始转向角或归一化油门。

modeld 发布 `modelV2`、`drivingModelData`、`cameraOdometry`；`fill_model_msg` 还处理 frame id、
时间戳、概率与轨迹字段。部署 adapter 必须保持消息结构和所有下游实际读取字段兼容。

## 设备切换与异常路径

当前代码包含小模型和 chestnut 外接计算设备路径，按硬件存在性和编译状态选择，加载失败或
特定运行错误会回退。不要把某条设备分支的检查误认为所有分支都提供同样保障，例如显式
非有限值检查位于特定 chestnut 分支。系统是否允许控制还由 selfdrived、controlsd、card 和固件共同决定。

## 实现与验收路径

先记录 artifact specs、标定/图像布局、条件和循环状态。对同一短片段做 dense 与 streaming
推理对齐，检查 reset、双摄像头同步、前后帧缓存、action_t、动作单位及后处理。
分别测试正常帧、缺帧、消息超时、无效输出和设备回退，记录容差与延迟；不能只确认模型文件能加载。

对应 [02_model_inference](../../experiments/02_model_inference/README.md)。当前未下载模型、未执行这些推理验证。

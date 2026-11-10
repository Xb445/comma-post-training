# 自有数据 episode 合同：comma-episodes-v1

开始采集目标环境数据前，先阅读 [数据需求与交付规范](target-environment-data-requirements.md)，
其中按当前 SFT、闭环后训练和后续世界模型区分必需信息、采集建议与审计门槛。

采用驾驶/机器人数据中常见的 episode、逐帧时间戳、观测/动作分离和整体任务拆分方式。
磁盘使用普通 JSON + NPY / 图片路径，便于 NumPy/PyTorch 读取，避免绑定特定数据库。
这是本项目的中间格式，不声称兼容原生 LeRobot、RLDS 或 openpilot 文件。

官方 Teacher 需要额外的双视角标定和 5 Hz 数据。新的 `comma-driving record-sim` 可直接产生
符合尺寸、时序和专家动作合同的仿真 episode，详细目录与 adapter 接口见
[仿真数据合同与操作手册](official-sim-post-training.md#p1仿真器--comma-episodes-v1)。

## 目录与 manifest

```text
data/local/own-v1/
  manifest.json
  episodes/
    route001-segment000/
      episode.json
      timestamps.npy
      ego_state.npy
      executed_actions.npy
      expert_actions.npy
      poses.npy
      images.npy                 # 与 image_paths 二选一
```

```json
{
  "format": "comma-episodes-v1",
  "kind": "recorded",
  "revision": "own-v1",
  "episodes": [
    {
      "id": "route001-segment000",
      "route_id": "route001",
      "domain": "real",
      "split": "train",
      "metadata": "episodes/route001-segment000/episode.json"
    }
  ]
}
```

实际训练还需单独的 val episode，正式评估需要 test。`kind` 为 `recorded|synthetic`；
domain 为 `real|sim|synthetic`，分别记录真实采集、仿真采集、合成 fixture。
同一路线连续采集的相邻片段必须共享可信 route_id，禁止跨 train/val/test。
身份不能凭 segment hash 伪造；如果没有 route 信息，应先补采集会话/任务分组。
episode id 只能包含字母数字、点、下划线和连字符，且以字母数字开头。

可用 `build-dataset` 从 `episodes/*/episode.json` 中读取身份并生成 manifest。
同一版本不覆盖；修订数据请另建版本目录。manifest 与 episode 内的身份必须一致。

## episode.json

```json
{
  "format": "comma-episode-v1",
  "episode": {
    "id": "route001-segment000", "route_id": "route001",
    "domain": "real", "split": "train"
  },
  "coordinate_frame": "vehicle_frd",
  "timestamp_units": "s",
  "action_order": ["a_lat", "a_long"],
  "action_units": "m/s^2",
  "camera_names": ["front_narrow", "front_wide"],
  "state_features": ["speed_mps", "yaw_rate_radps"],
  "preprocessing_id": "own-calibrated-rgb-v1",
  "arrays": {
    "timestamps": "timestamps.npy", "images": "images.npy",
    "ego_state": "ego_state.npy", "executed_actions": "executed_actions.npy",
    "expert_actions": "expert_actions.npy", "poses": "poses.npy"
  },
  "calibration": {"revision": "vehicle01-camera-v3"},
  "label_provenance": {"expert": "simulator expert controller v1"}
}
```

单摄像头使用一个 camera name 和 `V=1`，不需要复制图像伪造双视角。
同一 manifest 的 camera_names、state_features、preprocessing_id、坐标必须完全一致。
校准/监督来源字段由转换器保留，但内容的真实性需要数据负责人核实。

## 数组语义

| 名称 | 形状 / 类型 | 语义 |
| --- | --- | --- |
| `timestamps` | `[N]`，推荐 float64 | 同一时基的秒，严格递增，至少两帧 |
| `images` | `[N,V,H,W,3] uint8` | RGB，按 camera_names 排列；可用图片索引替换 |
| `ego_state` | `[N,D]` finite numeric | 对应 state_features，按名称/单位版本化 |
| `executed_actions` | `[N,2]` finite numeric | 时刻 t 之后区间内实际实现的加速度；输入时向后移一帧 |
| `commanded_actions` | `[N,2]`，可选 | 发给控制/执行器的加速度指令，独立保存用于诊断 |
| `expert_actions` | `[N,2]`，BC 需要 | 在观测 t 上由专家给出的监督动作 |
| `teacher_actions` | `[N,2]`，KD 需要 | 匹配预处理/历史状态的 Teacher 动作 |
| `teacher_features` | `[N,F]`，官方残差任务需要 | `comma-teacher cache` 产生的冻结基座特征；必须带 teacher_identity |
| `teacher_desire` | `[N,8]`，可选 | one-hot 意图状态；Teacher adapter 计算上升沿脉冲 |
| `teacher_traffic` / `teacher_action_t` | `[N,2]`，可选 | 逐帧通行侧 one-hot / 横纵向动作延迟秒数 |
| `expert_valid` / `teacher_valid` | `[N] bool`，可选 | 对应标签存在且有效；未给 mask 时该源全部有效 |
| `valid` | `[N] bool`，可选 | 传感器同步、状态等整体有效区间，未给时全部有效 |
| `poses` | `[N,4,4]`，世界模型需要 | `T_world_from_vehicle(t)`，米制平移、右手正交旋转 |
| `latents` | `[N,*latent_shape]`，缓存生成 | 数值 latent；元数据必须记录 encoder_fingerprint |

数组不允许 pickle/object。所有数组首维与 N 一致，数值状态/执行动作必须有限。
无效 expert/teacher 标签可以是 NaN，但相应 mask 必须明确为 false；不能用 NaN 表示有效动作。
无效原始状态应先删除/重建或拆成不同 episode，不把 NaN 状态直接交给模型。
一个 mask 不能在没有对应标签数组时单独存在。

每个 episode 的所有摄像头已同步到同一 N 帧索引。训练器只做 RGB float [0,1] 和双线性缩放，
不执行畸变矫正、透视 warp、BGR→RGB 猜测、坐标估计或输入状态标准化。
需要的 crop/视角对齐/归一化应在转换时固定，并更新 preprocessing_id；统计量只用 train 拟合。

## 时间和动作对齐

设 `o_t` 是第 t 帧图像/状态，标签 `a_t` 是基于该观测的目标动作，实际动作历史输入为
`executed_actions[t−1]`。窗口结束于 t，监督只取 t；当前动作标签和未来帧不进入策略输入。
episode / 有效区间开头的过去动作置零，避免负索引绕到末帧。

对控制系统延迟应先确定含义：传感器时间、计算指令时间、执行器接受时间和加速度测量时间
通常不同。转换器必须记录并校正偏移，不能把全体数组按行截断后视作同步。
CAN/定位可以按明确方法插值到相机时钟；图像通常取最近帧并限制偏差；无可靠对应时标无效。
大时间间断和 `valid=false` 都会切断窗口，history 与未来目标不得跨过去。
当前不自动补帧或做变采样率重建；每帧实际 dt 是模型输入。

若 BC 标签来自定位重建而非控制命令，应在 provenance 中写明 achieved-action proxy，
单独分析控制滞后与噪声。不要将 commanded、achieved、Teacher/reference 三类量混为一列。

## 坐标与位姿

规范车辆系 FRD：x 向前、y 向右、z 向下；横向加速度右转为正，纵向加速度向前为正。
位姿是车辆系到同一固定右手 world 系的变换，旋转满足 `RᵀR=I`、`det(R)=+1`。
所有参考位姿使用同一 episode/world 原点与方向，平移单位米。

相对运动 `ΔT = inv(T_world_from_vehicle(t)) @ T_world_from_vehicle(t+1)`。
Flow 使用平移加 roll/pitch/yaw，约定 `R=Rz(yaw) Ry(pitch) Rx(roll)`，单位弧度；
近 Euler 奇异点明确报错。Plan 使用同一自车坐标下未来位置的前两维。

ROS 中常见 FLU（前左上）与本合同不同。`flu_to_frd_poses` 明确变换 world 与 body 两套基，
实现 `C @ T @ C`、`C=diag(1,-1,-1,1)`；若保持原 world 系，只改变 body，则应使用 `T @ C`。
两种操作含义不同。CARLA/其他左手系需要单独写适配，不能直接调用该 helper 就认定正确。
左右符号、相机外参和轨迹方向至少做一段直行/左右转的人工可视检查。

## 延迟图片读取

episode.json 删除 `arrays.images`，增加 `"image_paths":"frames.json"`。
`frames.json` 是 N 行、每行 V 个路径：

```json
[
  ["images/narrow/000000.jpg", "images/wide/000000.jpg"],
  ["images/narrow/000001.jpg", "images/wide/000001.jpg"]
]
```

每个图片路径相对 frames.json；图片索引路径相对 episode.json，可使用绝对路径。
程序在校验时检查文件存在，在采样时用 Pillow 解码；损坏图片会在读取时报错。
视频应在转换阶段先抽帧，保留原视频、帧编号与时间映射，后续可再实现按需视频解码适配器。

## NPZ 导入示例

先复制 `configs/training/episode.example.json`，更换身份、相机/状态和预处理版本。
其中 `array_keys` 将规范数组名称映射到你的 NPZ key。例如原始 key 叫 `camera_rgb`：

```json
{
  "array_keys": {
    "timestamps": "time_s", "images": "camera_rgb",
    "ego_state": "state", "executed_actions": "achieved_accel",
    "expert_actions": "expert_accel", "poses": "T_world_vehicle"
  }
}
```

上面只是 metadata 中的一个字段，完整 metadata 仍需前面的格式/单位声明。
如果 key 已是规范名称，也可去掉 array_keys，让导入器选择已识别数组。
不相关的调试/未来信息不会自动成为模型输入。数据已经是 NPY 时可以直接写 episode.json，
再执行 `build-dataset` 与 `validate-dataset`。

从对齐后的 CSV / ROS / LeRobot / HDF5 转换时，可在自己的导出脚本中构造：

```python
import numpy as np

np.savez(
    "aligned-episode.npz",
    timestamps=np.asarray(time_s, dtype=np.float64),
    images=np.asarray(rgb_frames, dtype=np.uint8),  # N,V,H,W,3
    ego_state=np.asarray(state, dtype=np.float32),
    executed_actions=np.asarray(achieved_accel, dtype=np.float32),
    expert_actions=np.asarray(expert_accel, dtype=np.float32),
    poses=np.asarray(T_world_vehicle, dtype=np.float64),
)
```

这里只示意已对齐数组的导出，不推断你原始日志字段的意义。大图像建议改为逐文件 NPY
或图片索引，避免一次把全量 RGB 放入内存；NPZ importer 仍按一个 episode 转换。

## 与主流来源的映射

| 来源 | 可映射内容 | 转换时仍需解决 |
| --- | --- | --- |
| LeRobot / Parquet + 视频 | episode_index、timestamp、observation.images.*、state、action | action 实际含义、视频 seek 时间与帧率、route 分组 |
| RLDS / TFRecord | episode、steps.observation、action、is_first/is_last | 自有 schema、动作单位、图像视角；不能把 is_last 都当 terminal |
| ROS bag / MCAP | Image、Odometry、IMU、CAN/控制 topic | 时基、TF、传感器外参、执行延迟 |
| HDF5 / NPZ / NPY | episode group 与多维数值数组 | 名称、坐标/单位、压缩视频抽帧、无效区间 |
| 仿真器录制 | camera、pose、velocity、expert controller action | 原始 steer/throttle/brake 到加速度的定义、场景与地图隔离 |
| comma 原始数据 | 道路视频、定位/日志、Teacher | 匹配版本、frame_info、warp/标定、真实 route 映射 |

不要把相邻帧随机拆分为训练/测试。先按真实 route/session/场景池划分，再处理窗口；
模型选择用 val，最终报告用 test。数据版本应同时固定原始来源、转换脚本、标定与监督策略。

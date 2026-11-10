# 目标环境数据需求与交付规范

日期：2026-11-10（北京时间）。依据仓库 `87c8d26` 的读取器、Teacher adapter、训练窗口与审计实现整理。
适用目标：**自有目标环境数据 → 官方冻结基座的残差 SFT / Student SFT → 闭环 DAgger / Residual TD3 → 后续世界模型适配**。

这份文档面向数据采集、仿真导出、标注和训练人员，回答“采什么、怎么对齐、怎么保存、怎样验收”。
现有格式的实现细节见 [episode 合同](training-data.md)，训练步骤见 [新环境适配路线](adaptation-roadmap.md)。

## 1. 先明确当前需要交付什么

当前最有价值的第一份交付是：

> 在目标环境中采集连续、同步的双视角道路图像、车辆速度/偏航角速度、实际加速度和可信专家动作，
> 按独立 route/session 划分 train/val/test，转换为 `comma-episodes-v1`。

之后由项目运行官方 Teacher，自动生成教师动作、512 维特征及有效性 mask。
**采集端不需要生成 Teacher 特征，不需要先拥有 Student checkpoint，也不需要等官方 4B 世界模型。**

你上传的本地测试摘要报告：142 项回归通过、已有 4 routes/600 帧的数据审计通过，
真实官方 ONNX 在程序化道路的 15 场景基线中无终止，9 个扰动场景恢复 2 个，速度 MAE 约 2.28 m/s。
这些是用户本地报告，本轮未在云端重新运行该 GPU 实验。它们说明可以开始接真实目标数据，
也提示应完整记录速度目标、纵向命令与实际响应；单凭这些指标还不能确定速度误差的原因。

## 2. 各阶段需要的数据不同

下文区分四种要求，避免把建议当成代码限制：

- **格式必需**：当前 `Episode` / `Catalogue` 校验器要求，缺失时不能读取。
- **任务必需**：具体 Teacher、SFT 或闭环入口要求。
- **审计门槛**：`audit-dataset` 当前采用的就绪规则，可能比读取器更严格。
- **采集建议**：为可重复实验和后续扩展保留的信息，当前模型未必直接读取。

| 数据或能力 | 通用 episode | 官方残差 SFT | Student 目标域 BC/SFT | 闭环 DAgger / TD3 | 后续世界模型 |
| --- | --- | --- | --- | --- | --- |
| 图像、时间戳、车辆状态、实际动作 | 格式必需 | 必需 | 必需 | 环境每步提供 | 必需 |
| 当前观测上的专家动作 | 可缺省 | 当前任务必需 | 当前目标域 SFT 必需 | DAgger 必须可在线查询；TD3 专家正则启用时也需要 | 参考驾驶行为监督需要 |
| `teacher_rgb` 与模型视角标定 | 通用格式不强制 | 生成官方缓存必需 | 官方蒸馏时需要；纯 BC 可不用 | 官方在线路径必需 | 不能替代世界模型的完整几何标定 |
| Teacher 动作、特征和有效 mask | 不由采集端提供 | 缓存后必需 | KD 需要 Teacher 标签；默认官方蒸馏配置要求官方缓存来源 | 官方路径在线生成 | 可留作参考，不等于真实运动 |
| 实际发送的命令 | 通用格式可选 | 强烈建议 | 强烈建议 | 在线训练必须明确最终命令 | 当前 world 审计必需 |
| 位姿 `poses` | 可选 | 轨迹诊断建议 | 轨迹诊断建议 | 轨迹/恢复评估建议 | 当前 world 审计和原生世界模型任务需要 |
| 完整相机内外参、动作来源 | 通用读取器不强制 | 应随真实标定归档 | 强烈建议 | 官方输入、几何诊断需要 | 当前 world 审计必需 |
| 奖励、终止、时间截断、最终观测 | 普通 SFT 不必提供 | 不用于当前离线损失 | 不用于当前离线损失 | 在线 TD3 必需 | 作为交互环境时需要 |
| `reset/step` 和动作响应 | 固定数据集不提供 | 离线不需要 | 离线不需要 | 真正闭环必需 | 作为策略训练环境时需要 |

纯 Teacher 蒸馏在算法上可以没有独立专家标签，但它只能学习 Teacher 已有行为。
你的目标是改善新环境驾驶，因此第一轮目标域数据应优先提供独立、可信的专家监督。

## 3. 第一批数据的建议规模和场景

### 3.1 最小工程交付

建议第一批先准备 4 条独立 route：2 train、1 val、1 test，每条连续 30–60 秒，导出为 5 Hz。
这与当前审计的最小 route 数一致，用于验证采集、标定、标签和训练接口。
读取器允许更小的数据；4 条 route 并不代表足以证明模型泛化。

以 30 秒、150 帧、双路 256×512 RGB 为例：

| 项目 | 无间断、标签完整时的数量 |
| --- | --- |
| 单 episode 原始帧 | 150 |
| 官方 Teacher 预热后有效帧 | 125 |
| Student，history=4、stride=1 的专家窗口 | 147 |
| 4 episodes 的原始 RGB 大小 | 约 472 MB，约 450 MiB，不含额外原始视频 |

这些数量会因无效帧、时间间断和标签缺失而减少。
你本地审计的 train=294、val/test 各 147 对应 history=4 的专家窗口计数，
不能直接当成官方残差任务的有效 Teacher 样本数；后者还需要 `teacher_valid` 和专家标签同时有效。

### 3.2 扩展到研究 pilot

接口稳定后再扩大到约 50–100 条独立片段，并按实际任务覆盖场景。
可从 route 级约 70%/15%/15% 的 train/val/test 起步；比例是建议，代码不固定该比例。
保留源环境验证集，用于检查目标域适配是否损害原有能力。

第一轮应有意覆盖：

- 直行、左右弯、不同曲率和不同速度；不能只录制恒速直道。
- 加速、减速和稳定跟踪；低速/停车是否纳入，由本轮目标场景决定。
- 目标环境中的主要视觉变化，例如道路材质、相机安装、照明或天气。
- 可控仿真中的横向、航向和速度偏差，以及动作执行延迟。
- 相同初始状态下专家能够合理纠偏的片段，供后续恢复实验使用。

先限定车辆、相机 rig 和目标场景，再逐步增加变化。应记录各场景覆盖量，
防止少量罕见纠偏样本被大量正常直行帧淹没。

## 4. 原始采集信息与训练中间格式

建议同时保存两层数据：

1. **原始层**：原视频/图片、原始传感器时间戳、CAN/IMU/定位/控制日志、相机标定、仿真 seed/地图和版本。
2. **训练层**：完成同步、单位/坐标转换、标定预处理后的 `comma-episodes-v1`。

5 Hz 是当前官方 Teacher 的使用频率，不要求采集设备只能录制 5 Hz。
应保留设备能够提供的原始频率，另行生成 5 Hz 派生视图，方便后续重新同步或训练其他模型。

推荐交付目录：

```text
data/local/target-v1/
  manifest.json
  dataset-card.md                 # 场景、来源、拆分、转换版本、已知缺口
  calibration/                    # 原始与处理后标定文件
  quality/                        # 同步报告、分布统计、人工审核图
  episodes/
    target-train-001/
      episode.json
      timestamps.npy
      ego_state.npy
      executed_actions.npy
      commanded_actions.npy
      expert_actions.npy
      valid.npy
      expert_valid.npy
      poses.npy
      images.npy                  # 或 image_paths + 图片文件
```

`dataset-card.md`、`calibration/`、`quality/` 的目录名是交付建议，当前程序不会自动扫描它们。
需要由代码读取的标定/来源字段，应同时写入 `episode.json`，不能仅放在附件里。
原始数据可另存其他目录，在 dataset card 中记录可定位的来源与版本。

## 5. 逐帧数据字段、形状和单位

记 `N` 为 episode 帧数、`V` 为相机数、`D` 为状态维度；默认官方流程为 `V=2,D=2`。
数组采用普通数值 `.npy`，推荐浮点字段用 float32，时间戳和位姿用 float64。

| 字段 | 形状 / 类型 | 含义 | 当前要求 |
| --- | --- | --- | --- |
| `timestamps` | `[N] float64` 推荐 | 同一时基的观测时刻，秒 | 格式必需，至少两帧、有限、严格递增 |
| `images` | `[N,V,H,W,3] uint8` | RGB 图像，按 camera_names 排列 | 与图片索引二选一 |
| `ego_state` | `[N,D]` 数值，推荐 float32 | 与 state_features 一一对应 | 格式必需，全部有限 |
| `executed_actions` | `[N,2]` 数值 | 观测 t 之后区间实际达到的 `[a_lat,a_long]` | 格式必需，全部有限 |
| `commanded_actions` | `[N,2]` 数值 | 观测 t 后实际发出的最终加速度命令 | 通用可选；建议从第一天采集 |
| `expert_actions` | `[N,2]` 数值 | 当前观测对应的可信参考命令 | 当前目标域 SFT 必需 |
| `expert_valid` | `[N] bool` | 当前专家标签是否有效 | 标签缺失/不可靠时必须显式提供 |
| `valid` | `[N] bool` | 当前观测整体是否有效 | 可选；无效同步/质量区间建议显式标记 |
| `poses` | `[N,4,4]` 数值 | `T_world_from_vehicle(t)` | 当前 SFT 可选，建议保留以支持未来世界模型 |
| `teacher_actions` | `[N,2] float32` | 官方网络输出的两维动作 | 由 Teacher cache 生成 |
| `teacher_features` | `[N,512] float32` | 当前官方版本的冻结特征 | 由 Teacher cache 生成；官方残差训练需要 |
| `teacher_valid` | `[N] bool` | Teacher 预热与状态有效性 | 官方缓存显式生成，不手工改为全 true |
| `teacher_desire` | `[N,8]` 数值 | 逐帧意图状态，one-hot 或全零 | 可选离线 Teacher 控制 |
| `teacher_traffic` | `[N,2]` 数值 | 逐帧通行约定 one-hot | 可选，否则用 profile 固定值 |
| `teacher_action_t` | `[N,2]` 数值 | 横/纵向动作延迟条件，非负秒数 | 可选，否则用 profile 固定值 |
| `latents` | `[N,*latent_shape]` 数值 | 编码器生成的视觉 latent | 后续原生世界模型缓存，不由初始采集端伪造 |

所有写进 `arrays` 的数组首维都必须等于 N；不接受 object/pickle 数组。
mask 必须是 NumPy bool，不能用整数 0/1 数组代替。
缺省 `valid` 表示观测全部有效；缺省某来源的 label mask 表示该来源标签全部有效。
mask 不能在缺少对应标签数组时单独存在。

无效专家/Teacher 标签可以填 NaN，但对应 `expert_valid` / `teacher_valid` 必须为 false。
`valid=false` 不能替代标签自己的有效性声明；状态和实际动作即使在无效观测位置也要求有限。
缺失图像路径也不能靠 mask 绕过文件存在检查。损坏/缺失原始数据应重建、剔除或切段，并保存索引映射。

### 5.1 默认状态字段

```text
state_features = ["speed_mps", "yaw_rate_radps"]
ego_state[t] = [车辆速度，绕车辆 z 轴的偏航角速度]
```

速度单位 m/s，角速度单位 rad/s；不要直接保存 km/h、deg/s 后只更改字段名。
速度采用车速大小还是前向速度，应在数据说明中固定；存在倒车时应单独定义适用范围和符号。
当前读取器不自动估计状态、拟合标准化统计量或推断字段单位。

### 5.2 与速度目标有关的附加信息

建议另外记录 `target_speed_mps`、限速、路线/机动意图、道路曲率、参考中心线、控制延迟和可用的前车状态，
用于解释误差和定义场景。**这些名称目前不是默认模型输入字段。**
若相同图像和车辆状态对应不同目标速度，专家动作会因目标不同而不同；
模型没有获得目标条件时，SFT 可能学到相互冲突的标签。
需要将目标加入模型时，应显式扩展 `state_features` / `ego_state` 或策略条件接口，
同时调整环境与 checkpoint 的维度合同；不能只增加一个 JSON 字段就认为模型已接收目标。

## 6. 三种动作必须分别定义

当前动作统一为 `[a_lat,a_long]`，单位 `m/s^2`：

- `expert_actions[t]`：在当前观测上应发出的专家参考命令。
- `commanded_actions[t]`：实际行为策略/驾驶员/控制器最终发出的命令。
- `executed_actions[t]`：命令之后的时间区间内，车辆实际达到的加速度。

在专家采集 rollout 中，expert 与 commanded 可以相等；由于执行延迟、滤波和动力学，
executed 一般不会与它们严格相等。策略采集或 DAgger 时，expert 与 commanded 也可能不同。

```text
观测 o_t
  ├── 当前专家参考 expert[t]
  └── 实际命令 commanded[t]
               │ 控制/执行器/动力学
               ▼
        区间 [t,t+1) 的实际响应 executed[t]
               ▼
观测 o_(t+1)，其历史动作可以使用 executed[t]
```

当前训练窗口会将实际动作向后移一帧：`past_actions(t)=executed_actions[t-1]`。
当前 expert、当前区间的 executed 和未来观测不能作为 t 时刻的策略输入。
有效区间起点没有上一实际动作时，窗口构造器置零；采集端不用再手工生成 past_actions。

executed_actions 长度为 N，意味着最后一个观测之后也要有对应区间的实际测量。
采集应保留额外的末端状态/响应，或丢弃无法取得后续响应的最后观测，并一致调整所有数组；
不要用 0 冒充未知的最后实际动作。在线 TD3 还需保留该区间之后的最终观测。

原始 steer/throttle/brake、方向盘角、转向轮角均不直接等于本项目加速度动作。
从车辆运动学近似重建时可考虑 `a_lat≈v²·tan(road_wheel_angle)/wheelbase`，
但需要转向比、轴距、角度单位、符号和适用速度范围；这不是通用真值转换。
纵向实际加速度可由同步、滤波后的速度差分或重力补偿 IMU 估计，必须说明滤波和时延。

如果真实驾驶日志只有运动估计，没有专家控制命令，可构造 measured/achieved-action proxy 作为 BC 标签，
但必须在 provenance 中明确它是代理标签，并评估噪声和滞后。不要把它标成已知的专家命令。
默认新配置最终命令范围为 ±3 m/s²，这是实验控制约束，不是通用数据格式范围；
超出范围的数据应检查任务适用性和单位，不要静默裁剪来通过校验。

## 7. 时间同步与连续性

### 7.1 同步以观测时刻为基准

每个 t 的两路图像、ego_state 和 poses 应对应同一观测时刻；
expert/commanded 是基于该观测的命令，executed 是之后区间的实际响应。
图像编号、控制命令时间、执行器接受时间和 IMU 测量时间需要分开记录。

转换器应完成：

1. 统一相机、定位、IMU、CAN 与仿真时钟，保存原始时钟到统一时钟的映射。
2. 在共同的 5 Hz 网格上选择图像，按明确规则插值车辆状态，记录每次匹配的时间偏差。
3. 对实际动作按对应时间区间聚合；对命令按发送/保持语义采样，不能机械地平均所有控制字段。
4. 处理缺帧、重帧、长间断和 reset；切段或标无效，保留原因。

双视图和状态尽量采用同步触发或足够小的时间偏差。
10–20 ms 可作为初期同步误差的诊断目标，具体门槛应按速度和传感器条件制定；当前代码不自动验收这个建议值。
高精度 epoch 时间戳推荐 float64，避免 float32 在大数值下无法分辨 0.2 秒间隔。

### 7.2 当前代码的时间边界

| 入口 | 当前行为 |
| --- | --- |
| 普通 episode 校验 | 只要求有限且严格递增，不自动证明为 5 Hz |
| 官方 Teacher cache | profile 要求 context_hz=5；相邻有效输入 Δt<0.18 s 报错；Δt>0.22 s 重置 Teacher |
| Teacher 处理 `valid=false` | 当前帧无标签，并重置历史；后续连续有效帧重新预热 |
| Student/原生窗口 | 默认 max_gap_s=0.3；更长间断或 valid=false 切断窗口；不跨 episode |
| 原生世界模型窗口 | 历史、未来 target/anchor/plan 都不能跨越这些边界 |

目标仍是稳定 0.2 秒，0.18–0.22 秒是当前容差，不应被当成混合采样率方案。
在 5 Hz 派生数据上没有可靠同步映射时，先修转换器，不通过补零、乱序重排或重复帧伪造连续性。

## 8. 图像与官方 Teacher 的额外要求

### 8.1 摄像头与颜色

当前官方配置使用两路：`front_narrow` 和 `front_wide`，显式映射到 ONNX 的 `img`、`big_img`。
两路应来自独立、正确标定的视角；不能复制一张图来冒充双摄。
通用 Student 格式允许单摄像头，但现成双摄 checkpoint、官方 Teacher 和在线配置不能因此自动兼容。

输入训练层为 RGB uint8。用 OpenCV 解码得到 BGR 时，应在转换端明确转换。
视频需要先按真实时间戳抽帧；现有 episode 读取器不直接解码 HEVC/MP4、ROS bag 或 MCAP。

`images.npy` 要求所有帧/视角可以装成同一 `[N,V,H,W,3]` 数组。
不同分辨率的原始相机可采用图片索引，或在真实标定后统一成模型视角尺寸。
Student 的普通 loader 会双线性缩放并转为 float RGB [0,1]，不会自动矫畸变或修正相机视角。

### 8.2 两种 Teacher 图像模式

| `teacher_rgb.mode` | 采集/转换端交付 | Teacher adapter 行为 |
| --- | --- | --- |
| `prewarped_model_rgb` | 每路已经按实际标定转换为严格 `[256,512,3]` RGB uint8 | 检查尺寸并打包；不会用 resize 补成“标定” |
| `source_to_model_homography` | 原始 RGB，加每路真实计算的 source→model 3×3 单应矩阵 | 透视变换到 256×512，再打包 |

单应矩阵应有限、可逆且方向正确。镜头畸变需在适当阶段处理，
单应 warp 也不能自动消除任意场景的视差或替代外参标定。
仅把普通图片 resize 到 256×512，并不满足“已标定模型视角”的含义。

ONNX 内部的 `uint8[1,12,128,256]` 是 adapter 处理后的时序 YUV 输入，
**采集端仍交付两路 RGB 图像，不需要自己保存为 12 通道。**
Student 配置的 128×256 是它自身的缩放输入，也不能拿来替代 Teacher 的 256×512 RGB。

### 8.3 Teacher 控制 profile

`teacher_rgb` 至少包含：

| 字段 | 形状 / 值 | 要求 |
| --- | --- | --- |
| `format` | `openpilot-supercombo-rgb-v1` | 当前 adapter 的固定格式名 |
| `mode` | 上述两种之一 | 与实际图像准备方式一致 |
| `calibration_id` | 非空字符串 | 指向实际车辆/相机/预处理标定版本 |
| `camera_mapping` | `img`、`big_img` 各映射一个 camera name | 顺序与实际 narrow/wide 视角一致 |
| `context_hz` | 5 | 与实际采样频率一致 |
| `desire_mode` | `none` 或 `array` | array 模式需要 teacher_desire；不是提前生成的 desire pulse |
| `traffic_convention` | `[2]` one-hot | 上游以 is_rhd 为索引：[1,0] 对应 false，[0,1] 对应 true；按实际车辆约定配置 |
| `action_t_s` | `[2]` 非负秒数 | 横/纵向动作延迟条件；示例 [0.2,0.2] 不代表所有设备实测延迟 |
| `source_to_model` | 每路一个 3×3 矩阵 | 仅 homography 模式需要，不能保留示例 null |

`action_t_s` 与 5 Hz 的采样间隔是不同概念，需要分别记录。
逐帧 desire 的类别应使用对应官方版本的枚举并保存来源，不能自行给 8 个位置重新编号。
当前官方在线 policy v1 要求固定 `desire_mode=none`；离线支持 array 不代表在线导航意图适配已经实现。

## 9. 坐标、位姿与完整相机标定

统一使用车辆 FRD 右手系：x 向前、y 向右、z 向下。
正 a_lat 朝车辆右侧，正 a_long 朝前；偏航角速度绕 z 轴正向，单位 rad/s。
相机光学系采用 x 右、y 下、z 前时，应通过外参明确转换到车辆系。

`poses[t]` 是车辆到固定世界坐标的 `T_world_from_vehicle(t)`，平移单位米，
旋转满足 `RᵀR=I, det(R)=+1`，齐次末行为 `[0,0,0,1]`。
不能直接把位置和四元数拼成任意 4×4 矩阵；应明确四元数顺序、主动/被动旋转和变换方向。
同一 episode 的 world 原点/方向不能无标记跳变。FLU/ENU/NED/CARLA 坐标需要显式转换。

第一天就应归档的几何信息：

| 信息 | 含义 |
| --- | --- |
| 原始相机内参和畸变 | 分辨率、fx/fy/cx/cy、畸变模型与系数 |
| 相机外参 | 每路相机到车辆系的旋转、平移，包含安装位置/高度 |
| 处理后图像几何 | 去畸变、crop、warp 后对应的尺寸、内参/投影与版本 |
| 相机同步信息 | 时间戳来源、曝光/触发约定、不同相机时差 |
| 车辆参数 | 轴距、转向比、传感器安装、控制/执行时延及适用车辆 |

对于当前 world 审计，需在 episode.json 内提供 `calibration`：

- `id`：非空标定版本。
- `intrinsics`：按 camera_name 映射的真实 3×3 矩阵，焦距为正，末行为 `[0,0,1]`。
- `T_vehicle_from_optical`：合法 4×4 刚体变换；可以是共享矩阵，也可以按 camera_name 映射。

两台物理位置不同的摄像头应分别记录外参。
仅有 `teacher_rgb.calibration_id` 可满足 Teacher 的元数据检查，但不会通过 world 的完整几何审计。
矩阵数值合法也不证明几何正确，必须用投影/直行/左右转可视化核实。

## 10. 数据划分、身份与版本

### 10.1 按真实驾驶过程分组

`episode` 是一次连续片段；`route_id` 应对应可信的连续驾驶过程、采集会话或防泄漏分组。
同一路线连续片段不能因切成不同文件就跨 train/val/test，修改光照或裁剪也不能绕过分组。
若目标是跨地图泛化，进一步按地图/地理区域/场景模板分组；代码不会自动识别地理重复。

先划分 route/session，再抽帧、切窗和增强；val 用于选模型，test 保留到最终评估。
跨 episode 的地图、车辆、天气、场景模板、采集日期和 seed 应另行记录，以便检查相关性。

### 10.2 当前格式的身份约束

| 字段 | 当前要求 |
| --- | --- |
| manifest `format` | `comma-episodes-v1` |
| episode `format` | `comma-episode-v1` |
| `kind` | manifest 为 recorded 或 synthetic |
| `domain` | 每个 episode 为 real、sim 或 synthetic |
| `split` | train、val 或 test |
| `id` | manifest 内唯一；以字母/数字开头，仅字母数字、点、下划线、连字符 |
| `route_id` | 非空，同一 route 不能出现在不同 split |
| `revision` | 非空、不可含混的数据版本标记 |

真实驾驶可使用 recorded/real，高保真仿真器采集可使用 recorded/sim。
内置程序化 RoadEnvironment 当前使用 synthetic/sim；公式生成的 demo 使用 synthetic/synthetic。
`domain=sim` 不能单独证明是高保真仿真数据，真实性还要看 kind 和来源。

manifest 与 `episode.json.episode` 中的 id、route_id、domain、split 必须完全一致。
`episode` 对象仅放这四个身份字段；其他采集/场景信息放到顶层扩展字段。

同一 manifest 的 camera_names、state_features、preprocessing_id、coordinate_frame 必须相同。
带 teacher_features 时还要求 Teacher identity 相同；缓存/原始 episode 不能随意混在一个清单。

`preprocessing_id` 表示真实的预处理合同版本，不是场景名称。
源域/目标域采用相同标准化流程后才能共享同一 ID；不能只改字符串来假装输入兼容。
不同相机的具体 calibration_id 可以不同，但必须实际转换到共同输入约定。
`domain_mix` 只区分 real/sim/synthetic，不自动区分 source/target。

## 11. 可直接参考的 JSON 与图片索引

### 11.1 manifest 示例

这是最小 2/1/1 route 组织示例，实际目录需包含各自 episode.json 和数组。

```json
{
  "format": "comma-episodes-v1",
  "kind": "recorded",
  "revision": "target-v1",
  "episodes": [
    {"id": "target-train-001", "route_id": "session-001", "domain": "real", "split": "train", "metadata": "episodes/target-train-001/episode.json"},
    {"id": "target-train-002", "route_id": "session-002", "domain": "real", "split": "train", "metadata": "episodes/target-train-002/episode.json"},
    {"id": "target-val-001", "route_id": "session-003", "domain": "real", "split": "val", "metadata": "episodes/target-val-001/episode.json"},
    {"id": "target-test-001", "route_id": "session-004", "domain": "real", "split": "test", "metadata": "episodes/target-test-001/episode.json"}
  ]
}
```

### 11.2 episode 示例

以下假设图像已按真实标定转为 256×512 模型视角；字符串是命名示例，采集端需替换为实际版本与来源。
该示例覆盖当前 SFT/Teacher 的元数据；world 审计所需的真实 calibration 矩阵另按第 9 节补入。
若使用仿真数据，manifest 与 episode 同时改为实际 domain/kind，保持身份一致。

```json
{
  "format": "comma-episode-v1",
  "episode": {"id": "target-train-001", "route_id": "session-001", "domain": "real", "split": "train"},
  "coordinate_frame": "vehicle_frd",
  "timestamp_units": "s",
  "action_order": ["a_lat", "a_long"],
  "action_units": "m/s^2",
  "camera_names": ["front_narrow", "front_wide"],
  "state_features": ["speed_mps", "yaw_rate_radps"],
  "preprocessing_id": "target-calibrated-model-rgb-v1",
  "arrays": {
    "timestamps": "timestamps.npy",
    "images": "images.npy",
    "ego_state": "ego_state.npy",
    "executed_actions": "executed_actions.npy",
    "commanded_actions": "commanded_actions.npy",
    "expert_actions": "expert_actions.npy",
    "valid": "valid.npy",
    "expert_valid": "expert_valid.npy",
    "poses": "poses.npy"
  },
  "teacher_rgb": {
    "format": "openpilot-supercombo-rgb-v1",
    "mode": "prewarped_model_rgb",
    "calibration_id": "vehicle01-camera-rig-calibration-v1",
    "camera_mapping": {"img": "front_narrow", "big_img": "front_wide"},
    "context_hz": 5,
    "desire_mode": "none",
    "traffic_convention": [1, 0],
    "action_t_s": [0.2, 0.2]
  },
  "scenario_id": "target-cityA-day-001",
  "label_provenance": {
    "expert": "replace-with-actual-expert-or-proxy-definition-and-version",
    "commanded": "replace-with-actual-final-command-source-and-timestamp-convention",
    "executed": "replace-with-actual-motion-estimator-filter-and-interval-definition"
  }
}
```

如果某个可选字段暂不具备，应从 arrays 删除对应引用；不要留下指向不存在文件的路径，
也不要伪造 poses 或 commanded_actions 来满足后续审计。
普通 SFT 可以先缺这些可选字段，world 审计会明确报告尚未就绪。

### 11.3 图片索引替代 images.npy

从 arrays 删除 images，顶层增加 `"image_paths":"frames.json"`。
frames.json 包含 N 行、每行 V 个路径，按 camera_names 排列，例如：

```json
[
  ["images/front_narrow/000000.png", "images/front_wide/000000.png"],
  ["images/front_narrow/000001.png", "images/front_wide/000001.png"]
]
```

上面仅展示前两行，实际 N 行都必须存在。图片路径相对 frames.json，
episode 中的数组路径/索引路径相对 episode.json，manifest.metadata 相对 manifest。
绝对路径也支持，但跨机器搬运更适合相对路径。
Teacher cache 与 combine-datasets 的派生引用可能为绝对路径，迁移后应重新构建或重新生成派生清单。

## 12. 专家与数据来源如何记录

每个 episode 建议能追溯：

- 采集设备/车辆、相机 rig、地图/道路、场景、日期、采集 session、原始数据位置及版本。
- 转换脚本版本、同步算法、时钟偏移、图像处理步骤、坐标/单位转换和滤波设置。
- 专家是人工控制器、仿真规划器、人工驾驶命令还是运动估计代理；版本、限制和失效条件。
- commanded 是哪个行为策略发出的最终命令，executed 如何估计；控制延迟和传感器延迟如何区分。
- 每种无效 mask 的原因、剔除/切段规则、可用帧数与有效标签比例。

`label_provenance.expert/commanded/executed` 在当前 world 审计中要求为非空字符串，
但字符串本身不证明标签质量。建议同时保留可追溯的结构化采集记录。
Teacher 与 expert 分开：Teacher 的权重 hash 不能替代真实专家来源，Teacher 在新环境也可能给出不合理动作。

额外的 target speed、原始逐相机时间戳、碰撞/离路标志等可以放入独立日志、顶层扩展元数据或额外 NPY。
当前 `import-npz` 只导入代码白名单内的规范数组；rewards/terminated/truncated 等 recorder 字段不在该 importer 白名单，
需要保留时可直接写 NPY/episode.json 或扩展转换器。额外字段不会自动成为模型输入或 TD3 replay。

## 13. 闭环后训练还需要哪些信息

固定 episode 可用于离线 SFT、蒸馏、评估和离线回放。
要让策略通过动作改变下一观测，还需要可交互仿真器，或另行验证的动作条件世界模型环境。

当前环境接口必须提供：

| 接口/输出 | 要求 |
| --- | --- |
| `reset(seed=...)` | 返回确定性初始观测；官方 Teacher 每 episode reset，不跨 episode 泄漏状态 |
| `step(action)` | 接受最终 `[a_lat,a_long]`，推进动力学，返回真实 reached-state 观测 |
| `expert_action()` | 在当前策略访问状态上查询专家；不能取原录像同帧序号动作代替 |
| `describe()` | split、来源；官方 runner 还要求 scenario_id、scenario_pool、expert_source、reward_version |
| Transition | observation、有限 reward、terminated、truncated、achieved_action |
| `current_pose()` / `recording_metadata()` | 使用现有 recorder 导出时需要；记录标定与动作来源 |

在线 observation 的形状是 images=`[history,V,3,H,W]`、state=`[history,D]`、
past_actions=`[history,2]`、dt=`[history,1]`；RGB 为 float [0,1]，状态/动作仍使用上述物理单位。
官方在线输入为 5 Hz、history=1；Student 使用自身 checkpoint 的 history/尺寸和因果窗口。
控制器频率更高时，需要显式规定保持/降采样策略。

DAgger 采集应记录当前行为策略版本、访问状态、当前专家标签和场景来源。
TD3 另外需要最终命令、实际动作、reward 版本、下一观测、next base、动作范围、terminated/truncated。
启用 actor BC 需要当前状态专家标签，启用 smoothness 需要上一实际动作与正时间间隔。
真正终止不 bootstrap，时间截断从最终观测 bootstrap；不能返回 reset 后的帧冒充终止下一帧。

初始数据集不要求预先生成奖励。后续实验环境定义奖励及参考轨迹，
评测还应保存 cross-track、heading/speed error、动作 jerk、扰动时间和失败原因。
当前固定 recovery suite 使用程序化道路；自有平台仍需适配器及与真实目标一致的评测场景。

## 14. 为后续世界模型提前保留什么

在当前 SFT 数据之外，尽量从第一天保留：

1. 未降采样的原始视觉数据、原始时间戳和实际相机几何，支持后续视觉表示适配。
2. 连续位姿、速度/角速度、最终命令和实际动作，支持区分视觉变化与运动/执行器规律。
3. 独立专家动作/参考轨迹、目标速度、路线意图与其来源，支持参考驾驶行为监督。
4. reset/终止/截断、最后观测、地图/场景/车辆版本，支持多步 rollout 与误差诊断。

当前 `audit-dataset --stage world` 的具体门槛：

- train/val/test 至少 2/1/1 条独立 route，每个 episode 有可用的连续 history+future 窗口。
- 有 poses、commanded_actions、executed_actions、合法内外参和三个动作来源说明。
- 每个 episode 的有效帧中，独立 expert 标签覆盖比例至少 80%。

这是当前数据准备审计门槛，不是所有世界模型算法的理论最低要求。
原生 Flow 训练还需要由固定编码器生成 latents 和 encoder_fingerprint；
可用窗口的未来范围由 plan_steps、anchor_offsets 和 next target 共同决定，审计 future_steps 应覆盖该范围。
官方 4B 的精确训练接口、latent/anchor 布局与可训练检查点仍需后续适配，
不能因 world 审计 ready=true 就认为已经兼容官方 4B 训练。

## 15. 数据交付后的验证顺序与命令

以下在仓库根目录执行，沿用你已有 `.venv-gpu`；所有输出使用新目录/新文件名。

### 15.1 导入已经对齐的 NPZ，或直接注册 NPY

NPZ 不是原始日志解析器。先完成同步、坐标/单位转换和图像预处理，再使用：

```bash
.venv-gpu/bin/comma-driving import-npz \
  --input /path/to/aligned-target-train-001.npz \
  --metadata /path/to/target-train-001.metadata.json \
  --output-dir data/local/target-v1/episodes/target-train-001
```

metadata 使用上述身份/单位/teacher_rgb 声明；NPZ key 已为规范名称时可自动识别。
若需改名，增加 array_keys，例如 `"timestamps":"time_s"`、`"images":"camera_rgb"`；
该映射只改字段名，不转换单位、时间或含义。其余 3 个 episode 也需分别导入。
若已有正确的 NPY 与 episode.json，可跳过 import。

```bash
.venv-gpu/bin/comma-driving build-dataset \
  --root data/local/target-v1 --revision target-v1 --kind recorded
.venv-gpu/bin/comma-driving validate-dataset data/local/target-v1/manifest.json
.venv-gpu/bin/comma-driving audit-dataset \
  --manifest data/local/target-v1/manifest.json --stage student --history 4 \
  --output outputs/acceptance/target-v1-raw-audit.json
```

如果已经手写 manifest.json，则跳过 build-dataset；它拒绝覆盖既有清单。
纯 Student BC 的审计不要求 Teacher 来源，目标域专家监督必须有可用窗口。

### 15.2 生成官方缓存并检查有效帧

```bash
.venv-gpu/bin/comma-teacher cache \
  --model models/big_driving_supercombo.onnx --provider cuda \
  --manifest data/local/target-v1/manifest.json \
  --output-dir data/local/target-teacher-v1
.venv-gpu/bin/comma-driving audit-dataset \
  --manifest data/local/target-teacher-v1/manifest.json --stage student --history 4 \
  --require-official-teacher \
  --output outputs/acceptance/target-v1-official-audit.json
```

Teacher 从每次 reset 后的 frame index 25 开始有效，所以连续区间至少 26 帧才可能有一个有效输出。
建议 30–60 秒，并检查 cache 报告中每个 episode 的 valid_frames，避免全是短片段/频繁 reset。
缓存自动写入 teacher_identity、profile hash、有效帧数和来源引用；不手填特征或伪造官方 kind/hash。
原始 episode 不被修改，源图像引用必须持续可访问。

当前 `require_official_teacher` 审计/训练检查缓存记录的 pinned commit/hash；
实际模型字节校验在生成缓存的 TeacherSession 完成。仅手工加入 Teacher 标签不等于通过完整官方缓存流程。

### 15.3 人工检查与世界模型准备审计

```bash
.venv-gpu/bin/comma-driving report \
  --manifest data/local/target-teacher-v1/manifest.json \
  --output-dir outputs/reports/target-v1-data --max-episodes 4 --max-frames 150
.venv-gpu/bin/comma-driving audit-dataset \
  --manifest data/local/target-v1/manifest.json --stage world \
  --history 4 --future-steps 10 \
  --output outputs/acceptance/target-v1-world-readiness.json
```

world 审计缺项不会妨碍先开展满足条件的 SFT。
`ready=false` 返回码 1 并保留原因，输入格式/文件错误通常返回 2。
当前审计不会加载模型训练，也不会自动修正数据；默认只额外解码图像首尾，
图片索引的全部帧仍应由采集验收做完整可解码检查。
数据审计通过也不等于后续残差 Gate 2 或闭环配对准入通过，它们分别检验模型在 held-out 数据上的表现。

通过数据验收后，可将官方残差配置的 manifest 指向新缓存运行 SFT，
或在准备好兼容 Student 初始化后运行目标域 Student SFT，具体命令见 [技术路线](adaptation-roadmap.md#9-按顺序执行的命令)。

## 16. 交付验收清单与常见不合格情况

数据交付至少应有：

- manifest/episode/数组全部可读，N 和相机/状态字段一致，原始来源与转换版本可追溯。
- 两路真实视角；图像颜色、warp、分辨率、标定 ID 经人工确认。
- 稳定 5 Hz 派生时序，跨传感器同步误差和时延有报告，无跨 reset/间断拼接。
- expert/commanded/executed 含义分明，单位/左右转符号正确；有直行、左右转、加减速的人工检查。
- 缺失标签显式 mask，状态/实际动作有限，坏图像和缺文件已处理。
- 独立 train/val/test route；场景相关性已检查，不仅依靠不同文件名。
- Teacher 缓存每段有有效样本，专家与 Teacher 标签交集足以训练官方残差。
- 速度、加速度、命令—实际动作差异、样本覆盖量有分布统计，异常原因有说明。

| 情况 | 影响 / 应对 |
| --- | --- |
| 只有道路视频 | 缺当前必需状态/实际动作及可信目标监督，先补日志与对齐 |
| 只有方向盘/油门百分比 | 不能直接作为 m/s²，加速度定义和车辆/时延转换需先确定 |
| 一路图复制成两路 | 不满足真实双视角；采集正确相机，或另选兼容的单摄 Student 路线 |
| 20 Hz 原帧直接交 Teacher | 当前 cache 拒绝过快输入，先生成同步 5 Hz 派生数据 |
| 每段约 5 秒或频繁无效 | Teacher 反复预热，有效监督可能为空；延长连续段并修同步 |
| 把当前 actual action 当 past action | 引入未来响应信息；按窗口的 t−1 约定交付 |
| 仅填 calibration_id / 单位矩阵 | 元数据或数值检查不等于实际标定，先完成几何校准 |
| 只保留成功纠偏结果 | 无法解释失败分布；后续闭环评测保留全部场景和失败 |
| 同图像/状态却用不同速度目标打标签 | 目标条件未进入模型时监督冲突，先固定任务或扩展显式条件 |
| 全部数据 audit ready=true | 只证明结构与规则满足，仍需标定/标签质量和 held-out 性能验收 |

当前阶段应先交付“可追溯、连续同步、有独立专家标签的目标环境 pilot”，
再扩数据规模和训练预算。上述字段满足后，即可先进行官方残差 SFT；
闭环专家接口和世界模型扩展可以按后续阶段逐项补齐。

## 17. 代码依据

| 代码 | 负责的实际规则 |
| --- | --- |
| [episodes.py](../src/comma_post_training/data_pipeline/episodes.py) | 身份、数组形状/数值、图片索引、跨 split route、输入签名 |
| [windows.py](../src/comma_post_training/data_pipeline/windows.py) | 因果历史、上一实际动作、mask、间断、Teacher/latent 窗口 |
| [prepare.py](../src/comma_post_training/data_pipeline/prepare.py) | NPZ 白名单导入和 build-dataset |
| [preprocess.py](../src/comma_post_training/teacher/preprocess.py) | Teacher RGB 标定/profile/打包 |
| [cache.py](../src/comma_post_training/teacher/cache.py) | 5 Hz 边界、reset、Teacher 缓存与来源 |
| [audit.py](../src/comma_post_training/data_pipeline/audit.py) | Student/world 数据就绪与最低 route 规则 |
| [api.py](../src/comma_post_training/simulator/api.py) | 在线观测、动作与 Transition 合同 |
| [recorder.py](../src/comma_post_training/simulator/recorder.py) | 同步仿真录制、命令/实际动作、最终观测与来源 |

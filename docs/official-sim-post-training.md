# 官方基座的仿真数据与后训练：P0–P4 操作手册

当前路径：**固定官方 ONNX → 5 Hz 双视角 episode → Teacher 缓存 → 残差 SFT → Gate 2 → 在线 DAgger / Residual TD3**。
官方网络始终冻结；本轮不训练官方世界模型，也不重建完整 openpilot 车端控制系统。

## 阶段状态与验收边界

| 阶段 | 已交付 | 还需在你的机器完成 |
| --- | --- | --- |
| P0 | v2 输出解析、完整序列 reset、暖启动延迟统计、可选 ORT profile；本地已报告完成 60 帧验收 | 保留原始 inspect/smoke/profile 与环境版本供复查 |
| P1 | 5 Hz 录制器、独立双视角投影、专家/实际动作区分、位姿、route/scenario 分组、图表 | 按目标仿真器或真实数据核实标定、左右符号和专家质量 |
| P2 | 官方特征缓存、残差 SFT、逐 route/逐轴 MAE 与 jerk、Gate 2；本地已报告合成道路 val 改善 | 保留原始缓存/预测报告，扩展独立 route 和 test 验证 |
| P3 | 实时 Teacher policy、DAgger、TD3、三组同场景评估、策略重载 | Gate 2 通过后，用官方权重运行足够预算、多 seed 和扰动评测 |
| P4 | 保留独立扩展接口和任务路径 | 上游 Student 多任务蒸馏、官方 VAE/4B 世界模型需要独立环境与资源 |

云端测试使用显式标记的 `synthetic_fixture` ONNX，复现真实输入尺寸和输出布局，
不包含官方网络权重。你上传的本地日志证明官方权重通过身份校验并执行了两帧黑图，
两帧尚未 warmup；[历史文件摘要与哈希](acceptance/local-rtx3090-20261106/README.md) 已保存。
最新 [11 月 9 日本地摘要](acceptance/local-rtx3090-20261109/README.md) 已报告 60 帧 v2 CUDA
验收、官方仿真标签、SFT 改善与闭环退化；逐帧原始产物仍在用户本地。
这些证据不等同于驾驶行为或后训练效果验收。
本次 115 项测试和 30 秒四路端到端实验的结果、预览与 TD3 退化记录见
[云端工程验收](acceptance/cloud-official-sim-20261109/README.md)。
新增 [可视化与评估指南](visualization-and-evaluation.md) 提供 report 与 evaluation trace 命令。

## P0：固定模型、环境和动作合同

固定模型版本与下载链接见 [Teacher 指南](official-teacher-first.md)。
当前 adapter 为 `supercombo-084747-stream-5hz-v2`：

- `action` 原始 slice 为 `[2062:2066]`，宽度 4；仅前两项作为 `[a_lat, a_long]`。
  后两项保留为 `action_auxiliary`，不作为控制指令。`hidden_state=[2066:2578]`，宽度 512。
- 允许输出元数据包含 `slice(-2, None)`；使用受限解析和边界检查。
- 动作单位 `m/s²`，坐标 `vehicle_frd`：前、右、下；横向正值向右。
- 输入固定 5 Hz。前 25 帧作为 warmup，首个有效决策位于 `t=5.0s`。
- 默认 `traffic_convention=[1,0]`、`desire_mode=none`、`action_t=[0.2,0.2]s`，随 episode 记录。

在仓库根目录，沿用你已验证的 `.venv-gpu`，避免 CPU 锁文件替换 CUDA 依赖：

```bash
mkdir -p outputs/acceptance/p0-v2
.venv-gpu/bin/comma-teacher verify --model models/big_driving_supercombo.onnx
.venv-gpu/bin/comma-teacher inspect \
  --model models/big_driving_supercombo.onnx --provider cuda \
  --output outputs/acceptance/p0-v2/inspect.json
.venv-gpu/bin/comma-teacher smoke \
  --model models/big_driving_supercombo.onnx --provider cuda --frames 60 \
  --profile-prefix outputs/acceptance/p0-v2/ort \
  --output outputs/acceptance/p0-v2/smoke.json
.venv-gpu/bin/python -m pip freeze > outputs/acceptance/p0-v2/pip-freeze.txt
nvidia-smi > outputs/acceptance/p0-v2/nvidia-smi.txt
git rev-parse HEAD > outputs/acceptance/p0-v2/code-commit.txt
```

检查 `reset_replayed_frames=60`、`warmed_frames=35`、动作/特征有限、完整序列重放可重复。
同时报告冷启动、warmup 后 P50/P95/P99；`latency_ms` 仅含 ORT 推理与输出转换，
在线策略另测 RGB 处理与 Teacher 的总耗时。ORT profile 列出实际执行节点的 provider，
用于区分“CUDA 可用”和“图中哪些节点在 CUDA 执行”。profile 会包含第二次 reset 重放的执行。

v2 的缓存身份含 adapter 和输出语义；旧 v1 缓存及残差 checkpoint 需重新生成，
不能通过手改元数据假装兼容。模型放在忽略的 `models/` 目录，Git 只保存身份与验收摘要。
仓库之前误提交的单行普通文件 `models` 已移除，以免阻止创建这个权重目录。

## P1：仿真器 → comma-episodes-v1

本轮提供无需额外游戏引擎的 `RoadEnvironment`，在平面道路上执行加速度命令、
一阶执行器响应和车辆运动积分；两路图像分别按自己的 pinhole 内参进行道路地面投影。
场景几何与纹理固定在 world 系中，所以不同动作会改变下一帧图像。
专家为独立的道路跟踪/速度控制器，每次读取**当前仿真状态**，与 Teacher 输出无关。
这个 backend 明确标记为 `synthetic`，适合验证数据、时序和训练接口，不能据此宣称驾驶性能。

MetaDrive 仍是下一步可接入的引擎。已核查版本 `metadrive-simulator==0.4.3` 要求 Python `<3.12`，
与项目 Python 3.12 不兼容。推荐独立 Python 3.11 worker，通过进程通信适配下述合同；
需要实际验证相机、车辆响应和 headless 渲染后再接入。其原生 steer/throttle 不可直接标成加速度。
CARLA 也需要独立的坐标、传感器同步和低层控制器转换。本轮未交付这两个引擎的可用适配器。

```bash
.venv-gpu/bin/comma-driving record-sim \
  --output-dir data/local/sim-pilot-v1 --duration-s 30 --seed 42 \
  --train-routes 2 --val-routes 1 --test-routes 1
.venv-gpu/bin/comma-driving validate-dataset data/local/sim-pilot-v1/manifest.json
```

默认输出 4 条互异 route，每条 150 帧、两路 256×512 RGB；图像数组约 472 MB。
每个 split 采用独立随机流，记录 scenario 参数与 seed。30 秒是采集区间长度：
观测时间为 `0.0…29.8s`，另保存 `30.0s` 的最终观测。
重复运行必须换目录；不覆盖已有非空目录。短于 30 秒的数据可用于测试，但 `pilot_ready=false`。

```text
data/local/sim-pilot-v1/
  manifest.json
  recording-report.json
  episodes/sim-train-000-42/
    episode.json
    images.npy                  [N,2,256,512,3] uint8，narrow / wide
    timestamps.npy              [N] float64 秒，dt=0.2
    ego_state.npy               [N,2]，speed_mps / yaw_rate_radps
    poses.npy                   [N,4,4] T_world_from_vehicle
    expert_actions.npy          [N,2] 当前状态专家加速度命令
    commanded_actions.npy       [N,2] 本次实际下发的命令；录制时等于 expert
    executed_actions.npy        [N,2] 执行器之后实际达到的区间加速度
    valid.npy / expert_valid.npy
    rewards.npy / terminated.npy / truncated.npy
    final-observation.npz       采集截止后最终 RGB、state、past_actions、pose、时间
    camera-preview.png          双视角并排预览
    diagnostics.svg             轨迹、横向/纵向专家与实际动作曲线
```

每帧顺序是 `observation_t → expert_t → step(command_t) → achieved_t → observation_(t+1)`。
模型输入的 `past_actions_t=executed_actions[t-1]`，不能把当前目标动作输入残差头。
采集时长结束使用 `truncated=true`，真实失效使用 `terminated=true`。
最终观测独立保存，避免用 reset 后帧充当截断转移的 next observation。

`episode.json` 包括 `teacher_rgb`、唯一 `calibration_id`、每路内参、
`T_vehicle_from_optical`、相机高度、模型视角尺寸和 expert/executed 的来源。
模式为 `prewarped_model_rgb`，这里的图像直接按模型空间内参合成，未进行缩放。
接入原始相机时使用 `source_to_model_homography` 并给出两路实际单应矩阵。
校验器会拒绝缺失标定、尺寸不符和直接输入 20 Hz；高频仿真器由 adapter 显式执行子步并输出 5 Hz。

先打开 `camera-preview.png` 和 `diagnostics.svg`，检查双视角、转向符号、速度和执行器滞后。
程序检查有限数值、时间间隔、位姿合法性及 route 泄漏；语义和画面仍需人工核实。

### 自有仿真器接口

通过 `--factory your_package.adapter:YourEnvironment --env-kwargs args.json` 接入。
工厂参数固定为 `spec, split, **kwargs`：

```python
class YourEnvironment:
    def reset(self, *, seed: int) -> dict: ...
    def step(self, action) -> Transition: ...
    def expert_action(self): ...
    def describe(self) -> dict: ...
    def current_pose(self): ...
    def recording_metadata(self) -> dict: ...
    def close(self): ...
```

observation 为 CPU tensor：`images=[1,2,3,256,512]` RGB float `[0,1]`、`state=[1,2]`、
`past_actions=[1,2]`、`dt=[1,1]`。reset 的 dt 为 0，step 的 dt 为 0.2 秒。
`Transition` 包含下一观测、reward、两个 done 标志、`achieved_action=[2]`。
recording_metadata 提供 teacher_rgb 和真实标定；describe 至少提供 `kind`、`split`、
`scenario_id`、`route_id`、`scenario_pool`、`expert_source`、`reward_version`。
同 seed reset 必须可重复；训练与验证场景池必须隔离；done 时返回最终观测，不能自动 reset。

## P2：Teacher 缓存和离线残差

```bash
.venv-gpu/bin/comma-teacher cache \
  --model models/big_driving_supercombo.onnx --provider cuda \
  --manifest data/local/sim-pilot-v1/manifest.json --output-dir data/local/sim-teacher
.venv-gpu/bin/comma-driving validate-dataset data/local/sim-teacher/manifest.json
.venv-gpu/bin/comma-driving train --config configs/training/official_residual_sim.toml
.venv-gpu/bin/comma-teacher predict-adapted \
  --model models/big_driving_supercombo.onnx --provider cuda \
  --manifest data/local/sim-pilot-v1/manifest.json \
  --checkpoint outputs/official-residual-sim/best.pt \
  --split val --output-dir outputs/official-residual-sim-val
.venv-gpu/bin/comma-teacher gate-residual \
  --metrics outputs/official-residual-sim-val/metrics.json \
  --output outputs/official-residual-sim-val/gate2.json
```

每条连续 150 帧 episode 预期产生 125 个有效 Teacher 帧；缓存 revision 带模型 hash。
源 episode 与图像不被改写，派生缓存使用绝对源文件引用，搬迁时需一起迁移来源。
缓存记录 Teacher identity、adapter、RGB profile hash、provider、warmup 和有效样本数。

残差输入为 `512 features + 2 state + 2 past_actions + 2 official_action = 518`，
输出 `official_action + scale*tanh(head)`。零初始化时严格等于官方动作，默认每轴修正 ≤0.5 m/s²。
SFT 使用专家监督和偏离基座的 KD 约束。权重文件只包含 head/buffer 和基座身份；
完整离线训练 checkpoint 另含 optimizer/RNG 等续训状态，不含官方 ONNX 权重。
缓存生成需要 GPU Teacher；这个很小的 residual head 默认用 CPU 训练，也可 `--device cuda`。

预测结果包含逐帧 JSONL、全体及逐 route/逐轴 MAE、修正饱和比例和 jerk RMS。
这里 jerk 是 `Δa/Δt`，单位 `m/s³`；跨 episode、无效帧或时间间断不计算差分。

Gate 2 默认要求：所有 val route 都有有效专家标签，每 route 每轴 MAE 退化不超过 `1e-4 m/s²`，
至少一项改善超过 `1e-3 m/s²`。阈值可显式配置；失败返回 exit code 1 并保存报告。
这只是进入下一阶段的工程门槛，不能替代多 seed、误差分布和目标仿真器的质量评估。
如果失败，先检查标定、动作语义、残差饱和、数据对齐和训练预算，再重训；不以 test 选择阈值。

## P3：实时官方基座的闭环后训练

```bash
# 默认读取 Gate 2，并核对 checkpoint SHA 与 live Teacher identity
.venv-gpu/bin/comma-driving train-official --config configs/training/official_dagger.toml
.venv-gpu/bin/comma-driving train-official --config configs/training/official_td3.toml

# 完成模型选择后再执行独立 test 场景评估
.venv-gpu/bin/comma-driving evaluate-official \
  --config configs/training/official_dagger.toml \
  --checkpoint outputs/official-dagger/final.pt --split test
.venv-gpu/bin/comma-driving evaluate-official \
  --config configs/training/official_td3.toml \
  --checkpoint outputs/official-td3/final.pt --split test
```

配置内路径相对配置文件；`teacher_provider=cuda` 与训练 head 的 `device=cpu` 独立。
Gate 2 未通过时训练会停止。`--engineering-only` 是显式接口调试开关，报告会记录 bypass；
它仍执行官方模型身份校验，也不会使失败的 Gate 自动变成通过。
本轮云端测试仅通过 Python 测试注入使用合成 ONNX；CLI 不接受将其冒充官方模型。

### TeacherPolicy 与时序

`OfficialTeacherPolicy.reset()` 清空 ONNX 的所有历史队列。
`observe(obs, observation_id=n)` 每个环境观测仅推进一次 Teacher，返回可复用的 TeacherContext。
同一 id 的重复查询复用结果；乱序或跳号会失败。
`act(context)` 的多次调用和 learner update 不会再次推进 Teacher。
25 帧 warmup 期间由环境专家控制，样本不加入 learner；基座/离线残差/在线残差使用相同规则。
在线 v1 仅支持显式固定无意图 profile；逐帧导航/desire 需扩展接口后使用。

DAgger 在实际访问状态调用 `env.expert_action()`，更新现有离线残差头。
聚合池采用有界 FIFO，默认 10000 条；默认 minibatch 一半来自离线 **train** 专家样本，
另一半来自在线访问状态。默认以 0.2 概率执行专家动作，其他时候执行当前策略。
事件记录 scenario、episode、env step、行为策略更新版本、专家来源、命令与实际动作，
replay 中的 env_step 可追溯到对应事件；不重用原始回放动作作为偏离状态的专家标签。

TD3 冻结“官方 Teacher + 已训练离线 head”作为 anchor，再增加一个有界 residual actor。
这样保留 P2 成果；不会在进入 TD3 时丢掉 SFT head。
`features=[512 Teacher + 2 state + 2 past_actions]=516`，Actor 拼接 anchor 动作后为 518。
默认 TD3 额外修正每轴 ±0.25 m/s²，叠加离线 head 后总修正上限为两者之和，
最终仍受 `action_low/high=±3 m/s²` 限制。噪声按加速度单位定义。
双 Critic 接收最终下发的命令；`achieved_action` 单独保存为真实响应与下一步动作历史。
next features 和 next anchor 都来自 step 返回的下一观测，包括截断的最终观测。
true terminal 不 bootstrap，time limit truncation 仍 bootstrap。

内置奖励版本 `road_tracking_speed_jerk_v1`：每步为
`-dt*(cross_track² + 0.1*speed_error² + 0.01*sum(jerk²))`，越出道路/超速终止额外扣 50。
这是一项本项目实验奖励，不能称为 comma 官方奖励。Actor 另有偏离 anchor 的 L2 约束。

### 输出与对照

每轮输出 `run.json`、`events.jsonl`、`metrics.json`、`final.pt`、`learning_state.pt`。
run 固定配置、source hash、Teacher 身份、init checkpoint hash、Gate 和环境/标签/奖励版本。
同一组 val seed、同样 episode 上限下分别报告官方基座、离线残差初值、训练后策略，
区分包含 warmup 的总回报与模型控制阶段回报，并报告控制步数及终止/截断。
两种算法的交互预算应保持相同；正式比较至少多个 seed，并扩展到固定扰动集合。

`final.pt` 可由 evaluate-official 重载；TD3 文件包含冻结 anchor head 和额外 actor。
`learning_state.pt` 保留 replay、optimizer/target 供审计，但 **不支持精确在线续训**：
还缺少仿真器快照及 ONNX 内部历史的完整恢复。不要把重新 reset 当作从中断处继续。
原生 CNN/GRU 的 `train-closed-loop` 保留，官方模型使用独立的 `train-official`。

## 关键代码入口与 P4

| 文件 | 职责 |
| --- | --- |
| `simulator/recording.py` | 5 Hz 动力学、当前状态专家、独立双相机模型投影 |
| `simulator/recorder.py` | 流式 NPY 录制、标定/时间/身份、最终观测、诊断图 |
| `teacher/runtime.py` | 固定官方身份、输出布局、历史队列、reset、ORT profiling |
| `teacher/policy.py` | 单次观测上下文、实时 Teacher 与残差动作封装 |
| `teacher/evaluation.py` | 逐 route 统计、jerk、Gate 2 |
| `simulator/official_runner.py` | 场景隔离、在线 DAgger/TD3、同预算对照、组合导出重载 |
| `tests/test_official_sim_pipeline.py` | 仿真 → Teacher → SFT、实时 policy、两种 learner 回归 |

P4 再单独复现上游 convnext_pico + causal Transformer、多任务蒸馏损失和真实
`frame_info.safetensors / localizer.safetensors` 解码。官方 VAE、4B 世界模型及其
量化算子需要单独核对 GPU/精度支持；届时用新的环境、资产清单和验收记录，
不混入当前已验证的 `.venv-gpu`。

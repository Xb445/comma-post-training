# 本地结果怎样评估与可视化

你提供的本地摘要表明：**官方 Teacher v2 的 CUDA 推理已经通过完整 warmup/reset 初步验收，
离线残差在当前验证数据上改善，在线 DAgger / TD3 均退化。** 当前应保留离线残差为对照，
先诊断闭环失败，再扩大训练。

本页的本地数字来自你上传的文本摘要；云端没有取得其中的逐帧图像、events、profile 原文件和 checkpoint。
摘要的来源哈希、数值与汇总图见 [本地结果记录](acceptance/local-rtx3090-20261109/README.md)。
下面的工具在你的机器上直接读取已有产物，生成实测曲线与相机回放。

## 1. 怎样理解这次结果

| 层次 | 本地摘要 | 可以得出的结论 |
| --- | --- | --- |
| 工程测试 | 116 passed，21.63 秒；Ruff 已由你补装 | 工程回归通过；Ruff 安装本身不衡量驾驶效果 |
| 官方 Teacher | 60 帧，35 帧 warmed；完整 60 帧重放，动作/特征最大差为 0 | 官方权重在本地 CUDA 可推理、历史重置可重复；不是只测了两帧 |
| 推理延迟 | 冷启动 12453.77 ms；warmed P50/P95/P99 = 8.56/9.41/9.80 ms | 稳态 Teacher 推理较快；冷启动与 RGB/端到端处理应分别报告 |
| 仿真数据 | 4 route、600 帧、两路 256×512、5 Hz；500 个有效 Teacher 标签 | 本轮录制/缓存链路已跑通；数据仍来自内置合成道路 |
| 离线残差 | 横向 MAE 0.00379→0.00352；纵向 0.11780→0.05990 m/s² | 当前 val 上横向约降 7.1%、纵向约降 49.2%；横向绝对改善仅 0.00027 m/s² |
| 离线目标 | loss 0.009370→0.003598；Gate 2 val passed | 有监督适配在当前数据上有效；仍需独立 test / 更多 route 检验 |
| DAgger | 控制回报 -0.4092→-2.1697，差值 -1.7605 | 本次闭环退化，不能因训练完成就认定有益 |
| TD3 | 控制回报 -0.4092→-48.8842，差值 -48.4750 | 明显退化，需要定位动作、奖励、终止和 Critic，而非直接继续加步数 |

这里的 MAE 单位为 m/s²，动作顺序 `[a_lat,a_long]`。纵向相对改善较大，横向原本已很小；
不把两轴直接平均后隐藏差异。Gate 2 通过是工程门槛，不是统计显著性或真实道路泛化证明。

TD3 的回报与环境终止罚值同量级，但**仅凭这个均值不能确定发生了越界终止**。
先查看各 episode 的 `terminated`、`truncated`、控制步数、逐步 reward 和 reached-state 偏差。
一条验证 route 也不等于每次评估只有一个 episode，实际数量以 metrics 中的 episodes 为准。
当前 backend 的 terminated 表示道路越界/超速等任务终止，不等于已经测量碰撞率。

100 环境步、37 次 learner 更新确实较短；若沿用 `policy_delay=2`，TD3 Actor 只更新约 18 次。
这能解释证据不足，不能排除控制语义、探索噪声、优化目标或状态分布变化导致的退化。

## 2. 一条命令生成离线报告

在仓库根目录执行，沿用已验证的 `.venv-gpu`。报告本身不加载 ONNX / .pt，不启动模型推理，
只需要 NumPy、Pillow 和新增的可选 Matplotlib：

```bash
uv pip install --python .venv-gpu/bin/python "matplotlib>=3.10,<4"

.venv-gpu/bin/comma-driving report \
  --manifest data/local/sim-teacher-v2/manifest.json \
  --smoke outputs/acceptance/p0-v2/smoke.json \
  --run-dir outputs/official-residual-sim-v2 \
  --run-dir outputs/official-dagger-v2 \
  --run-dir outputs/official-td3-v2 \
  --prediction-dir outputs/official-residual-sim-v2-val \
  --output-dir outputs/reports/local-v2-20261109 \
  --max-episodes 4 --max-frames 150 --gif
```

上面路径对应你的摘要；如果 smoke 文件的真实名字不同，修改 `--smoke`。
推荐使用带 Teacher 缓存的 manifest：它引用原始 RGB，并额外提供 Teacher 动作和有效 mask。
也可使用 `data/local/sim-pilot-v2/manifest.json`，此时仅在提供了逐帧预测的 episode 上补充 Teacher 曲线。

打开 `outputs/reports/local-v2-20261109/index.html`。通常可双击离线打开；如果浏览器限制 file URL，
在仓库根目录运行：

```bash
.venv-gpu/bin/python -m http.server 8000 --bind 127.0.0.1
# 浏览器打开 http://127.0.0.1:8000/outputs/reports/local-v2-20261109/index.html
```

页面不依赖 CDN、Plotly 服务、W&B 或网络 API。相机缩略图和图表预览内嵌 HTML；
分享 HTML 可保留回放，分享完整目录则还保留 PNG/SVG/GIF/CSV 下载。
云端 CPU 开发环境可执行 `uv sync --locked --extra teacher --extra visualization`；
不要用 CPU 锁文件同步命令覆盖已经验证的 GPU 环境。

## 3. 报告有哪些内容

| 展示 | 输入 | 用途 |
| --- | --- | --- |
| 同步双相机播放、拖动时间、逐帧切换、1×/2×/4×/8× | episode RGB、timestamp | 检查两视角独立、时间连续、画面与动作对应 |
| World XY、速度状态、横纵向动作 | poses、state、expert/Teacher/commanded/executed | 检查左右符号、控制滞后、异常振荡和修正范围 |
| warmup/无效标签 mask | teacher_valid、valid | 确认前 25 帧不进入 Teacher 监督评价 |
| 训练与验证目标、逐轴 val MAE | SFT events.jsonl | 看收敛、验证退化、训练与验证差距 |
| 逐 route/逐轴基座与适配后 MAE | predictions.jsonl | 防止总体均值掩盖某条路线或单个动作轴退化 |
| learner loss、逐步 reward、终止/截断标记、在线动作 | 闭环 events.jsonl | 看 Critic/Actor/监督更新与异常动作发生的时间 |
| 官方基座、离线 anchor、在线策略对照 | 闭环 metrics.json | 同时看控制回报、每控制步回报、控制步数和终止数 |
| 重放真实策略访问的观测、执行后横向偏差 | 新增 evaluation trace | 定位失败过程；不是把专家录制图像当作在线策略画面 |

报告拒绝预测与 episode 的 frame/timestamp/route 对不上、重复预测和非有限日志。
无效标签保留为空；不会当成零误差。没有 events 就不重建训练曲线，也不把汇总值插值成实测点。
多组实验的 Teacher、初始 checkpoint、奖励、场景或预算缺失/不一致时，会标明不能直接横比。
即使同一设置，也需多 seed 才能评估稳定性；报告不会给一个场景伪造置信区间。

```text
outputs/reports/local-v2-20261109/
  index.html                   离线交互回放与图表
  summary.json                 统计、来源、输入文件哈希和抽样范围
  figures/*.png / *.svg        可直接插入汇报或继续编辑的图表
  episode-*.csv                已选 episode 的完整时间序列
  episode-*.gif                --gif 时导出已抽样的数据相机回放
  events-*.csv                 实测训练事件
  predictions-*.csv            逐帧开环预测
  trace-*.csv                  策略评估真实观测/动作/奖励/位姿
```

`max-episodes` 默认每个来源最多 4 段；`max-frames` 默认每段 150 张显示帧。
CSV 和静态图使用所选 episode 的完整数值序列，播放器/GIF 按源时间戳抽样；抽样范围记录在 summary。
图像缩略图只用于展示，不作为模型输入。图表坐标使用英文，减少本地缺少中文绘图字体的问题。
输出目录拒绝覆盖非空结果，重跑请换版本目录。所有原始输入只读，报告默认放在 Git 忽略的 outputs。

### 只看某一部分

```bash
# 只看仿真数据，无需加载 Teacher
.venv-gpu/bin/comma-driving report \
  --manifest data/local/sim-pilot-v2/manifest.json \
  --output-dir outputs/reports/sim-only --gif

# 只看现有 SFT / DAgger / TD3 训练与闭环汇总，无需图像
.venv-gpu/bin/comma-driving report \
  --run-dir outputs/official-residual-sim-v2 \
  --run-dir outputs/official-dagger-v2 --run-dir outputs/official-td3-v2 \
  --output-dir outputs/reports/training-only
```

## 4. 记录真正的闭环画面

旧版训练 events 没有保存每帧相机图像和位姿，因此无法从旧摘要恢复那次训练的视频。
新增 `evaluate-official --trace-dir` 会在**重新评估固定策略**时记录实际访问的画面。
它不重新训练，25 帧 warmup 仍由环境专家控制，Teacher 每 episode 重置。

沿用训练时的配置，保持 seed、场景、episode 上限与奖励相同：

```bash
# 官方基座
.venv-gpu/bin/comma-driving evaluate-official \
  --config configs/training/official_dagger.toml --official-only --split val \
  --trace-dir outputs/evaluation-v2/official

# 离线 residual anchor：省略 --checkpoint，使用配置的 init_checkpoint
.venv-gpu/bin/comma-driving evaluate-official \
  --config configs/training/official_dagger.toml --split val \
  --trace-dir outputs/evaluation-v2/offline

# DAgger 与 TD3 的最终组合策略
.venv-gpu/bin/comma-driving evaluate-official \
  --config configs/training/official_dagger.toml \
  --checkpoint outputs/official-dagger-v2/final.pt --split val \
  --trace-dir outputs/evaluation-v2/dagger
.venv-gpu/bin/comma-driving evaluate-official \
  --config configs/training/official_td3.toml \
  --checkpoint outputs/official-td3-v2/final.pt --split val \
  --trace-dir outputs/evaluation-v2/td3

.venv-gpu/bin/comma-driving report \
  --trace-dir outputs/evaluation-v2/official --trace-dir outputs/evaluation-v2/offline \
  --trace-dir outputs/evaluation-v2/dagger --trace-dir outputs/evaluation-v2/td3 \
  --run-dir outputs/official-dagger-v2 --run-dir outputs/official-td3-v2 \
  --output-dir outputs/reports/closed-loop-v2
```

你本地已有的 init checkpoint 路径符号链接可继续使用；如果更改目录，请在配置中明确设置对应路径。
TD3 重载必须保持训练时的 `action_low/high` 和 `residual_scale`，否则会拒绝错误解释 actor。

trace 保存 `trace.json`、`metrics.json`、逐 episode JPEG 与 steps.json。
动作和图像对应执行前观测，reward、next_pose、diagnostics_after_step 对应执行后的状态，
`interval_s` 明确时间差。内置道路 backend 额外记录道路横向偏差、速度误差和道路边界。
JPEG 为显示用途，不替代原始训练图像。未实现 current_pose/diagnostics 的自有 backend 会明确缺少相应曲线。

## 5. 建议下一步实验顺序

1. 固定当前通过 Gate 2 的离线 checkpoint，生成本页总报告，先检查逐 route/逐轴 MAE、jerk 和饱和比例。
2. 对四组策略重新记录相同 val seeds 的评估 trace。优先检查 TD3 是否 terminated，横向是否越界，
   commanded 是否触及上下限，achieved 是否滞后，以及 Critic/Actor loss 何时异常。
3. 排除输入/奖励/控制问题后再做单变量消融。DAgger 可先比较更低 learning_rate、更高 offline_fraction，
   以及与离线 SFT 接近的基座约束强度；TD3 可比较更小额外残差和探索噪声。每次只改一项并保留对照。
4. 增加训练 route、至少 3 个训练 seed、固定扰动和独立验证场景，再比较相同环境交互与计算预算。
   增大步数本身不是验收条件；先看退化是否得到解释和改善。
5. 方法与参数固定后再用独立 test；真实驾驶分布和更可信动力学仍需 MetaDrive/CARLA 或自有数据验证。

注意：DAgger 继承 init checkpoint 中 head 的 residual_scale，改闭环配置里的同名字段不会改写已有 head。
如果需要改变离线 head 的修正范围，应新建相应 SFT 配置、重训并重做 Gate 2。
TD3 配置的 residual_scale 则控制它在冻结 anchor 上新增的修正范围。
世界模型适配继续按 [P0–P4 路线](official-sim-post-training.md) 延后，当前优先把这套驾驶策略评估做扎实。

Ruff 和完整回归可在你已补装依赖的 GPU 环境中执行；ROS pytest 插件按你本地已验证方式隔离：

```bash
.venv-gpu/bin/ruff check .
.venv-gpu/bin/ruff format --check .
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv-gpu/bin/python -m pytest -q
```

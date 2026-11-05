# 驾驶数据与评测合同 v1

本仓库提供研究工具，尚未解析原始 HEVC、Cap'n Proto 或定位 safetensors。
以下是本项目的中间格式，不是 openpilot 原生消息格式。

## 数据段清单

样例见 `data/examples/driving/segments.json`。`kind=synthetic` 明确表示文件路径是示意占位，
可以用 `--metadata-only` 校验结构。真实数据使用 `kind=recorded` 并记录 Hugging Face 数据仓库的
40 位不可变 commit；清单中的路径相对于该清单，允许绝对本地路径。

每个 segment 有唯一 `segment_id`、可信的 `route_id`、`split=train|val|test` 与 artifacts 映射。
同一 route 的所有段必须属于同一 split。必须提供的三个 artifact 是 `fcamera.hevc`、
`frame_info.safetensors`、`localizer.safetensors`，可以另外提供 `ecamera.hevc`。
默认检查文件存在且非空；这只证明文件存在，不证明视频能解码、tensor 字段正确或已经对齐。

```bash
comma-driving validate-manifest data/examples/driving/segments.json --metadata-only
# 真实清单需要执行默认文件检查；样例路径尚无数据，下面的检查会失败：
comma-driving validate-manifest data/examples/driving/segments.json
```

## 评测 JSONL

每行一条时序预测，见 `data/examples/driving/predictions.jsonl`。

| 字段 | 合同 |
| --- | --- |
| `schema_version` | 整数 1 |
| `sample_id` / `route_id` | 非空字符串，sample_id 在报告内唯一 |
| `source` | `synthetic` 或 `recorded`，同一报告不得混合 |
| `action_order` / `action_units` | 固定 `["a_lat", "a_long"]`，`m/s^2` |
| `action_times_s` | 非负、严格递增的实际秒数，至少两个点 |
| `predicted_actions` / `reference_actions` | 与 action 时间数一致的 N×2 数组 |
| `coordinate_frame` | `ego_at_start`；本中间格式采用片段初始自车原点、x 向前、y 向右 |
| `trajectory.times_s` | 从 0 开始的独立未来时间网格，至少两个点；报告内所有记录相同 |
| `trajectory.predicted_xy_m` / `reference_xy_m` | M×2 米制位置，已转换为上述同一自车坐标系 |

提取器负责相机、定位、Teacher 与参考动作的时间同步、坐标变换、有效区间以及动作延迟。
评测器拒绝 shape 不匹配、重复/倒退时间戳和非有限值，不会静默截断或插值。
它接收已经产生的预测，不执行 Teacher/Student 推理，也不通过动作猜测缺失轨迹。

指标定义：每轴加速度 MAE、点平均轨迹 ADE、最后一点 FDE、每轴 `da/dt` 的时间加权 jerk RMS。
报告的汇总是**各记录指标的宏平均**；逐记录结果保留供 route 分组分析。它不等于将全部帧拼接后
计算的微平均，尤其不会在两个独立记录之间计算 jerk。

```bash
comma-driving evaluate data/examples/driving/predictions.jsonl
```

结果显式显示 `source=synthetic` 与 `mode=open_loop`。样例包含已知的线性加速度和 1 米侧向位置
误差，方便检查数值；不能作为真实驾驶模型的性能报告。

## BC/KD action-only loss

`comma_post_training.driving.losses.action_distillation_loss` 接受 `[..., 2]` 浮点动作，
`valid` 是对应的 `[...]` 布尔 mask。Teacher 与 expert 标签均 detach，避免反向更新监督源。
只启用 BC 或 KD 时无需提供另一种标签；有效位置为空、缺少启用的监督或权重无效时明确报错。
动作单位与归一化必须在调用前统一。当前 helper 不包含上游图像/pose/plan 等多任务损失，
也没有训练 runner；真实 Student 训练接入属于任务 E02。

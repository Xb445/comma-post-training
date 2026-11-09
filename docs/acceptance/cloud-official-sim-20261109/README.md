# 云端工程验收：2026-11-09

本记录使用 CPU、内置道路工程环境和 **synthetic_fixture ONNX**。
图的输入/输出布局与用户提供的官方报告一致，网络计算是本地生成的微型测试逻辑，
不是 comma 官方权重。官方权重与 CUDA 的本地证据另见
[RTX 3090 记录](../local-rtx3090-20261106/README.md)。

机器可读结果和各依赖版本见 [summary.json](summary.json)。

| 验证 | 结果 |
| --- | --- |
| 全量测试 | 115 项通过，其中 23 项 Teacher 测试、17 项仿真/在线集成测试 |
| Teacher smoke | 60 帧，35 帧 warmed，完整 60 帧 reset 重放一致，动作/特征有限 |
| 仿真录制 | 2 train / 1 val / 1 test 独立 route，各 30 秒、150 帧、5 Hz，共 600 帧 |
| 双视角 | 每路 256×512 RGB，分别按各自内参投影；下方预览保留了不同主点/视场 |
| 缓存 | 每条 route 有 125 个有效帧；源 episode 不改写；Teacher identity 带 v2 与 hash |
| 残差 SFT | 20 step，val MSE 从 0.316370 降到 0.141443；零修正与基座严格相同的测试通过 |
| Gate 2 | 这个合成图实验的 val 逐轴无退化且改善，通过；不代表官方权重下会通过 |
| 在线 DAgger / TD3 | 各 70 环境 step、18 个有效 Teacher 控制 step、17 次 learner update；容量 8 的 FIFO 验证淘汰 |
| 组合模型重载 | 两算法都在相同 val seed 重载，逐 episode 结果完全一致 |
| 工程检查 | Ruff、格式、whitespace、wheel/sdist、5 个上游源码 checkout 与 44 个源码锚点通过 |

短预算在线对照的模型控制阶段回报如下（同一验证 seed，episode 上限 34 step，warmup 25 帧）：

| 方法 | 控制阶段回报 |
| --- | ---: |
| 合成 Teacher 基座 | -0.500892 |
| 离线残差 anchor | -0.234861 |
| 在线 DAgger | -0.019950 |
| 在线 TD3 | -0.256514 |

TD3 相比离线 anchor 退化，不能据此声称 RL 有益。此处只有一个简单场景、单个评估 seed，
且 online 用 `engineering_only=true` 显式验证接口；Gate bypass 被保存在产物中。
正式实验按 [P2/P3 手册](../../official-sim-post-training.md) 使用官方权重、Gate 2、
目标仿真器、相同预算、多 seed 和独立测试集。

![独立 narrow / wide 投影](camera-preview.png)

[轨迹与横纵向加速度诊断图](diagnostics.svg) 用于人工检查符号、单位与执行器滞后。
检查实际命令和达到的加速度不同、动作改变下一观测、DAgger 标签对应当前状态、
Teacher reset 不跨 episode、TD3 next base 来自下一观测、terminated/truncated bootstrap 不同。

本次工作区中的完整运行产物位于忽略目录：

- `data/local/acceptance-sim-pilot-20261109/`
- `outputs/acceptance/cloud-official-sim-20261109/`

Git 只保存摘要和一组小型预览/诊断图；不包含数据图像数组、运行 checkpoint 或模型权重。
数据指纹的 scope 是 metadata/file stats，重新录制会因文件时间等产生不同指纹；
图像、动力学与专家输出的确定性由同 seed 重放测试验证。

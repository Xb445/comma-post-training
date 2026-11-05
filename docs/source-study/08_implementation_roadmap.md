# 08 实施路线与交付检查

先保持每一层的输入/输出可追溯，再扩大模型、数据和闭环长度。
已新增 [自有数据训练框架](../training-framework.md)，具备轻量原生模型与闭环算法的 CPU
验证；下面继续跟踪上游系统、官方资产和真实运行实验。

| 阶段 | 实现路径 | 交付/验收 | 当前状态 |
| --- | --- | --- | --- |
| S0 源码底座 | 五个固定子模块、openpilot 六个源码依赖、源码地图与版本检查 | HEAD/gitlink/锁一致；上游无本地修改；符号可定位 | 已完成源码获取；HF 模型子模块除外 |
| S1 系统观察 | `experiments/01_openpilot_replay` | 系统依赖和模型资产就绪；真实回放；service 时序表 | 待运行，当前未构建上游 |
| S2 模型接口 | `experiments/02_model_inference` | artifact specs、条件/缓存/动作单位、dense/streaming 对齐 | 源码分析完成，推理待模型 |
| S3 视频压缩 | `experiments/03_video_compression` | 小样本编码/解码、重建及 latent 报告 | 待权重、数据和独立环境 |
| S4 世界模型 | `experiments/04_world_model` | VAE round-trip、predict、动作条件响应、可重复 reset | 待 CUDA/模型包/数据 |
| S5 策略后训练 | `experiments/05_policy_distillation` | 原样多任务基线，再 BC/KD/混合、DAgger 对照 | 原生基线已具备；上游真实训练待接入 |
| S6 残差 RL | `experiments/06_residual_rl` | 双 Critic、n-step、冻结/有界 residual，等预算闭环对照 | 原生 one-step TD3 已实现；官方 recipe / n-step 对照待研究 |
| S7 部署连接 | exporter/adapter → tinygrad → modeld → 仿真/硬件 | 数值、状态、时序、控制接口及失败场景逐层通过 | 尚未实现完整转换和部署 |

## 首先完成的三个闭环

### 源码闭环（当前已具备）

```bash
python scripts/verify_sources.py
python scripts/source_map.py --check
```

源码校验返回 ok，但单列 `commavq/gpt2m` 模型子模块缺失。它不把“代码可读”解释为“模型可运行”。
父项目的 CPU 训练/评测测试与上游独立运行结果分别记录。

### 数据与教师闭环（下一阶段）

先取得 2 段真实样本、固定的 Teacher 和对应标定/定位数据。记录 SHA/revision，核查
`frame_info.safetensors`，建立可信 route 分组并检查空 split。完成 Teacher 一批输入到动作/plan
的实际推理与缓存，再扩大到 50–100 段。

当前 `configs/driving/upstream.lock.json` 中 dataset/world-model revision 仍为空，不能跳过
资产固定步骤。Student 的 timm 预训练骨干也是需要记录的资产。

### Student 与闭环环境闭环

先完成监督单步和小样本过拟合、保存/重载、独立验证。世界模型服务通过 health 后，还要做
encode/decode/predict 和受控 action-response 实验；随后才在相同 benchmark 下跑 DAgger/TD3。

## 实验产物的统一目录

每次在 `results/<experiment>/<run_id>/` 或现有忽略的 `outputs/` 下保存：

```text
run_config.json        算法、数据、种子、设备、预算
provenance.json        父项目/upstream/model/dataset/VAE/reward 版本与 hash
metrics.json           含单位、聚合方式、样本数与失败分母
events.jsonl           episode、step、时间、模型版本、约束与失败事件
checkpoints/           模型/优化器/target/replay 游标等可恢复状态
figures/               轨迹、动作、延迟与失败样例
```

源码 Git snapshot、checkpoint 版本、dataset revision 与 rollout behavior version 都要记录，
不能用一个“latest”替代。原始数据和大权重放 `datasets/`、`data/local/`、`models/` 等忽略目录；
需要审查的结果摘要与协议另行写入文档。

## 本次没有替代执行的步骤

完整的 2B/4B 世界模型训练、官方 TD3 内部 recipe、Student 的车端导出兼容性尚未取得或实现。
本次没有安装五个仓库的全部依赖，没有执行上游 build、Replay、GPU rollout 或硬件测试。
现有 CPU 合成结果验证了数据、原生模型训练/续训和闭环算法；尚不能证明真实驾驶性能。

对应更细的训练任务和依赖，继续使用 [原任务表](../tasks/openpilot.md)。

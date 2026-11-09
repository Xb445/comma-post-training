# comma-post-training

当前主线：**官方驾驶 Teacher 推理 → 冻结官方基座 → 用自有数据训练附加残差头**。
已接入仿真数据录制、离线残差 SFT 和实时官方基座 DAgger/TD3；世界模型适配训练留待资源就绪后开展。
既有原生策略/AE/Flow、DAgger/TD3 保留为研究基线。

已实现固定版本 ONNX Teacher adapter、显式 RGB 标定、状态队列、特征/动作缓存、
冻结基座残差 SFT 与组合推理接口。用户最新本地摘要已报告 RTX 3090 上的官方 v2 Teacher
60 帧 CUDA smoke、35 帧 warmup 后有效输出、500 个仿真标签与离线残差改善；DAgger/TD3 回报退化。
真实驾驶数据与闭环收益仍待验证，见 [本地结果分析](docs/acceptance/local-rtx3090-20261109/README.md)。

## 从这里开始

- [官方 Teacher 与冻结基座后训练](docs/official-teacher-first.md)：当前主线、命令、真实验收与阻塞。
- [P0–P4 仿真后训练操作手册](docs/official-sim-post-training.md)：数据录制、标定、缓存、Gate 2、实时 DAgger/TD3。
- [可视化与评估](docs/visualization-and-evaluation.md)：本地结果分析、离线 HTML 相机回放、PNG/SVG/CSV/GIF、真实策略评估 trace。
- [完整训练说明](docs/training-framework.md)：模块结构、三条训练路径、命令、续训和闭环接口。
- [自有数据接入合同](docs/training-data.md)：NPZ/NPY、图像索引、时间对齐、位姿和动作单位。
- [训练配置](configs/training/)：policy、compressor、world_model、DAgger、TD3 和 learned 环境。
- [源码研究手册](docs/source-study/README.md)：官方系统结构、modeld、世界模型、后训练与部署研究。
- [任务表](docs/tasks/openpilot.md)：区分原生基线能力与上游真实复现任务。
- [验证记录](docs/training-validation.md)：实际执行的链路、损失、闭环结果和适用范围。

## 训练与可视化示例

已有实验产物可直接生成报告，不需要重新推理或训练：

```bash
# 沿用本地 CUDA 环境；图表为可选依赖
uv pip install --python .venv-gpu/bin/python "matplotlib>=3.10,<4"
.venv-gpu/bin/comma-driving report \
  --manifest data/local/sim-teacher-v2/manifest.json \
  --run-dir outputs/official-residual-sim-v2 \
  --run-dir outputs/official-dagger-v2 --run-dir outputs/official-td3-v2 \
  --prediction-dir outputs/official-residual-sim-v2-val \
  --output-dir outputs/reports/local-v2 --gif
```

下面使用 CPU 环境运行原生小模型工程示例；官方基座流程见上面的 Teacher 指南。

```bash
cd /workspace/comma-post-training
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_SYSTEM_CERTS=true
uv sync --locked --extra teacher --extra visualization --python 3.12

uv run --locked --offline comma-driving make-demo --output-dir data/local/demo
uv run --locked --offline comma-driving validate-dataset data/local/demo/manifest.json
uv run --locked --offline comma-driving train \
  --config configs/training/policy.toml \
  --manifest data/local/demo/manifest.json --output-dir outputs/policy
uv run --locked --offline comma-driving evaluate-checkpoint \
  --checkpoint outputs/policy/best.pt --split test
```

将 `--manifest` 换为自有数据即可复用训练器。数据生成和训练拒绝覆盖已有非空目录；重复运行
请使用新目录，已有训练可用 `--resume` 接续。路径在配置内相对配置文件，在 CLI 内相对工作目录。
完整的压缩器、latent 缓存、世界模型及闭环命令见 [训练说明](docs/training-framework.md)。
原始数据、缓存、模型与输出请放入 `data/local/`、`datasets/`、`models/`、`outputs/` 等忽略目录。

## 关键结构

```text
configs/training/                    原生离线/闭环训练配置
src/comma_post_training/
  data_pipeline/                     episode 校验、导入、几何、因果窗口、域采样
  driving_policy/                    多视角 CNN + GRU + 加速度动作头
  world_model/                       Autoencoder、latent cache、条件 Flow、Plan Head
  engine/                            配置、损失、验证、初始化与严格离线续训
  teacher/                           官方权重校验、ONNX 流式推理、缓存与残差头
  simulator/                         环境接口、learned 环境、DAgger、残差 TD3
  reporting/                         离线 HTML 回放、Matplotlib 图表、CSV/GIF 与结果判读
  driving/                           CLI、既有清单和轨迹指标工具
docs/                                接入合同、训练教程、源码手册与研究路线
third_party/                         固定版本上游源码，保持原样
tests/                               数学/数据合同与实际 CPU 训练集成测试
```

闭环 backend 包括：合成点质量环境用于算法验证；双 pinhole 道路环境支持官方 Teacher 的
256×512 双视角数据录制与在线后训练；learned latent 环境以平面运动学产生
动作条件并回灌生成图像，仍使用记录中的未来 anchors 和代理奖励。真实仿真器通过环境工厂接入，
需要自行验证场景划分、动作响应、传感器和车辆模型；当前结果不代表驾驶泛化或车辆部署就绪。

## 官方源码与既有工具

`third_party/` 包含官方 openpilot、opendbc、panda、commaVQ，以及社区 openpilot.distill；
来源与恢复方法见 [上游说明](third_party/README.md)。openpilot 六个源码依赖也已固定。
commaVQ 的可选 HF `gpt2m` 模型子模块未下载，其他大模型/真实数据载荷也未取得。
源码可读、模型可运行和真实训练通过分别验收。

```bash
python scripts/verify_sources.py
python scripts/source_map.py --check
uv run --locked --offline comma-driving validate-manifest \
  data/examples/driving/segments.json --metadata-only
uv run --locked --offline comma-driving evaluate data/examples/driving/predictions.jsonl
```

旧清单和轨迹评测格式见 [评测合同](docs/driving-contracts.md)。初始环境保留了 `comma-train`
文本 SFT 示例，它不训练驾驶策略，使用方法见 [环境说明](docs/environment.md)。

## 验证与运行环境

```bash
uv run --locked --offline --extra teacher --extra visualization ruff check .
uv run --locked --offline --extra teacher --extra visualization ruff format --check .
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --locked --offline --extra teacher --extra visualization pytest -q
uv build
```

当前 126 项测试通过，包含实际策略/压缩器/Flow 训练、测试集评估、冻结编码器、数据缓存、精确 CPU 续训、
learned 环境动作响应、DAgger 更新，以及 TD3 双 Critic/延迟更新/截断 bootstrap。
另覆盖官方形状的合成 ONNX、完整 reset 重放、sim→cache→SFT、Gate 2 和实时 Teacher DAgger/TD3。
可视化测试覆盖 mask/时间对齐、输入只读、缺失日志、退化判读、trace 回放与导出。
依赖首次安装需要访问 PyPI / PyTorch CPU 索引；安装后原生合成链路可以离线运行。

默认锁文件固定 CPU Torch。CUDA 原生实验与上游复现应使用独立环境，见
[GPU 说明](docs/gpu-reproduction.md)。当前云实例无 GPU，未执行真实 Replay、CUDA 训练或硬件测试。
本仓库尚未选择开源许可证。

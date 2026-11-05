# comma-post-training

研究目标：基于 openpilot 教师模型进行驾驶策略蒸馏，建立 BC/KD 基线，再开展世界模型内
DAgger 和 Residual TD3 后训练。路线分析见 [研究说明](docs/openpilot-research.md)，
分阶段任务和验收见 [任务表](docs/tasks/openpilot.md)。

## 官方源码与源码研究手册

已将 openpilot、opendbc、panda、commaVQ 四个官方仓库和社区 openpilot.distill 作为固定版本
Git 子模块克隆到 `third_party/`，并补齐 openpilot 引用的六个源码依赖。
来源、版本和恢复方式见 [上游源码说明](third_party/README.md)。

从 [源码研究手册](docs/source-study/README.md) 开始，依次阅读系统架构、modeld、世界模型、
Rectified Flow、策略训练、残差 RL 和部署连接；[源码地图](docs/source-study/00_source_map.md)
提供实际符号、行号与上游永久链接。六个 [实验目录](experiments/README.md) 给出实现路径和验收协议。

```bash
# 已有源码 checkout 的完整核查；新 clone 的初始化步骤见 third_party/README.md
python scripts/verify_sources.py
python scripts/source_map.py --check
```

commaVQ 的 HF 模型子模块 `gpt2m` 因当前网络 403 尚未取得，大模型载荷也未下载。
源码齐备与实际编译/训练/Replay 就绪分别验证；本次只整理源码与研究路径。

## 当前可运行的驾驶研究工具

```bash
cd /workspace/comma-post-training
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_SYSTEM_CERTS=true
uv sync --locked --python 3.12
uv run --locked --offline comma-driving validate-manifest \
  data/examples/driving/segments.json --metadata-only
uv run --locked --offline comma-driving evaluate data/examples/driving/predictions.jsonl
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --locked --offline pytest -q
```

当前已实现 route 划分泄漏与数据文件检查、加速度 MAE/jerk/轨迹 ADE/FDE 评测、带 mask 的
action-only BC/KD 损失。示例为**合成 fixture**，用于验证合同和数学实现。真实数据清单应省略
`--metadata-only` 以检查文件，文件存在仍须进一步验证解码和对齐。

Teacher 推理、真实 Student 训练、世界模型 rollout、DAgger 和 TD3 的接入尚未完成。
当前云实例无 GPU，模型和真实数据未下载；目标 3090 Ti 的上游兼容性也未验证。
固定的上游版本及资产状态见 [upstream.lock.json](configs/driving/upstream.lock.json)，
[中间数据合同](docs/driving-contracts.md) 和 [GPU 复现准备](docs/gpu-reproduction.md) 给出后续入口。
`configs/driving/experiments.toml` 仅登记实验矩阵，尚未连接训练器。

## 原始文本 SFT 环境示例

初次搭建时保留的离线文本示例，用来检查 Python/PyTorch 安装；它不训练驾驶策略。

```bash
cd /workspace/comma-post-training
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_SYSTEM_CERTS=true
uv sync --locked --python 3.12
uv run --locked comma-train doctor
uv run --locked comma-train validate-data data/examples/sft.jsonl
uv run --locked comma-train train --config configs/tiny_sft.toml
uv run --locked comma-train generate --checkpoint outputs/tiny-sft --prompt "你好"
```

安装基础依赖需要访问 PyPI 与 PyTorch 官方 CPU wheel 索引。下述文本示例和 CPU 测试无需模型
下载、GPU、外部服务或 API 密钥。云环境的安装与启动细节见 [环境说明](docs/environment.md)。

示例随机初始化两层 GPT-2 模型，采用固定的字节分词器，只对 assistant 回复与结束符计算
交叉熵损失。它用于验证 SFT 工程链路；少量示例训练后的生成内容不代表真实大模型质量。
`metrics.json` 中的初始和最终损失均来自训练集，不是独立评测结果。

默认输出到 `outputs/tiny-sft/`，包含模型、分词器、训练配置和指标。为保护已有结果，训练拒绝
写入非空输出目录。再次运行时通过 `--output-dir outputs/another-run` 选择新目录。
配置文件中的数据和输出路径均相对于配置文件，命令行的输出路径则相对于当前目录。

## 开发与验证

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --locked pytest -q
uv build
```

测试覆盖聊天格式、assistant 损失掩码、变长批次，以及真实的 CPU 训练、检查点重新加载
和生成。集成测试在临时目录中创建模型，不覆盖手动实验结果。

## 数据与目录

每行一段完整对话，格式为 `{"messages":[{"role":"user","content":"问题"},
{"role":"assistant","content":"回答"}]}`。允许开头有一条 system 消息，随后 user 与
assistant 交替，最后一条必须是 assistant。输入中的控制标记保留给分词器。
超过 `max_length` 的样本会报错，避免截断后丢失训练目标。

```text
configs/                  训练配置
data/examples/            可提交的微型示例数据
data/local/               本地数据（Git 忽略）
src/comma_post_training/   数据校验、分词器、训练与 CLI
tests/                    单元与训练集成测试
outputs/                  检查点和指标（Git 忽略）
```

当前锁文件固定使用 CPU 版 PyTorch；上游驾驶模型训练应使用独立的 GPU 环境与锁文件。
本仓库尚未选择开源许可证。

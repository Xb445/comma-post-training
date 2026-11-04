# comma-post-training

Python 大模型后训练研究的基础工程，默认使用 Linux CPU、Python 3.12 和 uv。
先用完全离线的微型 SFT 示例验证数据、训练、检查点和生成链路，再扩展真实模型。

## 快速开始

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

安装依赖需要访问 PyPI 与 PyTorch 官方 CPU wheel 索引。训练和测试无需模型下载、GPU、
外部服务或 API 密钥。云环境的安装与启动细节见 [环境说明](docs/environment.md)。

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

真实预训练模型、LoRA、DPO、分布式训练和独立评测尚未实现。扩展这些能力时，需要分别确定
模型许可、数据来源、GPU/CUDA 环境和依赖版本；当前锁文件固定使用 CPU 版 PyTorch。
本仓库尚未选择开源许可证。

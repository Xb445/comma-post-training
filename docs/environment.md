# 开发环境

当前目标是 Linux CPU 上可重复的 Python 开发、数据校验和微型 SFT 训练。
Python 使用 3.12，依赖由 `uv.lock` 固定，PyTorch 使用官方 CPU wheel。
默认训练使用两个 CPU 线程；不需要启动数据库或 Web 服务。

## 云环境安装

在当前已有的 checkout 中作业；云任务本身已隔离，无需额外创建 Git worktree。
云机器已提供 Python 与 uv。设置可写缓存和系统证书信任，然后安装：

```bash
cd /workspace/comma-post-training
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_SYSTEM_CERTS=true
uv sync --locked --python 3.12
uv run --locked --no-sync comma-train doctor
uv run --locked --no-sync comma-train validate-data data/examples/sft.jsonl
```

保留环境注入的 HTTP/HTTPS 代理、认证和 CA 设置。安装使用 `pypi.org`、
`files.pythonhosted.org`、`download.pytorch.org`，wheel 可能重定向到
`download-r2.pytorch.org`。这些目的地包含在当前包管理器网络预设中。
不要关闭 TLS 或包校验。离线示例无需 Hugging Face 下载权限或密钥。

## 新任务启动与验证

环境发布可保留安装结果，运行进程不会随快照保留。本项目的 CLI 按需运行，无需常驻进程。
新任务先切换到上述目录，设置同样的缓存与系统证书变量，然后执行：

```bash
uv sync --locked --offline --python 3.12
uv run --locked --offline comma-train doctor
uv run --locked --offline ruff check .
uv run --locked --offline ruff format --check .
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --locked --offline pytest -q
```

如果缓存缺失，按安装步骤重新同步依赖。修改依赖时更新 `pyproject.toml` 与 `uv.lock`，
在正常开发和后续环境设置中使用 `--locked`。不要把 `.venv`、缓存、私有数据、密钥或检查点
加入 Git。源码和锁文件需要由后续开发流程提交；环境设置不会自动推送仓库。

## 可复现的手动实验

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --locked --offline \
  comma-train train --config configs/tiny_sft.toml
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --locked --offline \
  comma-train generate --checkpoint outputs/tiny-sft --prompt "你好" --max-new-tokens 16
uv build
```

第一次训练生成 `outputs/tiny-sft/`；再次训练应选择新的 `--output-dir`。
测试中的训练使用独立临时目录，可以重复执行。打包会生成忽略的 `dist/`。
当前配置尚未针对 CUDA、真实模型微调或分布式训练验证。

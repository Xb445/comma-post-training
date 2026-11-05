# 上游 GPU 复现准备

当前云环境已验证原生策略/AE/Flow、闭环算法和原始文本 SFT 的 CPU 合成链路；没有 GPU。
下列上游训练命令来自固定版本源码，**本次未执行**。

## 自有数据原生框架的 CUDA 环境

原生训练实现见 [训练框架](training-framework.md)，它与下面的上游 distill 环境分别管理。
根目录 uv.lock 指向 CPU Torch，即使配置 `device="cuda"` 也不会替换 wheel。
建议创建 `.venv-gpu/`，按实际 NVIDIA 驱动选择 PyTorch 官方支持的 CUDA wheel，并记录版本。
例如下面是 Torch 2.8 / CUDA 12.8 的独立安装示例，**未在此实例执行或验证**：

```bash
uv venv --python 3.12 .venv-gpu
uv pip install --python .venv-gpu/bin/python torch==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-gpu/bin/python -e .
.venv-gpu/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
mkdir -p outputs
uv pip freeze --python .venv-gpu/bin/python > outputs/gpu-environment.txt
.venv-gpu/bin/comma-driving train --config configs/training/policy.toml \
  --manifest data/local/own-v1/manifest.json --output-dir outputs/gpu-policy --device cuda
```

先建立 `outputs/` 再保存 freeze 文件。实际安装版本须与驱动兼容；用这个环境自己的 CLI，
不要再运行根目录 `uv sync` 将其当作 CPU 环境同步。先 batch=1 做前向/反向、保存/重载和
短训练，记录显存和数值，再扩大规模。闭环配置中的 device 和 latent 环境自己的 device
都需要显式设置。不要同时用不同 encoder 版本的缓存和世界模型。

以上安装遵循原生项目的 `torch>=2.6,<3`，不是对上游 Torch 2.13 环境的替代。
当前没有验证 CUDA 逐位复现；跨 Torch/CUDA/源码版本改变时应 warm start 新实验。

## 环境与资产

在独立的 `openpilot.distill` checkout/虚拟环境中使用其自己的 `uv.lock`，固定 commit
`215f33d4a860f380df4b4f4f075c6c9493f5c205`。不要将上游 Torch/CUDA 依赖同步到当前 CPU `.venv`。
现有云任务本身已隔离，不需要创建额外 Git worktree。

上游固定 Python `>=3.12.3,<3.13` 和 Torch `2.13.0`。先验证目标 GPU 的驱动、CUDA PyTorch、
HEVC 解码、ORT CUDA provider 和 BF16 前向/反向，再做 Student 单步。3090 Ti 24GB 是否能容纳
上游默认 Student、Teacher 和激活必须实测；首次 batch_size=1、workers=0，记录峰值显存。

模型地址与 openpilot dependency 必须都使用 `084747c75d2cbd23af65ab7a9e770bbd7b98bac9`。
数据根下应有 `data/<segment_id>/fcamera.hevc`、`frame_info.safetensors` 和
`localizer.safetensors`。先检查定位 tensor 字段和帧时间对齐；输入列表中不得混入验证/测试路段。
上游默认 loader 的 split 规则不等同于 route 级划分，需要在数据适配任务中处理。

当前下载访问尚未满足。Teacher 的上游地址位于 `media.githubusercontent.com`，数据、world model
和 VAE 位于 `huggingface.co`。先通过环境设置获得对应目的地访问，再解析 revision、下载少量
资产并核验哈希；大文件的重定向目标以实际请求为准。

## 监督单步验收

在准备好的上游 checkout 中，先按其锁文件同步环境，然后使用实际的数据路径：

```bash
# 上游命令；当前 CPU 环境未验证，不可直接在本仓库目录执行。
uv sync --locked
WANDB_MODE=offline uv run torchrun --standalone --nproc-per-node=1 \
  -m supervised.train --dataset /path/to/comma1M \
  --teacher /path/to/big_driving_supercombo.onnx \
  --steps 1 --batch-size 1 --workers 0 --output /path/to/new-run/supercombo.pt
```

验收：非空有效 batch、有限 loss/梯度、CUDA/ORT 真正执行、保存与重载一致；随后 tiny-overfit，
再扩大到 50–100 段。该命令运行上游完整多任务训练，不对应纯 BC 或纯 KD 消融。
记录实际配置/seed、数据与 Teacher 哈希、git commit、step time 和峰值显存。

## 世界模型与后训练

世界模型服务另外需要 `model.fp8_nvfp4.torchpackage` 和 VAE `commaai/vit-ae-2x-f8c32`。
权重格式并不能证明 3090 Ti 可以执行对应 kernel；先在选定硬件上跑单步和 VAE round-trip。
显存估计必须包含 latent/KV cache、VAE、diffusion 采样与并行 Student，而不仅是参数体积。

上游 `rl.server` → `rl.train --model <监督检查点>` 是 DAgger-style 路径；它需要 server 真正通过
健康检查与短 rollout。Residual TD3 另行实现，不能用运行这条命令代替 TD3 验收。
CPU fixture 或 MetaDrive 基准也不能替代该世界模型路径的真实验证。

# 05 监督蒸馏与 DAgger-style 后训练

目标：先复现上游多任务 loss，再比较 BC、KD、BC+KD 与闭环更新。
配套阅读：[策略训练](../../docs/source-study/05_policy_training.md)。

必需：Teacher/定位标签、route 级拆分、能解码的数据、独立 CUDA 环境和固定 Student 配置。
以下上游示例未在当前 CPU 环境执行：

```bash
# 在准备好的 distill checkout 中；使用新的输出目录
WANDB_MODE=offline uv run torchrun --standalone --nproc-per-node=1 \
  -m supervised.train --dataset /path/to/comma1M \
  --teacher /path/to/big_driving_supercombo.onnx --steps 1 \
  --batch-size 1 --workers 0 --output /path/to/new-run/supercombo.pt

# 世界模型服务和小规模 rollout 已通过后
uv run python -m rl.train --model /path/to/new-run/supercombo.pt \
  --segment VALID_COMMA_FOUR_SEGMENT_ID --steps 1 --batch-size 1 \
  --output /path/to/new-run/on_policy.pt
```

监督验收：非空 batch、有限 loss/梯度、小样本可过拟合、保存/重载一致、held-out route 评测。
闭环验收：访问状态和标注一一对应，冻结共享模型，只更新指定 head，保存 base+head 的配套版本。

本项目 `configs/driving/experiments.toml` 是登记表，尚不是上游 runner 的 CLI 配置。
action-only loss 已有 CPU 测试；三组方法接入、显式 seed、辅助 loss 控制和数据聚合需要继续实现。
DAgger-style 的 MSE head 更新与 Residual TD3 分开验收。

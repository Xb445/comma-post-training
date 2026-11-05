# 04 世界模型服务与动作响应

> 实现状态更新：原生 Flow/Plan 与 learned latent 闭环基线已提供；官方大模型 fidelity 与服务验收仍待执行。入口见 [训练框架](../../docs/training-framework.md)。

目标：跑通真实模型包的 encode/decode/predict，再验证动作对生成观测及 plan 的影响。
配套阅读：[结构](../../docs/source-study/03_world_model_architecture.md)、
[RF 实现设计](../../docs/source-study/04_rectified_flow.md)。

必需：兼容 CUDA 设备、固定 world-model package/VAE、comma four 双摄像头段和定位元数据。
`rl/env.py:Episode` 当前检查 `device_type=mici`；不将任意数据段视为兼容。
以下来自上游的运行示例本次未执行：

```bash
# 在准备好的 distill GPU 环境中
uv run python -m rl.server /path/to/model.fp8_nvfp4.torchpackage --device cuda
```

健康接口成功后还需要实际编码/解码/预测。对同一历史与 anchor，改变动作、控制延迟和 seed，
观察相对运动、视觉结构与参考 plan；记录 model_io、生成步长、采样 schedule/CFG、延迟和显存。
默认 Env 步长为 0.2s，不能将其等同车端 20Hz 策略或 100Hz 控制器。

交付：多步 rollout、reset 一致性、动作响应对照、来源完整的 plan/latent 缓存。
若 world model 在目标 3090 Ti 无法执行其量化 kernel，记录实际失败并选经过验证的兼容后端/设备；
不将 CPU toy RF 或另一模拟器的成功计入该模型复现结果。

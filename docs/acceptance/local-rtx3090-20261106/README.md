# 用户本地官方 Teacher 的初步 GPU 证据

来源为用户上传的四份日志；不是云端运行结果。原始文件 SHA256 和字节数见
[provenance.json](provenance.json)。保留了匿名化的模型路径及环境版本，移除了绝对 Python
模块路径、进程列表等机器信息。没有保存或上传 1.76 GB 模型权重。

- GPU 是 **RTX 3090 24 GB**，日志没有显示 Ti。驱动 580.178.04；PyTorch 2.8.0+cu128，
  Torch CUDA runtime 12.8；nvidia-smi 的 13.0 表示驱动可支持的最高 CUDA 版本。
- [GPU 环境](gpu-python.sanitized.txt) 与 [小网络计算](gpu-compute.txt) 确认 CUDA 可用，
  4096×4096 矩阵乘法有限，小网络五次更新的 loss/梯度有限。
- [smoke 报告](smoke-cuda-fixed.sanitized.json) 中官方权重的 size/SHA 与固定模型一致。
  CUDA provider 注册成功，原始图执行两帧；输出有限，报告 reset 可重复。
- 两帧都未 warmup，输入是黑色合成图；首次约 12.49 秒、第二次约 8.88 毫秒。
  第二次只有一个样本，不能据此认定稳定吞吐或实时性。provider 列表也不能说明所有节点在 CUDA。
- 小网络的 0.305 GB 分配和 nvidia-smi 的 885 MiB 桌面快照，都不是 Teacher 的峰值显存。

该日志来自用户本地修复的 `supercombo-084747-stream-5hz-v1`，当前代码为 v2。
v2 增加真实 4 通道 action slice 解析、负边界 slice、完整序列 reset 对比、暖启动延迟统计和 profiling。
必须在本地按 [P0 命令](../../official-sim-post-training.md#p0固定模型环境和动作合同) 重跑，
再将新的结果与这个历史证据分别归档。当前还没有真实驾驶片段、官方模型后训练改善或世界模型验收。

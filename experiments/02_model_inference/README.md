# 02 Teacher / Student 推理接口

目标：固定模型与预处理，验证一小批视频到 action/plan 的实际输出和时序缓存。
配套阅读：[modeld](../../docs/source-study/02_modeld_analysis.md)。

研究对象分别是 modeld 编译产物、distill 固定 Teacher 和可训练 Student。它们使用不同的模型
格式与接口，先各自验证再转换。distill Teacher 必须来自 `084747c…`，不替换成阅读快照的任意模型。

在匹配的独立 GPU 环境中，数据检查入口为上游 `supervised/dataset.py` 的 CLI；具体参数见其
`_main`。先确认选定 route/split 能产生有效 batch，再运行以下未执行的示例：

```bash
# 在准备好的 distill checkout 中，路径替换为真实资产
WANDB_MODE=offline uv run python -m supervised.dataset \
  --dataset-path /path/to/comma1M --model /path/to/big_driving_supercombo.onnx \
  --batch-size 1 --num-workers 0 --gpu-id 0
```

交付：输入/输出名称、dtype、shape、物理单位、图像布局、action_t、reset 约定；同一短序列的
dense/streaming 对照和失帧/低速测试；记录目标设备、provider、峰值显存与延迟分布。
Teacher 标签缓存同时保存模型、数据和预处理版本，避免只缓存没有来源的动作数组。

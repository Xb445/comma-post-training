# 03 视频压缩与 latent 研究

目标：先理解 commaVQ 的离散 token，再比较连续 VAE 的接口。
配套阅读：[世界模型结构](../../docs/source-study/03_world_model_architecture.md)。

入口：`commavq/utils/vqvae.py` 的 Encoder/VectorQuantizer/Decoder、`utils/gpt.py`、
`notebooks/encode.ipynb` / `decode.ipynb`、`nanogpt/prepare.py`。

步骤：先取得已固定 revision 的小样本和编码/解码权重；检查像素布局与 code grid；做
image→indices→image round-trip，记录重建、token 范围/频率、时序变化、延迟和内存。
再接相同输入的连续 VAE，以相同像素/场景比较，避免把不同数据和压缩协议放在同一结果表。

当前只完成源码读取，模型子模块尚不可达，未执行 notebook 或得到重建分数。
`nanogpt/prepare.py` 是整库准备 helper，代码注释涉及约 30GB 输出和额外缓存、40 workers；
不要将它当作当前 5 核云实例的小样本示例，先实现受限段数/worker 的 pilot 准备流程。
还需处理 EOT=1025 与已有 GPT vocab 的差异。

交付：样本来源、权重 hash、预处理、shape/单位、可复核的重建指标和图像；不能用压缩网络
定义文件证明完整 VQ/ViT 官方训练 recipe 已复现。

# 04 Rectified Flow：从数学目标到可复现实现

> 实现状态更新：已新增条件 Flow、Plan Head、缓存与采样基线；本文保留更大规模模型及上游接入的研究设计。入口见 [训练框架](../training-framework.md)。

本章给出用于后续开发的实现设计。五个已克隆仓库没有被本项目认定为包含完整的官方 RF 训练器；
`distill/rl/server.py` 是已有模型的推理调用入口。下面的伪代码与模块名是本项目拟实现方案，
不是已运行或经确认的 comma.ai 完整训练算法。

## 速度场目标

采用统一时间方向：`z_noise ~ N(0,I)`，`z_data` 为数据 latent，`τ∈[0,1]`。

\[
z_\tau=(1-\tau)z_{noise}+\tau z_{data},\qquad
v^*=z_{data}-z_{noise}
\]

模型 `vθ(zτ, τ, C)` 学习这个速度，基础损失为有效预测位置的 MSE。
这是常见的线性路径 flow-matching 目标，尚未包含官方可能使用的时间采样、权重、plan head
或其他辅助目标。若采用相反的噪声时间方向，目标符号与积分方向也必须一起改变。

## 一个训练样本应保存什么

| 部分 | 必需信息 |
| --- | --- |
| 原始来源 | dataset revision、route/segment、采样时刻、摄像头/标定与预处理版本 |
| latent | VAE revision、归一化、latent_shape、dtype、编码/缓存 hash |
| 条件 | 历史帧、动作或相对位姿、语义帧索引、明确允许的未来锚点 |
| 掩码 | 有效帧、已知条件帧、待预测帧、padding，分别保存 |
| 监督 | 数据 latent；如果训练 plan head，还需要时间/坐标系一致的参考 plan |

未来锚定使用的是明确声明的条件，不应把任意未来标签放进输入。按 route 先拆分，再提取上下文、
锚定帧与缓存；同一测试路段的未来图像不能回流训练。生成任务与真实部署可用条件的差异应单独记录。

## 推荐训练步骤（伪代码）

```text
读取同一 route 内的 clip，构建 history / anchors / targets / valid masks
用固定 VAE 编码；若 VAE 冻结，latent cache 必须记录 VAE hash
采样 τ 与 noise；只对待预测帧构建 zτ
输入 zτ、τ、条件 C 和符合布局的 attention mask
预测 v；只对有效 target 元素计算 MSE
如有 plan head：追加显式定义、版本化的 plan loss
反向传播，检查梯度/数值，更新 optimizer 与 EMA（若方案采用）
保存模型、optimizer、采样器/随机数、数据游标、mask 约定和配置
```

损失分母使用参与监督的元素数，而不是整个 padded tensor 的元素数。不可用目标应先排除，
避免 `NaN * 0` 污染损失。历史和锚定条件是否参与去噪/监督取决于任务设计，不能默认全部都参与。

## 条件注入与 attention 结构

可将动作/相对位姿、噪声时间、语义帧位置编码后注入 Transformer，例如条件归一化或额外 token。
具体层数、AdaLN 方式和参数尺寸须与目标版本核对，不由当前 wrapper 反推。

帧内的视觉 patch 可互相注意，预测帧通常只能读取其允许的历史与已声明条件。
“block-causal”表达的是帧/块关系；commaVQ 的普通 token 级 `torch.tril` 只是另一种具体实现。
distill 把 future anchor 放在序列前部，并另传 `fidxs`，所以不能简单用数组下标的下三角掩码
替代真实的语义时间与条件 mask。完整世界模型 attention 代码尚未从模型包中检查。

需要验证四类查询：同帧 patch、历史帧、已知 anchor、未知未来 target；逐项测试谁可访问谁。
padding、异步 clip 和不同长度 batch 也要检查。输出变化的实验只能证明特定实现行为，不能替代
对实际 attention mask 的检查。

## Euler 采样与 CFG

在上述时间方向下，从噪声起步，按 schedule 迭代：

\[
z_{k+1}=z_k+\Delta\tau_k\,v_\theta(z_k,\tau_k,C)
\]

CFG 的常见形式为 `v_uncond + s*(v_cond-v_uncond)`；使用了哪些条件 dropout，以及哪些条件
在 unconditioned 分支仍保留，应固定在训练/采样协议中。15/30 steps 和 CFG=2.0 只是在规划或
调用 wrapper 中出现的配置，不应直接当作所有版本的最佳参数。

KV cache 仅在被缓存部分的 K/V 对当前去噪迭代确实保持有效时复用。带噪 target 随采样步骤改变，
不能不经证明就复用它的缓存。通过“全量前向 vs 缓存前向”数值一致性检查后再评估加速。

## 后续代码落点与验收

建议新增独立 `world_model` 实验模块（当前尚不存在）：

| 模块 | 职责 | 首要验收 |
| --- | --- | --- |
| `latent_dataset` | 版本化 clip、条件、掩码和 VAE cache | 无 route 泄漏，时间与形状可追踪 |
| `conditioning` | 动作/位姿编码和语义索引 | 左/右等扰动按已知约定产生不同条件 |
| `flow_objective` | 时间采样、插值、监督 mask | 用解析 toy velocity 验证目标符号和 masked loss |
| `dit` | 条件 Transformer / plan head | 因果访问测试和极小 batch 过拟合 |
| `sampler` | schedule、Euler、CFG、cache | 可重复采样，全量/缓存一致，步数/耗时可测 |

先做小 latent 的解析任务验证 RF 数学与 mask，再接真实 VAE 和驾驶数据。它属于本项目的方法
实现，不能在没有权重/recipe 对齐的情况下称为“官方 2B/4B 训练复现”。

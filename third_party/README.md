# 上游源码集合

本目录使用 Git 子模块保存独立仓库的固定提交。源码锁文件为 [sources.lock.json](sources.lock.json)，
不是把五个上游合并成一个 Python 环境。各仓库的 LICENSE、提交历史、依赖和代码风格保留。

| 路径 | 来源 | 本次固定提交 | 用途 |
| --- | --- | --- | --- |
| `openpilot/` | comma.ai 官方 | `493ae5ad0eb6f1ef4d44817749d83dc089ce0001` | 系统架构、modeld、消息、规划、控制、Replay |
| `opendbc/` | comma.ai 官方 | `d1bd15ecb18e273e4af740dbbfb7def555828668` | 独立阅读最新车型接口、DBC 与 safety |
| `panda/` | comma.ai 官方 | `16c88abdecdc0e1dc802a963da979ade4a7d966c` | 主机通信、板端固件与 CAN 发送路径 |
| `commavq/` | comma.ai 官方 | `623a27f85689de4db3822cc81654cdf4db1b3da0` | 离散视觉压缩、自回归视频预测参考 |
| `openpilot.distill/` | YassineYousfi 社区研究项目 | `215f33d4a860f380df4b4f4f075c6c9493f5c205` | 定位、Student 蒸馏、DAgger-style 更新 |

openpilot 自己固定了 `msgq_repo`、`opendbc_repo`、`panda`、`rednose_repo`、`teleoprtc_repo`、
`tinygrad_repo` 六个依赖，它们也已取得。这些版本由 openpilot 的 gitlink 决定，不一定等于
上表独立阅读仓库的最新版本。分析一套可运行 openpilot 时，以其内嵌版本为准。

## 在新 checkout 恢复源码

从项目根目录执行以下已在当前工作区验证的命令。先恢复五个主仓库，再递归恢复 openpilot
的源码依赖，避免把 commaVQ 的模型资产下载混入基础源码获取：

```bash
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --depth 1 --filter=blob:none --jobs 3
GIT_LFS_SKIP_SMUDGE=1 git -C third_party/openpilot submodule update \
  --init --recursive --depth 1 --filter=blob:none --jobs 3
python scripts/verify_sources.py
python scripts/source_map.py --check
```

尚未初始化的是 `commavq/gpt2m`：它指向 Hugging Face 上的预训练模型仓库，当前代理返回 403。
这不影响阅读 `commavq/utils/gpt.py` 等已取得的实现。其他模型大文件仍可能是小型 LFS 指针，
不代表真实模型载荷已下载。openpilot 当前提交的 `lfs.py` 使用 Hugging Face bucket 作为后端；
取得代码不能替代该资产下载步骤。

只需核查项目元数据、暂不拉取子模块时：

```bash
python scripts/verify_sources.py --metadata-only
```

脚本只读锁文件、`.gitmodules`、Git index 和 checkout。完整模式还检查源码子模块的 HEAD、
工作树及递归依赖；缺失的已声明模型子模块单独列出。它不验证模型推理或训练可用性。

升级源码时一起更新：gitlink、锁文件、源码阅读地图及受影响的接口说明。不要在日常复现中
运行 `git submodule update --remote` 来漂移版本。建议将研究修改放在父项目的 adapter、配置和
实验目录，若必须改上游则记录独立补丁及基线，避免将源码阅读快照与实验修改混在一起。

本文中的 openpilot 是 2026-11-05 的研究快照，不声称等同某个 release。distill 教师仍固定
`084747c…`，详见 [训练资产锁](../configs/driving/upstream.lock.json)。

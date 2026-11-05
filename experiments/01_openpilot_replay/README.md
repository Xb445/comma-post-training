# 01 系统回放与消息时序

目标：观察摄像头/状态/模型/规划/控制消息，绘制跨进程时序，区分日志回放与模型重新推理。
配套阅读：[系统架构](../../docs/source-study/01_system_architecture.md)。

源码入口：openpilot 的 `tools/README.md`、`openpilot/tools/replay/README.md`、
`openpilot/cereal/services.py`、`process_config.py`。

以下上游运行步骤本次未执行。先在独立普通 checkout 和匹配系统环境取得相同源码版本、
子模块与模型载荷，处理好该版本 setup/hook 的目录约定：

```bash
# 在准备好的独立 openpilot runtime checkout 中
tools/op.sh setup
source .venv/bin/activate
scons -u -j2
openpilot/tools/replay/replay --demo
```

本版本的模型构建可能因缺少 LLVM 被跳过；验证实际运行产物，不能只依据 scons 退出码。
模型下载和 demo route 还需要实际目的地的网络访问。当前未完成这些前提。

交付：每个 service 的源/接收时间、频率、valid、丢帧/异常计数，以及选定片段的时序图。
至少跟踪 modelV2、longitudinalPlan、controlsState、carControl、carState。明确标注哪些是历史
记录、哪些由当前进程生成。输出放 `results/01_openpilot_replay/<run_id>/`。

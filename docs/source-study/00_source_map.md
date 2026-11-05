# 固定版本源码阅读地图

由 `python scripts/source_map.py` 从本地源码静态生成；升级版本后须重新核对。
本地路径相对于项目根目录；链接指向上游固定提交及实际符号所在行。

| 仓库 | 本地文件 | 符号/入口 | 行 | 上游源码 |
| --- | --- | --- | ---: | --- |
| openpilot | `third_party/openpilot/openpilot/selfdrive/controls/plannerd.py` | `main` | 11 | [定位](https://github.com/commaai/openpilot/blob/493ae5ad0eb6f1ef4d44817749d83dc089ce0001/openpilot/selfdrive/controls/plannerd.py#L11) |
| openpilot | `third_party/openpilot/openpilot/selfdrive/pandad/pandad.cc` | `void can_send_thread(` | 58 | [定位](https://github.com/commaai/openpilot/blob/493ae5ad0eb6f1ef4d44817749d83dc089ce0001/openpilot/selfdrive/pandad/pandad.cc#L58) |
| openpilot | `third_party/openpilot/openpilot/selfdrive/modeld/SConscript` | `def compile_model(` | 42 | [定位](https://github.com/commaai/openpilot/blob/493ae5ad0eb6f1ef4d44817749d83dc089ce0001/openpilot/selfdrive/modeld/SConscript#L42) |
| openpilot | `third_party/openpilot/openpilot/selfdrive/modeld/constants.py` | `ModelConstants` | 6 | [定位](https://github.com/commaai/openpilot/blob/493ae5ad0eb6f1ef4d44817749d83dc089ce0001/openpilot/selfdrive/modeld/constants.py#L6) |
| openpilot | `third_party/openpilot/tools/op.sh` | `function op_setup()` | 181 | [定位](https://github.com/commaai/openpilot/blob/493ae5ad0eb6f1ef4d44817749d83dc089ce0001/tools/op.sh#L181) |
| openpilot | `third_party/openpilot/openpilot/system/manager/process_config.py` | `procs = [` | 73 | [定位](https://github.com/commaai/openpilot/blob/493ae5ad0eb6f1ef4d44817749d83dc089ce0001/openpilot/system/manager/process_config.py#L73) |
| openpilot | `third_party/openpilot/openpilot/selfdrive/modeld/modeld.py` | `ModelState` | 147 | [定位](https://github.com/commaai/openpilot/blob/493ae5ad0eb6f1ef4d44817749d83dc089ce0001/openpilot/selfdrive/modeld/modeld.py#L147) |
| openpilot | `third_party/openpilot/openpilot/selfdrive/modeld/modeld.py` | `get_action_from_model` | 53 | [定位](https://github.com/commaai/openpilot/blob/493ae5ad0eb6f1ef4d44817749d83dc089ce0001/openpilot/selfdrive/modeld/modeld.py#L53) |
| openpilot | `third_party/openpilot/openpilot/selfdrive/modeld/helpers.py` | `load_oob` | 15 | [定位](https://github.com/commaai/openpilot/blob/493ae5ad0eb6f1ef4d44817749d83dc089ce0001/openpilot/selfdrive/modeld/helpers.py#L15) |
| openpilot | `third_party/openpilot/openpilot/selfdrive/modeld/parse_model_outputs.py` | `Parser` | 20 | [定位](https://github.com/commaai/openpilot/blob/493ae5ad0eb6f1ef4d44817749d83dc089ce0001/openpilot/selfdrive/modeld/parse_model_outputs.py#L20) |
| openpilot | `third_party/openpilot/openpilot/selfdrive/modeld/fill_model_msg.py` | `fill_model_msg` | 73 | [定位](https://github.com/commaai/openpilot/blob/493ae5ad0eb6f1ef4d44817749d83dc089ce0001/openpilot/selfdrive/modeld/fill_model_msg.py#L73) |
| openpilot | `third_party/openpilot/openpilot/cereal/log.capnp` | `struct ModelDataV2` | 1054 | [定位](https://github.com/commaai/openpilot/blob/493ae5ad0eb6f1ef4d44817749d83dc089ce0001/openpilot/cereal/log.capnp#L1054) |
| openpilot | `third_party/openpilot/openpilot/cereal/log.capnp` | `struct DrivingModelData` | 1010 | [定位](https://github.com/commaai/openpilot/blob/493ae5ad0eb6f1ef4d44817749d83dc089ce0001/openpilot/cereal/log.capnp#L1010) |
| openpilot | `third_party/openpilot/openpilot/cereal/services.py` | `SERVICE_LIST` | 99 | [定位](https://github.com/commaai/openpilot/blob/493ae5ad0eb6f1ef4d44817749d83dc089ce0001/openpilot/cereal/services.py#L99) |
| openpilot | `third_party/openpilot/openpilot/cereal/messaging/__init__.py` | `SubMaster` | 175 | [定位](https://github.com/commaai/openpilot/blob/493ae5ad0eb6f1ef4d44817749d83dc089ce0001/openpilot/cereal/messaging/__init__.py#L175) |
| openpilot | `third_party/openpilot/openpilot/selfdrive/controls/lib/longitudinal_planner.py` | `LongitudinalPlanner.update` | 49 | [定位](https://github.com/commaai/openpilot/blob/493ae5ad0eb6f1ef4d44817749d83dc089ce0001/openpilot/selfdrive/controls/lib/longitudinal_planner.py#L49) |
| openpilot | `third_party/openpilot/openpilot/selfdrive/controls/controlsd.py` | `Controls.state_control` | 73 | [定位](https://github.com/commaai/openpilot/blob/493ae5ad0eb6f1ef4d44817749d83dc089ce0001/openpilot/selfdrive/controls/controlsd.py#L73) |
| openpilot | `third_party/openpilot/openpilot/selfdrive/car/card.py` | `Car.controls_update` | 225 | [定位](https://github.com/commaai/openpilot/blob/493ae5ad0eb6f1ef4d44817749d83dc089ce0001/openpilot/selfdrive/car/card.py#L225) |
| openpilot | `third_party/openpilot/openpilot/selfdrive/selfdrived/selfdrived.py` | `SelfdriveD` | 49 | [定位](https://github.com/commaai/openpilot/blob/493ae5ad0eb6f1ef4d44817749d83dc089ce0001/openpilot/selfdrive/selfdrived/selfdrived.py#L49) |
| openpilot | `third_party/openpilot/lfs.py` | `pull` | 386 | [定位](https://github.com/commaai/openpilot/blob/493ae5ad0eb6f1ef4d44817749d83dc089ce0001/lfs.py#L386) |
| opendbc | `third_party/opendbc/opendbc/car/interfaces.py` | `CarInterfaceBase.apply` | 114 | [定位](https://github.com/commaai/opendbc/blob/d1bd15ecb18e273e4af740dbbfb7def555828668/opendbc/car/interfaces.py#L114) |
| opendbc | `third_party/opendbc/opendbc/car/toyota/carcontroller.py` | `CarController.update` | 78 | [定位](https://github.com/commaai/opendbc/blob/d1bd15ecb18e273e4af740dbbfb7def555828668/opendbc/car/toyota/carcontroller.py#L78) |
| opendbc | `third_party/opendbc/opendbc/car/car.capnp` | `struct CarControl` | 343 | [定位](https://github.com/commaai/opendbc/blob/d1bd15ecb18e273e4af740dbbfb7def555828668/opendbc/car/car.capnp#L343) |
| opendbc | `third_party/opendbc/opendbc/safety/safety.h` | `bool safety_tx_hook(` | 238 | [定位](https://github.com/commaai/opendbc/blob/d1bd15ecb18e273e4af740dbbfb7def555828668/opendbc/safety/safety.h#L238) |
| panda | `third_party/panda/board/drivers/can_common.h` | `void can_send(` | 161 | [定位](https://github.com/commaai/panda/blob/16c88abdecdc0e1dc802a963da979ade4a7d966c/board/drivers/can_common.h#L161) |
| panda | `third_party/panda/board/main.c` | `int main(` | 288 | [定位](https://github.com/commaai/panda/blob/16c88abdecdc0e1dc802a963da979ade4a7d966c/board/main.c#L288) |
| commavq | `third_party/commavq/utils/vqvae.py` | `Encoder` | 177 | [定位](https://github.com/commaai/commavq/blob/623a27f85689de4db3822cc81654cdf4db1b3da0/utils/vqvae.py#L177) |
| commavq | `third_party/commavq/utils/vqvae.py` | `VectorQuantizer` | 136 | [定位](https://github.com/commaai/commavq/blob/623a27f85689de4db3822cc81654cdf4db1b3da0/utils/vqvae.py#L136) |
| commavq | `third_party/commavq/utils/vqvae.py` | `Decoder` | 256 | [定位](https://github.com/commaai/commavq/blob/623a27f85689de4db3822cc81654cdf4db1b3da0/utils/vqvae.py#L256) |
| commavq | `third_party/commavq/utils/gpt.py` | `GPT` | 110 | [定位](https://github.com/commaai/commavq/blob/623a27f85689de4db3822cc81654cdf4db1b3da0/utils/gpt.py#L110) |
| commavq | `third_party/commavq/utils/gpt.py` | `KVCache` | 43 | [定位](https://github.com/commaai/commavq/blob/623a27f85689de4db3822cc81654cdf4db1b3da0/utils/gpt.py#L43) |
| commavq | `third_party/commavq/nanogpt/prepare.py` | `import` | 4 | [定位](https://github.com/commaai/commavq/blob/623a27f85689de4db3822cc81654cdf4db1b3da0/nanogpt/prepare.py#L4) |
| openpilot-distill | `third_party/openpilot.distill/supervised/dataset.py` | `_load_targets` | 207 | [定位](https://github.com/YassineYousfi/openpilot.distill/blob/215f33d4a860f380df4b4f4f075c6c9493f5c205/supervised/dataset.py#L207) |
| openpilot-distill | `third_party/openpilot.distill/supervised/model.py` | `VisionConfig` | 22 | [定位](https://github.com/YassineYousfi/openpilot.distill/blob/215f33d4a860f380df4b4f4f075c6c9493f5c205/supervised/model.py#L22) |
| openpilot-distill | `third_party/openpilot.distill/supervised/model.py` | `TemporalPolicy` | 262 | [定位](https://github.com/YassineYousfi/openpilot.distill/blob/215f33d4a860f380df4b4f4f075c6c9493f5c205/supervised/model.py#L262) |
| openpilot-distill | `third_party/openpilot.distill/supervised/train.py` | `loss_fn` | 27 | [定位](https://github.com/YassineYousfi/openpilot.distill/blob/215f33d4a860f380df4b4f4f075c6c9493f5c205/supervised/train.py#L27) |
| openpilot-distill | `third_party/openpilot.distill/supervised/model_for_inference.py` | `InputQueues` | 19 | [定位](https://github.com/YassineYousfi/openpilot.distill/blob/215f33d4a860f380df4b4f4f075c6c9493f5c205/supervised/model_for_inference.py#L19) |
| openpilot-distill | `third_party/openpilot.distill/supervised/model_for_inference.py` | `ORTSupercomboForInference` | 281 | [定位](https://github.com/YassineYousfi/openpilot.distill/blob/215f33d4a860f380df4b4f4f075c6c9493f5c205/supervised/model_for_inference.py#L281) |
| openpilot-distill | `third_party/openpilot.distill/rl/server.py` | `Runtime.predict` | 78 | [定位](https://github.com/YassineYousfi/openpilot.distill/blob/215f33d4a860f380df4b4f4f075c6c9493f5c205/rl/server.py#L78) |
| openpilot-distill | `third_party/openpilot.distill/rl/env.py` | `Episode` | 71 | [定位](https://github.com/YassineYousfi/openpilot.distill/blob/215f33d4a860f380df4b4f4f075c6c9493f5c205/rl/env.py#L71) |
| openpilot-distill | `third_party/openpilot.distill/rl/env.py` | `Physics.step` | 136 | [定位](https://github.com/YassineYousfi/openpilot.distill/blob/215f33d4a860f380df4b4f4f075c6c9493f5c205/rl/env.py#L136) |
| openpilot-distill | `third_party/openpilot.distill/rl/env.py` | `Env.step` | 184 | [定位](https://github.com/YassineYousfi/openpilot.distill/blob/215f33d4a860f380df4b4f4f075c6c9493f5c205/rl/env.py#L184) |
| openpilot-distill | `third_party/openpilot.distill/rl/actor.py` | `SupercomboActor.act` | 112 | [定位](https://github.com/YassineYousfi/openpilot.distill/blob/215f33d4a860f380df4b4f4f075c6c9493f5c205/rl/actor.py#L112) |
| openpilot-distill | `third_party/openpilot.distill/rl/train.py` | `main` | 21 | [定位](https://github.com/YassineYousfi/openpilot.distill/blob/215f33d4a860f380df4b4f4f075c6c9493f5c205/rl/train.py#L21) |

## 快照标识

- `openpilot`: `493ae5ad0eb6f1ef4d44817749d83dc089ce0001`
- `opendbc`: `d1bd15ecb18e273e4af740dbbfb7def555828668`
- `panda`: `16c88abdecdc0e1dc802a963da979ade4a7d966c`
- `commavq`: `623a27f85689de4db3822cc81654cdf4db1b3da0`
- `openpilot-distill`: `215f33d4a860f380df4b4f4f075c6c9493f5c205`

## 已读取文件的 SHA256

- `third_party/commavq/nanogpt/prepare.py`: `5f751585c82b6c1a26b87db78841685b8519cf8a7262336d5c19e195024fd557`
- `third_party/commavq/utils/gpt.py`: `6c602b6c4b5f45dddcf2a6881093a7eae4274755cb0175ea5dec295ce0247c88`
- `third_party/commavq/utils/vqvae.py`: `1742b9ae5c1d3aa0c47298b2f779d0f3fe6585fba015184cc6393cbd567158ad`
- `third_party/opendbc/opendbc/car/car.capnp`: `ef3454f5add03a98f31b08c2c89473e28ddaf1bfd4d787f1c71cfbe4093abdac`
- `third_party/opendbc/opendbc/car/interfaces.py`: `584136faf3296668d6cb8d42d9218fbe312cd1af5be8a2a77a3f19c6da07079c`
- `third_party/opendbc/opendbc/car/toyota/carcontroller.py`: `99846e1e9a3780a106e3bc6c4c31b0546e9cae119af201bf63b15d36f7411946`
- `third_party/opendbc/opendbc/safety/safety.h`: `64fcd117d1105ecbf4d3fd2275f74162adc929db274e3d2e8c4ad3e4a4b1561e`
- `third_party/openpilot/lfs.py`: `f7160070a20c909134d50edc61d3260cac22e0bcb9dc20228a50fa9a0c1c121f`
- `third_party/openpilot/openpilot/cereal/log.capnp`: `b114cf1a70f64219830632b4ccc04bf09d725f88649973014af2c0e1b40d1808`
- `third_party/openpilot/openpilot/cereal/messaging/__init__.py`: `8ed5b50b36c26dbe4cbb742eff1be8c0d831bbf57ff56396ceae0fdb63a451b0`
- `third_party/openpilot/openpilot/cereal/services.py`: `c9548e1612ab73bf609fee3c20313c678ea7edb0d9df1b2b93950e6d81bd8fd7`
- `third_party/openpilot/openpilot/selfdrive/car/card.py`: `895db4e95712f8af0dedcca22fa18e9f6e91dfcda0061bce3dd7b83fd7eecf93`
- `third_party/openpilot/openpilot/selfdrive/controls/controlsd.py`: `cce4689f8942472cf89ce22dfd5a787859a76168475b00821fa547d889939914`
- `third_party/openpilot/openpilot/selfdrive/controls/lib/longitudinal_planner.py`: `260d9f26bd6d0af25255fba40e81dba68991f038969a43f094e6e818ffc18b5a`
- `third_party/openpilot/openpilot/selfdrive/controls/plannerd.py`: `57a99dcdcb0c3cfefdb1f3ccff33f7792957fabf4284c0ffe9287df88037aeef`
- `third_party/openpilot/openpilot/selfdrive/modeld/SConscript`: `9835f4c053316078ed61f74fa2d900a5f9182d03168cbdca96261539ebbd501d`
- `third_party/openpilot/openpilot/selfdrive/modeld/constants.py`: `1e8af533a981e0234a89cdaf3b6e033f22887393e0fa9e365795fb911d30a55e`
- `third_party/openpilot/openpilot/selfdrive/modeld/fill_model_msg.py`: `2dde5e76457413d9a578cb1905e00a2060c392e38738eee389a0e842c1665717`
- `third_party/openpilot/openpilot/selfdrive/modeld/helpers.py`: `c725b4005da1587e7a4f087cdad26edd29c561ab91b736347fe407d45a19682c`
- `third_party/openpilot/openpilot/selfdrive/modeld/modeld.py`: `3020257697a77550a7635a3128f3fc97946206b3bbb2e19ea2dbe1edc168ae94`
- `third_party/openpilot/openpilot/selfdrive/modeld/parse_model_outputs.py`: `aefd4353cfd396f99abb017da0933b01fa18eea4eb6ca92f09c99751ab1731e7`
- `third_party/openpilot/openpilot/selfdrive/pandad/pandad.cc`: `8a30586560a2c7efd7e33da9b8e5ee8993f4a516ef1a819f153715c3b3972f30`
- `third_party/openpilot/openpilot/selfdrive/selfdrived/selfdrived.py`: `3eece05937c9755ee7c68ca0444723a4b7e6f8a8f76defa4f7f2d75142f379c1`
- `third_party/openpilot/openpilot/system/manager/process_config.py`: `352cc1ef42425f6a35e0482419dd6bf71ece07e83e5d050c1c48ac0e4753b5bc`
- `third_party/openpilot/tools/op.sh`: `aff7f243f671cf220d0a2d8d120cb3a06ff05b71a69d1172a8401bb8f3ca1f5f`
- `third_party/openpilot.distill/rl/actor.py`: `b6963a1e9d48a1bc8235d55f844ddb7edc9894fbfd50204e549384bc1cf03bff`
- `third_party/openpilot.distill/rl/env.py`: `e4c3fc3eabba8eb1301529da7be6a742da8aafda0976f7fdb11fbd3d00e8a021`
- `third_party/openpilot.distill/rl/server.py`: `9e29c49f239983d0d0fb3ce31c6bf5da1f912e0eb5c52ef512634c0fa85eb559`
- `third_party/openpilot.distill/rl/train.py`: `4b068a4a958c0c04baf05f26833047d67a984d8765ce669bc9e0bdf2b72884a5`
- `third_party/openpilot.distill/supervised/dataset.py`: `cac5af7b993550b18c7e8129cacef8815ee75e5ab1dbf98ce03c5dfce6cb3ec5`
- `third_party/openpilot.distill/supervised/model.py`: `4dab3097625c38714a93b3fd73ea43b3eef4db768b81e42e7d59f755c1d11e22`
- `third_party/openpilot.distill/supervised/model_for_inference.py`: `d05fdb1aa641e7101d5d4ee59aeeb816d771474fb3f1a69e52115773d61a164f`
- `third_party/openpilot.distill/supervised/train.py`: `41b64c7f316ca853faa50aa66d5eb27e37ff19558453ff5b2bbeff39e12b1dbf`
- `third_party/panda/board/drivers/can_common.h`: `0e6e5481fd3bb0a15945a239ea457571799c5966cfe57eb625f8e065f71ba1c1`
- `third_party/panda/board/main.c`: `e47ba07362c3d8af59ed9a5fd7715a7bc9778d4b1bed3caa6b4f0647895a4521`

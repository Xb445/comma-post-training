"""Generate a deterministic reading map from pinned source symbols without importing them."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/source-study/00_source_map.md"
ENTRIES = [
    ("openpilot", "openpilot/selfdrive/controls/plannerd.py", "symbol", "main"),
    ("openpilot", "openpilot/selfdrive/pandad/pandad.cc", "text", "void can_send_thread("),
    ("openpilot", "openpilot/selfdrive/modeld/SConscript", "text", "def compile_model("),
    ("openpilot", "openpilot/selfdrive/modeld/constants.py", "symbol", "ModelConstants"),
    ("openpilot", "tools/op.sh", "text", "function op_setup()"),
    ("openpilot", "openpilot/system/manager/process_config.py", "text", "procs = ["),
    ("openpilot", "openpilot/selfdrive/modeld/modeld.py", "symbol", "ModelState"),
    ("openpilot", "openpilot/selfdrive/modeld/modeld.py", "symbol", "get_action_from_model"),
    ("openpilot", "openpilot/selfdrive/modeld/helpers.py", "symbol", "load_oob"),
    ("openpilot", "openpilot/selfdrive/modeld/parse_model_outputs.py", "symbol", "Parser"),
    ("openpilot", "openpilot/selfdrive/modeld/fill_model_msg.py", "symbol", "fill_model_msg"),
    ("openpilot", "openpilot/cereal/log.capnp", "text", "struct ModelDataV2"),
    ("openpilot", "openpilot/cereal/log.capnp", "text", "struct DrivingModelData"),
    ("openpilot", "openpilot/cereal/services.py", "text", "SERVICE_LIST"),
    ("openpilot", "openpilot/cereal/messaging/__init__.py", "symbol", "SubMaster"),
    (
        "openpilot",
        "openpilot/selfdrive/controls/lib/longitudinal_planner.py",
        "symbol",
        "LongitudinalPlanner.update",
    ),
    ("openpilot", "openpilot/selfdrive/controls/controlsd.py", "symbol", "Controls.state_control"),
    ("openpilot", "openpilot/selfdrive/car/card.py", "symbol", "Car.controls_update"),
    ("openpilot", "openpilot/selfdrive/selfdrived/selfdrived.py", "symbol", "SelfdriveD"),
    ("openpilot", "lfs.py", "symbol", "pull"),
    ("opendbc", "opendbc/car/interfaces.py", "symbol", "CarInterfaceBase.apply"),
    ("opendbc", "opendbc/car/toyota/carcontroller.py", "symbol", "CarController.update"),
    ("opendbc", "opendbc/car/car.capnp", "text", "struct CarControl"),
    ("opendbc", "opendbc/safety/safety.h", "text", "bool safety_tx_hook("),
    ("panda", "board/drivers/can_common.h", "text", "void can_send("),
    ("panda", "board/main.c", "text", "int main("),
    ("commavq", "utils/vqvae.py", "symbol", "Encoder"),
    ("commavq", "utils/vqvae.py", "symbol", "VectorQuantizer"),
    ("commavq", "utils/vqvae.py", "symbol", "Decoder"),
    ("commavq", "utils/gpt.py", "symbol", "GPT"),
    ("commavq", "utils/gpt.py", "symbol", "KVCache"),
    ("commavq", "nanogpt/prepare.py", "text", "import"),
    ("openpilot-distill", "supervised/dataset.py", "symbol", "_load_targets"),
    ("openpilot-distill", "supervised/model.py", "symbol", "VisionConfig"),
    ("openpilot-distill", "supervised/model.py", "symbol", "TemporalPolicy"),
    ("openpilot-distill", "supervised/train.py", "symbol", "loss_fn"),
    ("openpilot-distill", "supervised/model_for_inference.py", "symbol", "InputQueues"),
    (
        "openpilot-distill",
        "supervised/model_for_inference.py",
        "symbol",
        "ORTSupercomboForInference",
    ),
    ("openpilot-distill", "rl/server.py", "symbol", "Runtime.predict"),
    ("openpilot-distill", "rl/env.py", "symbol", "Episode"),
    ("openpilot-distill", "rl/env.py", "symbol", "Physics.step"),
    ("openpilot-distill", "rl/env.py", "symbol", "Env.step"),
    ("openpilot-distill", "rl/actor.py", "symbol", "SupercomboActor.act"),
    ("openpilot-distill", "rl/train.py", "symbol", "main"),
]


def symbol_line(text: str, symbol: str) -> int:
    nodes = ast.parse(text).body
    for part in symbol.split("."):
        node = next(
            (
                node
                for node in nodes
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == part
            ),
            None,
        )
        if node is None:
            raise ValueError(f"missing source symbol: {symbol}")
        nodes = node.body
    return node.lineno


def render() -> str:
    lock = json.loads((ROOT / "third_party/sources.lock.json").read_text())
    sources = {item["id"]: item for item in lock["sources"]}
    lines = [
        "# 固定版本源码阅读地图",
        "",
        "由 `python scripts/source_map.py` 从本地源码静态生成；升级版本后须重新核对。",
        "本地路径相对于项目根目录；链接指向上游固定提交及实际符号所在行。",
        "",
        "| 仓库 | 本地文件 | 符号/入口 | 行 | 上游源码 |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for identity, relative, kind, target in ENTRIES:
        source = sources[identity]
        path = ROOT / source["path"] / relative
        text = path.read_text()
        if kind == "symbol":
            line = symbol_line(text, target)
        else:
            line = next(
                (i for i, value in enumerate(text.splitlines(), 1) if target in value), None
            )
            if line is None:
                raise ValueError(f"missing source text: {relative}: {target}")
        url = source["url"].removesuffix(".git")
        link = f"{url}/blob/{source['commit']}/{relative}#L{line}"
        local = f"{source['path']}/{relative}"
        lines.append(f"| {identity} | `{local}` | `{target}` | {line} | [定位]({link}) |")
    lines.extend(["", "## 快照标识", ""])
    for source in sources.values():
        lines.append(f"- `{source['id']}`: `{source['commit']}`")
    lines.extend(["", "## 已读取文件的 SHA256", ""])
    for identity, relative in sorted({(row[0], row[1]) for row in ENTRIES}):
        source = sources[identity]
        path = ROOT / source["path"] / relative
        lines.append(
            f"- `{source['path']}/{relative}`: `{hashlib.sha256(path.read_bytes()).hexdigest()}`"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the saved map is stale")
    args = parser.parse_args()
    try:
        document = render()
        if args.check:
            if not OUTPUT.is_file() or OUTPUT.read_text() != document:
                raise ValueError(
                    "source map is missing or stale; regenerate after reviewing sources"
                )
        else:
            OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            OUTPUT.write_text(document)
    except (OSError, ValueError, SyntaxError) as error:
        parser.exit(1, f"error: {error}\n")
    print(f"Verified {len(ENTRIES)} source anchors" if args.check else f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

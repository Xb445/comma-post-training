"""Command-line entry points for the development workflow."""

import argparse
import json
import platform
from pathlib import Path

from comma_post_training.data import load_chats


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline CPU SFT development tools")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="show installed runtime versions and device availability")
    validate = commands.add_parser("validate-data", help="validate chat-format JSONL")
    validate.add_argument("path", type=Path)
    training = commands.add_parser("train", help="train a randomly initialized tiny model on CPU")
    training.add_argument("--config", type=Path, required=True)
    training.add_argument("--output-dir", type=Path, help="override the configured output path")
    generation = commands.add_parser("generate", help="reload a local checkpoint and generate")
    generation.add_argument("--checkpoint", type=Path, required=True)
    generation.add_argument("--prompt", required=True)
    generation.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    try:
        if args.command == "doctor":
            import torch
            import transformers

            result = {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "cuda_available": torch.cuda.is_available(),
                "default_device": "cpu",
            }
        elif args.command == "validate-data":
            chats = load_chats(args.path)
            result = {"examples": len(chats), "messages": sum(len(chat) for chat in chats)}
        elif args.command == "train":
            from comma_post_training.training import load_config, train

            result = train(load_config(args.config, args.output_dir))
        else:
            from comma_post_training.training import generate

            result = {
                "text": generate(args.checkpoint, args.prompt, args.max_new_tokens),
            }
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")

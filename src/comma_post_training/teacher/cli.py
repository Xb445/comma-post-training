import argparse
import json
from pathlib import Path

from comma_post_training.teacher.assets import fetch_model, verify_model

DEFAULT_MODEL = Path("models/openpilot-084747c/big_driving_supercombo.onnx")


def main():
    parser = argparse.ArgumentParser(
        description="Pinned official comma Teacher and frozen-base adaptation"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("fetch", "verify", "inspect", "smoke", "cache", "predict-adapted"):
        command = commands.add_parser(name)
        command.add_argument("--model", type=Path, default=DEFAULT_MODEL)
        if name not in ("fetch", "verify"):
            command.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
            command.add_argument("--threads", type=int, default=2)
        if name in ("cache", "predict-adapted"):
            command.add_argument("--manifest", type=Path, required=True)
            command.add_argument("--output-dir", type=Path, required=True)
        if name == "smoke":
            command.add_argument("--frames", type=int, default=2)
            command.add_argument("--output", type=Path)
        if name == "predict-adapted":
            command.add_argument("--checkpoint", type=Path, required=True)
            command.add_argument("--split", choices=("train", "val", "test"), default="test")
            command.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    args = parser.parse_args()
    try:
        if args.command == "fetch":
            result = fetch_model(args.model)
        elif args.command == "verify":
            result = verify_model(args.model)
        else:
            from comma_post_training.teacher.runtime import TeacherSession

            if args.command == "smoke" and args.output and args.output.exists():
                raise ValueError("smoke output exists; choose a new report path")
            teacher = TeacherSession(args.model, provider=args.provider, threads=args.threads)
            if args.command == "inspect":
                result = teacher.describe()
            elif args.command == "smoke":
                from comma_post_training.teacher.cache import smoke_teacher

                result = smoke_teacher(teacher, frames=args.frames)
                if args.output:
                    from comma_post_training.engine.checkpoints import atomic_json

                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    atomic_json(args.output, result)
            elif args.command == "cache":
                from comma_post_training.teacher.cache import cache_teacher

                result = cache_teacher(args.manifest, teacher, args.output_dir)
            else:
                from comma_post_training.teacher.predict import predict_adapted

                result = predict_adapted(
                    args.manifest,
                    args.checkpoint,
                    teacher,
                    args.output_dir,
                    split=args.split,
                    device=args.device,
                )
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()

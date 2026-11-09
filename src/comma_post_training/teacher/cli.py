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
    gate = commands.add_parser(
        "gate-residual", help="Gate 2 from held-out validation prediction metrics"
    )
    gate.add_argument("--metrics", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)
    gate.add_argument("--tolerance-mps2", type=float, default=1e-4)
    gate.add_argument("--min-improvement-mps2", type=float, default=1e-3)
    for name in ("fetch", "verify", "inspect", "smoke", "cache", "predict-adapted"):
        command = commands.add_parser(name)
        command.add_argument("--model", type=Path, default=DEFAULT_MODEL)
        if name not in ("fetch", "verify"):
            command.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
            command.add_argument("--threads", type=int, default=2)
            command.add_argument("--profile-prefix", type=Path)
        if name in ("cache", "predict-adapted"):
            command.add_argument("--manifest", type=Path, required=True)
            command.add_argument("--output-dir", type=Path, required=True)
        if name == "smoke":
            command.add_argument("--frames", type=int, default=2)
        if name in ("smoke", "inspect"):
            command.add_argument("--output", type=Path)
        if name == "predict-adapted":
            command.add_argument("--checkpoint", type=Path, required=True)
            command.add_argument("--split", choices=("train", "val", "test"), default="test")
            command.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    args = parser.parse_args()
    try:
        if args.command == "gate-residual":
            from comma_post_training.engine.checkpoints import atomic_json
            from comma_post_training.teacher.evaluation import gate_residual

            if args.output.exists():
                raise ValueError("gate output exists; choose a new report path")
            result = gate_residual(
                json.loads(args.metrics.read_text()),
                tolerance_mps2=args.tolerance_mps2,
                min_improvement_mps2=args.min_improvement_mps2,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(args.output, result)
            print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
            if not result["passed"]:
                parser.exit(1, "Gate 2 failed; online experiments remain blocked\n")
            return
        elif args.command == "fetch":
            result = fetch_model(args.model)
        elif args.command == "verify":
            result = verify_model(args.model)
        else:
            from comma_post_training.teacher.runtime import TeacherSession

            if args.command in ("smoke", "inspect") and args.output and args.output.exists():
                raise ValueError("smoke output exists; choose a new report path")
            teacher = TeacherSession(
                args.model,
                provider=args.provider,
                threads=args.threads,
                profile_prefix=args.profile_prefix,
            )
            if args.command == "inspect":
                result = teacher.describe()
            elif args.command == "smoke":
                from comma_post_training.teacher.cache import smoke_teacher

                result = smoke_teacher(teacher, frames=args.frames)
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
            if args.profile_prefix:
                result["profiling"] = teacher.finish_profiling()
            if args.command in ("smoke", "inspect") and args.output:
                from comma_post_training.engine.checkpoints import atomic_json

                args.output.parent.mkdir(parents=True, exist_ok=True)
                atomic_json(args.output, result)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()

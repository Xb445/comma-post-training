"""Tools for preparing and measuring driving-policy experiments."""

import argparse
import json
from pathlib import Path

from comma_post_training.driving.manifest import validate_manifest
from comma_post_training.driving.metrics import evaluate_file


def main() -> None:
    parser = argparse.ArgumentParser(description="openpilot post-training research tools")
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("validate-manifest", help="check provenance and route splits")
    manifest.add_argument("path", type=Path)
    manifest.add_argument(
        "--metadata-only", action="store_true", help="skip artifact file-presence checks"
    )
    evaluate = commands.add_parser("evaluate", help="measure aligned open-loop predictions")
    evaluate.add_argument("path", type=Path)
    demo = commands.add_parser("make-demo", help="generate an explicitly synthetic episode dataset")
    demo.add_argument("--output-dir", type=Path, required=True)
    demo.add_argument("--frames", type=int, default=24)
    imported = commands.add_parser(
        "import-npz", help="import already aligned numeric episode arrays"
    )
    imported.add_argument("--input", type=Path, required=True)
    imported.add_argument("--metadata", type=Path, required=True)
    imported.add_argument("--output-dir", type=Path, required=True)
    register = commands.add_parser("build-dataset", help="register episodes/*/episode.json")
    register.add_argument("--root", type=Path, required=True)
    register.add_argument("--revision", required=True)
    register.add_argument("--kind", choices=("recorded", "synthetic"), default="recorded")
    dataset = commands.add_parser("validate-dataset", help="validate a native training dataset")
    dataset.add_argument("path", type=Path)
    training = commands.add_parser(
        "train", help="train native models or a residual head over cached frozen Teacher features"
    )
    training.add_argument("--config", type=Path, required=True)
    training.add_argument("--manifest", type=Path)
    training.add_argument("--output-dir", type=Path)
    training.add_argument("--resume", type=Path)
    training.add_argument("--init-checkpoint", type=Path)
    training.add_argument("--max-steps", type=int)
    training.add_argument("--device", choices=("cpu", "cuda", "auto"))
    checkpoint = commands.add_parser(
        "evaluate-checkpoint", help="evaluate a held-out episode split"
    )
    checkpoint.add_argument("--checkpoint", type=Path, required=True)
    checkpoint.add_argument("--manifest", type=Path)
    checkpoint.add_argument("--split", choices=("val", "test"), default="test")
    checkpoint.add_argument("--device", choices=("cpu", "cuda", "auto"))
    cache = commands.add_parser(
        "cache-latents", help="encode a derived dataset with a frozen encoder"
    )
    cache.add_argument("--manifest", type=Path, required=True)
    cache.add_argument("--checkpoint", type=Path, required=True)
    cache.add_argument("--output-dir", type=Path, required=True)
    cache.add_argument("--batch-size", type=int, default=16)
    cache.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    rollout = commands.add_parser("rollout-world", help="autoregressive held-out latent prediction")
    rollout.add_argument("--checkpoint", type=Path, required=True)
    rollout.add_argument("--output", type=Path, required=True)
    rollout.add_argument("--frames", type=int, default=3)
    rollout.add_argument("--sampling-steps", type=int)
    rollout.add_argument("--seed", type=int, default=0)
    rollout.add_argument("--device", choices=("cpu", "cuda", "auto"))
    online = commands.add_parser(
        "train-closed-loop", help="DAgger or residual TD3 with an environment adapter"
    )
    online.add_argument("--config", type=Path, required=True)
    online.add_argument("--policy-checkpoint", type=Path)
    online.add_argument("--output-dir", type=Path)
    online.add_argument("--steps", type=int)
    online_eval = commands.add_parser(
        "evaluate-closed-loop", help="evaluate a policy in held-out simulator scenarios"
    )
    online_eval.add_argument("--config", type=Path, required=True)
    online_eval.add_argument("--checkpoint", type=Path, required=True)
    online_eval.add_argument("--split", choices=("val", "test"), default="test")
    args = parser.parse_args()
    try:
        if args.command == "validate-manifest":
            result = validate_manifest(args.path, check_files=not args.metadata_only)
        elif args.command == "evaluate":
            result = evaluate_file(args.path)
        elif args.command == "make-demo":
            from comma_post_training.data_pipeline.demo import create_demo

            result = {
                "manifest": str(create_demo(args.output_dir, frames=args.frames)),
                "kind": "synthetic",
            }
        elif args.command == "validate-dataset":
            from comma_post_training.data_pipeline.episodes import Catalogue

            result = Catalogue(args.path).validate()
        elif args.command == "import-npz":
            from comma_post_training.data_pipeline.prepare import import_npz

            result = import_npz(args.input, args.metadata, args.output_dir)
        elif args.command == "build-dataset":
            from comma_post_training.data_pipeline.prepare import build_manifest

            result = build_manifest(args.root, revision=args.revision, kind=args.kind)
        elif args.command == "train":
            from dataclasses import replace

            from comma_post_training.engine.config import load_config
            from comma_post_training.engine.trainer import train

            config = load_config(args.config)
            data, settings, model = config.data, config.training, config.model
            if args.manifest:
                data = replace(data, manifest=args.manifest.resolve())
            if args.output_dir:
                settings = replace(settings, output_dir=args.output_dir.resolve())
            if args.max_steps is not None:
                settings = replace(settings, max_steps=args.max_steps)
            if args.device:
                settings = replace(settings, device=args.device)
            if args.init_checkpoint:
                model = replace(model, init_checkpoint=args.init_checkpoint.resolve())
            result = train(
                replace(config, data=data, training=settings, model=model), resume=args.resume
            )
        elif args.command == "evaluate-checkpoint":
            from comma_post_training.engine.trainer import evaluate

            result = evaluate(
                args.checkpoint, manifest=args.manifest, split=args.split, device=args.device
            )
        elif args.command == "cache-latents":
            from comma_post_training.world_model.cache import cache_latents

            result = cache_latents(
                args.manifest,
                args.checkpoint,
                args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
            )
        elif args.command == "rollout-world":
            from comma_post_training.world_model.rollout import predict_rollout

            result = predict_rollout(
                args.checkpoint,
                args.output,
                frames=args.frames,
                seed=args.seed,
                sample_steps=args.sampling_steps,
                device=args.device,
            )
        else:
            from dataclasses import replace

            from comma_post_training.simulator.config import load_closed_loop_config
            from comma_post_training.simulator.runner import evaluate_closed_loop, train_closed_loop

            config = load_closed_loop_config(args.config)
            if args.command == "evaluate-closed-loop":
                result = evaluate_closed_loop(config, args.checkpoint, split=args.split)
            else:
                changes = {}
                if args.policy_checkpoint:
                    changes["policy_checkpoint"] = args.policy_checkpoint.resolve()
                if args.output_dir:
                    changes["output_dir"] = args.output_dir.resolve()
                if args.steps is not None:
                    changes["steps"] = args.steps
                result = train_closed_loop(replace(config, **changes))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")

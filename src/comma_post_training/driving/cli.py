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
    args = parser.parse_args()
    try:
        if args.command == "validate-manifest":
            result = validate_manifest(args.path, check_files=not args.metadata_only)
        else:
            result = evaluate_file(args.path)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")

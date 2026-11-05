"""Validate segment provenance, route splits, and local artifact presence."""

import json
import re
from collections import Counter
from pathlib import Path

REQUIRED_ARTIFACTS = {"fcamera.hevc", "frame_info.safetensors", "localizer.safetensors"}
SPLITS = {"train", "val", "test"}


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def validate_manifest(path: Path, *, check_files: bool = True) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or type(document.get("schema_version")) is not int:
        raise ValueError("manifest must contain an integer schema_version")
    if document["schema_version"] != 1:
        raise ValueError("unsupported manifest schema_version")
    kind = document.get("kind")
    if kind not in ("synthetic", "recorded"):
        raise ValueError("kind must be synthetic or recorded")
    dataset = _text(document.get("dataset"), "dataset")
    revision = _text(document.get("dataset_revision"), "dataset_revision")
    if kind == "recorded" and not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("recorded data requires an immutable 40-character dataset commit")
    segments = document.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("segments must be a nonempty list")
    ids: set[str] = set()
    route_splits: dict[str, str] = {}
    counts: Counter = Counter()
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"segment {index} must be an object")
        identity = _text(segment.get("segment_id"), "segment_id")
        route = _text(segment.get("route_id"), f"{identity}: route_id")
        split = segment.get("split")
        if identity in ids:
            raise ValueError(f"duplicate segment_id: {identity}")
        ids.add(identity)
        if not isinstance(split, str) or split not in SPLITS:
            raise ValueError(f"{identity}: split must be train, val, or test")
        if route in route_splits and route_splits[route] != split:
            raise ValueError(f"route leakage: {route} occurs in multiple splits")
        route_splits[route] = split
        counts[split] += 1
        artifacts = segment.get("artifacts")
        if not isinstance(artifacts, dict) or not REQUIRED_ARTIFACTS.issubset(artifacts):
            raise ValueError(f"{identity}: required artifacts are {sorted(REQUIRED_ARTIFACTS)}")
        for name, location in artifacts.items():
            artifact = path.parent / _text(location, f"{identity}: {name}")
            if check_files and (not artifact.is_file() or artifact.stat().st_size == 0):
                raise ValueError(f"{identity}: missing or empty artifact {artifact}")
    return {
        "kind": kind,
        "dataset": dataset,
        "dataset_revision": revision,
        "segments": len(segments),
        "routes": len(route_splits),
        "split_segments": dict(sorted(counts.items())),
        "asset_validation": "file_presence_only" if check_files else "not_checked",
    }

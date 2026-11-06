import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import torch


def atomic_json(path: Path, value: dict) -> None:
    content = json.dumps(value, default=str, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path: Path, value: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix="checkpoint-", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_checkpoint(path: Path, *, inference: bool = False) -> dict:
    result = torch.load(path, map_location="cpu", weights_only=True)
    formats = {"comma-trainer-v1"}
    if inference:
        formats.add("comma-policy-export-v1")
    if not isinstance(result, dict) or result.get("format") not in formats:
        raise ValueError(
            "use a native comma-trainer-v1 checkpoint; upstream models need an adapter"
        )
    return result


@contextmanager
def staged_directory(destination: Path):
    """Publish a new directory only after its full contents have been validated."""
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError(f"destination must be empty: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".preparing-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "content"
        staging.mkdir()
        yield staging
        if destination.exists():
            destination.rmdir()  # Refuse if another process populated the directory.
        staging.replace(destination)


def source_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for directory in (
        "data_pipeline",
        "engine",
        "driving_policy",
        "world_model",
        "simulator",
        "teacher",
    ):
        for path in sorted((root / directory).rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

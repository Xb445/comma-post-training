"""Canonical NPZ import and dataset registration; raw sensor synchronization is explicit."""

import json
import re
import tempfile
from pathlib import Path

import numpy as np

from comma_post_training.data_pipeline.episodes import Catalogue, Episode, EpisodeRef
from comma_post_training.engine.checkpoints import atomic_json, staged_directory

ARRAYS = {
    "timestamps",
    "images",
    "ego_state",
    "executed_actions",
    "commanded_actions",
    "expert_actions",
    "teacher_actions",
    "poses",
    "valid",
    "expert_valid",
    "teacher_valid",
    "teacher_features",
    "teacher_desire",
    "teacher_traffic",
    "teacher_action_t",
    "latents",
}


def import_npz(source: Path, metadata: Path, output: Path) -> dict:
    document = json.loads(metadata.read_text(encoding="utf-8"))
    identity = document.get("episode", {})
    if not all(identity.get(key) for key in ("id", "route_id", "domain", "split")):
        raise ValueError("metadata.episode requires id, route_id, domain, split")
    with staged_directory(output) as staging:
        archive = np.load(source, allow_pickle=False)
        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise ValueError(
                "import-npz requires an NPZ archive; native NPY uses episode.json directly"
            )
        with archive:
            mapping = document.pop(
                "array_keys", {key: key for key in archive.files if key in ARRAYS}
            )
            if not isinstance(mapping, dict) or set(mapping) - ARRAYS:
                raise ValueError("array_keys maps canonical training names to NPZ keys")
            if not {"timestamps", "ego_state", "executed_actions"}.issubset(mapping):
                raise ValueError("import requires timestamps, ego_state, and executed_actions")
            if any(
                not isinstance(value, str) or value not in archive.files
                for value in mapping.values()
            ):
                raise ValueError("array_keys references a missing or invalid NPZ key")
            document["arrays"] = {}
            for name, original in mapping.items():
                value = archive[original]
                if value.dtype.hasobject:
                    raise ValueError("pickled/object arrays are not supported")
                destination = staging / f"{name}.npy"
                np.save(destination, value, allow_pickle=False)
                document["arrays"][name] = destination.name
        if "image_paths" in document:
            document["image_paths"] = str((metadata.parent / document["image_paths"]).resolve())
        atomic_json(staging / "episode.json", document)
        reference = EpisodeRef(
            **{key: identity[key] for key in ("id", "route_id", "domain", "split")},
            metadata=staging / "episode.json",
        )
        frames = Episode(reference).validate()
    return {"metadata": str((output / "episode.json").resolve()), "frames": frames}


def build_manifest(root: Path, *, revision: str, kind: str = "recorded") -> dict:
    destination = root / "manifest.json"
    if destination.exists():
        raise ValueError("manifest.json exists; write a new dataset version instead of overwriting")
    rows = []
    for metadata in sorted((root / "episodes").glob("*/episode.json")):
        identity = json.loads(metadata.read_text(encoding="utf-8"))["episode"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", identity["id"]):
            raise ValueError("episode ids must be safe directory names")
        rows.append({**identity, "metadata": str(metadata.relative_to(root))})
    document = {"format": "comma-episodes-v1", "kind": kind, "revision": revision, "episodes": rows}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", dir=root, delete=False) as stream:
        json.dump(document, stream)
        pending = Path(stream.name)
    try:
        quality = Catalogue(pending).validate()
        pending.replace(destination)
    finally:
        pending.unlink(missing_ok=True)
    return {"manifest": str(destination), **quality}

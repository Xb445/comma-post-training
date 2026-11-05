"""Create a derived latent dataset without modifying source episode metadata."""

import copy
import json
from pathlib import Path

import numpy as np
import torch

from comma_post_training.data_pipeline.episodes import Catalogue, Episode
from comma_post_training.engine.checkpoints import (
    atomic_json,
    file_sha256,
    load_checkpoint,
    staged_directory,
)
from comma_post_training.engine.tasks import build_model
from comma_post_training.engine.trainer import select_device


@torch.no_grad()
def cache_latents(
    manifest: Path, checkpoint: Path, output: Path, *, device: str = "cpu", batch_size: int = 16
) -> dict:
    if batch_size < 1:
        raise ValueError("cache batch_size must be positive")
    if output.exists() and any(output.iterdir()):
        raise ValueError("latent cache destination must be empty")
    saved = load_checkpoint(checkpoint)
    if saved["task"] != "compressor":
        raise ValueError("latent cache requires a native compressor checkpoint")
    spec = saved["model_spec"]
    target = select_device(device)
    model = build_model(spec).to(target).eval()
    model.load_state_dict(saved["model_state"], strict=True)
    encoder_id = model.encoder_fingerprint()
    catalogue = Catalogue(manifest)
    quality = catalogue.validate()
    for key in ("camera_names",):
        if quality["signature"][key] != spec[key]:
            raise ValueError(f"compressor and dataset disagree on {key}")
    with staged_directory(output) as staging:
        rows = []
        for ref in catalogue.episodes:
            episode = Episode(ref)
            n = len(episode.array("timestamps"))
            folder = staging / "episodes" / ref.id
            folder.mkdir(parents=True)
            temporary = folder / "latents.pending.npy"
            mapped = None
            for start in range(0, n, batch_size):
                indices = list(range(start, min(n, start + batch_size)))
                images = torch.from_numpy(episode.images(indices, tuple(spec["image_size"]))).to(
                    target
                )
                latent = model.encode(images).cpu().float().numpy()
                if not np.isfinite(latent).all():
                    raise RuntimeError(f"nonfinite encoder output in {ref.id}")
                if mapped is None:
                    mapped = np.lib.format.open_memmap(
                        temporary, mode="w+", dtype=np.float32, shape=(n, *latent.shape[1:])
                    )
                mapped[start : start + len(latent)] = latent
            mapped.flush()
            del mapped
            temporary.replace(folder / "latents.npy")
            metadata = copy.deepcopy(episode.metadata)
            metadata["arrays"] = {key: str(value) for key, value in episode.paths.items()}
            metadata["arrays"]["latents"] = "latents.npy"
            if episode.image_paths is not None:
                metadata["image_paths"] = str(episode.image_paths[0])
            metadata.update(
                encoder_fingerprint=encoder_id, encoder_checkpoint_sha256=file_sha256(checkpoint)
            )
            atomic_json(folder / "episode.json", metadata)
            rows.append(
                {
                    "id": ref.id,
                    "route_id": ref.route_id,
                    "domain": ref.domain,
                    "split": ref.split,
                    "metadata": str((folder / "episode.json").relative_to(staging)),
                }
            )
        original = json.loads(manifest.read_text())
        atomic_json(
            staging / "manifest.json",
            {
                "format": "comma-episodes-v1",
                "kind": original["kind"],
                "revision": str(catalogue.revision) + "+encoder-" + encoder_id[:16],
                "episodes": rows,
                "source_manifest_sha256": file_sha256(manifest),
                "encoder_fingerprint": encoder_id,
            },
        )
        Catalogue(staging / "manifest.json").validate()
    return {
        "manifest": str(output / "manifest.json"),
        "encoder_fingerprint": encoder_id,
        "episodes": len(rows),
        "source_metadata_unchanged": True,
    }

"""Compose versioned read-only episode references for source/target replay."""

from pathlib import Path

from comma_post_training.data_pipeline.episodes import Catalogue
from comma_post_training.engine.checkpoints import atomic_json, file_sha256, staged_directory


def combine_datasets(manifests: list[Path], output: Path, *, revision: str):
    if len(manifests) < 2 or not revision.strip():
        raise ValueError("combine-datasets needs >=2 manifests and a nonempty revision")
    catalogues = [Catalogue(path) for path in manifests]
    qualities = [catalogue.validate() for catalogue in catalogues]
    if any(quality["signature"] != qualities[0]["signature"] for quality in qualities):
        raise ValueError(
            "source/target camera, state, preprocessing and Teacher contracts must match"
        )
    episodes = [
        {
            "id": ref.id,
            "route_id": ref.route_id,
            "domain": ref.domain,
            "split": ref.split,
            "metadata": str(ref.metadata),
        }
        for catalogue in catalogues
        for ref in catalogue.episodes
    ]
    with staged_directory(output) as staging:
        atomic_json(
            staging / "manifest.json",
            {
                "format": "comma-episodes-v1",
                "revision": revision,
                "kind": "synthetic"
                if any(c.kind == "synthetic" for c in catalogues)
                else "recorded",
                "episodes": episodes,
                "sources": [
                    {
                        "manifest": str(c.path),
                        "sha256": file_sha256(c.path),
                        "fingerprint": c.fingerprint,
                    }
                    for c in catalogues
                ],
            },
        )
        quality = Catalogue(staging / "manifest.json").validate()
    return {
        "manifest": str(output / "manifest.json"),
        "quality": quality,
        "sampling": "uniform over usable windows unless domain_mix is configured",
        "source_files_modified": False,
    }

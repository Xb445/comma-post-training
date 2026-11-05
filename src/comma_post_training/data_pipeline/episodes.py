"""Memory-mapped NPY episodes and lazy image-path decoding; no pickled arrays."""

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from comma_post_training.data_pipeline.geometry import validate_poses


@dataclass(frozen=True)
class EpisodeRef:
    id: str
    route_id: str
    domain: str
    split: str
    metadata: Path


class Episode:
    def __init__(self, reference: EpisodeRef):
        self.reference = reference
        self.metadata = json.loads(reference.metadata.read_text(encoding="utf-8"))
        meta = self.metadata
        if not isinstance(meta, dict):
            raise ValueError("episode metadata must be an object")
        if meta.get("format") != "comma-episode-v1":
            raise ValueError(f"{reference.id}: unsupported episode format")
        if meta.get("episode") != {
            key: getattr(reference, key) for key in ("id", "route_id", "domain", "split")
        }:
            raise ValueError("episode identity must match its manifest reference")
        if meta.get("coordinate_frame") != "vehicle_frd":
            raise ValueError(f"{reference.id}: convert coordinates explicitly to vehicle_frd")
        if meta.get("timestamp_units") != "s":
            raise ValueError("timestamp_units must explicitly be seconds: s")
        if meta.get("action_order") != ["a_lat", "a_long"] or meta.get("action_units") != "m/s^2":
            raise ValueError(f"{reference.id}: actions must be [a_lat,a_long] in m/s^2")
        for key in ("preprocessing_id", "camera_names", "state_features"):
            if not meta.get(key):
                raise ValueError(f"{reference.id}: missing {key}")
        cameras = meta["camera_names"]
        features = meta["state_features"]
        if not isinstance(cameras, list) or not isinstance(features, list):
            raise ValueError("camera_names and state_features must be lists")
        if not all(isinstance(v, str) and v for v in cameras + features):
            raise ValueError("camera_names and state_features must contain names")
        if len(set(cameras)) != len(cameras) or len(set(features)) != len(features):
            raise ValueError("camera and state feature names must be unique")
        arrays = meta.get("arrays")
        if not isinstance(arrays, dict) or not all(
            isinstance(value, str) and value.endswith(".npy") for value in arrays.values()
        ):
            raise ValueError("arrays must map names to numeric .npy file paths")
        self.paths = {
            key: (reference.metadata.parent / value).resolve() for key, value in arrays.items()
        }
        self._arrays: dict[str, np.ndarray] = {}
        self.image_paths = None
        if "image_paths" in meta:
            index = (reference.metadata.parent / meta["image_paths"]).resolve()
            self.image_paths = (index, json.loads(index.read_text(encoding="utf-8")))

    def array(self, name: str) -> np.ndarray:
        if name not in self.paths:
            raise ValueError(f"{self.reference.id}: required array {name!r} is missing")
        if name not in self._arrays:
            value = np.load(self.paths[name], mmap_mode="r", allow_pickle=False)
            if not isinstance(value, np.ndarray) or value.dtype.kind not in "biuf":
                raise ValueError(f"{name}: use a numeric .npy array, not a pickle/NPZ archive")
            self._arrays[name] = value
        return self._arrays[name]

    def validate(self) -> int:
        timestamps = self.array("timestamps")
        if timestamps.ndim != 1 or len(timestamps) < 2 or not np.isfinite(timestamps).all():
            raise ValueError("timestamps must contain at least two finite seconds")
        if np.any(timestamps[1:] <= timestamps[:-1]):
            raise ValueError("timestamps must be strictly increasing")
        n = len(timestamps)
        for name in self.paths:
            values = self.array(name)
            if values.ndim < 1 or values.shape[0] != n:
                raise ValueError(f"{self.reference.id}: {name} length differs from timestamps")
        state = self.array("ego_state")
        if state.shape != (n, len(self.metadata["state_features"])) or not np.isfinite(state).all():
            raise ValueError("ego_state shape/finite values do not match state_features")
        executed = self.array("executed_actions")
        if executed.shape != (n, 2) or not np.isfinite(executed).all():
            raise ValueError("executed_actions must be finite [N,2]")
        if "commanded_actions" in self.paths:
            commands = self.array("commanded_actions")
            if commands.shape != (n, 2) or not np.isfinite(commands).all():
                raise ValueError("commanded_actions must be finite [N,2]")
        for name in ("expert_actions", "teacher_actions"):
            if name in self.paths:
                labels = self.array(name)
                mask_name = name.removesuffix("_actions") + "_valid"
                mask = self.array(mask_name) if mask_name in self.paths else np.ones(n, dtype=bool)
                if mask.shape != (n,) or mask.dtype != np.bool_:
                    raise ValueError(f"{mask_name} must be a boolean [N] array")
                if labels.shape != (n, 2) or not np.isfinite(labels[mask]).all():
                    raise ValueError(f"{name}: invalid active labels")
            elif name.removesuffix("_actions") + "_valid" in self.paths:
                raise ValueError("label validity masks require their corresponding actions")
        if "valid" in self.paths:
            valid = self.array("valid")
            if valid.shape != (n,) or valid.dtype != np.bool_:
                raise ValueError("valid must be a boolean [N] array")
        if "poses" in self.paths:
            validate_poses(self.array("poses"))
        if "images" in self.paths:
            images = self.array("images")
            if images.ndim != 5 or images.shape[1] != len(self.metadata["camera_names"]):
                raise ValueError("images must be [N,V,H,W,3] in declared camera order")
            if images.shape[-1] != 3 or images.dtype != np.uint8:
                raise ValueError("images must contain RGB uint8 pixels")
            if min(images.shape[2:4]) < 1:
                raise ValueError("image height and width must be positive")
        elif self.image_paths is not None:
            index, paths = self.image_paths
            if (
                not isinstance(paths, list)
                or len(paths) != n
                or any(
                    not isinstance(row, list) or len(row) != len(self.metadata["camera_names"])
                    for row in paths
                )
            ):
                raise ValueError("image_paths must contain N rows of V camera paths")
            for row in paths:
                if any(not isinstance(p, str) or not (index.parent / p).is_file() for p in row):
                    raise ValueError("image_paths contains a missing image")
        else:
            raise ValueError("provide images.npy or a lazy image_paths JSON index")
        if "latents" in self.paths:
            if not self.metadata.get("encoder_fingerprint"):
                raise ValueError("latent arrays require an encoder_fingerprint")
            if self.array("latents").ndim < 2:
                raise ValueError("latents must have shape [N,*latent_shape]")
        return n

    def images(self, indices: list[int], size: tuple[int, int]) -> np.ndarray:
        height, width = size
        frames = []
        for frame in indices:
            views = []
            for view in range(len(self.metadata["camera_names"])):
                if "images" in self.paths:
                    image = Image.fromarray(np.asarray(self.array("images")[frame, view]))
                else:
                    index, paths = self.image_paths
                    with Image.open(index.parent / paths[frame][view]) as source:
                        image = source.convert("RGB")
                image = image.resize((width, height), Image.Resampling.BILINEAR)
                views.append(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)
            frames.append(np.stack(views))
        return np.stack(frames)


class Catalogue:
    def __init__(self, path: Path):
        self.path = path.resolve()
        document = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or document.get("kind") not in ("synthetic", "recorded"):
            raise ValueError("dataset must declare kind=synthetic or recorded")
        if document.get("format") != "comma-episodes-v1" or not document.get("revision"):
            raise ValueError("dataset requires format=comma-episodes-v1 and a declared revision")
        self.revision = document["revision"]
        if not isinstance(self.revision, str) or not self.revision.strip():
            raise ValueError("dataset revision must be a nonempty version string")
        self.kind = document["kind"]
        self.episodes = []
        routes, identities = {}, set()
        if not isinstance(document.get("episodes"), list):
            raise ValueError("episodes must be a list")
        for row in document["episodes"]:
            if not isinstance(row, dict):
                raise ValueError("episode references must be objects")
            if not all(
                isinstance(row.get(key), str) and row[key]
                for key in ("id", "route_id", "domain", "split", "metadata")
            ):
                raise ValueError("each episode requires id, route_id, domain, split, metadata")
            if row["id"] in identities:
                raise ValueError(f"duplicate episode id: {row['id']}")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", row["id"]):
                raise ValueError("episode ids must be safe directory names")
            identities.add(row["id"])
            if row["domain"] not in ("real", "sim", "synthetic") or row["split"] not in (
                "train",
                "val",
                "test",
            ):
                raise ValueError("invalid episode domain or split")
            if row["route_id"] in routes and routes[row["route_id"]] != row["split"]:
                raise ValueError(f"route leakage: {row['route_id']}")
            routes[row["route_id"]] = row["split"]
            self.episodes.append(
                EpisodeRef(
                    **{key: row[key] for key in ("id", "route_id", "domain", "split")},
                    metadata=(self.path.parent / row["metadata"]).resolve(),
                )
            )
        if not self.episodes:
            raise ValueError("dataset has no episodes")
        self.fingerprint = self._fingerprint()

    def _fingerprint(self) -> str:
        digest = hashlib.sha256(self.path.read_bytes())
        for ref in self.episodes:
            episode = Episode(ref)
            digest.update(ref.metadata.read_bytes())
            paths = list(episode.paths.values())
            if episode.image_paths is not None:
                index, rows = episode.image_paths
                digest.update(index.read_bytes())
                paths += [index.parent / p for row in rows for p in row]
            for path in paths:
                stat = path.stat()
                digest.update(f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode())
        return digest.hexdigest()

    def validate(self) -> dict:
        counts, frames, signature = Counter(), 0, None
        for reference in self.episodes:
            episode = Episode(reference)
            frames += episode.validate()
            current = {
                key: episode.metadata[key]
                for key in (
                    "camera_names",
                    "state_features",
                    "preprocessing_id",
                    "coordinate_frame",
                )
            }
            if signature is not None and signature != current:
                raise ValueError("episodes have incompatible camera/state/preprocessing contracts")
            signature = current
            counts[f"{reference.split}/{reference.domain}"] += 1
        return {
            "episodes": len(self.episodes),
            "frames": frames,
            "groups": dict(counts),
            "kind": self.kind,
            "revision": self.revision,
            "fingerprint": self.fingerprint,
            "fingerprint_scope": "manifest_metadata_and_file_stats",
            "signature": signature,
        }

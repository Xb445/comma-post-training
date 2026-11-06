"""Sequential Teacher inference with reset boundaries and a derived cache manifest."""

import copy
from pathlib import Path

import numpy as np

from comma_post_training.data_pipeline.episodes import Catalogue, Episode
from comma_post_training.engine.checkpoints import atomic_json, file_sha256, staged_directory
from comma_post_training.teacher.preprocess import RGBAdapter, pack_yuv420


def iter_teacher_episode(episode, teacher):
    adapter = RGBAdapter(episode, teacher.image_size)
    times = episode.array("timestamps")
    valid = episode.array("valid") if "valid" in episode.paths else np.ones(len(times), dtype=bool)
    teacher.reset()
    previous = None
    for frame, timestamp in enumerate(times):
        if not valid[frame]:
            teacher.reset()
            previous = None
            yield frame, None, adapter
            continue
        if previous is not None:
            delta = float(timestamp - previous)
            if delta < 0.18:
                raise ValueError(
                    "Teacher input is faster than 5 Hz; explicitly resample the episode"
                )
            if delta > 0.22:
                teacher.reset()
        previous = timestamp
        result = teacher.step(adapter.images(frame), **adapter.controls(frame))
        yield frame, result, adapter


def cache_teacher(manifest: Path, teacher, output: Path) -> dict:
    catalogue = Catalogue(manifest)
    quality = catalogue.validate()
    rows, counts = [], {}
    with staged_directory(output) as staging:
        for ref in catalogue.episodes:
            episode = Episode(ref)
            n = len(episode.array("timestamps"))
            folder = staging / "episodes" / ref.id
            folder.mkdir(parents=True)
            actions = np.lib.format.open_memmap(
                folder / "teacher_actions.npy", mode="w+", dtype=np.float32, shape=(n, 2)
            )
            features = np.lib.format.open_memmap(
                folder / "teacher_features.npy",
                mode="w+",
                dtype=np.float32,
                shape=(n, teacher.feature_dim),
            )
            mask = np.zeros(n, dtype=bool)
            actions[:] = 0
            features[:] = 0
            for frame, result, _adapter in iter_teacher_episode(episode, teacher):
                if result is not None:
                    actions[frame], features[frame] = result["action"], result["features"]
                    mask[frame] = result["valid"]
            actions.flush()
            features.flush()
            del actions, features
            if not mask.any():
                raise ValueError(
                    f"{ref.id}: no warmed Teacher frames; use longer continuous 5 Hz episodes"
                )
            np.save(folder / "teacher_valid.npy", mask, allow_pickle=False)
            metadata = copy.deepcopy(episode.metadata)
            metadata["arrays"] = {key: str(value) for key, value in episode.paths.items()}
            metadata["arrays"].update(
                {
                    key: key + ".npy"
                    for key in ("teacher_actions", "teacher_features", "teacher_valid")
                }
            )
            if episode.image_paths is not None:
                metadata["image_paths"] = str(episode.image_paths[0])
            metadata["teacher_identity"] = teacher.cache_identity()
            metadata["teacher_run"] = {
                "profile_sha256": _adapter.fingerprint,
                "source_episode_metadata_sha256": file_sha256(ref.metadata),
                "warmup_frames": teacher.warmup_frames,
                "valid_frames": int(mask.sum()),
                "provider": teacher.session.get_providers(),
                "rgb_preprocessing": "opencv_i420_distill_rgb_adapter_not_camerad_nv12",
            }
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
            counts[ref.id] = int(mask.sum())
        atomic_json(
            staging / "manifest.json",
            {
                "format": "comma-episodes-v1",
                "kind": catalogue.kind,
                "revision": catalogue.revision + "+teacher-" + teacher.identity.sha256[:16],
                "episodes": rows,
                "teacher_identity": teacher.cache_identity(),
                "source_manifest_sha256": file_sha256(manifest),
            },
        )
        Catalogue(staging / "manifest.json").validate()
    return {
        "manifest": str(output / "manifest.json"),
        "teacher_identity": teacher.cache_identity(),
        "dataset_kind": quality["kind"],
        "valid_frames": counts,
        "source_metadata_unchanged": True,
    }


def smoke_teacher(teacher, *, frames=2):
    if type(frames) is not int or frames < 1:
        raise ValueError("smoke frames must be positive")
    height, width = teacher.image_size
    image = pack_yuv420(np.zeros((height, width, 3), dtype=np.uint8))
    controls = {"desire": np.zeros(8), "traffic": [1, 0], "action_t": [0.2, 0.2]}
    results = []
    teacher.reset()
    for _ in range(frames):
        results.append(teacher.step({"img": image, "big_img": image}, **controls))
    teacher.reset()
    repeated = teacher.step({"img": image, "big_img": image}, **controls)
    reproducible = all(
        np.allclose(results[0][key], repeated[key], rtol=1e-5, atol=1e-6)
        for key in ("action", "features")
    )
    if not reproducible:
        raise RuntimeError("Teacher reset did not reproduce its first prediction")
    return {
        "teacher": teacher.describe(),
        "input_kind": "synthetic_black_calibrated_model_space_rgb",
        "frames": frames,
        "reset_reproducible": True,
        "actions": [result["action"].tolist() for result in results],
        "warmed": [result["valid"] for result in results],
        "latency_ms": [result["latency_ms"] for result in results],
        "real_driving_validation": False,
    }

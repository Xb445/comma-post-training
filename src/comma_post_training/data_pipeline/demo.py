"""Generate small, explicitly synthetic episodes for offline pipeline verification."""

from pathlib import Path

import numpy as np

from comma_post_training.data_pipeline.prepare import build_manifest
from comma_post_training.engine.checkpoints import atomic_json


def create_demo(root: Path, *, frames: int = 24) -> Path:
    if frames < 12:
        raise ValueError("demo requires at least twelve frames per episode")
    if root.exists() and any(root.iterdir()):
        raise ValueError("demo destination must be empty")
    root.mkdir(parents=True, exist_ok=True)
    for number in range(6):
        identity = {
            "id": f"demo-{number}",
            "route_id": f"synthetic-route-{number}",
            "domain": "synthetic",
            "split": ("train", "val", "test")[number // 2],
        }
        folder = root / "episodes" / identity["id"]
        folder.mkdir(parents=True)
        times = np.arange(frames, dtype=np.float64) * 0.2
        phase = times * 0.4 + number * 0.07
        speed, yaw = 5 + 0.5 * np.sin(phase), 0.1 * np.sin(phase)
        yaw_rate = 0.04 * np.cos(phase)
        actions = np.stack((speed * yaw_rate, 0.2 * np.cos(phase)), axis=-1).astype(np.float32)
        state = np.stack((speed, yaw_rate), axis=-1).astype(np.float32)
        poses = np.tile(np.eye(4), (frames, 1, 1))
        poses[:, 0, 0], poses[:, 1, 1] = np.cos(yaw), np.cos(yaw)
        poses[:, 1, 0], poses[:, 0, 1] = np.sin(yaw), -np.sin(yaw)
        poses[1:, 0, 3] = np.cumsum(speed[:-1] * np.cos(yaw[:-1]) * 0.2)
        poses[1:, 1, 3] = np.cumsum(speed[:-1] * np.sin(yaw[:-1]) * 0.2)
        images = np.empty((frames, 2, 16, 32, 3), dtype=np.uint8)
        for frame in range(frames):
            for view in range(2):
                images[frame, view, :, :, 0] = int(100 + 80 * np.sin(phase[frame]))
                images[frame, view, :, :, 1] = int(120 + 70 * np.cos(phase[frame]))
                images[frame, view, :, :, 2] = np.arange(32)[None] * 4 + view * 12
        arrays = {
            "timestamps": times,
            "ego_state": state,
            "executed_actions": actions,
            "expert_actions": actions,
            "teacher_actions": actions * 0.95,
            "images": images,
            "poses": poses,
            "valid": np.ones(frames, dtype=bool),
        }
        for name, value in arrays.items():
            np.save(folder / f"{name}.npy", value, allow_pickle=False)
        atomic_json(
            folder / "episode.json",
            {
                "format": "comma-episode-v1",
                "episode": identity,
                "coordinate_frame": "vehicle_frd",
                "timestamp_units": "s",
                "action_order": ["a_lat", "a_long"],
                "action_units": "m/s^2",
                "camera_names": ["front_narrow", "front_wide"],
                "state_features": ["speed_mps", "yaw_rate_radps"],
                "preprocessing_id": "synthetic-rgb-v1",
                "arrays": {name: name + ".npy" for name in arrays},
                "calibration": {"status": "synthetic_fixture_not_a_real_camera"},
                "teacher_provenance": "synthetic labels, not openpilot Teacher inference",
            },
        )
    build_manifest(root, revision="synthetic-native-v1", kind="synthetic")
    return root / "manifest.json"

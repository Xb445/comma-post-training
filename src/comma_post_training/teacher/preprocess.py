"""Explicit calibrated RGB adapter; pixel packing matches the pinned ONNX input."""

import hashlib
import json

import numpy as np
from PIL import Image

from comma_post_training.teacher.runtime import IMAGE_NAMES


def pack_yuv420(rgb: np.ndarray) -> np.ndarray:
    try:
        import cv2
    except ImportError as error:
        raise ValueError(
            "install Teacher image dependencies: uv sync --locked --extra teacher"
        ) from error
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Teacher camera frames must be RGB uint8 [H,W,3]")
    h, w = rgb.shape[:2]
    if h % 2 or w % 2:
        raise ValueError("YUV420 requires even RGB image dimensions")
    yuv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2YUV_I420)
    y = yuv[:h]
    packed_y = np.stack((y[0::2, 0::2], y[1::2, 0::2], y[0::2, 1::2], y[1::2, 1::2]))
    uv = yuv[h:].reshape(2, h // 2, w // 2)
    return np.ascontiguousarray(np.concatenate((packed_y, uv)))


class RGBAdapter:
    def __init__(self, episode, image_size):
        self.episode, self.image_size = episode, tuple(image_size)
        profile = episode.metadata.get("teacher_rgb")
        if not isinstance(profile, dict):
            raise ValueError("episode requires explicit teacher_rgb calibration/control metadata")
        self.profile = profile
        if profile.get("format") != "openpilot-supercombo-rgb-v1":
            raise ValueError("unsupported teacher_rgb format")
        if profile.get("mode") not in ("prewarped_model_rgb", "source_to_model_homography"):
            raise ValueError("teacher_rgb must explicitly specify prewarped RGB or homographies")
        if (
            not isinstance(profile.get("calibration_id"), str)
            or not profile["calibration_id"].strip()
        ):
            raise ValueError("teacher_rgb requires a recorded calibration_id")
        mapping = profile.get("camera_mapping", {})
        names = episode.metadata["camera_names"]
        if set(mapping) != set(IMAGE_NAMES) or any(
            value not in names for value in mapping.values()
        ):
            raise ValueError("teacher_rgb camera_mapping must explicitly map img and big_img")
        self.camera_indices = {key: names.index(value) for key, value in mapping.items()}
        if profile.get("context_hz") != 5:
            raise ValueError(
                "this adapter supports explicit 5 Hz context data only; resample first"
            )
        if profile.get("desire_mode") not in ("none", "array"):
            raise ValueError("declare teacher desire_mode=none or array")
        for key in ("traffic_convention", "action_t_s"):
            value = np.asarray(profile.get(key), dtype=np.float32)
            if value.shape != (2,) or not np.isfinite(value).all():
                raise ValueError(f"teacher_rgb requires explicit finite {key} [2]")
        self.transforms = {}
        if profile["mode"] == "source_to_model_homography":
            for key in IMAGE_NAMES:
                matrix = np.asarray(profile.get("source_to_model", {}).get(key), dtype=np.float64)
                if (
                    matrix.shape != (3, 3)
                    or not np.isfinite(matrix).all()
                    or abs(np.linalg.det(matrix)) < 1e-12
                ):
                    raise ValueError(
                        "source_to_model requires invertible finite camera homographies"
                    )
                self.transforms[key] = matrix
        self.fingerprint = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()

    def _rgb(self, frame, camera):
        if "images" in self.episode.paths:
            return np.asarray(self.episode.array("images")[frame, camera])
        index, paths = self.episode.image_paths
        with Image.open(index.parent / paths[frame][camera]) as image:
            return np.asarray(image.convert("RGB"))

    def images(self, frame):
        result = {}
        height, width = self.image_size
        for key, camera in self.camera_indices.items():
            rgb = self._rgb(frame, camera)
            if key in self.transforms:
                import cv2

                rgb = cv2.warpPerspective(
                    rgb,
                    self.transforms[key],
                    (width, height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
            elif rgb.shape != (height, width, 3):
                raise ValueError(
                    f"prewarped {key} RGB must already be {height}x{width}; "
                    "resizing is not camera calibration"
                )
            result[key] = pack_yuv420(rgb)
        return result

    def controls(self, frame):
        desire = np.zeros(8, dtype=np.float32)
        if self.profile["desire_mode"] == "array":
            desire = self.episode.array("teacher_desire")[frame]
        traffic = (
            self.episode.array("teacher_traffic")[frame]
            if "teacher_traffic" in self.episode.paths
            else self.profile["traffic_convention"]
        )
        delays = (
            self.episode.array("teacher_action_t")[frame]
            if "teacher_action_t" in self.episode.paths
            else self.profile["action_t_s"]
        )
        return {"desire": desire, "traffic": traffic, "action_t": delays}

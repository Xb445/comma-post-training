"""Deterministic planar-road simulator with two independent calibrated pinhole renders.

This is an explicitly synthetic engineering backend, not MetaDrive/CARLA and not a
validated driving benchmark.  It provides physical acceleration commands and an
independent analytic expert so recorder / official-policy integration can be tested.
"""

import copy
import hashlib
import json
from collections import deque

import numpy as np
import torch
from PIL import Image

from comma_post_training.simulator.api import Transition, check_action

CAMERA_NAMES = ["front_narrow", "front_wide"]
IMAGE_SIZE = (256, 512)
INTRINSICS = {
    "front_narrow": [[910.0, 0.0, 256.0], [0.0, 910.0, 47.6], [0.0, 0.0, 1.0]],
    "front_wide": [[455.0, 0.0, 256.0], [0.0, 455.0, 151.8], [0.0, 0.0, 1.0]],
}
CAMERA_HEIGHT_M = 1.22
DT = 0.2


def recording_spec() -> dict:
    return {
        "history": 1,
        "cameras": 2,
        "camera_names": list(CAMERA_NAMES),
        "image_size": list(IMAGE_SIZE),
        "state_dim": 2,
        "state_features": ["speed_mps", "yaw_rate_radps"],
        "preprocessing_id": "procedural-road-model-rgb-v1",
    }


def teacher_rgb_profile() -> dict:
    return {
        "format": "openpilot-supercombo-rgb-v1",
        "mode": "prewarped_model_rgb",
        "calibration_id": "procedural-frd-model-pinhole-h1.22-v1",
        "camera_mapping": {"img": "front_narrow", "big_img": "front_wide"},
        "context_hz": 5,
        "desire_mode": "none",
        "traffic_convention": [1, 0],
        "action_t_s": [DT, DT],
    }


class RoadEnvironment:
    """One 5 Hz step applies a commanded acceleration and renders the reached state."""

    def __init__(
        self,
        *,
        spec: dict,
        split: str,
        horizon: int = 1000,
        actuator_alpha: float = 0.8,
        visual_profile: str = "clear",
        control_delay_steps: int = 0,
        initial_lateral_offset_m: float = 0.0,
        initial_heading_error_rad: float = 0.0,
        initial_speed_offset_mps: float = 0.0,
        perturb_at_step: int | None = None,
        lateral_perturb_m: float = 0.0,
        heading_perturb_rad: float = 0.0,
        speed_perturb_mps: float = 0.0,
    ):
        expected = recording_spec()
        self.native_policy = spec.get("kind") == "policy"
        for key in ("cameras", "state_features"):
            if spec.get(key) != expected[key]:
                raise ValueError(f"RoadEnvironment requires canonical {key}={expected[key]}")
        if type(spec.get("history")) is not int or spec["history"] < 1:
            raise ValueError("history must be a positive integer")
        if not self.native_policy and spec["history"] != 1:
            raise ValueError("official recording/Teacher requires history=1")
        if not self.native_policy and list(spec.get("image_size", [])) != expected["image_size"]:
            raise ValueError("RoadEnvironment renders calibrated 256x512 views, without resizing")
        if len(spec.get("image_size", [])) != 2 or any(
            type(v) is not int or v < 4 for v in spec["image_size"]
        ):
            raise ValueError("image_size must contain two integer dimensions >=4")
        if "camera_names" in spec and list(spec["camera_names"]) != CAMERA_NAMES:
            raise ValueError("RoadEnvironment camera order must be front_narrow,front_wide")
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train/val/test")
        if type(horizon) is not int or horizon < 2:
            raise ValueError("horizon must be an integer >=2")
        if not np.isfinite(actuator_alpha) or not 0 < actuator_alpha <= 1:
            raise ValueError("actuator_alpha must be in (0,1]")
        if visual_profile not in ("clear", "dusk", "fog"):
            raise ValueError("visual_profile must be clear/dusk/fog")
        if type(control_delay_steps) is not int or not 0 <= control_delay_steps <= 50:
            raise ValueError("control_delay_steps must be an integer in [0,50]")
        if perturb_at_step is not None and (
            type(perturb_at_step) is not int or not 1 <= perturb_at_step < horizon
        ):
            raise ValueError("perturb_at_step must occur strictly inside the episode")
        self.conditions = {
            "visual_profile": visual_profile,
            "control_delay_steps": control_delay_steps,
            "initial_lateral_offset_m": initial_lateral_offset_m,
            "initial_heading_error_rad": initial_heading_error_rad,
            "initial_speed_offset_mps": initial_speed_offset_mps,
            "perturb_at_step": perturb_at_step,
            "lateral_perturb_m": lateral_perturb_m,
            "heading_perturb_rad": heading_perturb_rad,
            "speed_perturb_mps": speed_perturb_mps,
        }
        for name, value in self.conditions.items():
            if name not in ("visual_profile", "perturb_at_step", "control_delay_steps"):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (float, int))
                    or not np.isfinite(value)
                ):
                    raise ValueError(f"{name} must be finite")
        if perturb_at_step is None and any(
            (lateral_perturb_m, heading_perturb_rad, speed_perturb_mps)
        ):
            raise ValueError("a nonzero disturbance requires perturb_at_step")
        self.spec, self.split, self.horizon = copy.deepcopy(spec), split, horizon
        self.actuator_alpha = float(actuator_alpha)
        self.scenario_id = None
        self.route_id = None
        self._done = True
        self._pixel_grids = np.meshgrid(np.arange(512), np.arange(256))
        self._history = deque(maxlen=spec["history"])

    def recording_metadata(self):
        optical_from_vehicle = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
        vehicle_from_optical = np.eye(4)
        vehicle_from_optical[:3, :3] = optical_from_vehicle.T
        vehicle_from_optical[:3, 3] = [0, 0, -CAMERA_HEIGHT_M]
        return {
            "teacher_rgb": teacher_rgb_profile(),
            "calibration": {
                "id": teacher_rgb_profile()["calibration_id"],
                "status": "exact_synthetic_pinhole_projection_not_real_camera_calibration",
                "intrinsics": copy.deepcopy(INTRINSICS),
                "T_vehicle_from_optical": vehicle_from_optical.tolist(),
                "image_size_hw": list(IMAGE_SIZE),
                "camera_height_m": CAMERA_HEIGHT_M,
                "render_method": "independent_ground_plane_projection_per_intrinsic",
            },
            "label_provenance": {
                "expert": "analytic_road_tracking_controller_v1_on_current_simulator_state",
                "commanded": "expert command in [a_lat,a_long] m/s^2 before actuator lag",
                "executed": "achieved interval acceleration after first_order_actuator_v1",
            },
        }

    def describe(self):
        return {
            "name": "procedural-road-pinhole-v1",
            "kind": "synthetic",
            "split": self.split,
            "scenario_pool": f"procedural-road-pinhole-v1:{self.split}",
            "scenario_id": self.scenario_id,
            "route_id": self.route_id,
            "scenario_parameters": getattr(self, "scenario_parameters", None),
            "context_hz": 5,
            "expert_source": "analytic_road_tracking_controller_v1",
            "reward_version": "road_tracking_speed_jerk_v1",
            "actuator_version": "first_order_actuator_v1",
            "actuator_alpha": self.actuator_alpha,
            "fidelity": "synthetic_kinematic_engineering_backend_not_a_driving_benchmark",
            "teacher_rgb": teacher_rgb_profile(),
            "preprocessing_id": recording_spec()["preprocessing_id"],
            "conditions": copy.deepcopy(self.conditions),
            "native_input_resize_hw": list(self.spec["image_size"]) if self.native_policy else None,
            "disturbance_semantics": "exogenous_state_change_before_reached_state_reward",
        }

    def reset(self, *, seed: int):
        if type(seed) is not int or seed < 0:
            raise ValueError("scenario seed must be a nonnegative integer")
        group = {"train": 0, "val": 1, "test": 2}[self.split]
        rng = np.random.default_rng(np.random.SeedSequence([seed, group, 42017]))
        self.scenario_parameters = {
            "amplitude_m": float(rng.uniform(0.8, 2.0)),
            "wave_number_per_m": float(rng.uniform(0.012, 0.022)),
            "phase_rad": float(rng.uniform(-np.pi, np.pi)),
            "speed_phase_rad": float(rng.uniform(-np.pi, np.pi)),
        }
        if self.conditions != {
            "visual_profile": "clear",
            "control_delay_steps": 0,
            "initial_lateral_offset_m": 0.0,
            "initial_heading_error_rad": 0.0,
            "initial_speed_offset_mps": 0.0,
            "perturb_at_step": None,
            "lateral_perturb_m": 0.0,
            "heading_perturb_rad": 0.0,
            "speed_perturb_mps": 0.0,
        }:
            self.scenario_parameters["conditions"] = copy.deepcopy(self.conditions)
        digest = hashlib.sha256(
            json.dumps(self.scenario_parameters, sort_keys=True).encode()
        ).hexdigest()[:12]
        self.scenario_id = f"road-{self.split}-{seed}-{digest}"
        self.route_id = self.scenario_id
        self.x = 0.0
        self.y = float(self.road_center(0)) + float(rng.uniform(-0.12, 0.12))
        self.yaw = float(np.arctan(self.road_slope(0)))
        self.speed = float(rng.uniform(5.0, 6.5))
        self.yaw_rate = 0.0
        self.y += float(self.conditions["initial_lateral_offset_m"]) * np.sqrt(
            1 + self.road_slope(0) ** 2
        )
        self.yaw += float(self.conditions["initial_heading_error_rad"])
        self.speed = max(0.0, self.speed + float(self.conditions["initial_speed_offset_mps"]))
        self.previous_action = np.zeros(2, dtype=np.float32)
        self._delay = deque(
            np.zeros(2, dtype=np.float32) for _ in range(self.conditions["control_delay_steps"])
        )
        self._history.clear()
        self.perturbation_applied = False
        self.steps = 0
        self._done = False
        return self._observation(dt=0.0)

    def road_center(self, x):
        p = self.scenario_parameters
        return p["amplitude_m"] * np.sin(p["wave_number_per_m"] * x + p["phase_rad"])

    def road_slope(self, x):
        p = self.scenario_parameters
        return (
            p["amplitude_m"]
            * p["wave_number_per_m"]
            * np.cos(p["wave_number_per_m"] * x + p["phase_rad"])
        )

    def _reference(self):
        p = self.scenario_parameters
        center = float(self.road_center(self.x))
        slope = float(self.road_slope(self.x))
        second = (
            -p["amplitude_m"]
            * p["wave_number_per_m"] ** 2
            * np.sin(p["wave_number_per_m"] * self.x + p["phase_rad"])
        )
        curvature = float(second / (1 + slope**2) ** 1.5)
        target_speed = 6.0 + 0.6 * np.sin(0.009 * self.x + p["speed_phase_rad"])
        return center, slope, curvature, float(target_speed)

    def expert_action(self):
        if self._done:
            raise ValueError("reset the environment before querying its expert")
        center, slope, curvature, target_speed = self._reference()
        cross_track = (self.y - center) / np.sqrt(1 + slope**2)
        heading_error = np.arctan2(
            np.sin(self.yaw - np.arctan(slope)), np.cos(self.yaw - np.arctan(slope))
        )
        lateral = self.speed**2 * curvature - 0.7 * cross_track - 2 * self.speed * heading_error
        longitudinal = 0.8 * (target_speed - self.speed)
        return np.clip([lateral, longitudinal], [-2.5, -2.5], [2.5, 2.5]).astype(np.float32)

    def current_pose(self):
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        pose = np.eye(4)
        pose[:2, :2] = [[c, -s], [s, c]]
        pose[:2, 3] = [self.x, self.y]
        return pose

    def diagnostics(self):
        center, slope, _, target_speed = self._reference()
        return {
            "cross_track_m": float((self.y - center) / np.sqrt(1 + slope**2)),
            "target_speed_mps": target_speed,
            "speed_error_mps": self.speed - target_speed,
            "road_half_width_m": 5.4,
            "heading_error_rad": float(
                np.arctan2(np.sin(self.yaw - np.arctan(slope)), np.cos(self.yaw - np.arctan(slope)))
            ),
            "perturbation_applied": self.perturbation_applied,
            "perturbation_step": self.conditions["perturb_at_step"],
        }

    def _render_camera(self, name):
        # Inverse ray/ground intersection; each camera uses its own calibrated K.
        u, v = self._pixel_grids
        matrix = np.asarray(INTRINSICS[name])
        focal, cx, cy = matrix[0, 0], matrix[0, 2], matrix[1, 2]
        down = v - cy
        ground = down > 0.5
        depth = np.where(ground, focal * CAMERA_HEIGHT_M / np.maximum(down, 0.5), 0)
        right = (u - cx) * depth / focal
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        world_x = self.x + c * depth - s * right
        world_y = self.y + s * depth + c * right
        error = world_y - self.road_center(world_x)
        image = np.empty((256, 512, 3), dtype=np.uint8)
        image[:] = [115, 170, 219]
        # All ground textures are fixed in world coordinates, so motion changes pixels.
        grass_pattern = ((np.floor(world_x / 4) + np.floor(world_y / 3)) % 2) * 10
        for channel, level in enumerate((62, 114, 57)):
            image[:, :, channel][ground] = (level + grass_pattern[ground]).astype(np.uint8)
        asphalt = ground & (np.abs(error) < 5.4)
        pattern = ((np.floor(world_x / 5) % 2) * 3).astype(np.uint8)
        for channel, level in enumerate((62, 65, 69)):
            image[:, :, channel][asphalt] = level + pattern[asphalt]
        lines = ground & (
            (np.abs(np.abs(error) - 1.8) < 0.075) | (np.abs(np.abs(error) - 5.25) < 0.1)
        )
        dashed = (np.floor(world_x / 3) % 2 == 0) | (np.abs(error) > 4)
        image[lines & dashed] = [235, 232, 208]
        # Roadside posts share the same world geometry in the two camera projections.
        posts = ground & (np.abs(np.abs(error) - 6.1) < 0.15) & (world_x % 12 < 0.3)
        image[posts] = [210, 65, 44]
        if self.conditions["visual_profile"] == "dusk":
            image = np.rint(image.astype(float) * np.array([0.58, 0.53, 0.48])).astype(np.uint8)
        elif self.conditions["visual_profile"] == "fog":
            blend = np.where(ground, 1 - np.exp(-depth / 35), 0.55)[..., None]
            image = np.rint(image * (1 - blend) + np.array([185, 190, 195]) * blend).astype(
                np.uint8
            )
        return image

    def _observation(self, *, dt):
        images = np.stack([self._render_camera(name) for name in CAMERA_NAMES])
        if self.native_policy and tuple(self.spec["image_size"]) != IMAGE_SIZE:
            height, width = self.spec["image_size"]
            images = np.stack(
                [
                    np.asarray(
                        Image.fromarray(rgb).resize((width, height), Image.Resampling.BILINEAR)
                    )
                    for rgb in images
                ]
            )
        frame = {
            "images": torch.from_numpy(images.transpose(0, 3, 1, 2).copy()).float() / 255,
            "state": torch.tensor([self.speed, self.yaw_rate], dtype=torch.float32),
            "past_actions": torch.from_numpy(self.previous_action.copy()),
            "dt": torch.tensor([dt], dtype=torch.float32),
        }
        if not self._history:
            for _ in range(self.spec["history"] - 1):
                self._history.append({key: torch.zeros_like(value) for key, value in frame.items()})
        self._history.append(frame)
        result = {key: torch.stack([item[key] for item in self._history]) for key in frame}
        result["dt"][0] = 0  # The first dt within every causal window is zero in EpisodeWindows.
        if self.spec["history"] == 1:
            result["dt"][0] = dt
        return result

    def step(self, action):
        if self._done:
            raise ValueError("reset the environment after termination/truncation")
        command = check_action(action)
        if (np.abs(command) > 3.0 + 1e-6).any():
            raise ValueError("RoadEnvironment commands must lie within +/-3 m/s^2")
        if self.conditions["control_delay_steps"]:
            self._delay.append(command.copy())
            command = self._delay.popleft()
        achieved = (
            self.actuator_alpha * command + (1 - self.actuator_alpha) * self.previous_action
        ).astype(np.float32)
        next_speed = max(0.0, self.speed + float(achieved[1]) * DT)
        average_speed = (self.speed + next_speed) / 2
        self.yaw_rate = float(achieved[0]) / max(average_speed, 1.0)
        turn = self.yaw_rate * DT
        self.x += average_speed * np.cos(self.yaw + turn / 2) * DT
        self.y += average_speed * np.sin(self.yaw + turn / 2) * DT
        self.yaw += turn
        self.steps += 1
        if self.steps == self.conditions["perturb_at_step"]:
            self.y += float(self.conditions["lateral_perturb_m"]) * np.sqrt(
                1 + self.road_slope(self.x) ** 2
            )
            self.yaw += float(self.conditions["heading_perturb_rad"])
            next_speed = max(0.0, next_speed + float(self.conditions["speed_perturb_mps"]))
            self.perturbation_applied = True
        center, slope, _, target_speed = self._reference()
        cross_track = (self.y - center) / np.sqrt(1 + slope**2)
        jerk = (achieved - self.previous_action) / DT
        terminated = bool(abs(cross_track) > 5.4 or next_speed > 18)
        reward = -float(
            (cross_track**2 + 0.1 * (next_speed - target_speed) ** 2 + 0.01 * (jerk @ jerk)) * DT
        )
        if terminated:
            reward -= 50
        self.speed, self.previous_action = next_speed, achieved
        truncated = self.steps >= self.horizon
        self._done = terminated or truncated
        return Transition(self._observation(dt=DT), reward, terminated, truncated, achieved.copy())

    def close(self):
        self._done = True

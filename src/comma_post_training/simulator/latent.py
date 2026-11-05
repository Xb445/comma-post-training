"""Experimental planar dynamics + native learned video; uses recorded future anchors."""

from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from comma_post_training.data_pipeline.episodes import Catalogue, Episode
from comma_post_training.data_pipeline.geometry import motion_vector
from comma_post_training.engine.checkpoints import file_sha256, load_checkpoint
from comma_post_training.engine.tasks import build_model, build_spec
from comma_post_training.engine.trainer import (
    config_from_dict,
    make_windows,
    move_batch,
    select_device,
)
from comma_post_training.simulator.api import Transition, check_action


class LatentDrivingEnvironment:
    """Research adapter, with explicit privileged reference frames and proxy rewards.

    Action determines a planar SE(3) increment, which conditions each generated frame.
    This does not reproduce full vehicle/contact dynamics or validate world-model fidelity.
    """

    def __init__(
        self,
        *,
        spec,
        split,
        world_checkpoint,
        compressor_checkpoint,
        device="cpu",
        horizon=20,
        sampling_steps=4,
    ):
        if (
            type(horizon) is not int
            or horizon < 1
            or type(sampling_steps) is not int
            or sampling_steps < 1
        ):
            raise ValueError("latent horizon/sampling_steps must be positive integers")
        if spec["state_features"] != ["speed_mps", "yaw_rate_radps"]:
            raise ValueError("planar latent adapter supports only speed_mps,yaw_rate_radps")
        self.spec, self.split, self.horizon, self.sampling_steps = (
            spec,
            split,
            horizon,
            sampling_steps,
        )
        self.device = select_device(device)
        world = load_checkpoint(Path(world_checkpoint))
        compressor = load_checkpoint(Path(compressor_checkpoint))
        if world["task"] != "world_model" or compressor["task"] != "compressor":
            raise ValueError(
                "latent environment requires native world_model and compressor checkpoints"
            )
        self.world = build_model(world["model_spec"]).to(self.device).eval().requires_grad_(False)
        self.compressor = (
            build_model(compressor["model_spec"]).to(self.device).eval().requires_grad_(False)
        )
        self.world.load_state_dict(world["model_state"], strict=True)
        self.compressor.load_state_dict(compressor["model_state"], strict=True)
        world_spec, compressor_spec = world["model_spec"], compressor["model_spec"]
        for key in ("camera_names", "image_size", "preprocessing_id"):
            if spec[key] != world_spec[key] or spec[key] != compressor_spec[key]:
                raise ValueError(f"latent environment policy/world/compressor disagree on {key}")
        if (
            spec["history"] != world_spec["history"]
            or world_spec["encoder_fingerprint"] != self.compressor.encoder_fingerprint()
        ):
            raise ValueError(
                "latent environment requires matching history and cached latent encoder"
            )
        cfg = config_from_dict(world["config"])
        if cfg.data.stride != 1:
            raise ValueError("latent environment requires world-model stride=1")
        self.catalogue = Catalogue(cfg.data.manifest)
        quality = self.catalogue.validate()
        self.dataset = make_windows(cfg, self.catalogue, split)
        if build_spec(cfg, quality["signature"], self.dataset) != world_spec:
            raise ValueError("latent dataset contract changed")
        self.asset_ids = {
            "world_sha256": file_sha256(Path(world_checkpoint)),
            "compressor_sha256": file_sha256(Path(compressor_checkpoint)),
            "dataset_fingerprint": self.catalogue.fingerprint,
        }
        self.frames = deque(maxlen=spec["history"])

    def describe(self):
        return {
            "name": "native-latent-planar-v1",
            "kind": "simulator",
            "split": self.split,
            "dataset_kind": self.catalogue.kind,
            "fidelity": "experimental_not_validated",
            "uses_recorded_future_anchors": bool(self.world.spec["anchor_offsets"]),
            "expert_source": "learned_plan_head_proxy_controller",
            "reward_source": "reference_pose_speed_and_jerk_proxy",
            **self.asset_ids,
        }

    def _observation(self):
        return {key: torch.stack([frame[key] for frame in self.frames]) for key in self.frames[0]}

    def reset(self, *, seed):
        rng = np.random.default_rng(seed)
        part = int(rng.integers(len(self.dataset.spans)))
        span = self.dataset.spans[part]
        offset = int(rng.integers(max(1, span.count - self.horizon + 1)))
        first = (self.dataset.ends[part - 1] if part else 0) + offset
        self.samples = [
            self.dataset[first + i] for i in range(min(self.horizon, span.count - offset))
        ]
        ref = self.catalogue.episodes[span.episode]
        self.episode = Episode(ref)
        sample = self.samples[0]
        t = sample["frame"]
        indices = list(range(t - self.spec["history"] + 1, t + 1))
        images = self.episode.images(indices, tuple(self.spec["image_size"]))
        times = self.episode.array("timestamps")
        self.frames.clear()
        for number, frame in enumerate(indices):
            previous = (
                np.array(self.episode.array("executed_actions")[frame - 1], dtype=np.float32)
                if frame > span.run_start
                else np.zeros(2, dtype=np.float32)
            )
            self.frames.append(
                {
                    "images": torch.from_numpy(images[number]),
                    "state": torch.tensor(
                        np.array(self.episode.array("ego_state")[frame]), dtype=torch.float32
                    ),
                    "past_actions": torch.from_numpy(previous),
                    "dt": torch.tensor(
                        [times[frame] - times[frame - 1] if number else 0], dtype=torch.float32
                    ),
                }
            )
        self.history = sample["history_latents"].to(self.device)
        self.pose = np.array(self.episode.array("poses")[t], dtype=np.float64)
        self.speed = float(self.episode.array("ego_state")[t, 0])
        self.previous = self.frames[-1]["past_actions"].numpy().copy()
        self.count, self.seed = 0, seed
        return self._observation()

    def _conditions(self):
        sample = self.samples[self.count]
        batch = move_batch(default_collate([sample]), self.device)
        batch["history_latents"] = self.history[None]
        # Recorded next motion is never used by the interactive environment.
        batch["motion"] = torch.zeros((1, 6), device=self.device)
        del batch["target_latent"], batch["plan"]
        return batch

    @torch.no_grad()
    def expert_action(self):
        batch = self._conditions()
        noisy = torch.zeros((1, *self.world.spec["latent_shape"]), device=self.device)
        plan = (
            self.world(noisy, torch.zeros(1, device=self.device), batch)["plan"][0, -1]
            .cpu()
            .numpy()
        )
        seconds = float(batch["plan_times"][0, -1])
        # This is a proxy expert on the visited latent history, not an openpilot Teacher.
        return np.clip(
            [2 * plan[1] / seconds**2, (np.linalg.norm(plan) / seconds - self.speed) / seconds],
            -3,
            3,
        ).astype(np.float32)

    @torch.no_grad()
    def step(self, action):
        if self.count >= len(self.samples):
            raise ValueError("reset the latent environment after termination/truncation")
        command = check_action(action)
        achieved = (0.8 * command + 0.2 * self.previous).astype(np.float32)
        batch = self._conditions()
        dt = float(batch["frame_times"][0, -1])
        next_speed = max(0.0, self.speed + float(achieved[1]) * dt)
        average_speed = (self.speed + next_speed) / 2
        yaw = float(achieved[0]) / max(average_speed, 1.0) * dt
        distance = average_speed * dt
        delta = np.eye(4)
        delta[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
        delta[:2, 3] = [distance * np.cos(yaw / 2), distance * np.sin(yaw / 2)]
        next_pose = self.pose @ delta
        batch["motion"] = torch.from_numpy(motion_vector(self.pose, next_pose)[None]).to(
            self.device
        )
        generated = self.world.sample(batch, steps=self.sampling_steps, seed=self.seed + self.count)
        if not torch.isfinite(generated["latent"]).all():
            raise RuntimeError("nonfinite interactive world-model prediction")
        image = self.compressor.decode(generated["latent"])[0].cpu()
        self.history = torch.cat((self.history[1:], generated["latent"]), dim=0)
        frame = self.samples[self.count]["frame"] + 1
        reference_pose = self.episode.array("poses")[frame]
        position_error = float(np.linalg.norm(next_pose[:2, 3] - reference_pose[:2, 3]))
        speed_error = next_speed - float(self.episode.array("ego_state")[frame, 0])
        jerk = (achieved - self.previous) / dt
        reward = -(position_error**2 + 0.1 * speed_error**2 + 0.01 * float(jerk @ jerk)) * dt
        self.frames.append(
            {
                "images": image,
                "state": torch.tensor([next_speed, yaw / dt], dtype=torch.float32),
                "past_actions": torch.from_numpy(achieved.copy()),
                "dt": torch.tensor([dt]),
            }
        )
        self.pose, self.speed, self.previous = next_pose, next_speed, achieved
        self.count += 1
        terminated = position_error > 10 or next_speed > 40
        return Transition(
            self._observation(), reward, terminated, self.count >= len(self.samples), achieved
        )

    def close(self):
        self.dataset.cache.clear()

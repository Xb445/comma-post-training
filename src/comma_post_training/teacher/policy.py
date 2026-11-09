"""Online frozen Teacher context, separated from trainable residual evaluation.

The caller identifies each environment observation. Repeated act/update calls reuse
that context and never advance the ONNX history a second time.
"""

import time
from dataclasses import dataclass

import numpy as np
import torch

from comma_post_training.simulator.api import check_observation
from comma_post_training.teacher.preprocess import RGBAdapter
from comma_post_training.teacher.residual import OfficialResidualPolicy


class _RGBObservation:
    def __init__(self, images, spec, profile):
        self.metadata = {"camera_names": spec["camera_names"], "teacher_rgb": profile}
        self.paths = {"images": "in_memory"}
        self.frames = images

    def array(self, name):
        if name != "images":
            raise ValueError("online Teacher controls must be explicit fixed profile values")
        return self.frames


@dataclass
class TeacherContext:
    batch: dict[str, torch.Tensor]
    valid: bool
    observation_id: int
    teacher_frame: int
    teacher_latency_ms: float
    end_to_end_ms: float

    @property
    def features(self):
        return torch.cat(
            (
                self.batch["teacher_features"],
                self.batch["state"][:, -1],
                self.batch["past_actions"][:, -1],
            ),
            dim=-1,
        )

    @property
    def official_action(self):
        return self.batch["teacher"]


class OfficialTeacherPolicy:
    def __init__(self, teacher, spec, *, teacher_rgb, device="cpu", residual_model=None):
        if spec.get("kind") != "official_residual" or spec.get("history") != 1:
            raise ValueError("online official policy requires an official_residual history=1 spec")
        if spec.get("teacher_identity") != teacher.cache_identity():
            raise ValueError("policy requires the exact recorded frozen Teacher identity")
        if tuple(spec["image_size"]) != tuple(teacher.image_size):
            raise ValueError("online camera image size differs from Teacher model space")
        if teacher_rgb.get("desire_mode") != "none":
            raise ValueError("online v1 requires explicit fixed no-intent controls")
        self.teacher, self.spec, self.profile = teacher, spec, teacher_rgb
        self.device = torch.device(device)
        self.model = residual_model or OfficialResidualPolicy(spec)
        self.model.to(self.device)
        self.context = None
        self.last_id = -1
        self.reset_count = 0
        self.teacher_calls = 0

    @property
    def feature_dim(self):
        return self.spec["teacher_feature_dim"] + self.spec["state_dim"] + 2

    def reset(self):
        self.teacher.reset()
        self.context, self.last_id = None, -1
        self.reset_count += 1

    @torch.no_grad()
    def observe(self, observation, *, observation_id: int):
        if type(observation_id) is not int or observation_id < 0:
            raise ValueError("observation_id must be a nonnegative integer within an episode")
        if observation_id == self.last_id:
            return self.context
        if observation_id != self.last_id + 1:
            raise ValueError("observations must arrive once in sequence; reset at every episode")
        start = time.perf_counter()
        obs = check_observation(observation, self.spec)
        if observation_id and not np.isclose(float(obs["dt"][-1, 0]), 0.2, atol=0.02):
            raise ValueError("online Teacher requires explicit 5 Hz observations")
        rgb = (
            (obs["images"][-1].permute(0, 2, 3, 1).numpy().clip(0, 1) * 255)
            .round()
            .astype(np.uint8)[None]
        )
        source = _RGBObservation(rgb, self.spec, self.profile)
        adapter = RGBAdapter(source, self.teacher.image_size)
        result = self.teacher.step(adapter.images(0), **adapter.controls(0))
        self.teacher_calls += 1
        batch = {
            "teacher_features": torch.as_tensor(result["features"], device=self.device)[None],
            "teacher": torch.as_tensor(result["action"], device=self.device)[None],
            "state": obs["state"].to(self.device)[None],
            "past_actions": obs["past_actions"].to(self.device)[None],
            "dt": obs["dt"].to(self.device)[None],
        }
        self.context = TeacherContext(
            batch,
            bool(result["valid"]),
            observation_id,
            result["frame"],
            float(result["latency_ms"]),
            (time.perf_counter() - start) * 1000,
        )
        self.last_id = observation_id
        return self.context

    def act(self, context=None, *, official_only=False):
        context = context or self.context
        if context is None:
            raise ValueError("observe the environment before requesting an action")
        return context.official_action if official_only else self.model(context.batch)

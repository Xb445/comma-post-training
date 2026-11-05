"""Small action-conditioned synthetic test environment; no road simulation claims."""

from collections import deque

import numpy as np
import torch

from comma_post_training.simulator.api import Transition, check_action


class ReferenceEnvironment:
    def __init__(self, *, spec: dict, split: str, horizon: int = 40):
        if spec["state_features"] != ["speed_mps", "yaw_rate_radps"]:
            raise ValueError(
                "reference fixture only supports speed_mps,yaw_rate_radps state features"
            )
        if type(horizon) is not int or horizon < 2:
            raise ValueError("reference horizon must be an integer >=2")
        self.spec, self.split, self.horizon = spec, split, horizon
        self.dt = 0.2
        self.frames = deque(maxlen=spec["history"])

    def describe(self):
        return {"name": "reference-point-mass-v1", "kind": "synthetic", "split": self.split}

    def _frame(self, past, dt):
        v, _, h, w = self.spec["cameras"], 3, *self.spec["image_size"]
        images = np.zeros((v, 3, h, w), dtype=np.float32)
        images[:, 0] = np.clip(0.5 + self.y / 8, 0, 1)
        images[:, 1] = np.clip(self.speed / 12, 0, 1)
        images[:, 2] = np.linspace(0, 1, w)[None, None]
        return {
            "images": torch.from_numpy(images),
            "state": torch.tensor(
                [self.speed, self.lateral_speed / max(self.speed, 1)], dtype=torch.float32
            ),
            "past_actions": torch.from_numpy(past.copy()),
            "dt": torch.tensor([dt], dtype=torch.float32),
        }

    def _observation(self):
        return {key: torch.stack([frame[key] for frame in self.frames]) for key in self.frames[0]}

    def reset(self, *, seed):
        rng = np.random.default_rng(seed)
        self.y, self.lateral_speed = float(rng.uniform(-1.5, 1.5)), 0.0
        self.speed, self.count = float(rng.uniform(3.5, 6.5)), 0
        self.previous = np.zeros(2, dtype=np.float32)
        self.frames.clear()
        for _ in range(self.spec["history"]):
            self.frames.append(self._frame(self.previous, 0.0))
        return self._observation()

    def expert_action(self):
        return np.array([-self.y - 1.5 * self.lateral_speed, 5 - self.speed], dtype=np.float32)

    def step(self, action):
        command = check_action(action)
        achieved = (0.7 * command + 0.3 * self.previous).astype(np.float32)
        jerk = (achieved - self.previous) / self.dt
        self.lateral_speed += float(achieved[0]) * self.dt
        self.y += self.lateral_speed * self.dt
        self.speed = max(0.0, self.speed + float(achieved[1]) * self.dt)
        self.count += 1
        reward = -(self.y**2 + (self.speed - 5) ** 2 + 0.01 * float(jerk @ jerk)) * self.dt
        self.previous = achieved
        self.frames.append(self._frame(achieved, self.dt))
        terminated = abs(self.y) > 5 or self.speed > 15
        return Transition(
            self._observation(), reward, terminated, self.count >= self.horizon, achieved
        )

    def close(self):
        pass

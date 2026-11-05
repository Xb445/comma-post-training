"""The environment owns synchronization, dynamics, rewards, and expert queries."""

import importlib
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch


@dataclass
class Transition:
    observation: dict[str, torch.Tensor]
    reward: float
    terminated: bool
    truncated: bool
    achieved_action: np.ndarray


class Environment(Protocol):
    def describe(self) -> dict: ...
    def reset(self, *, seed: int) -> dict[str, torch.Tensor]: ...
    def step(self, action: np.ndarray) -> Transition: ...
    def expert_action(self) -> np.ndarray: ...
    def close(self) -> None: ...


def check_action(action) -> np.ndarray:
    result = np.asarray(action, dtype=np.float32)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError("actions must be finite [a_lat,a_long] in m/s^2")
    return result


def check_observation(observation: dict, spec: dict) -> dict:
    shapes = {
        "images": (spec["history"], spec["cameras"], 3, *spec["image_size"]),
        "state": (spec["history"], spec["state_dim"]),
        "past_actions": (spec["history"], 2),
        "dt": (spec["history"], 1),
    }
    result = {}
    for key, shape in shapes.items():
        value = torch.as_tensor(observation[key], dtype=torch.float32).detach().cpu()
        if tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError(f"environment observation {key} must be finite {shape}")
        result[key] = value.clone()
    if (result["images"] < 0).any() or (result["images"] > 1).any():
        raise ValueError("environment images must be RGB floats in [0,1]")
    if (result["dt"] < 0).any():
        raise ValueError("environment dt must be nonnegative seconds")
    return result


def check_transition(value: Transition, spec: dict) -> Transition:
    if not isinstance(value, Transition):
        raise ValueError("environment step must return simulator.api.Transition")
    if (
        not np.isfinite(value.reward)
        or type(value.terminated) is not bool
        or type(value.truncated) is not bool
    ):
        raise ValueError("environment reward must be finite and done flags must be bool")
    return Transition(
        check_observation(value.observation, spec),
        float(value.reward),
        value.terminated,
        value.truncated,
        check_action(value.achieved_action),
    )


def load_environment(factory: str, spec: dict, split: str, kwargs: dict) -> Environment:
    module, separator, name = factory.partition(":")
    if not separator or not module or not name:
        raise ValueError("environment factory must be module:callable")
    creator = getattr(importlib.import_module(module), name)
    env = creator(spec=spec, split=split, **kwargs)
    description = env.describe()
    if (
        description.get("kind") not in ("synthetic", "simulator")
        or description.get("split") != split
    ):
        env.close()
        raise ValueError(
            "environment must describe kind=synthetic|simulator and its requested split"
        )
    return env

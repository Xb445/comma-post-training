import math
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass(frozen=True)
class ClosedLoopConfig:
    algorithm: str
    policy_checkpoint: Path
    output_dir: Path
    env_factory: str = "comma_post_training.simulator.reference:ReferenceEnvironment"
    env_kwargs: dict = field(default_factory=dict)
    steps: int = 100
    seed: int = 42
    device: str = "cpu"
    num_threads: int = 2
    batch_size: int = 8
    warmup_steps: int = 8
    capacity: int = 10000
    learning_rate: float = 0.001
    episode_limit: int = 200
    eval_episodes: int = 2
    eval_seed: int = 1000000
    offline_fraction: float = 0.5
    expert_mix: float = 0.2
    action_low: tuple[float, float] = (-3.0, -3.0)
    action_high: tuple[float, float] = (3.0, 3.0)
    residual_scale: tuple[float, float] = (0.5, 0.5)
    exploration_noise: float = 0.1
    target_noise: float = 0.2
    noise_clip: float = 0.5
    gamma: float = 0.99
    tau: float = 0.005
    policy_delay: int = 2
    residual_penalty: float = 0.001
    actor_warmup_updates: int = 0
    actor_bc_weight: float = 0.0
    actor_smoothness_weight: float = 0.0

    def __post_init__(self):
        if self.algorithm not in ("dagger", "td3"):
            raise ValueError("algorithm must be dagger or td3")
        if type(self.actor_warmup_updates) is not int or self.actor_warmup_updates < 0:
            raise ValueError("actor_warmup_updates must be a nonnegative integer")
        if self.device not in ("cpu", "cuda", "auto"):
            raise ValueError("device must be cpu, cuda, or auto")
        for name in (
            "steps",
            "num_threads",
            "batch_size",
            "warmup_steps",
            "capacity",
            "episode_limit",
            "eval_episodes",
            "policy_delay",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("seed", "eval_seed"):
            if (
                type(getattr(self, name)) is not int
                or not 0 <= getattr(self, name) < 2**32 - self.steps - self.eval_episodes
            ):
                raise ValueError(
                    f"{name} must be a nonnegative 32-bit seed with room for episode offsets"
                )
        if self.capacity < self.batch_size or self.steps < max(
            self.warmup_steps, self.batch_size
        ) + (self.policy_delay if self.algorithm == "td3" else 0):
            raise ValueError("steps/capacity must allow actual minibatch and delayed actor updates")
        for name in (
            "learning_rate",
            "exploration_noise",
            "target_noise",
            "noise_clip",
            "gamma",
            "tau",
            "offline_fraction",
            "expert_mix",
            "residual_penalty",
            "actor_bc_weight",
            "actor_smoothness_weight",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.learning_rate <= 0 or not 0 < self.tau <= 1 or self.gamma > 1:
            raise ValueError("learning_rate/tau must be positive and gamma/tau at most one")
        if not 0 <= self.offline_fraction < 1 or not 0 <= self.expert_mix <= 1:
            raise ValueError("offline_fraction must be in [0,1) and expert_mix in [0,1]")
        for name in ("action_low", "action_high", "residual_scale"):
            value = getattr(self, name)
            if len(value) != 2 or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                for v in value
            ):
                raise ValueError(f"{name} must contain two finite acceleration values")
        if any(
            low >= high for low, high in zip(self.action_low, self.action_high, strict=True)
        ) or any(v <= 0 for v in self.residual_scale):
            raise ValueError("action bounds must be ordered and residual scales positive")
        if not isinstance(self.env_kwargs, dict) or not isinstance(self.env_factory, str):
            raise ValueError("env_factory must be a string and env_kwargs a table")


def load_closed_loop_config(path: Path) -> ClosedLoopConfig:
    with path.open("rb") as stream:
        document = tomllib.load(stream)
    unknown = set(document) - {f.name for f in fields(ClosedLoopConfig)}
    if unknown:
        raise ValueError(f"unknown closed-loop options: {sorted(unknown)}")
    for key in ("policy_checkpoint", "output_dir"):
        if key not in document or not isinstance(document[key], str) or not document[key].strip():
            raise ValueError(f"{key} must be a path")
        document[key] = (path.parent / document[key]).resolve()
    for key in ("action_low", "action_high", "residual_scale"):
        if key in document:
            document[key] = tuple(document[key])
    if (
        document.get("env_factory")
        == "comma_post_training.simulator.latent:LatentDrivingEnvironment"
    ):
        for key in ("world_checkpoint", "compressor_checkpoint"):
            value = document.get("env_kwargs", {}).get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"latent env_kwargs requires {key}")
            document["env_kwargs"][key] = str((path.parent / value).resolve())
    return ClosedLoopConfig(**document)

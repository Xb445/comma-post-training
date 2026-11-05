"""Bounded DAgger aggregation and TD3 with separate train/validation environments."""

import json
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from comma_post_training.data_pipeline.episodes import Catalogue
from comma_post_training.data_pipeline.windows import EpisodeWindows
from comma_post_training.engine.checkpoints import (
    atomic_json,
    file_sha256,
    load_checkpoint,
    save_checkpoint,
    source_fingerprint,
)
from comma_post_training.engine.tasks import build_model, build_spec
from comma_post_training.engine.trainer import config_from_dict, move_batch, select_device
from comma_post_training.simulator.api import (
    check_action,
    check_observation,
    check_transition,
    load_environment,
)
from comma_post_training.simulator.config import ClosedLoopConfig
from comma_post_training.simulator.td3 import TD3


class Replay:
    def __init__(self, capacity: int):
        self.rows = deque(maxlen=capacity)

    def add(self, **values):
        self.rows.append(
            {key: torch.as_tensor(value).detach().cpu().clone() for key, value in values.items()}
        )

    def sample(self, size: int, rng) -> dict:
        return default_collate([self.rows[int(i)] for i in rng.integers(len(self.rows), size=size)])

    def state_dict(self):
        return default_collate(list(self.rows)) if self.rows else {}


def _batch(observation, device):
    return move_batch({key: value[None] for key, value in observation.items()}, device)


@torch.no_grad()
def _features(model, observation, device, low, high):
    features = model.features(_batch(observation, device))
    base = model.action_head(features).clamp(low, high)
    return features, base


@torch.no_grad()
def _evaluate(model, agent, cfg, spec, *, split="val"):
    env = load_environment(cfg.env_factory, spec, split, cfg.env_kwargs)
    results = []
    model.eval()
    device = next(model.parameters()).device
    low, high = (
        torch.tensor(cfg.action_low, device=device),
        torch.tensor(cfg.action_high, device=device),
    )
    try:
        for episode in range(cfg.eval_episodes):
            # Test seeds occupy another range; the adapter also owns disjoint scenario pools.
            seed = cfg.eval_seed + episode + (100000 if split == "test" else 0)
            observation = check_observation(env.reset(seed=seed), spec)
            total = 0.0
            for step in range(1, cfg.episode_limit + 1):
                features, base = _features(model, observation, device, low, high)
                action = agent.action(features, base) if agent else base
                transition = check_transition(env.step(action[0].cpu().numpy()), spec)
                total += transition.reward
                observation = transition.observation
                if transition.terminated or transition.truncated or step == cfg.episode_limit:
                    break
            results.append(
                {
                    "seed": seed,
                    "return": total,
                    "steps": step,
                    "terminated": transition.terminated,
                    "truncated": transition.truncated or step == cfg.episode_limit,
                }
            )
        return {
            "environment": env.describe(),
            "split": split,
            "mean_return": float(np.mean([row["return"] for row in results])),
            "episodes": results,
        }
    finally:
        env.close()


def _offline_windows(saved):
    cfg = config_from_dict(saved["config"])
    catalogue = Catalogue(cfg.data.manifest)
    quality = catalogue.validate()
    dataset = EpisodeWindows(
        catalogue,
        split="train",
        task="policy",
        history=cfg.data.history,
        image_size=cfg.data.image_size,
        stride=cfg.data.stride,
        max_gap_s=cfg.data.max_gap_s,
        supervision=("expert",),
    )
    if build_spec(cfg, quality["signature"], dataset) != saved["model_spec"]:
        raise ValueError("offline replay dataset contract differs from the base policy")
    return dataset


def train_closed_loop(cfg: ClosedLoopConfig) -> dict:
    if cfg.output_dir.exists() and any(cfg.output_dir.iterdir()):
        raise ValueError("closed-loop output directory must be empty")
    device = select_device(cfg.device)
    torch.set_num_threads(cfg.num_threads)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    saved = load_checkpoint(cfg.policy_checkpoint, inference=True)
    if saved["task"] != "policy":
        raise ValueError("closed-loop training requires a native policy checkpoint")
    spec = saved["model_spec"]
    model = build_model(spec).to(device)
    model.load_state_dict(saved["model_state"], strict=True)
    model.encoder.requires_grad_(False)
    if cfg.algorithm == "td3":
        model.requires_grad_(False)
    agent = TD3(spec["hidden_size"], cfg, device) if cfg.algorithm == "td3" else None
    optimizer = (
        torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.learning_rate)
        if agent is None
        else None
    )
    offline = _offline_windows(saved) if agent is None and cfg.offline_fraction else None
    replay = Replay(cfg.capacity)
    low, high = (
        torch.tensor(cfg.action_low, device=device),
        torch.tensor(cfg.action_high, device=device),
    )
    baseline = _evaluate(model, agent, cfg, spec)
    env = load_environment(cfg.env_factory, spec, "train", cfg.env_kwargs)
    try:
        observation = check_observation(env.reset(seed=cfg.seed), spec)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        run = {
            "algorithm": cfg.algorithm,
            "config": asdict(cfg),
            "environment": env.describe(),
            "base_checkpoint_sha256": file_sha256(cfg.policy_checkpoint),
            "model_spec": spec,
            "native_source_sha256": source_fingerprint(),
            "online_resume_supported": False,
            "action_semantics": (
                "critic uses final commanded acceleration; achieved acceleration stored separately"
            ),
        }
        atomic_json(cfg.output_dir / "run.json", run)
        episode_number, episode_steps, updates = 0, 0, 0
        for step in range(1, cfg.steps + 1):
            model.eval()
            with torch.no_grad():
                batch = _batch(observation, device)
                features, base = _features(model, observation, device, low, high)
                if agent is None:
                    expert = check_action(env.expert_action())
                    replay.add(
                        vision=model.visual_features(batch)[0],
                        state=batch["state"][0],
                        past_actions=batch["past_actions"][0],
                        dt=batch["dt"][0],
                        label=expert,
                    )
                    action = (
                        torch.as_tensor(expert, device=device)[None]
                        if rng.random() < cfg.expert_mix
                        else base
                    )
                    action = action.clamp(low, high)
                else:
                    noise = torch.from_numpy(
                        rng.normal(0, cfg.exploration_noise, (1, 2)).astype(np.float32)
                    ).to(device)
                    action = agent.clamp(agent.action(features, base) + noise)
                transition = check_transition(env.step(action[0].cpu().numpy()), spec)
                episode_steps += 1
                truncated = transition.truncated or episode_steps >= cfg.episode_limit
                if agent is not None:
                    next_features, next_base = _features(
                        model, transition.observation, device, low, high
                    )
                    replay.add(
                        features=features[0],
                        base=base[0],
                        action=action[0],
                        achieved_action=transition.achieved_action,
                        reward=transition.reward,
                        next_features=next_features[0],
                        next_base=next_base[0],
                        terminated=transition.terminated,
                        truncated=truncated,
                    )
            event = {
                "step": step,
                "episode": episode_number,
                "reward": transition.reward,
                "commanded_action": action[0].cpu().tolist(),
                "achieved_action": transition.achieved_action.tolist(),
                "terminated": transition.terminated,
                "truncated": truncated,
            }
            if step >= cfg.warmup_steps and len(replay.rows) >= cfg.batch_size:
                if agent is not None:
                    event.update(agent.update(replay.sample(cfg.batch_size, rng)))
                else:
                    online = move_batch(replay.sample(cfg.batch_size, rng), device)
                    if offline is not None:
                        count = int(cfg.batch_size * cfg.offline_fraction)
                        if count:
                            raw = move_batch(
                                default_collate(
                                    [
                                        offline[int(i)]
                                        for i in rng.integers(len(offline), size=count)
                                    ]
                                ),
                                device,
                            )
                            with torch.no_grad():
                                encoded = {
                                    "vision": model.visual_features(raw),
                                    "state": raw["state"],
                                    "past_actions": raw["past_actions"],
                                    "dt": raw["dt"],
                                    "label": raw["expert"],
                                }
                            online = {
                                key: torch.cat((value[: cfg.batch_size - count], encoded[key]), 0)
                                for key, value in online.items()
                            }
                    model.train()
                    prediction = model.action_head(model.features_from_encoded(online))
                    loss = (prediction - online["label"]).square().mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
                    optimizer.step()
                    event["supervised_loss"] = float(loss.detach())
                updates += 1
            with (cfg.output_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
            observation = transition.observation
            if transition.terminated or truncated:
                episode_number += 1
                episode_steps = 0
                observation = check_observation(env.reset(seed=cfg.seed + episode_number), spec)
        final = _evaluate(model, agent, cfg, spec)
        export = {
            "format": "comma-residual-export-v1" if agent else "comma-policy-export-v1",
            "task": "policy",
            "model_spec": spec,
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "config": saved["config"],
            "closed_loop_config": json.loads(json.dumps(asdict(cfg), default=str)),
            "base_checkpoint_sha256": run["base_checkpoint_sha256"],
        }
        if agent is not None:
            export["residual_state"] = {
                key: value.detach().cpu() for key, value in agent.actor.state_dict().items()
            }
        save_checkpoint(cfg.output_dir / "final.pt", export)
        save_checkpoint(
            cfg.output_dir / "learning_state.pt",
            {
                "format": "comma-closed-loop-learning-v1",
                "online_resume_supported": False,
                "learner": agent.state_dict() if agent else optimizer.state_dict(),
                "replay": replay.state_dict(),
                "rng_state": torch.get_rng_state(),
            },
        )
        report = {
            "algorithm": cfg.algorithm,
            "environment": env.describe(),
            "steps": cfg.steps,
            "updates": updates,
            "replay_size": len(replay.rows),
            "base_frozen": bool(agent),
            "encoder_frozen": True,
            "initial_validation": baseline,
            "final_validation": final,
            "checkpoint": str(cfg.output_dir / "final.pt"),
            "evaluation_note": (
                "synthetic environments verify algorithms only; "
                "driving conclusions require a validated simulator"
            ),
        }
        atomic_json(cfg.output_dir / "metrics.json", report)
        return report
    finally:
        env.close()


def evaluate_closed_loop(cfg: ClosedLoopConfig, checkpoint: Path, *, split: str = "test") -> dict:
    device = select_device(cfg.device)
    torch.set_num_threads(cfg.num_threads)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        saved.get("format")
        not in ("comma-trainer-v1", "comma-policy-export-v1", "comma-residual-export-v1")
        or saved.get("task") != "policy"
    ):
        raise ValueError("closed-loop evaluation requires a native policy/residual checkpoint")
    spec = saved["model_spec"]
    model = build_model(spec).to(device).eval()
    model.load_state_dict(saved["model_state"], strict=True)
    agent = None
    if saved["format"] == "comma-residual-export-v1":
        for key in ("action_low", "action_high", "residual_scale"):
            if list(getattr(cfg, key)) != list(saved["closed_loop_config"][key]):
                raise ValueError(f"residual evaluation must preserve trained {key}")
        agent = TD3(spec["hidden_size"], cfg, device)
        agent.actor.load_state_dict(saved["residual_state"], strict=True)
        agent.actor.eval()
    return _evaluate(model, agent, cfg, spec, split=split)

"""Live official-base DAgger / TD3, with explicit Gate 2 and scenario provenance."""

import json
import time
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from comma_post_training.data_pipeline.episodes import Catalogue
from comma_post_training.data_pipeline.windows import EpisodeWindows
from comma_post_training.engine.checkpoints import (
    atomic_json,
    file_sha256,
    save_checkpoint,
    source_fingerprint,
)
from comma_post_training.engine.trainer import move_batch, select_device
from comma_post_training.simulator.api import (
    check_action,
    check_observation,
    check_transition,
    load_environment,
)
from comma_post_training.simulator.config import ClosedLoopConfig
from comma_post_training.simulator.metrics import EpisodeMetrics
from comma_post_training.simulator.runner import Replay
from comma_post_training.simulator.td3 import TD3
from comma_post_training.simulator.trace import evaluation_trace, pose_of
from comma_post_training.teacher.policy import OfficialTeacherPolicy
from comma_post_training.teacher.residual import OfficialResidualPolicy
from comma_post_training.teacher.runtime import TeacherSession


@dataclass(frozen=True)
class OfficialConfig:
    algorithm: str
    teacher_model: Path
    output_dir: Path
    init_checkpoint: Path | None = None
    offline_manifest: Path | None = None
    gate2_report: Path | None = None
    engineering_only: bool = False
    teacher_provider: str = "cpu"
    teacher_threads: int = 2
    teacher_rgb: dict = field(default_factory=dict)
    env_factory: str = "comma_post_training.simulator.recording:RoadEnvironment"
    env_kwargs: dict = field(default_factory=dict)
    steps: int = 500
    seed: int = 42
    device: str = "cpu"
    num_threads: int = 2
    batch_size: int = 32
    warmup_steps: int = 32
    capacity: int = 10000
    learning_rate: float = 0.001
    episode_limit: int = 300
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
    hidden_size: int = 64
    state_features: tuple[str, ...] = ("speed_mps", "yaw_rate_radps")
    camera_names: tuple[str, ...] = ("front_narrow", "front_wide")
    preprocessing_id: str = "procedural-road-model-rgb-v1"

    def __post_init__(self):
        shared = {
            f.name: getattr(self, f.name) for f in fields(ClosedLoopConfig) if hasattr(self, f.name)
        }
        shared["policy_checkpoint"] = self.init_checkpoint or self.teacher_model
        ClosedLoopConfig(**shared)
        if self.teacher_provider not in ("cpu", "cuda"):
            raise ValueError("teacher_provider must be cpu/cuda")
        if type(self.teacher_threads) is not int or self.teacher_threads < 1:
            raise ValueError("teacher_threads must be positive")
        if type(self.hidden_size) is not int or self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if type(self.engineering_only) is not bool:
            raise ValueError("engineering_only must be boolean")
        if not isinstance(self.teacher_rgb, dict):
            raise ValueError("teacher_rgb must be a profile table")
        if not self.state_features or len(set(self.state_features)) != len(self.state_features):
            raise ValueError("state_features must contain unique names")
        if len(self.camera_names) != 2 or len(set(self.camera_names)) != 2:
            raise ValueError("official online policy requires two explicitly named camera views")


def load_official_config(path: Path) -> OfficialConfig:
    with path.open("rb") as stream:
        document = tomllib.load(stream)
    unknown = set(document) - {f.name for f in fields(OfficialConfig)}
    if unknown:
        raise ValueError(f"unknown official-loop options: {sorted(unknown)}")
    for name in (
        "teacher_model",
        "output_dir",
        "init_checkpoint",
        "offline_manifest",
        "gate2_report",
    ):
        if name in document:
            if not isinstance(document[name], str) or not document[name].strip():
                raise ValueError(f"{name} must be a nonempty path")
            document[name] = (path.parent / document[name]).resolve()
    for name in ("action_low", "action_high", "residual_scale", "state_features", "camera_names"):
        if name in document:
            document[name] = tuple(document[name])
    try:
        return OfficialConfig(**document)
    except TypeError as error:
        raise ValueError(f"invalid official-loop config: {error}") from error


def _gate(cfg):
    if cfg.engineering_only:
        return {
            "passed": False,
            "bypassed": True,
            "reason": "engineering smoke; no improvement claim",
        }
    if cfg.init_checkpoint is None or cfg.gate2_report is None:
        raise ValueError(
            "official closed-loop training requires init_checkpoint and a passing Gate 2 report"
        )
    report = json.loads(cfg.gate2_report.read_text())
    if (
        report.get("format") != "comma-gate2-v1"
        or report.get("passed") is not True
        or report.get("split") != "val"
    ):
        raise ValueError(
            "Gate 2 must pass on held-out val routes before official closed-loop training"
        )
    if report.get("checkpoint_sha256") != file_sha256(cfg.init_checkpoint):
        raise ValueError("Gate 2 checkpoint differs from init_checkpoint")
    return {**report, "bypassed": False}


def _initial(cfg, teacher, device):
    saved = None
    if cfg.init_checkpoint is not None:
        saved = torch.load(cfg.init_checkpoint, map_location="cpu", weights_only=True)
        if saved.get("task") != "official_residual" or saved.get("format") not in (
            "comma-trainer-v1",
            "comma-official-online-v1",
        ):
            raise ValueError("init_checkpoint must contain an official residual head")
        if saved.get("td3_actor_state") is not None:
            raise ValueError(
                "a TD3 export contains another actor; "
                "it cannot be silently discarded during initialization"
            )
        spec = saved["model_spec"]
    else:
        spec = {
            "kind": "official_residual",
            "history": 1,
            "cameras": 2,
            "camera_names": list(cfg.camera_names),
            "state_features": list(cfg.state_features),
            "state_dim": len(cfg.state_features),
            "image_size": list(teacher.image_size),
            "preprocessing_id": cfg.preprocessing_id,
            "hidden_size": cfg.hidden_size,
            "vision_channels": 16,
            "latent_channels": 4,
            "teacher_identity": teacher.cache_identity(),
            "teacher_feature_dim": teacher.feature_dim,
            "residual_scale": list(cfg.residual_scale),
        }
    if spec.get("teacher_identity") != teacher.cache_identity():
        raise ValueError("initial residual head and live Teacher identity differ")
    if tuple(spec["image_size"]) != tuple(teacher.image_size) or spec["history"] != 1:
        raise ValueError("initial residual head image/history contract differs from live Teacher")
    model = OfficialResidualPolicy(spec).to(device)
    if saved is not None:
        model.load_state_dict(saved["model_state"], strict=True)
    return spec, model, saved


def _profile(env, cfg):
    metadata = env.recording_metadata() if hasattr(env, "recording_metadata") else env.describe()
    profile = cfg.teacher_rgb or metadata.get("teacher_rgb")
    if not isinstance(profile, dict):
        raise ValueError("official environments must expose calibrated teacher_rgb metadata")
    return profile


def _provenance(env):
    info = env.describe()
    for name in ("scenario_id", "scenario_pool", "expert_source", "reward_version"):
        if not isinstance(info.get(name), str) or not info[name]:
            raise ValueError(f"official environment must record {name}")
    return info


def _new_teacher(cfg, factory):
    return (
        factory()
        if factory
        else TeacherSession(
            cfg.teacher_model, provider=cfg.teacher_provider, threads=cfg.teacher_threads
        )
    )


def _clip(action, cfg):
    return action.clamp(
        torch.tensor(cfg.action_low, device=action.device),
        torch.tensor(cfg.action_high, device=action.device),
    )


@torch.no_grad()
def _evaluate(
    cfg,
    teacher,
    spec,
    model,
    *,
    agent=None,
    split="val",
    official_only=False,
    trace_dir=None,
    metric_options=None,
    allow_warmup_failure=False,
):
    device = next(model.parameters()).device
    env = load_environment(cfg.env_factory, spec, split, cfg.env_kwargs)
    policy = OfficialTeacherPolicy(
        teacher, spec, teacher_rgb=_profile(env, cfg), device=device, residual_model=model
    )
    results, profiles = [], []
    model.eval()
    try:
        with evaluation_trace(
            trace_dir, spec, split, "official" if official_only else "td3" if agent else "residual"
        ) as trace:
            for episode in range(cfg.eval_episodes):
                seed = cfg.eval_seed + episode + (100000 if split == "test" else 0)
                obs = check_observation(env.reset(seed=seed), spec)
                provenance = _provenance(env)
                profiles.append(provenance)
                if trace:
                    trace.start(episode, seed, provenance)
                elapsed_s = 0.0
                reached = EpisodeMetrics(metric_options)
                policy.reset()
                context = policy.observe(obs, observation_id=0)
                total, controlled, valid_steps = 0.0, 0.0, 0
                for number in range(1, cfg.episode_limit + 1):
                    expert = (
                        check_action(env.expert_action()) if trace or not context.valid else None
                    )
                    before_pose = pose_of(env) if trace else None
                    if context.valid:
                        base = _clip(policy.act(context, official_only=official_only), cfg)
                        action = agent.action(context.features, base) if agent else base
                        valid_steps += 1
                    else:
                        action = torch.tensor(expert, device=device)[None]
                    transition = check_transition(
                        env.step(_clip(action, cfg)[0].cpu().numpy()), spec
                    )
                    diagnostics = env.diagnostics() if hasattr(env, "diagnostics") else None
                    reached.add(transition, controlled=context.valid, diagnostics=diagnostics)
                    if trace:
                        trace.append(
                            obs,
                            timestamp_s=elapsed_s,
                            pose=before_pose,
                            next_pose=pose_of(env),
                            expert=expert.tolist(),
                            official=context.official_action[0].cpu().tolist(),
                            command=_clip(action, cfg)[0].cpu().tolist(),
                            transition=transition,
                            warmed=context.valid,
                            truncated=transition.truncated or number == cfg.episode_limit,
                            diagnostics=diagnostics,
                        )
                    elapsed_s += float(transition.observation["dt"][-1, 0])
                    obs = transition.observation
                    total += transition.reward
                    if context.valid:
                        controlled += transition.reward
                    if transition.terminated or transition.truncated or number == cfg.episode_limit:
                        break
                    context = policy.observe(transition.observation, observation_id=number)
                if not valid_steps and not allow_warmup_failure:
                    raise ValueError(
                        "evaluation episodes end before Teacher warmup; increase the horizon"
                    )
                if trace:
                    trace.end()
                results.append(
                    {
                        "seed": seed,
                        "scenario_id": provenance["scenario_id"],
                        "return": total,
                        "controlled_return": controlled,
                        "controlled_steps": valid_steps,
                        "steps": number,
                        "terminated": transition.terminated,
                        "truncated": transition.truncated or number == cfg.episode_limit,
                        "reached_state": reached.report(),
                    }
                )
            report = {
                "split": split,
                "mean_return": float(np.mean([r["return"] for r in results])),
                "mean_controlled_return": float(np.mean([r["controlled_return"] for r in results])),
                "episodes": results,
                "environments": profiles,
                "warmup_control": "environment_expert",
                "teacher_calls": policy.teacher_calls,
                "end_to_end_latency_includes_rgb": True,
            }
            if trace:
                trace.finish(report)
                report["trace_dir"] = str(trace_dir)
            return report
    finally:
        env.close()


def _offline(cfg, saved, spec):
    path = cfg.offline_manifest
    if path is None and saved and saved.get("config"):
        path = Path(saved["config"]["data"]["manifest"])
    if path is None:
        raise ValueError(
            "DAgger offline replay needs offline_manifest or an offline init checkpoint"
        )
    catalogue = Catalogue(path)
    signature = catalogue.validate()["signature"]
    for name in ("teacher_identity", "state_features", "camera_names", "preprocessing_id"):
        if signature.get(name) != spec[name]:
            raise ValueError(f"offline DAgger replay differs from the live contract on {name}")
    dataset = EpisodeWindows(
        catalogue,
        split="train",
        task="official_residual",
        history=1,
        image_size=tuple(spec["image_size"]),
        supervision=("expert",),
    )
    return dataset


def train_official_closed_loop(cfg: OfficialConfig, *, teacher_factory=None):
    if cfg.output_dir.exists() and any(cfg.output_dir.iterdir()):
        raise ValueError("official-loop output directory must be empty")
    gate = _gate(cfg)
    torch.set_num_threads(cfg.num_threads)
    torch.manual_seed(cfg.seed)
    rng, device = np.random.default_rng(cfg.seed), select_device(cfg.device)
    teacher = _new_teacher(cfg, teacher_factory)
    spec, model, saved = _initial(cfg, teacher, device)
    if not gate["bypassed"] and gate.get("teacher_identity") != teacher.cache_identity():
        raise ValueError("Gate 2 refers to a different Teacher identity")
    if cfg.algorithm == "td3":
        model.requires_grad_(False)
    offline = (
        _offline(cfg, saved, spec) if cfg.algorithm == "dagger" and cfg.offline_fraction else None
    )
    official_baseline = _evaluate(cfg, teacher, spec, model, official_only=True)
    initial = _evaluate(cfg, teacher, spec, model)
    env = load_environment(cfg.env_factory, spec, "train", cfg.env_kwargs)
    policy = OfficialTeacherPolicy(
        teacher, spec, teacher_rgb=_profile(env, cfg), device=device, residual_model=model
    )
    agent = TD3(policy.feature_dim, cfg, device) if cfg.algorithm == "td3" else None
    optimizer = (
        torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate) if agent is None else None
    )
    replay = Replay(cfg.capacity)
    validation_pools = {info["scenario_pool"] for info in initial["environments"]}
    validation_ids = {info["scenario_id"] for info in initial["environments"]}
    try:
        obs = check_observation(env.reset(seed=cfg.seed), spec)
        provenance = _provenance(env)
        if (
            provenance["scenario_pool"] in validation_pools
            or provenance["scenario_id"] in validation_ids
        ):
            raise ValueError("train/val scenario pools overlap")
        policy.reset()
        context = policy.observe(obs, observation_id=0)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        run = {
            "format": "comma-official-loop-run-v1",
            "algorithm": cfg.algorithm,
            "config": asdict(cfg),
            "model_spec": spec,
            "teacher_identity": teacher.cache_identity(),
            "environment": provenance,
            "gate2": gate,
            "source_sha256": source_fingerprint(),
            "initial_checkpoint_sha256": file_sha256(cfg.init_checkpoint)
            if cfg.init_checkpoint
            else None,
            "anchor_mode": "frozen_offline_residual_plus_official"
            if cfg.algorithm == "td3" and saved
            else "official_with_trainable_residual",
            "anchor_residual_scale": spec["residual_scale"],
            "additional_td3_residual_scale": list(cfg.residual_scale) if agent else None,
            "online_resume_supported": False,
            "action_semantics": "final_commanded_acceleration_mps2",
            "warmup_control": "environment_expert",
            "replay_eviction": "bounded_fifo",
            "offline_data_fingerprint": offline.catalogue.fingerprint if offline else None,
            "offline_train_routes": sorted(
                {ref.route_id for ref in offline.catalogue.episodes if ref.split == "train"}
            )
            if offline
            else [],
            "online_label_source": "expert_query_on_current_visited_state",
            "online_source": "events.jsonl maps replay env_step to scenario/episode/policy version",
        }
        atomic_json(cfg.output_dir / "run.json", run)
        episode, episode_steps, updates, valid_count = 0, 0, 0, 0
        training_start = time.perf_counter()
        latency = []
        for step in range(1, cfg.steps + 1):
            expert = check_action(env.expert_action())
            model.eval()
            with torch.no_grad():
                base = _clip(policy.act(context), cfg)
                if not context.valid:
                    command = torch.tensor(expert, device=device)[None]
                elif agent is not None:
                    noise = torch.tensor(
                        rng.normal(0, cfg.exploration_noise, (1, 2)),
                        dtype=torch.float32,
                        device=device,
                    )
                    command = agent.action(context.features, base) + noise
                else:
                    command = (
                        torch.tensor(expert, device=device)[None]
                        if rng.random() < cfg.expert_mix
                        else base
                    )
                command = _clip(command, cfg)
            behavior_version = updates
            transition = check_transition(env.step(command[0].cpu().numpy()), spec)
            episode_steps += 1
            truncated = transition.truncated or episode_steps >= cfg.episode_limit
            # Even a terminal transition has a final observation, never an auto-reset frame.
            next_context = policy.observe(transition.observation, observation_id=episode_steps)
            if context.valid:
                valid_count += 1
                latency.append(context.end_to_end_ms)
                if agent is not None:
                    with torch.no_grad():
                        next_base = _clip(policy.act(next_context), cfg)
                    if not next_context.valid:
                        raise ValueError(
                            "Teacher context lost validity without an explicit episode reset"
                        )
                    replay.add(
                        features=context.features[0],
                        base=base[0],
                        action=command[0],
                        achieved_action=transition.achieved_action,
                        reward=transition.reward,
                        next_features=next_context.features[0],
                        next_base=next_base[0],
                        terminated=transition.terminated,
                        truncated=truncated,
                        env_step=step,
                        episode=episode,
                        behavior_policy_update=behavior_version,
                        expert=expert,
                        previous_action=context.batch["past_actions"][0, -1],
                        interval_s=float(transition.observation["dt"][-1, 0]),
                    )
                else:
                    replay.add(
                        **{k: v[0] for k, v in context.batch.items()},
                        label=expert,
                        env_step=step,
                        episode=episode,
                        behavior_policy_update=behavior_version,
                    )
            event = {
                "step": step,
                "episode": episode,
                "scenario_id": provenance["scenario_id"],
                "scenario_pool": provenance["scenario_pool"],
                "behavior_policy_update": behavior_version,
                "expert_source": provenance["expert_source"],
                "teacher_warmed": context.valid,
                "teacher_frame": context.teacher_frame,
                "teacher_latency_ms": context.teacher_latency_ms,
                "observation_preprocess_teacher_ms": context.end_to_end_ms,
                "reward": transition.reward,
                "expert_action": expert.tolist(),
                "commanded_action": command[0].cpu().tolist(),
                "official_action": context.official_action[0].cpu().tolist(),
                "anchor_action": base[0].cpu().tolist(),
                "achieved_action": transition.achieved_action.tolist(),
                "terminated": transition.terminated,
                "truncated": truncated,
            }
            if context.valid and step >= cfg.warmup_steps and len(replay.rows) >= cfg.batch_size:
                batch = move_batch(replay.sample(cfg.batch_size, rng), device)
                if agent is not None:
                    event.update(agent.update(batch))
                else:
                    batch = {key: batch[key] for key in (*context.batch, "label")}
                    count = int(cfg.batch_size * cfg.offline_fraction) if offline else 0
                    if count:
                        raw = move_batch(
                            default_collate(
                                [offline[int(i)] for i in rng.integers(len(offline), size=count)]
                            ),
                            device,
                        )
                        batch = {
                            key: torch.cat(
                                (
                                    value[: cfg.batch_size - count],
                                    raw["expert"] if key == "label" else raw[key],
                                ),
                                dim=0,
                            )
                            for key, value in batch.items()
                        }
                    model.train()
                    prediction = model(batch)
                    bc = (prediction - batch["label"]).square().mean()
                    regularizer = (prediction - batch["teacher"]).square().mean()
                    loss = bc + cfg.residual_penalty * regularizer
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
                    optimizer.step()
                    event.update(supervised_loss=float(loss.detach()), bc_loss=float(bc.detach()))
                updates += 1
            with (cfg.output_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
            context = next_context
            if transition.terminated or truncated:
                episode += 1
                episode_steps = 0
                obs = check_observation(env.reset(seed=cfg.seed + episode), spec)
                provenance = _provenance(env)
                if (
                    provenance["scenario_pool"] in validation_pools
                    or provenance["scenario_id"] in validation_ids
                ):
                    raise ValueError("training reset selected a validation scenario pool")
                policy.reset()
                context = policy.observe(obs, observation_id=0)
        if not updates:
            raise ValueError(
                "no official learner updates: extend episodes/budget beyond Teacher warmup"
            )
        training_wall_s = time.perf_counter() - training_start
        final = _evaluate(cfg, teacher, spec, model, agent=agent)
        export = {
            "format": "comma-official-online-v1",
            "task": "official_residual",
            "algorithm": cfg.algorithm,
            "model_spec": spec,
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "config": saved.get("config") if saved else None,
            "official_loop_config": json.loads(json.dumps(asdict(cfg), default=str)),
            "teacher_identity": teacher.cache_identity(),
            "gate2_bypassed": gate["bypassed"],
        }
        if agent is not None:
            export["td3_actor_state"] = {
                k: v.detach().cpu() for k, v in agent.actor.state_dict().items()
            }
        save_checkpoint(cfg.output_dir / "final.pt", export)
        save_checkpoint(
            cfg.output_dir / "learning_state.pt",
            {
                "format": "comma-official-learning-v1",
                "online_resume_supported": False,
                "replay": replay.state_dict(),
                "learner": agent.state_dict() if agent else optimizer.state_dict(),
                "rng_state": torch.get_rng_state(),
            },
        )
        report = {
            "algorithm": cfg.algorithm,
            "teacher_identity": teacher.cache_identity(),
            "base_frozen": True,
            "steps": cfg.steps,
            "valid_teacher_steps": valid_count,
            "updates": updates,
            "actor_updates": agent.actor_updates if agent else 0,
            "expert_queries_train": cfg.steps,
            "training_wall_s": training_wall_s,
            "replay_size": len(replay.rows),
            "gate2": gate,
            "official_baseline": official_baseline,
            "initial_validation": initial,
            "final_validation": final,
            "latency_ms_percentiles": dict(
                zip(
                    ("p50", "p95", "p99"),
                    np.percentile(latency, [50, 95, 99]).tolist(),
                    strict=True,
                )
            ),
            "checkpoint": str(cfg.output_dir / "final.pt"),
            "validation_improved": final["mean_controlled_return"]
            > initial["mean_controlled_return"],
            "note": (
                "synthetic environments and engineering-only runs "
                "do not demonstrate driving improvement"
            ),
        }
        atomic_json(cfg.output_dir / "metrics.json", report)
        return report
    finally:
        env.close()


def evaluate_official_closed_loop(
    cfg: OfficialConfig,
    checkpoint: Path | None = None,
    *,
    split="test",
    teacher_factory=None,
    trace_dir: Path | None = None,
    official_only=False,
    metric_options=None,
    allow_warmup_failure=False,
):
    if official_only and checkpoint is not None:
        raise ValueError("choose official_only or a trained checkpoint, not both")
    torch.set_num_threads(cfg.num_threads)
    device = select_device(cfg.device)
    teacher = _new_teacher(cfg, teacher_factory)
    spec, model, _ = _initial(cfg, teacher, device)
    agent = None
    if checkpoint is not None:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            saved.get("format") != "comma-official-online-v1"
            or saved.get("teacher_identity") != teacher.cache_identity()
        ):
            raise ValueError("evaluation requires a matching official online export")
        if saved["model_spec"] != spec:
            raise ValueError("evaluation initial-head contract differs from saved policy")
        model.load_state_dict(saved["model_state"], strict=True)
        if saved.get("td3_actor_state") is not None:
            original = saved["official_loop_config"]
            for key in ("action_low", "action_high", "residual_scale"):
                if list(getattr(cfg, key)) != original[key]:
                    raise ValueError(f"TD3 evaluation must preserve trained {key}")
            agent = TD3(spec["teacher_feature_dim"] + spec["state_dim"] + 2, cfg, device)
            agent.actor.load_state_dict(saved["td3_actor_state"], strict=True)
            agent.actor.eval()
    return _evaluate(
        cfg,
        teacher,
        spec,
        model,
        agent=agent,
        split=split,
        official_only=official_only,
        trace_dir=trace_dir,
        metric_options=metric_options,
        allow_warmup_failure=allow_warmup_failure,
    )

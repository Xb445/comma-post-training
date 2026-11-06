"""Model creation and losses shared by the train and evaluation entry points."""

import torch

from comma_post_training.driving_policy.model import DrivingPolicy
from comma_post_training.engine.config import RunConfig
from comma_post_training.teacher.residual import OfficialResidualPolicy
from comma_post_training.world_model.compressor import Compressor
from comma_post_training.world_model.flow import LatentFlow


def build_spec(config: RunConfig, signature: dict, dataset) -> dict:
    spec = {
        "kind": config.task,
        "cameras": len(signature["camera_names"]),
        "camera_names": signature["camera_names"],
        "state_features": signature["state_features"],
        "state_dim": len(signature["state_features"]),
        "image_size": list(config.data.image_size),
        "preprocessing_id": signature["preprocessing_id"],
        "history": config.data.history,
        "hidden_size": config.model.hidden_size,
        "vision_channels": config.model.vision_channels,
        "latent_channels": config.model.latent_channels,
    }
    if config.task == "world_model":
        spec.update(
            latent_shape=list(dataset.latent_shape),
            encoder_fingerprint=dataset.encoder_fingerprint,
            plan_steps=config.world.plan_steps,
            heads=config.model.heads,
            transformer_layers=config.model.transformer_layers,
            anchor_offsets=list(config.world.anchor_offsets),
        )
    if config.task == "official_residual":
        spec.update(
            teacher_identity=dataset.teacher_identity,
            teacher_feature_dim=dataset.teacher_feature_dim,
            residual_scale=list(config.model.residual_scale),
        )
    return spec


def build_model(spec: dict):
    classes = {
        "policy": DrivingPolicy,
        "compressor": Compressor,
        "world_model": LatentFlow,
        "official_residual": OfficialResidualPolicy,
    }
    if spec.get("kind") not in classes:
        raise ValueError("unknown checkpoint model kind")
    return classes[spec["kind"]](spec)


def architecture(spec: dict) -> dict:
    keys = {
        "official_residual": (
            "teacher_identity",
            "teacher_feature_dim",
            "state_dim",
            "state_features",
            "hidden_size",
            "residual_scale",
            "history",
        ),
        "policy": (
            "cameras",
            "camera_names",
            "state_dim",
            "state_features",
            "image_size",
            "history",
            "hidden_size",
            "vision_channels",
        ),
        "compressor": (
            "cameras",
            "camera_names",
            "image_size",
            "vision_channels",
            "latent_channels",
        ),
        "world_model": (
            "latent_shape",
            "encoder_fingerprint",
            "history",
            "hidden_size",
            "heads",
            "transformer_layers",
            "plan_steps",
            "anchor_offsets",
        ),
    }
    return {"kind": spec["kind"], **{key: spec[key] for key in keys[spec["kind"]]}}


def task_loss(model, batch: dict, config: RunConfig) -> tuple[torch.Tensor, dict]:
    """Return a training scalar and per-example metrics with explicit validity masks."""
    metrics = {}
    if config.task in ("policy", "official_residual"):
        prediction = model(batch)
        loss = prediction.sum() * 0.0
        for source, weight in (
            ("expert", config.loss.bc_weight),
            ("teacher", config.loss.kd_weight),
        ):
            error = prediction - batch[source].detach()
            valid = batch[source + "_valid"]
            mse = error.square().mean(-1)
            metrics[source + "_mse"] = (mse, valid)
            metrics[source + "_lat_mae"] = (error[:, 0].abs(), valid)
            metrics[source + "_long_mae"] = (error[:, 1].abs(), valid)
            if weight and valid.any():
                loss = loss + weight * mse[valid].mean()
    elif config.task == "compressor":
        images = batch["images"][:, 0]
        errors = (model(images) - images).square().flatten(1).mean(-1)
        metrics["reconstruction_mse"] = (errors, torch.ones_like(errors, dtype=torch.bool))
        loss = errors.mean()
    else:
        target = batch["target_latent"]
        noise, tau = torch.randn_like(target), torch.rand(len(target), device=target.device)
        expanded = tau.reshape(len(tau), *([1] * (target.ndim - 1)))
        noisy = (1 - expanded) * noise + expanded * target
        output = model(noisy, tau, batch)
        flow = (output["velocity"] - (target - noise)).square().flatten(1).mean(-1)
        plan = (output["plan"] - batch["plan"]).square().flatten(1).mean(-1)
        valid = torch.ones_like(flow, dtype=torch.bool)
        metrics = {"flow_mse": (flow, valid), "plan_mse": (plan, valid)}
        loss = flow.mean() + config.loss.plan_weight * plan.mean()
    return loss, metrics


def metric_objective(metrics: dict[str, float], config: RunConfig) -> float:
    if config.task in ("policy", "official_residual"):
        result = 0.0
        for name, weight in (
            ("expert_mse", config.loss.bc_weight),
            ("teacher_mse", config.loss.kd_weight),
        ):
            if weight and name not in metrics:
                raise ValueError(f"no validation supervision for active loss {name}")
            result += weight * metrics.get(name, 0.0)
        return result
    if config.task == "compressor":
        return metrics["reconstruction_mse"]
    return metrics["flow_mse"] + config.loss.plan_weight * metrics["plan_mse"]

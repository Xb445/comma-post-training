"""Action-only BC/KD baseline; upstream multi-head reproduction is a separate task."""

import math

import torch


def action_distillation_loss(
    prediction: torch.Tensor,
    *,
    expert: torch.Tensor | None = None,
    teacher: torch.Tensor | None = None,
    bc_weight: float = 1.0,
    kd_weight: float = 1.0,
    valid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Masked MSE over [..., 2] accelerations; target tensors never receive gradients."""
    weights = (bc_weight, kd_weight)
    if any(not math.isfinite(weight) or weight < 0 for weight in weights) or not sum(weights):
        raise ValueError("BC/KD weights must be finite, nonnegative, and not both zero")
    if prediction.ndim < 2 or prediction.shape[-1] != 2 or not prediction.is_floating_point():
        raise ValueError("prediction must have floating shape [..., 2]")
    if not torch.isfinite(prediction).all():
        raise ValueError("prediction contains nonfinite values")
    if valid is None:
        valid = torch.ones(prediction.shape[:-1], dtype=torch.bool, device=prediction.device)
    if valid.dtype != torch.bool or valid.shape != prediction.shape[:-1]:
        raise ValueError("valid must be a boolean mask matching prediction.shape[:-1]")
    if valid.device != prediction.device or not valid.any():
        raise ValueError("valid must share the prediction device and select at least one sample")
    losses = {}
    for name, target, weight in (("bc", expert, bc_weight), ("kd", teacher, kd_weight)):
        if weight == 0:
            losses[name] = prediction.sum() * 0.0
            continue
        if target is None or target.shape != prediction.shape or target.device != prediction.device:
            raise ValueError(f"{name} target must match prediction shape and device")
        if not target.is_floating_point() or not torch.isfinite(target[valid]).all():
            raise ValueError(
                f"{name} targets must contain finite floating values at valid positions"
            )
        losses[name] = (prediction[valid] - target.detach()[valid]).square().mean()
    losses["total"] = bc_weight * losses["bc"] + kd_weight * losses["kd"]
    return losses

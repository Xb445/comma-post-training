"""Autoregressive held-out latent prediction with explicitly privileged future anchors."""

from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import default_collate

from comma_post_training.data_pipeline.episodes import Catalogue
from comma_post_training.engine.checkpoints import load_checkpoint, save_checkpoint
from comma_post_training.engine.tasks import build_model, build_spec
from comma_post_training.engine.trainer import (
    config_from_dict,
    make_windows,
    move_batch,
    select_device,
)


@torch.no_grad()
def predict_rollout(
    checkpoint: Path,
    output: Path,
    *,
    split: str = "test",
    frames: int = 3,
    seed: int = 0,
    sample_steps: int | None = None,
    device: str | None = None,
) -> dict:
    if frames < 1 or output.exists():
        raise ValueError("frames must be positive and rollout output must not exist")
    saved = load_checkpoint(checkpoint)
    if saved["task"] != "world_model":
        raise ValueError("rollout requires a native latent world-model checkpoint")
    config = config_from_dict(saved["config"])
    if sample_steps is None:
        sample_steps = config.world.sampling_steps
    if device is not None:
        config = replace(config, training=replace(config.training, device=device))
    if config.data.stride != 1:
        raise ValueError("autoregressive rollout requires stride=1")
    device = select_device(config.training.device)
    catalogue = Catalogue(config.data.manifest)
    quality = catalogue.validate()
    dataset = make_windows(config, catalogue, split)
    if build_spec(config, quality["signature"], dataset) != saved["model_spec"]:
        raise ValueError("rollout dataset no longer matches the checkpoint's latent contract")
    if len(dataset) < frames:
        raise ValueError("not enough contiguous windows for rollout")
    samples = [dataset[index] for index in range(frames)]
    first = samples[0]
    if any(
        sample["episode_id"] != first["episode_id"] or sample["frame"] != first["frame"] + i
        for i, sample in enumerate(samples)
    ):
        raise ValueError("rollout must not cross a route, invalid interval, or episode boundary")
    model = build_model(saved["model_spec"]).to(device).eval()
    model.load_state_dict(saved["model_state"], strict=True)
    history = first["history_latents"].to(device)
    predictions, errors, plans = [], [], []
    for number, sample in enumerate(samples):
        batch = move_batch(default_collate([sample]), device)
        batch["history_latents"] = history[None]
        generated = model.sample(batch, steps=sample_steps, seed=seed + number)
        latent = generated["latent"][0]
        errors.append(float((latent - batch["target_latent"][0]).square().mean()))
        predictions.append(latent.cpu())
        plans.append(generated["plan"][0].cpu())
        history = torch.cat((history[1:], latent[None]), dim=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        output,
        {
            "format": "comma-latent-rollout-v1",
            "latents": torch.stack(predictions),
            "plans": torch.stack(plans),
            "encoder_fingerprint": dataset.encoder_fingerprint,
            "uses_ground_truth_future_anchors": bool(config.world.anchor_offsets),
            "motion_source": "held_out_dataset_relative_poses",
            "dataset_kind": catalogue.kind,
        },
    )
    return {
        "frames": frames,
        "split": split,
        "episode_id": first["episode_id"],
        "latent_mse_per_frame": errors,
        "output": str(output),
        "mode": "autoregressive_prediction_with_recorded_motion_and_anchors",
    }

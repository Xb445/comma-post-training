"""Live frozen-Teacher + residual inference, preserving asset and data provenance."""

import json
from pathlib import Path

import numpy as np
import torch

from comma_post_training.data_pipeline.episodes import Catalogue, Episode
from comma_post_training.engine.checkpoints import (
    atomic_json,
    file_sha256,
    load_checkpoint,
    staged_directory,
)
from comma_post_training.engine.tasks import build_model
from comma_post_training.engine.trainer import select_device
from comma_post_training.teacher.cache import iter_teacher_episode
from comma_post_training.teacher.evaluation import RouteMetrics


@torch.no_grad()
def predict_adapted(
    manifest: Path, checkpoint: Path, teacher, output: Path, *, split="test", device="cpu"
):
    saved = load_checkpoint(checkpoint)
    if saved["task"] != "official_residual":
        raise ValueError("predict-adapted requires an official_residual checkpoint")
    spec = saved["model_spec"]
    if spec["teacher_identity"] != teacher.cache_identity():
        raise ValueError("residual head requires its exact frozen Teacher model and adapter")
    catalogue = Catalogue(manifest)
    quality = catalogue.validate()
    for key in ("camera_names", "state_features", "preprocessing_id"):
        if spec[key] != quality["signature"][key]:
            raise ValueError(f"residual input contract differs on {key}")
    if split not in ("train", "val", "test"):
        raise ValueError("prediction split must be train/val/test")
    references = [ref for ref in catalogue.episodes if ref.split == split]
    if not references:
        raise ValueError(f"no episodes in split={split}")
    target = select_device(device)
    model = build_model(spec).to(target).eval()
    model.load_state_dict(saved["model_state"], strict=True)
    errors, base_errors = np.zeros(2), np.zeros(2)
    count, labeled = 0, 0
    routes = {}
    with staged_directory(output) as staging:
        with (staging / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for ref in references:
                episode = Episode(ref)
                for frame, result, _ in iter_teacher_episode(episode, teacher):
                    if result is None or not result["valid"]:
                        continue
                    previous = (
                        episode.array("executed_actions")[frame - 1]
                        if result["frame"] > 0
                        else np.zeros(2)
                    )
                    batch = {
                        "teacher_features": torch.tensor(result["features"][None], device=target),
                        "teacher": torch.tensor(result["action"][None], device=target),
                        "state": torch.tensor(
                            np.array(episode.array("ego_state")[frame]),
                            dtype=torch.float32,
                            device=target,
                        )[None, None],
                        "past_actions": torch.tensor(
                            np.array(previous), dtype=torch.float32, device=target
                        )[None, None],
                    }
                    action = model(batch)[0].cpu().numpy()
                    if not np.isfinite(action).all():
                        raise RuntimeError("nonfinite adapted policy output")
                    row = {
                        "format": "comma-official-residual-prediction-v1",
                        "episode_id": ref.id,
                        "route_id": ref.route_id,
                        "frame": frame,
                        "timestamp_s": float(episode.array("timestamps")[frame]),
                        "teacher_action": result["action"].tolist(),
                        "adapted_action": action.tolist(),
                        "action_order": ["a_lat", "a_long"],
                        "action_units": "m/s^2",
                        "coordinate_frame": "vehicle_frd",
                        "source": catalogue.kind,
                    }
                    if "expert_actions" in episode.paths and (
                        "expert_valid" not in episode.paths or episode.array("expert_valid")[frame]
                    ):
                        expert = episode.array("expert_actions")[frame]
                        errors += np.abs(action - expert)
                        base_errors += np.abs(result["action"] - expert)
                        labeled += 1
                        row["expert_action"] = expert.tolist()
                    routes.setdefault(ref.route_id, RouteMetrics()).add(row, spec["residual_scale"])
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    count += 1
        if not count:
            raise ValueError("no warmed Teacher predictions in the requested split")
        report = {
            "mode": "open_loop_frozen_official_base_plus_residual",
            "split": split,
            "teacher_identity": teacher.cache_identity(),
            "teacher_asset_verified": True,
            "dataset_kind": catalogue.kind,
            "frames": count,
            "expert_frames": labeled,
            "teacher_mae_mps2": (base_errors / labeled).tolist() if labeled else None,
            "adapted_mae_mps2": (errors / labeled).tolist() if labeled else None,
            "checkpoint_sha256": file_sha256(checkpoint),
            "data_fingerprint": catalogue.fingerprint,
            "residual_scale_mps2": spec["residual_scale"],
            "predictions": str(output / "predictions.jsonl"),
            "per_route": {key: value.report() for key, value in routes.items()},
        }
        atomic_json(staging / "metrics.json", report)
    return report

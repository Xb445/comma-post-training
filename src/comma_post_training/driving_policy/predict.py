"""Independent visual Student inference; no ONNX model or Teacher session is loaded."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from comma_post_training.data_pipeline.episodes import Catalogue, Episode
from comma_post_training.data_pipeline.windows import EpisodeWindows
from comma_post_training.engine.checkpoints import (
    atomic_json,
    file_sha256,
    load_checkpoint,
    staged_directory,
)
from comma_post_training.engine.tasks import architecture, build_model, build_spec
from comma_post_training.engine.trainer import config_from_dict, move_batch, select_device
from comma_post_training.teacher.evaluation import RouteMetrics


@torch.no_grad()
def predict_policy(manifest: Path, checkpoint: Path, output: Path, *, split="val", device="cpu"):
    saved = load_checkpoint(checkpoint, inference=True)
    if saved["task"] != "policy":
        raise ValueError("predict-policy requires an independent visual policy checkpoint")
    if split not in ("train", "val", "test"):
        raise ValueError("split must be train/val/test")
    cfg = config_from_dict(saved["config"])
    torch.set_num_threads(cfg.training.num_threads)
    catalogue = Catalogue(manifest)
    quality = catalogue.validate()
    dataset = EpisodeWindows(
        catalogue,
        split=split,
        task="policy",
        history=cfg.data.history,
        image_size=cfg.data.image_size,
        max_gap_s=cfg.data.max_gap_s,
        inference=True,
    )
    spec = saved["model_spec"]
    expected = build_spec(cfg, quality["signature"], dataset)
    if (
        architecture(expected) != architecture(spec)
        or expected["preprocessing_id"] != spec["preprocessing_id"]
    ):
        raise ValueError("Student input contract differs from the dataset")
    target = select_device(device)
    model = build_model(spec).to(target).eval()
    model.load_state_dict(saved["model_state"], strict=True)
    episodes = {ref.id: Episode(ref) for ref in catalogue.episodes if ref.split == split}
    routes, frames = {}, 0
    with staged_directory(output) as staging:
        with (staging / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for original in DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=False):
                batch = move_batch(original, target)
                actions = model(batch).cpu()
                if not torch.isfinite(actions).all():
                    raise RuntimeError("nonfinite Student prediction")
                for index, action in enumerate(actions):
                    episode = episodes[original["episode_id"][index]]
                    frame = int(original["frame"][index])
                    row = {
                        "format": "comma-policy-prediction-v1",
                        "episode_id": episode.reference.id,
                        "route_id": episode.reference.route_id,
                        "frame": frame,
                        "timestamp_s": float(episode.array("timestamps")[frame]),
                        "policy_action": action.tolist(),
                        "adapted_action": action.tolist(),
                        "action_order": ["a_lat", "a_long"],
                        "action_units": "m/s^2",
                        "coordinate_frame": "vehicle_frd",
                        "source": catalogue.kind,
                    }
                    for source in ("expert", "teacher"):
                        if bool(original[source + "_valid"][index]):
                            row[source + "_action"] = original[source][index].tolist()
                    routes.setdefault(episode.reference.route_id, RouteMetrics()).add(
                        row, [float("inf")] * 2
                    )
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    frames += 1
        per_route = {key: value.report() for key, value in routes.items()}
        for values in per_route.values():
            values["saturation_fraction"] = None  # A full Student has no bounded residual scale.
        report = {
            "mode": "open_loop_independent_student",
            "task": "policy",
            "split": split,
            "frames": frames,
            "dataset_kind": catalogue.kind,
            "data_fingerprint": catalogue.fingerprint,
            "checkpoint_sha256": file_sha256(checkpoint),
            "teacher_identity": spec.get("distillation_teacher_identity"),
            "runtime_teacher_loaded": False,
            "teacher_asset_verified": False,
            "per_route": per_route,
            "predictions": str(output / "predictions.jsonl"),
            "comparison_note": "Teacher and paired Student MAE use their common labeled frames",
        }
        atomic_json(staging / "metrics.json", report)
    return report

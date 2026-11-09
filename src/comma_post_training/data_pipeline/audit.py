"""Data readiness for policy SFT and later action-conditioned world adaptation."""

from pathlib import Path

import numpy as np

from comma_post_training.data_pipeline.episodes import Catalogue, Episode
from comma_post_training.data_pipeline.geometry import validate_poses
from comma_post_training.engine.checkpoints import atomic_json
from comma_post_training.teacher.assets import OFFICIAL


def _calibrated(episode):
    calibration = episode.metadata.get("calibration")
    if not isinstance(calibration, dict) or not calibration.get("id"):
        return False
    intrinsics = calibration.get("intrinsics")
    if not isinstance(intrinsics, dict):
        return False
    try:
        for camera in episode.metadata["camera_names"]:
            matrix = np.asarray(intrinsics.get(camera), dtype=float)
            if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
                return False
            if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or not np.allclose(matrix[2], [0, 0, 1]):
                return False
        extrinsics = calibration.get("T_vehicle_from_optical")
        for camera in episode.metadata["camera_names"]:
            matrix = extrinsics.get(camera) if isinstance(extrinsics, dict) else extrinsics
            validate_poses(np.asarray(matrix, dtype=float)[None])
    except (ValueError, TypeError, IndexError):
        return False
    return True


def audit_dataset(
    manifest: Path,
    *,
    stage="student",
    history=4,
    future_steps=3,
    max_gap_s=0.3,
    require_official_teacher=False,
    output: Path | None = None,
):
    if stage not in ("student", "world"):
        raise ValueError("audit stage must be student/world")
    if any(type(value) is not int or value < 1 for value in (history, future_steps)):
        raise ValueError("history/future_steps must be positive integers")
    if not np.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("max_gap_s must be positive")
    catalogue = Catalogue(manifest)
    quality = catalogue.validate()
    issues, rows = [], []
    for ref in catalogue.episodes:
        episode = Episode(ref)
        times = episode.array("timestamps")
        n = len(times)
        valid = (
            np.array(episode.array("valid"), copy=True)
            if "valid" in episode.paths
            else np.ones(n, dtype=bool)
        )
        labels = {}
        for source in ("expert", "teacher"):
            labels[source] = np.zeros(n, dtype=bool)
            if source + "_actions" in episode.paths:
                mask = source + "_valid"
                labels[source] = (
                    np.array(episode.array(mask))
                    if mask in episode.paths
                    else np.ones(n, dtype=bool)
                )
                labels[source] &= valid
        usable = 0
        extra = future_steps if stage == "world" else 0
        for end in range(history - 1, n - extra):
            first, last = end - history + 1, end + extra
            if (
                valid[first : last + 1].all()
                and (np.diff(times[first : last + 1]) <= max_gap_s).all()
            ):
                if labels["expert"][end] or stage == "world":
                    usable += 1
        identity = episode.metadata.get("teacher_identity", {})
        official = (
            identity.get("kind") == "official"
            and identity.get("sha256") == OFFICIAL.sha256
            and identity.get("commit") == OFFICIAL.commit
        )
        # Decode lazy paths too; the episode validator only checks file existence.
        pixels = episode.images([0, n - 1], (16, 32))
        if not np.isfinite(pixels).all():
            issues.append(f"{ref.id}: invalid decoded image endpoints")
        reference_fraction = float(labels["expert"].sum() / max(1, valid.sum()))
        row = {
            "id": ref.id,
            "route_id": ref.route_id,
            "split": ref.split,
            "domain": ref.domain,
            "frames": n,
            "usable_windows": usable,
            "expert_frames": int(labels["expert"].sum()),
            "teacher_frames": int(labels["teacher"].sum()),
            "expert_fraction": reference_fraction,
            "official_teacher_provenance": official,
            "has_poses": "poses" in episode.paths,
            "has_commanded_actions": "commanded_actions" in episode.paths,
            "has_actual_actions": "executed_actions" in episode.paths,
            "has_calibration": _calibrated(episode),
            "has_action_provenance": isinstance(episode.metadata.get("label_provenance"), dict)
            and all(
                isinstance(episode.metadata["label_provenance"].get(key), str)
                and episode.metadata["label_provenance"][key].strip()
                for key in ("expert", "commanded", "executed")
            ),
            "median_hz": float(1 / np.median(np.diff(times))),
        }
        if not usable:
            issues.append(f"{ref.id}: no usable {stage} windows")
        if stage == "student" and not row["expert_frames"]:
            issues.append(f"{ref.id}: target adaptation needs independent expert labels")
        if require_official_teacher and (not official or not row["teacher_frames"]):
            issues.append(f"{ref.id}: verified official Teacher cache/labels required")
        if stage == "world":
            for flag in (
                "has_poses",
                "has_commanded_actions",
                "has_actual_actions",
                "has_calibration",
                "has_action_provenance",
            ):
                if not row[flag]:
                    issues.append(f"{ref.id}: missing {flag.removeprefix('has_')}")
            if reference_fraction < 0.8:
                issues.append(
                    f"{ref.id}: independent reference behavior covers less than 80% of valid frames"
                )
        rows.append(row)
    groups = {}
    for split, minimum in (("train", 2), ("val", 1), ("test", 1)):
        selected = [row for row in rows if row["split"] == split]
        routes = len({row["route_id"] for row in selected})
        groups[split] = {
            "routes": routes,
            "usable_windows": sum(row["usable_windows"] for row in selected),
        }
        if routes < minimum:
            issues.append(f"{split}: requires at least {minimum} independent routes")
    report = {
        "format": "comma-data-readiness-v1",
        "stage": stage,
        "ready": not issues,
        "issues": issues,
        "dataset": quality,
        "episodes": rows,
        "splits": groups,
        "history": history,
        "future_steps": extra,
        "max_gap_s": max_gap_s,
        "require_official_teacher": require_official_teacher,
        "scope": (
            "structural readiness only; timing/signs/calibration and label quality need review"
        ),
        "official_4b_training_supported": False,
    }
    if output:
        if output.exists():
            raise ValueError("audit output already exists")
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(output, report)
    return report

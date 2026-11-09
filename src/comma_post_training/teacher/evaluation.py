"""Route-level metrics and the explicit validation gate before online experiments."""

import math

import numpy as np


class RouteMetrics:
    def __init__(self):
        self.frames = self.expert_frames = 0
        self.base_error = np.zeros(2)
        self.adapted_error = np.zeros(2)
        self.saturated = np.zeros(2)
        self.jerk_base = np.zeros(2)
        self.jerk_adapted = np.zeros(2)
        self.elapsed = 0.0
        self.previous = {}

    def add(self, row, scale):
        base, adapted = (
            np.asarray(row[key], dtype=float) for key in ("teacher_action", "adapted_action")
        )
        if base.shape != (2,) or adapted.shape != (2,) or not np.isfinite([base, adapted]).all():
            raise ValueError("prediction actions must be finite [2]")
        self.frames += 1
        self.saturated += np.abs(adapted - base) >= np.asarray(scale) * 0.99
        if "expert_action" in row:
            expert = np.asarray(row["expert_action"], dtype=float)
            if expert.shape != (2,) or not np.isfinite(expert).all():
                raise ValueError("expert action must be finite [2]")
            self.expert_frames += 1
            self.base_error += np.abs(base - expert)
            self.adapted_error += np.abs(adapted - expert)
        prior = self.previous.get(row["episode_id"])
        if prior is not None:
            seconds = row["timestamp_s"] - prior["timestamp_s"]
            if row["frame"] == prior["frame"] + 1 and 0 < seconds <= 0.3:
                self.elapsed += seconds
                self.jerk_base += (base - prior["teacher_action"]) ** 2 / seconds
                self.jerk_adapted += (adapted - prior["adapted_action"]) ** 2 / seconds
        self.previous[row["episode_id"]] = row

    def report(self):
        return {
            "frames": self.frames,
            "expert_frames": self.expert_frames,
            "teacher_mae_mps2": (self.base_error / self.expert_frames).tolist()
            if self.expert_frames
            else None,
            "adapted_mae_mps2": (self.adapted_error / self.expert_frames).tolist()
            if self.expert_frames
            else None,
            "saturation_fraction": (self.saturated / self.frames).tolist() if self.frames else None,
            "teacher_jerk_rms_mps3": np.sqrt(self.jerk_base / self.elapsed).tolist()
            if self.elapsed
            else None,
            "adapted_jerk_rms_mps3": np.sqrt(self.jerk_adapted / self.elapsed).tolist()
            if self.elapsed
            else None,
            "jerk_elapsed_s": self.elapsed,
        }


def gate_residual(report: dict, *, tolerance_mps2=1e-4, min_improvement_mps2=1e-3) -> dict:
    """Engineering gate on held-out validation, not statistical proof of driving quality."""
    if not all(
        isinstance(v, (float, int)) and math.isfinite(v) and v >= 0
        for v in (tolerance_mps2, min_improvement_mps2)
    ):
        raise ValueError("gate tolerances must be finite and nonnegative")
    if report.get("split") != "val":
        raise ValueError("Gate 2 must use validation scenarios; keep test for final assessment")
    routes = report.get("per_route", {})
    results = {}
    for name, metrics in routes.items():
        base, adapted = metrics.get("teacher_mae_mps2"), metrics.get("adapted_mae_mps2")
        usable = metrics.get("expert_frames", 0) > 0 and base is not None and adapted is not None
        if usable:
            base, adapted = np.asarray(base, dtype=float), np.asarray(adapted, dtype=float)
            usable = (
                base.shape == (2,)
                and adapted.shape == (2,)
                and np.isfinite([base, adapted]).all()
                and np.all(base >= 0)
                and np.all(adapted >= 0)
            )
        delta = base - adapted if usable else None
        results[name] = {
            "expert_frames": metrics.get("expert_frames", 0),
            "passed": bool(usable and np.all(delta >= -tolerance_mps2)),
            "improved": bool(usable and np.any(delta > min_improvement_mps2)),
            "improvement_mps2": delta.tolist() if usable else None,
        }
    identity = report.get("teacher_identity", {})
    passed = (
        bool(results)
        and all(v["passed"] for v in results.values())
        and any(v["improved"] for v in results.values())
    )
    passed = (
        passed
        and bool(report.get("teacher_asset_verified"))
        and bool(report.get("checkpoint_sha256"))
    )
    return {
        "format": "comma-gate2-v1",
        "passed": passed,
        "split": "val",
        "teacher_identity": identity,
        "checkpoint_sha256": report.get("checkpoint_sha256"),
        "data_fingerprint": report.get("data_fingerprint"),
        "dataset_kind": report.get("dataset_kind"),
        "criteria": {
            "tolerance_mps2": tolerance_mps2,
            "min_improvement_mps2": min_improvement_mps2,
            "all_routes_and_axes_nondegrading": True,
            "at_least_one_improved": True,
        },
        "route_results": results,
        "interpretation": "engineering_validation_gate_not_driving_safety_or_statistical_evidence",
    }

"""Reached-state metrics with explicit controlled intervals and recovery evidence."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RecoveryCriteria:
    cross_track_m: float = 0.3
    heading_error_rad: float = 0.05
    speed_error_mps: float = 0.5
    hold_steps: int = 5

    def __post_init__(self):
        for name in ("cross_track_m", "heading_error_rad", "speed_error_mps"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"recovery {name} must be positive and finite")
        if type(self.hold_steps) is not int or self.hold_steps < 1:
            raise ValueError("recovery hold_steps must be a positive integer")


class EpisodeMetrics:
    def __init__(self, criteria=None):
        self.criteria = RecoveryCriteria(**(criteria or {}))
        self.errors = []
        self.jerks = []
        self.elapsed = 0.0
        self.previous_actual = None
        self.onset = None
        self.within_count = 0
        self.recovered_at = None
        self.terminated = False
        self.controlled_steps = 0

    def add(self, transition, *, controlled: bool, diagnostics=None):
        dt = float(transition.observation["dt"][-1, 0])
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("rollout metrics require a positive interval")
        self.elapsed += dt
        self.terminated |= transition.terminated
        actual = np.asarray(transition.achieved_action, dtype=float)
        if controlled:
            self.controlled_steps += 1
            if self.previous_actual is not None:
                self.jerks.append((actual - self.previous_actual) / dt)
        self.previous_actual = actual
        if not diagnostics:
            return
        keys = ("cross_track_m", "heading_error_rad", "speed_error_mps")
        if not all(key in diagnostics for key in keys):
            return
        errors = np.array([diagnostics[key] for key in keys], dtype=float)
        if not np.isfinite(errors).all():
            raise ValueError("nonfinite reached-state diagnostics")
        if controlled:
            self.errors.append(errors)
        if diagnostics.get("perturbation_applied") and self.onset is None:
            self.onset = self.elapsed
        # The perturbation frame itself is not a recovery; require subsequent policy steps.
        if self.onset is not None and self.elapsed > self.onset and controlled:
            limits = np.array([getattr(self.criteria, key) for key in keys])
            self.within_count = self.within_count + 1 if (np.abs(errors) <= limits).all() else 0
            if self.within_count >= self.criteria.hold_steps and self.recovered_at is None:
                self.recovered_at = self.elapsed - self.onset

    def report(self):
        errors = np.asarray(self.errors)
        jerks = np.asarray(self.jerks)
        return {
            "controlled_steps": self.controlled_steps,
            "diagnostic_steps": len(self.errors),
            "cross_track_mae_m": float(np.abs(errors[:, 0]).mean()) if len(errors) else None,
            "cross_track_max_m": float(np.abs(errors[:, 0]).max()) if len(errors) else None,
            "heading_mae_rad": float(np.abs(errors[:, 1]).mean()) if len(errors) else None,
            "speed_mae_mps": float(np.abs(errors[:, 2]).mean()) if len(errors) else None,
            "achieved_jerk_rms_mps3": np.sqrt(np.mean(jerks**2, axis=0)).tolist()
            if len(jerks)
            else None,
            "perturbation_observed": self.onset is not None,
            "recovered": (self.recovered_at is not None and not self.terminated)
            if self.onset is not None
            else None,
            "recovery_time_s": self.recovered_at if not self.terminated else None,
            "recovery_criteria": asdict(self.criteria),
            "recovery_time_semantics": (
                "elapsed after disturbance until hold_steps confirmed; "
                "a later terminal invalidates recovery"
            ),
        }

"""Open-loop metrics with explicit physical units and independent trajectory times."""

import json
import math
from itertools import pairwise
from pathlib import Path
from statistics import fmean


def _numbers(value: object, name: str, minimum: int = 2) -> list[float]:
    if not isinstance(value, list) or len(value) < minimum:
        raise ValueError(f"{name} must be a list of at least {minimum} numbers")
    if any(type(item) not in (int, float) or not math.isfinite(item) for item in value):
        raise ValueError(f"{name} must contain finite numbers")
    return [float(item) for item in value]


def _times(value: object, name: str) -> list[float]:
    times = _numbers(value, name)
    if times[0] < 0 or any(right <= left for left, right in pairwise(times)):
        raise ValueError(f"{name} must be nonnegative and strictly increasing")
    return times


def _pairs(value: object, name: str, length: int) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} length must match its timestamps")
    pairs = [_numbers(row, name) for row in value]
    if any(len(row) != 2 for row in pairs):
        raise ValueError(f"{name} must have exactly two components per sample")
    return pairs


def evaluate_record(record: dict) -> dict[str, float]:
    if record.get("action_order") != ["a_lat", "a_long"] or record.get("action_units") != "m/s^2":
        raise ValueError("actions must be [a_lat, a_long] in m/s^2")
    if record.get("coordinate_frame") != "ego_at_start":
        raise ValueError("trajectory coordinates must use ego_at_start")
    times = _times(record.get("action_times_s"), "action_times_s")
    predicted = _pairs(record.get("predicted_actions"), "predicted_actions", len(times))
    reference = _pairs(record.get("reference_actions"), "reference_actions", len(times))
    trajectory = record.get("trajectory")
    if not isinstance(trajectory, dict):
        raise ValueError("trajectory must contain aligned predictions and references in meters")
    trajectory_times = _times(trajectory.get("times_s"), "trajectory.times_s")
    if trajectory_times[0] != 0:
        raise ValueError("trajectory.times_s must start at zero")
    predicted_xy = _pairs(trajectory.get("predicted_xy_m"), "predicted_xy_m", len(trajectory_times))
    reference_xy = _pairs(trajectory.get("reference_xy_m"), "reference_xy_m", len(trajectory_times))
    errors = [math.dist(p, r) for p, r in zip(predicted_xy, reference_xy, strict=True)]
    duration = times[-1] - times[0]
    intervals = [right - left for left, right in pairwise(times)]
    result = {
        "trajectory_ade_m": fmean(errors),
        "trajectory_fde_m": errors[-1],
    }
    for axis, name in enumerate(("lateral", "longitudinal")):
        result[f"{name}_accel_mae_mps2"] = fmean(
            abs(p[axis] - r[axis]) for p, r in zip(predicted, reference, strict=True)
        )
        # RMS of da/dt, time-weighted; never differentiate across record boundaries.
        result[f"{name}_jerk_rms_mps3"] = math.sqrt(
            sum(
                ((predicted[i + 1][axis] - predicted[i][axis]) / dt) ** 2 * dt
                for i, dt in enumerate(intervals)
            )
            / duration
        )
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("metric computation produced nonfinite values")
    return result


def evaluate_file(path: Path) -> dict:
    results = []
    sources: set[str] = set()
    identities: set[str] = set()
    routes: set[str] = set()
    trajectory_grid = None
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if (
                not isinstance(record, dict)
                or type(record.get("schema_version")) is not int
                or record["schema_version"] != 1
            ):
                raise ValueError("expected a version 1 evaluation record")
            identity, route = record.get("sample_id"), record.get("route_id")
            if not all(isinstance(value, str) and value.strip() for value in (identity, route)):
                raise ValueError("sample_id and route_id must be nonempty strings")
            if identity in identities:
                raise ValueError(f"duplicate sample_id: {identity}")
            identities.add(identity)
            routes.add(route)
            if record.get("source") not in ("synthetic", "recorded"):
                raise ValueError("source must be synthetic or recorded")
            sources.add(record["source"])
            metrics = evaluate_record(record)
            grid = record["trajectory"]["times_s"]
            if trajectory_grid is not None and grid != trajectory_grid:
                raise ValueError("all records must use the same trajectory time grid")
            trajectory_grid = grid
            results.append({"sample_id": identity, **metrics})
        except (ValueError, OverflowError) as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
    if not results:
        raise ValueError("no evaluation records found")
    if len(sources) != 1:
        raise ValueError("synthetic and recorded results must be reported separately")
    return {
        "mode": "open_loop",
        "source": next(iter(sources)),
        "aggregation": "macro_per_record",
        "records": len(results),
        "routes": len(routes),
        "trajectory_times_s": trajectory_grid,
        "metrics": {key: fmean(row[key] for row in results) for key in metrics},
        "per_record": results,
    }

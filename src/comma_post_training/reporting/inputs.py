"""Read measured artifacts, retaining missing values and evaluation provenance."""

import json
import math
from pathlib import Path

import numpy as np

from comma_post_training.teacher.evaluation import RouteMetrics


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON constant {value}")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)


def read_rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, parse_constant=_reject_constant)
                if not isinstance(row, dict):
                    raise ValueError("expected a JSON object")
                yield row
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}:{number}: {error}") from error


def clean(value):
    """NaN labels masked by the episode contract become gaps, never zeroes."""
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (np.ndarray, list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def episode_values(episode):
    result = {
        "time_s": np.asarray(episode.array("timestamps")),
        "state": np.asarray(episode.array("ego_state")),
    }
    for name in ("expert", "teacher", "commanded", "executed"):
        key = name + "_actions"
        if key in episode.paths:
            values = np.array(episode.array(key), dtype=float)
            valid = np.ones(len(values), dtype=bool)
            for mask in ("valid", name + "_valid"):
                if mask in episode.paths:
                    valid &= episode.array(mask)
            values[~valid] = np.nan
            result[name] = values
    for name in ("valid", "expert_valid", "teacher_valid", "rewards", "terminated", "truncated"):
        if name in episode.paths:
            result[name] = np.asarray(episode.array(name))
    if "poses" in episode.paths:
        result["xy"] = np.asarray(episode.array("poses"))[:, :2, 3]
    return result


def evaluation_summary(evaluation):
    episodes = evaluation.get("episodes", [])
    controlled = sum(int(row.get("controlled_steps", 0)) for row in episodes)
    return {
        "episodes": len(episodes),
        "mean_controlled_return": evaluation.get("mean_controlled_return"),
        "mean_return": evaluation.get("mean_return"),
        "terminated": sum(row.get("terminated") is True for row in episodes),
        "truncated": sum(row.get("truncated") is True for row in episodes),
        "controlled_steps": controlled,
        "controlled_return_per_step": sum(row["controlled_return"] for row in episodes) / controlled
        if controlled
        else None,
        "scenarios": [[row.get("seed"), row.get("scenario_id")] for row in episodes],
    }


def load_run(path: Path):
    metrics = read_json(path / "metrics.json")
    provenance = read_json(path / "run.json") if (path / "run.json").exists() else {}
    events = list(read_rows(path / "events.jsonl")) if (path / "events.jsonl").exists() else []
    # Training losses and scalar diagnostics must be finite, unlike masked data labels.
    for row in events:
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{path}: nonfinite event field {key} at step {row.get('step')}")
    summary = {
        "name": path.name,
        "path": str(path),
        "task": metrics.get("task"),
        "events": len(events),
        "environment_kind": provenance.get("environment", {}).get("kind"),
        "dataset_kind": provenance.get("data", {}).get("kind"),
    }
    if "official_baseline" in metrics:
        summary["algorithm"] = metrics["algorithm"]
        summary["evaluations"] = {
            name: evaluation_summary(metrics[name])
            for name in ("official_baseline", "initial_validation", "final_validation")
        }
        before, after = (
            summary["evaluations"][name] for name in ("initial_validation", "final_validation")
        )
        cohorts = [row["scenarios"] for row in summary["evaluations"].values()]
        summary["paired_scenarios_match"] = (
            bool(cohorts[0])
            and all(
                seed is not None and isinstance(scenario, str) and bool(scenario)
                for seed, scenario in cohorts[0]
            )
            and all(cohort == cohorts[0] for cohort in cohorts)
        )
        summary["controlled_return_delta"] = (
            after["mean_controlled_return"] - before["mean_controlled_return"]
        )
        summary["verdict"] = (
            "not_comparable"
            if not summary["paired_scenarios_match"]
            else "degraded"
            if summary["controlled_return_delta"] < 0
            else "improved_in_this_run"
            if summary["controlled_return_delta"] > 0
            else "unchanged"
        )
        summary["gate2_bypassed"] = metrics.get("gate2", {}).get("bypassed")
        summary["evaluation_split"] = metrics["final_validation"].get("split")
    else:
        summary["initial_validation_loss"] = metrics.get("initial_validation", {}).get("loss")
        summary["final_validation_loss"] = metrics.get("final_validation", {}).get("loss")
        summary["resumed"] = bool(provenance.get("resume_from"))
    summary["teacher_identity"] = metrics.get("teacher_identity")
    summary["steps"] = metrics.get("steps")
    summary["updates"] = metrics.get("updates")
    return {"summary": summary, "metrics": metrics, "provenance": provenance, "events": events}


def compare_runs(runs):
    online = [run for run in runs if "algorithm" in run["summary"]]
    keys = []
    for run in online:
        provenance = run["provenance"]
        cfg = provenance.get("config", {})
        fields = (
            "env_factory",
            "env_kwargs",
            "eval_seed",
            "eval_episodes",
            "episode_limit",
            "action_low",
            "action_high",
        )
        required = [cfg.get(name) for name in fields]
        scenarios = run["summary"]["evaluations"]["initial_validation"]["scenarios"]
        rewards = [
            row.get("reward_version")
            for row in run["metrics"]["initial_validation"].get("environments", [])
        ]
        keys.append(
            {
                "complete": all(item is not None for item in required)
                and bool(provenance.get("initial_checkpoint_sha256"))
                and bool(run["metrics"].get("teacher_identity"))
                and bool(provenance.get("source_sha256"))
                and bool(rewards)
                and all(rewards)
                and bool(scenarios)
                and all(seed is not None and bool(scenario) for seed, scenario in scenarios),
                "environment_budget": required,
                "teacher_identity": run["metrics"].get("teacher_identity"),
                "initial_checkpoint": provenance.get("initial_checkpoint_sha256"),
                "source_sha256": provenance.get("source_sha256"),
                "scenarios": run["summary"]["evaluations"]["initial_validation"]["scenarios"],
                "rewards": [
                    row.get("reward_version")
                    for row in run["metrics"]["initial_validation"].get("environments", [])
                ],
                "interaction_steps": cfg.get("steps"),
            }
        )
    matched = len(keys) >= 2 and all(key["complete"] and key == keys[0] for key in keys)
    return {
        "online_runs": len(online),
        "same_recorded_setup": matched,
        "interpretation": "same setup; single seed does not establish generalization"
        if matched
        else "cross-run budgets/provenance differ or are incomplete; inspect each run separately",
    }


def load_predictions(path: Path, references=None):
    metrics = read_json(path / "metrics.json") if (path / "metrics.json").exists() else {}
    scale = metrics.get("residual_scale_mps2")
    rows = list(read_rows(path / "predictions.jsonl"))
    if not rows:
        raise ValueError(f"{path}: empty predictions.jsonl")
    routes, seen = {}, set()
    for row in rows:
        if not isinstance(row.get("timestamp_s"), (float, int)) or not math.isfinite(
            row["timestamp_s"]
        ):
            raise ValueError(f"{path}: prediction timestamp must be finite seconds")
        identity = (row["episode_id"], row["frame"])
        if identity in seen:
            raise ValueError(f"{path}: duplicate episode/frame prediction")
        seen.add(identity)
        if row.get("action_order") != ["a_lat", "a_long"] or row.get("action_units") != "m/s^2":
            raise ValueError(f"{path}: predictions must declare [a_lat,a_long] in m/s^2")
        if references is not None:
            episode = references.get(row["episode_id"])
            if episode is None or row["route_id"] != episode.reference.route_id:
                raise ValueError(f"{path}: prediction episode/route does not match the dataset")
            times = episode.array("timestamps")
            frame = row["frame"]
            if (
                type(frame) is not int
                or not 0 <= frame < len(times)
                or not np.isclose(times[frame], row["timestamp_s"], atol=1e-6, rtol=0)
            ):
                raise ValueError(f"{path}: prediction timestamp/frame does not align with dataset")
        routes.setdefault(row["route_id"], RouteMetrics()).add(row, scale or [float("inf")] * 2)
    per_route = {name: metric.report() for name, metric in routes.items()}
    if scale is None:
        for values in per_route.values():
            values["saturation_fraction"] = None
    return {
        "name": path.name,
        "path": str(path),
        "rows": rows,
        "per_route": per_route,
        "metrics": metrics,
        "gate2": read_json(path / "gate2.json") if (path / "gate2.json").exists() else None,
    }

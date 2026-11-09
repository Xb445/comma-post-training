"""Fixed paired engineering evaluations; model promotion is separate from training."""

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from comma_post_training.engine.checkpoints import (
    atomic_json,
    file_sha256,
    source_fingerprint,
    staged_directory,
)
from comma_post_training.simulator.config import ClosedLoopConfig
from comma_post_training.simulator.metrics import RecoveryCriteria
from comma_post_training.simulator.official_runner import (
    evaluate_official_closed_loop,
    load_official_config,
)
from comma_post_training.simulator.runner import evaluate_closed_loop
from comma_post_training.teacher.runtime import TeacherSession

ROAD_FACTORY = "comma_post_training.simulator.recording:RoadEnvironment"
ENV_OPTIONS = {
    "visual_profile",
    "actuator_alpha",
    "control_delay_steps",
    "initial_lateral_offset_m",
    "initial_heading_error_rad",
    "initial_speed_offset_mps",
    "perturb_at_step",
    "lateral_perturb_m",
    "heading_perturb_rad",
    "speed_perturb_mps",
}
DEFAULT_GATE = {
    "min_seeds_per_scenario": 3,
    "min_recovery_rate": 0.8,
    "controlled_return_tolerance": 0.05,
    "error_relative_tolerance": 0.05,
    "error_absolute_tolerance": 0.02,
    "jerk_relative_tolerance": 0.2,
    "jerk_absolute_tolerance": 0.1,
    "min_relative_improvement": 0.05,
    "min_recovery_rate_improvement": 0.05,
}


def load_suite(path: Path):
    suite = json.loads(path.read_text())
    if not isinstance(suite, dict) or suite.get("format") != "comma-recovery-suite-v1":
        raise ValueError("suite must have format=comma-recovery-suite-v1")
    if set(suite) - {
        "format",
        "name",
        "split",
        "horizon",
        "warmup_steps",
        "recovery",
        "gate",
        "cases",
    }:
        raise ValueError("unknown benchmark suite option")
    if suite.get("split") not in ("val", "test"):
        raise ValueError("benchmark split must be val/test")
    horizon, warmup = suite.get("horizon"), suite.get("warmup_steps")
    if type(horizon) is not int or type(warmup) is not int or not 0 <= warmup < horizon:
        raise ValueError("suite requires integer horizon > warmup_steps >=0")
    RecoveryCriteria(**suite.get("recovery", {}))
    gate = {**DEFAULT_GATE, **suite.get("gate", {})}
    if set(gate) != set(DEFAULT_GATE):
        raise ValueError("unknown benchmark gate threshold")
    for name, value in gate.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"invalid gate threshold {name}")
    if type(gate["min_seeds_per_scenario"]) is not int or gate["min_seeds_per_scenario"] < 3:
        raise ValueError("a paired gate requires at least three seeds per scenario")
    if gate["min_recovery_rate"] > 1 or gate["min_recovery_rate_improvement"] > 1:
        raise ValueError("recovery rate thresholds must be <=1")
    suite["gate"] = gate
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("suite cases must be a nonempty list")
    seen = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) - {
            "id",
            "scenario",
            "group",
            "domain",
            "seed",
            "env_kwargs",
        }:
            raise ValueError("invalid benchmark case")
        if any(
            not isinstance(case.get(key), str) or not case[key]
            for key in ("id", "scenario", "domain")
        ):
            raise ValueError("case id/scenario/domain must be nonempty strings")
        if not case["id"].replace("-", "").replace("_", "").isalnum() or case["id"] in seen:
            raise ValueError("case ids must be unique safe directory names")
        seen.add(case["id"])
        if case.get("group") not in ("nominal", "recovery"):
            raise ValueError("case group must be nominal/recovery")
        if type(case.get("seed")) is not int or not 0 <= case["seed"] < 2**32 - 100001:
            raise ValueError("case seed must be a nonnegative 32-bit seed")
        kwargs = case.get("env_kwargs", {})
        if not isinstance(kwargs, dict) or set(kwargs) - ENV_OPTIONS:
            raise ValueError("unknown RoadEnvironment case option")
        perturb = kwargs.get("perturb_at_step")
        magnitude = any(
            kwargs.get(key, 0)
            for key in ("lateral_perturb_m", "heading_perturb_rad", "speed_perturb_mps")
        )
        if case["group"] == "recovery":
            if not magnitude or type(perturb) is not int or not warmup <= perturb < horizon:
                raise ValueError(
                    "recovery perturbation must be nonzero and occur after expert warmup"
                )
        elif perturb is not None or magnitude:
            raise ValueError("nominal cases must not contain a scheduled disturbance")
    return suite


def _aggregate(cases):
    result = {}
    for group in ("all", "nominal", "recovery"):
        rows = [row for row in cases if group == "all" or row["case"]["group"] == group]
        if not rows:
            continue
        recovered = [
            row["reached_state"]["recovered"] is True
            for row in rows
            if row["case"]["group"] == "recovery"
        ]
        result[group] = {
            "episodes": len(rows),
            "terminations": sum(row["terminated"] for row in rows),
            "mean_controlled_return": float(np.mean([row["controlled_return"] for row in rows])),
            "mean_controlled_steps": float(np.mean([row["controlled_steps"] for row in rows])),
            "recovery_rate": float(np.mean(recovered)) if recovered else None,
            "recovery_cases": len(recovered),
            "recovered_episodes": sum(recovered),
        }
        for key in ("cross_track_mae_m", "heading_mae_rad", "speed_mae_mps", "recovery_time_s"):
            values = [
                row["reached_state"][key] for row in rows if row["reached_state"][key] is not None
            ]
            result[group][key] = float(np.mean(values)) if values else None
    return result


def run_benchmark(
    suite_path: Path,
    output: Path,
    *,
    checkpoint: Path | None = None,
    official_config: Path | None = None,
    official_only=False,
    record_traces=False,
    device=None,
    teacher_factory=None,
):
    suite = load_suite(suite_path)
    if official_only and (not official_config or checkpoint):
        raise ValueError("official_only requires an official config and no candidate checkpoint")
    if not official_config and not checkpoint:
        raise ValueError("provide a Student checkpoint or official config")
    if teacher_factory is not None and not official_config:
        raise ValueError("Teacher fixture injection requires official config")
    teacher, cfg = None, None
    identity = None
    if official_config:
        cfg = load_official_config(official_config)
        if cfg.env_factory != ROAD_FACTORY:
            raise ValueError("v1 benchmark uses the declared RoadEnvironment engineering backend")
        if suite["warmup_steps"] != 25:
            raise ValueError("official benchmark requires the standard 25-frame Teacher warmup")
        teacher = (
            teacher_factory()
            if teacher_factory
            else TeacherSession(
                cfg.teacher_model, provider=cfg.teacher_provider, threads=cfg.teacher_threads
            )
        )
        identity = teacher.cache_identity()
        if checkpoint is None and not official_only and cfg.init_checkpoint is None:
            raise ValueError("candidate official evaluation requires an initialized residual head")
        policy_path = checkpoint or (None if official_only else cfg.init_checkpoint)
    else:
        policy_path = checkpoint
    with staged_directory(output) as staging:
        rows = []
        for case in suite["cases"]:
            kwargs = {**case.get("env_kwargs", {}), "horizon": suite["horizon"]}
            trace = staging / "traces" / case["id"] if record_traces else None
            if cfg:
                case_config = replace(
                    cfg,
                    env_kwargs=kwargs,
                    eval_seed=case["seed"],
                    eval_episodes=1,
                    episode_limit=suite["horizon"],
                    **({"device": device} if device else {}),
                )
                report = evaluate_official_closed_loop(
                    case_config,
                    checkpoint,
                    split=suite["split"],
                    teacher_factory=lambda: teacher,
                    trace_dir=trace,
                    official_only=official_only,
                    metric_options=suite.get("recovery"),
                    allow_warmup_failure=True,
                )
            else:
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                action_settings = {
                    key: tuple(value)
                    for key, value in saved.get("closed_loop_config", {}).items()
                    if key in ("action_low", "action_high", "residual_scale")
                }
                case_config = ClosedLoopConfig(
                    "td3",
                    checkpoint,
                    staging,
                    env_factory=ROAD_FACTORY,
                    env_kwargs=kwargs,
                    eval_seed=case["seed"],
                    eval_episodes=1,
                    episode_limit=suite["horizon"],
                    device=device or "cpu",
                    **action_settings,
                )
                report = evaluate_closed_loop(
                    case_config,
                    checkpoint,
                    split=suite["split"],
                    warmup_steps=suite["warmup_steps"],
                    trace_dir=trace,
                    metric_options=suite.get("recovery"),
                )
                identity = saved["model_spec"].get("distillation_teacher_identity")
            row = {"case": case, **report["episodes"][0]}
            if trace:
                row["trace"] = str(Path("traces") / case["id"])
            rows.append(row)
        report = {
            "format": "comma-policy-benchmark-v1",
            "split": suite["split"],
            "suite": suite,
            "suite_sha256": file_sha256(suite_path),
            "source_sha256": source_fingerprint(),
            "policy": "official"
            if official_only
            else "official_residual"
            if cfg
            else "independent_student",
            "checkpoint_sha256": file_sha256(policy_path) if policy_path else None,
            "teacher_identity": identity,
            "runtime_teacher_loaded": teacher is not None,
            "engineering_backend": True,
            "eligible_for_real_driving_claim": False,
            "reward_version": "road_tracking_speed_jerk_v1",
            "evaluation_semantics": "5hz_expert_warmup_then_policy_reached_state_v1",
            "action_low": list(case_config.action_low),
            "action_high": list(case_config.action_high),
            "cases": rows,
            "summary": _aggregate(rows),
        }
        atomic_json(staging / "metrics.json", report)
        with (staging / "cases.csv").open("w", newline="") as stream:
            fields = [
                "id",
                "scenario",
                "domain",
                "seed",
                "return",
                "controlled_return",
                "controlled_steps",
                "terminated",
                "cross_track_mae_m",
                "speed_mae_mps",
                "recovered",
                "recovery_time_s",
            ]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                flat = {**row, **row["case"], **row["reached_state"]}
                writer.writerow({key: flat.get(key) for key in fields})
    return report


def compare_benchmarks(baseline: Path, candidate: Path, output: Path):
    left, right = (json.loads(path.read_text()) for path in (baseline, candidate))
    for report in (left, right):
        if report.get("format") != "comma-policy-benchmark-v1" or report.get("split") != "val":
            raise ValueError(
                "candidate selection requires held-out val benchmarks; test is final reporting only"
            )
    for key in (
        "suite_sha256",
        "suite",
        "source_sha256",
        "reward_version",
        "evaluation_semantics",
        "engineering_backend",
        "action_low",
        "action_high",
    ):
        if left.get(key) != right.get(key):
            raise ValueError(f"paired comparison requires identical {key}")
    if left.get("teacher_identity") != right.get("teacher_identity"):
        raise ValueError("paired comparison requires identical Teacher provenance")
    a, b = ({row["case"]["id"]: row for row in report["cases"]} for report in (left, right))
    expected_ids = {case["id"] for case in left["suite"]["cases"]}
    if (
        set(a) != expected_ids
        or set(b) != expected_ids
        or len(a) != len(left["cases"])
        or len(b) != len(right["cases"])
    ):
        raise ValueError("paired comparison requires one result per declared suite case")
    gate = left["suite"]["gate"]
    cohorts = {}
    checks = []
    for key in sorted(a):
        if (
            a[key]["case"] != b[key]["case"]
            or a[key]["seed"] != b[key]["seed"]
            or a[key].get("scenario_id") != b[key].get("scenario_id")
        ):
            raise ValueError("paired cases/scenario seeds differ")
        cohorts.setdefault((a[key]["case"]["scenario"], a[key]["case"]["domain"]), []).append(key)
        for row in (a[key], b[key]):
            if (
                not row["controlled_steps"]
                or row["reached_state"]["diagnostic_steps"] != row["controlled_steps"]
            ):
                checks.append((f"{key}:complete_controlled_diagnostics", False))
            for name in ("controlled_return", "cross_track_mae_m", "speed_mae_mps"):
                value = row.get(name, row["reached_state"].get(name))
                if value is None:
                    checks.append((f"{key}:{name}_available", False))
                elif not np.isfinite(value):
                    raise ValueError("paired metrics must be finite")
    for cohort, ids in cohorts.items():
        checks.append(
            (
                f"{cohort}:multi_seed",
                len({a[key]["seed"] for key in ids}) >= gate["min_seeds_per_scenario"],
            )
        )
        checks.append(
            (
                f"{cohort}:no_extra_terminals",
                sum(b[key]["terminated"] for key in ids)
                <= sum(a[key]["terminated"] for key in ids),
            )
        )
        old = np.mean([a[key]["controlled_return"] for key in ids])
        new = np.mean([b[key]["controlled_return"] for key in ids])
        checks.append(
            (f"{cohort}:controlled_return", new >= old - gate["controlled_return_tolerance"])
        )
        for metric in ("cross_track_mae_m", "speed_mae_mps"):
            if any(
                a[key]["reached_state"][metric] is None or b[key]["reached_state"][metric] is None
                for key in ids
            ):
                checks.append((f"{cohort}:{metric}_available", False))
                continue
            old = np.mean([a[key]["reached_state"][metric] for key in ids])
            new = np.mean([b[key]["reached_state"][metric] for key in ids])
            checks.append(
                (
                    f"{cohort}:{metric}",
                    new
                    <= old * (1 + gate["error_relative_tolerance"])
                    + gate["error_absolute_tolerance"],
                )
            )
        old_jerk = [a[key]["reached_state"]["achieved_jerk_rms_mps3"] for key in ids]
        new_jerk = [b[key]["reached_state"]["achieved_jerk_rms_mps3"] for key in ids]
        if any(value is None for value in old_jerk + new_jerk):
            checks.append((f"{cohort}:jerk_available", False))
        else:
            checks.append(
                (
                    f"{cohort}:jerk",
                    bool(
                        (
                            np.mean(new_jerk, axis=0)
                            <= np.mean(old_jerk, axis=0) * (1 + gate["jerk_relative_tolerance"])
                            + gate["jerk_absolute_tolerance"]
                        ).all()
                    ),
                )
            )
    recovery_ids = [key for key in a if a[key]["case"]["group"] == "recovery"]
    nominal_ids = [key for key in a if a[key]["case"]["group"] == "nominal"]
    checks.append(("contains_nominal_and_recovery", bool(recovery_ids and nominal_ids)))
    improvement = False
    if recovery_ids:
        for key in recovery_ids:
            checks.append(
                (
                    f"{key}:disturbance_observed",
                    bool(
                        a[key]["reached_state"]["perturbation_observed"]
                        and b[key]["reached_state"]["perturbation_observed"]
                    ),
                )
            )
        old_rate = np.mean([a[key]["reached_state"]["recovered"] is True for key in recovery_ids])
        new_rate = np.mean([b[key]["reached_state"]["recovered"] is True for key in recovery_ids])
        checks.append(("recovery_rate", bool(new_rate >= max(old_rate, gate["min_recovery_rate"]))))
        old_errors = [a[key]["reached_state"]["cross_track_mae_m"] for key in recovery_ids]
        new_errors = [b[key]["reached_state"]["cross_track_mae_m"] for key in recovery_ids]
        old_error = float(np.mean(old_errors)) if None not in old_errors else 0.0
        new_error = float(np.mean(new_errors)) if None not in new_errors else float("inf")
        improvement = bool(
            new_rate > old_rate
            and new_rate - old_rate >= gate["min_recovery_rate_improvement"]
            or old_error > 1e-6
            and new_error < old_error
            and new_error <= old_error * (1 - gate["min_relative_improvement"])
        )
    checks.append(("measured_recovery_improvement", improvement))
    checks = [(name, bool(value)) for name, value in checks]
    report = {
        "format": "comma-paired-promotion-v1",
        "split": "val",
        "passed": all(value for _, value in checks),
        "checks": dict(checks),
        "failed_checks": [name for name, value in checks if not value],
        "baseline_report_sha256": file_sha256(baseline),
        "candidate_report_sha256": file_sha256(candidate),
        "candidate_checkpoint_sha256": right["checkpoint_sha256"],
        "thresholds": gate,
        "engineering_backend": right["engineering_backend"],
        "weights_modified": False,
        "note": (
            "candidate selection on a synthetic engineering benchmark "
            "is not a real-driving acceptance"
        ),
    }
    if output.exists():
        raise ValueError("promotion report already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    return report

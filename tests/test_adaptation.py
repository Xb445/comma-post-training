"""Adaptation contracts and end-to-end CPU evidence; synthetic Teacher is never official."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from test_official_teacher import make_fixture

from comma_post_training.data_pipeline.audit import audit_dataset
from comma_post_training.data_pipeline.combine import combine_datasets
from comma_post_training.data_pipeline.episodes import Catalogue
from comma_post_training.data_pipeline.windows import EpisodeWindows
from comma_post_training.driving_policy.predict import predict_policy
from comma_post_training.engine.checkpoints import load_checkpoint
from comma_post_training.engine.config import (
    DataConfig,
    LossConfig,
    ModelConfig,
    RunConfig,
    TrainingConfig,
)
from comma_post_training.engine.trainer import evaluate, train
from comma_post_training.reporting.report import build_report
from comma_post_training.simulator.api import Transition, check_observation
from comma_post_training.simulator.benchmark import compare_benchmarks, load_suite, run_benchmark
from comma_post_training.simulator.config import ClosedLoopConfig
from comma_post_training.simulator.metrics import EpisodeMetrics
from comma_post_training.simulator.recorder import record_simulation
from comma_post_training.simulator.recording import RoadEnvironment, recording_spec
from comma_post_training.simulator.runner import evaluate_closed_loop, train_closed_loop
from comma_post_training.simulator.td3 import TD3
from comma_post_training.teacher import runtime
from comma_post_training.teacher.cache import cache_teacher
from comma_post_training.teacher.evaluation import RouteMetrics
from comma_post_training.teacher.runtime import TeacherSession


@pytest.fixture(scope="module")
def adaptation(tmp_path_factory):
    root = tmp_path_factory.mktemp("adaptation")
    graph = root / "fixture.onnx"
    identity = make_fixture(graph, official_layout=True)

    def teacher():
        return TeacherSession(graph, identity=identity, threads=2)

    record_simulation(root / "source", duration_s=6.4)
    record_simulation(
        root / "target",
        duration_s=6.4,
        seed=1042,
        env_kwargs={"visual_profile": "dusk", "control_delay_steps": 1},
    )
    source = root / "source/manifest.json"
    target = root / "target/manifest.json"
    cache_teacher(source, teacher(), root / "source-teacher")
    cfg = RunConfig(
        "policy",
        DataConfig(root / "source-teacher/manifest.json", history=4, image_size=(16, 32)),
        TrainingConfig(root / "student-source", max_steps=4, batch_size=4, val_every=2),
        ModelConfig(hidden_size=16, vision_channels=4, vision_pool=(2, 4)),
        LossConfig(bc_weight=1, kd_weight=1),
    )
    train(cfg)
    target_cfg = replace(
        cfg,
        data=replace(cfg.data, manifest=target),
        model=replace(cfg.model, init_checkpoint=root / "student-source/last.pt"),
        training=replace(cfg.training, output_dir=root / "student-target", max_steps=2),
        loss=LossConfig(bc_weight=1, kd_weight=0),
    )
    trained = train(target_cfg)
    return {
        "root": root,
        "source": source,
        "target": target,
        "cfg": cfg,
        "target_cfg": target_cfg,
        "trained": trained,
        "teacher": teacher,
        "graph": graph,
    }


def test_spatial_student_sft_updates_visual_weights_and_preserves_teacher_origin(
    adaptation, tmp_path
):
    root = adaptation["root"]
    before = load_checkpoint(root / "student-source/last.pt")
    after = load_checkpoint(root / "student-target/last.pt")
    assert before["model_spec"]["vision_pool"] == [2, 4]
    assert after["model_spec"]["distillation_teacher_identity"]["kind"] == "synthetic_fixture"
    assert (
        before["model_spec"]["distillation_teacher_identity"]
        == after["model_spec"]["distillation_teacher_identity"]
    )
    assert any(
        not torch.equal(value, after["model_state"][key])
        for key, value in before["model_state"].items()
        if key.startswith("encoder.")
    )
    assert adaptation["trained"]["visual_encoder_trainable"]
    assert not adaptation["trained"]["teacher_dependency_at_inference"]
    assert np.isfinite(evaluate(root / "student-target/last.pt", split="val")["loss"])
    cfg = adaptation["cfg"]
    with pytest.raises(ValueError, match="verified official Teacher"):
        train(
            replace(
                cfg,
                data=replace(cfg.data, require_official_teacher=True),
                training=replace(cfg.training, output_dir=tmp_path / "reject"),
            )
        )
    with pytest.raises(ValueError, match="architecture"):
        train(
            replace(
                adaptation["target_cfg"],
                model=replace(adaptation["target_cfg"].model, vision_pool=(1, 1)),
                training=replace(cfg.training, output_dir=tmp_path / "wrong-pool"),
            )
        )


def test_independent_student_predicts_unlabeled_inputs_and_pairs_teacher_metrics(
    adaptation, tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Student inference loaded a Teacher")

    monkeypatch.setattr(runtime, "TeacherSession", forbidden)
    root = adaptation["root"]
    checkpoint = root / "student-target/last.pt"
    report = predict_policy(
        root / "source-teacher/manifest.json", checkpoint, tmp_path / "predictions"
    )
    assert not report["runtime_teacher_loaded"]
    values = next(iter(report["per_route"].values()))
    assert values["expert_frames"] == 29 and values["teacher_expert_frames"] == 7
    rows = [
        json.loads(line)
        for line in (tmp_path / "predictions/predictions.jsonl").read_text().splitlines()
    ]
    assert "teacher_action" not in rows[0] and "teacher_action" in rows[-1]
    assert values["paired_adapted_mae_mps2"] is not None
    # Inference windows do not require labels; the training view still rejects them.
    catalogue = Catalogue(adaptation["target"])
    ref = next(ref for ref in catalogue.episodes if ref.split == "val")
    meta = json.loads(ref.metadata.read_text())
    meta["arrays"].pop("expert_actions")
    meta["arrays"].pop("expert_valid")
    for key, value in meta["arrays"].items():
        meta["arrays"][key] = str(ref.metadata.parent / value)
    alternate = tmp_path / "unlabeled.json"
    alternate.write_text(json.dumps(meta))
    document = json.loads(adaptation["target"].read_text())
    document["episodes"] = [
        {
            **{key: getattr(ref, key) for key in ("id", "route_id", "domain", "split")},
            "metadata": str(alternate),
        }
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(document))
    prediction = predict_policy(manifest, checkpoint, tmp_path / "unlabeled-predictions")
    assert next(iter(prediction["per_route"].values()))["adapted_mae_mps2"] is None
    with pytest.raises(ValueError, match="no usable"):
        EpisodeWindows(
            Catalogue(manifest),
            split="val",
            task="policy",
            history=4,
            image_size=(16, 32),
            supervision=("expert",),
        )
    visual = build_report(
        tmp_path / "report",
        manifest=root / "source-teacher/manifest.json",
        prediction_dirs=[tmp_path / "predictions"],
        max_episodes=1,
        max_frames=2,
    )
    assert visual["figures"] > 0


def test_road_domain_delay_history_and_post_warmup_perturbation():
    spec = {**recording_spec(), "kind": "policy", "history": 4, "image_size": [16, 32]}
    clear = RoadEnvironment(spec=spec, split="val", horizon=40)
    changed = RoadEnvironment(
        spec=spec,
        split="val",
        horizon=40,
        visual_profile="dusk",
        control_delay_steps=1,
        perturb_at_step=25,
        lateral_perturb_m=0.8,
    )
    first = check_observation(clear.reset(seed=99), spec)
    target = check_observation(changed.reset(seed=99), spec)
    torch.testing.assert_close(first["state"], target["state"])
    assert not torch.equal(first["images"], target["images"])
    assert first["images"].shape == (4, 2, 3, 16, 32)
    assert first["images"][:3].count_nonzero() == 0
    initial = changed.step(np.array([1.0, 0.0]))
    np.testing.assert_array_equal(initial.achieved_action, [0, 0])
    assert not changed.diagnostics()["perturbation_applied"]
    for _ in range(23):
        changed.step(changed.expert_action())
    assert not changed.diagnostics()["perturbation_applied"]
    previous = changed.diagnostics()["cross_track_m"]
    disturbed = changed.step(changed.expert_action())
    assert changed.diagnostics()["perturbation_applied"]
    assert changed.diagnostics()["cross_track_m"] - previous > 0.7
    assert disturbed.observation["dt"][0].item() == 0
    replay = changed.reset(seed=99)
    torch.testing.assert_close(replay["images"], target["images"], rtol=0, atol=0)
    with pytest.raises(ValueError, match="requires perturb_at_step"):
        RoadEnvironment(spec=spec, split="val", lateral_perturb_m=1)


def test_recovery_requires_a_post_disturbance_hold_and_no_later_terminal():
    metrics = EpisodeMetrics({"hold_steps": 2})
    transition = Transition({"dt": torch.tensor([[0.2]])}, 0.0, False, False, np.zeros(2))

    def add(error, controlled=True):
        metrics.add(
            transition,
            controlled=controlled,
            diagnostics={
                "cross_track_m": error,
                "heading_error_rad": 0.0,
                "speed_error_mps": 0.0,
                "perturbation_applied": True,
            },
        )

    add(0.8, controlled=False)
    add(0.1)
    assert metrics.report()["recovered"] is False
    add(0.1)
    assert metrics.report()["recovered"] is True
    assert metrics.report()["recovery_time_s"] == pytest.approx(0.4)
    transition.terminated = True
    add(0.1)
    assert metrics.report()["recovered"] is False
    missing = EpisodeMetrics()
    missing.add(transition, controlled=True)
    assert missing.report()["cross_track_mae_m"] is None
    assert missing.report()["recovered"] is None


def test_td3_critic_warmup_expert_guard_and_regularizers(tmp_path):
    cfg = ClosedLoopConfig(
        "td3",
        tmp_path / "base",
        tmp_path / "run",
        steps=20,
        batch_size=2,
        warmup_steps=2,
        actor_warmup_updates=2,
        actor_bc_weight=1.0,
        actor_smoothness_weight=0.01,
    )
    torch.manual_seed(1)
    agent = TD3(4, cfg, torch.device("cpu"))
    batch = {
        "features": torch.randn(2, 4),
        "base": torch.zeros(2, 2),
        "action": torch.zeros(2, 2),
        "next_features": torch.randn(2, 4),
        "next_base": torch.zeros(2, 2),
        "reward": torch.ones(2),
        "terminated": torch.zeros(2, dtype=torch.bool),
    }
    with pytest.raises(ValueError, match="expert labels"):
        agent.update(batch)
    batch.update(
        expert=torch.ones(2, 2) * 0.2,
        previous_action=torch.zeros(2, 2),
        interval_s=torch.ones(2) * 0.2,
    )
    before = copy.deepcopy(agent.actor.state_dict())
    for _ in range(3):
        assert agent.update(batch)["actor_updated"] is False
    for key, value in before.items():
        torch.testing.assert_close(agent.actor.state_dict()[key], value, rtol=0, atol=0)
    report = agent.update(batch)
    assert report["actor_updated"] and report["actor_updates"] == 1
    assert report["actor_bc_loss"] > 0 and report["actor_smoothness_loss"] == 0
    assert agent.state_dict()["actor_updates"] == 1
    batch["interval_s"].zero_()
    with pytest.raises(ValueError, match="positive interval"):
        agent.update(batch)


def write_suite(path, *, split="val", seeds=(1, 2, 3), horizon=35):
    cases = []
    for seed in seeds:
        for scenario in ("nominal", "lateral"):
            cases.append(
                {
                    "id": f"{scenario}-{seed}",
                    "scenario": scenario,
                    "domain": "target",
                    "group": "nominal" if scenario == "nominal" else "recovery",
                    "seed": seed,
                    "env_kwargs": {}
                    if scenario == "nominal"
                    else {"perturb_at_step": 25, "lateral_perturb_m": 0.8},
                }
            )
    path.write_text(
        json.dumps(
            {
                "format": "comma-recovery-suite-v1",
                "name": "test-engineering",
                "split": split,
                "horizon": horizon,
                "warmup_steps": 25,
                "cases": cases,
            }
        )
    )
    return path


def test_student_benchmark_traces_and_paired_selection_reject_false_success(adaptation, tmp_path):
    suite = write_suite(tmp_path / "suite.json")
    root = adaptation["root"]
    report = run_benchmark(
        suite,
        tmp_path / "benchmark",
        checkpoint=root / "student-target/last.pt",
        record_traces=True,
    )
    assert len(report["cases"]) == 6 and not report["runtime_teacher_loaded"]
    assert report["cases"][0]["controlled_steps"] == 10
    assert all(
        row["reached_state"]["perturbation_observed"]
        for row in report["cases"]
        if row["case"]["group"] == "recovery"
    )
    assert (tmp_path / "benchmark/cases.csv").is_file()
    trace = tmp_path / "benchmark/traces/nominal-1"
    document = json.loads((trace / "trace.json").read_text())
    steps = json.loads((trace / document["episodes"][0]["steps"]).read_text())["steps"]
    assert all(row["official_action"] is None for row in steps)
    promotion = compare_benchmarks(
        tmp_path / "benchmark/metrics.json",
        tmp_path / "benchmark/metrics.json",
        tmp_path / "gate.json",
    )
    assert not promotion["passed"]
    assert "measured_recovery_improvement" in promotion["failed_checks"]
    assert not promotion["weights_modified"]
    altered = copy.deepcopy(report)
    altered["split"] = "test"
    (tmp_path / "altered.json").write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="held-out val"):
        compare_benchmarks(
            tmp_path / "benchmark/metrics.json",
            tmp_path / "altered.json",
            tmp_path / "test-gate.json",
        )
    visual = build_report(
        tmp_path / "report",
        benchmark_dirs=[tmp_path / "benchmark"],
        trace_dirs=[trace],
        max_episodes=1,
        max_frames=2,
    )
    assert visual["viewer_episodes"] == 1 and visual["figures"] >= 2
    broken = json.loads(suite.read_text())
    broken["cases"][1]["env_kwargs"]["perturb_at_step"] = 2
    suite.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="after expert warmup"):
        load_suite(suite)


@pytest.mark.parametrize("algorithm", ["dagger", "td3"])
def test_spatial_student_online_uses_road_adapter_and_reports_actor_budget(
    adaptation, tmp_path, algorithm
):
    cfg = ClosedLoopConfig(
        algorithm,
        adaptation["root"] / "student-target/last.pt",
        tmp_path / algorithm,
        env_factory="comma_post_training.simulator.recording:RoadEnvironment",
        env_kwargs={"horizon": 35, "visual_profile": "dusk"},
        steps=8,
        batch_size=2,
        warmup_steps=2,
        eval_episodes=1,
        episode_limit=35,
        actor_warmup_updates=2,
        actor_bc_weight=1.0,
        actor_smoothness_weight=0.01,
    )
    report = train_closed_loop(cfg)
    assert report["updates"] > 0
    assert report["actor_updates"] > 0 if algorithm == "td3" else report["actor_updates"] == 0
    before = load_checkpoint(cfg.policy_checkpoint)
    exported = torch.load(cfg.output_dir / "final.pt", weights_only=True)
    for key in before["model_state"]:
        if algorithm == "td3" or key.startswith("encoder."):
            torch.testing.assert_close(
                before["model_state"][key], exported["model_state"][key], rtol=0, atol=0
            )
    evaluation = evaluate_closed_loop(
        cfg, cfg.output_dir / "final.pt", split="val", warmup_steps=25
    )
    assert evaluation["episodes"][0]["controlled_steps"] == 10
    assert report["expert_queries_train"] == cfg.steps
    visual = build_report(tmp_path / (algorithm + "-report"), run_dirs=[cfg.output_dir])
    summary = json.loads(Path(visual["summary"]).read_text())
    assert set(summary["runs"][0]["evaluations"]) == {"initial_validation", "final_validation"}
    assert visual["figures"] >= 3


def test_dataset_audit_and_source_target_composition(adaptation, tmp_path):
    assert audit_dataset(adaptation["target"], stage="student")["ready"]
    world = audit_dataset(adaptation["target"], stage="world", output=tmp_path / "world.json")
    assert world["ready"] and not world["official_4b_training_supported"]
    synthetic = audit_dataset(
        adaptation["root"] / "source-teacher/manifest.json", require_official_teacher=True
    )
    assert not synthetic["ready"]
    combined = combine_datasets(
        [adaptation["source"], adaptation["target"]],
        tmp_path / "combined",
        revision="source-target-v1",
    )
    assert combined["quality"]["episodes"] == 8
    assert not combined["source_files_modified"]
    with pytest.raises(ValueError, match="duplicate episode"):
        combine_datasets(
            [adaptation["source"], adaptation["source"]],
            tmp_path / "duplicate",
            revision="rejected",
        )
    assert not (tmp_path / "duplicate").exists()
    with pytest.raises(ValueError, match="contracts must match"):
        combine_datasets(
            [adaptation["target"], adaptation["root"] / "source-teacher/manifest.json"],
            tmp_path / "mixed",
            revision="reject",
        )


def test_metric_pairing_does_not_compare_full_student_to_partial_teacher():
    metrics = RouteMetrics()
    for i in range(3):
        metrics.add(
            {
                "episode_id": "a",
                "frame": i,
                "timestamp_s": 0.2 * i,
                "adapted_action": [i, i],
                "expert_action": [0.0, 0.0],
                **({"teacher_action": [4.0, 4.0]} if i == 2 else {}),
            },
            [float("inf")] * 2,
        )
    values = metrics.report()
    assert values["adapted_mae_mps2"] == [1.0, 1.0]
    assert values["paired_adapted_mae_mps2"] == [2.0, 2.0]
    assert values["teacher_mae_mps2"] == [4.0, 4.0]


def test_promotion_gate_positive_fixture_and_regressions(tmp_path):
    # Handcrafted report fixture tests gate arithmetic; these are not measured driving results.
    suite = load_suite(write_suite(tmp_path / "suite.json"))
    rows = []
    for case in suite["cases"]:
        rows.append(
            {
                "case": case,
                "seed": case["seed"],
                "scenario_id": case["id"],
                "controlled_steps": 10,
                "controlled_return": -1.0,
                "terminated": False,
                "reached_state": {
                    "diagnostic_steps": 10,
                    "cross_track_mae_m": 0.2,
                    "speed_mae_mps": 0.1,
                    "achieved_jerk_rms_mps3": [0.1, 0.1],
                    "perturbation_observed": case["group"] == "recovery",
                    "recovered": True if case["group"] == "recovery" else None,
                },
            }
        )
    base = {
        "format": "comma-policy-benchmark-v1",
        "split": "val",
        "suite": suite,
        "suite_sha256": "fixture",
        "source_sha256": "fixture",
        "reward_version": "fixture",
        "evaluation_semantics": "fixture",
        "engineering_backend": True,
        "teacher_identity": None,
        "checkpoint_sha256": "baseline",
        "cases": rows,
    }
    candidate = copy.deepcopy(base)
    candidate["checkpoint_sha256"] = "candidate"
    for row in candidate["cases"]:
        if row["case"]["group"] == "recovery":
            row["reached_state"]["cross_track_mae_m"] *= 0.8
    left, right = tmp_path / "baseline.json", tmp_path / "candidate.json"
    left.write_text(json.dumps(base))
    right.write_text(json.dumps(candidate))
    assert compare_benchmarks(left, right, tmp_path / "pass.json")["passed"]
    candidate["cases"][1]["controlled_return"] = -50.0
    right.write_text(json.dumps(candidate))
    failed = compare_benchmarks(left, right, tmp_path / "fail.json")
    assert not failed["passed"] and any(
        "controlled_return" in name for name in failed["failed_checks"]
    )
    candidate["cases"].pop()
    right.write_text(json.dumps(candidate))
    with pytest.raises(ValueError, match="per declared suite case"):
        compare_benchmarks(left, right, tmp_path / "missing.json")


def test_world_audit_reports_missing_geometry_and_action_provenance(tmp_path):
    from comma_post_training.data_pipeline.demo import create_demo

    manifest = create_demo(tmp_path / "demo")
    report = audit_dataset(manifest, stage="world")
    assert not report["ready"]
    assert any("calibration" in issue for issue in report["issues"])
    assert any("action_provenance" in issue for issue in report["issues"])


def test_target_sft_can_resume_after_retaining_distillation_identity(adaptation, tmp_path):
    cfg = replace(
        adaptation["target_cfg"],
        training=replace(
            adaptation["target_cfg"].training, output_dir=tmp_path / "resumed", max_steps=4
        ),
    )
    report = train(cfg, resume=adaptation["root"] / "student-target/last.pt")
    assert report["steps"] == 4
    assert report["teacher_identity"]["kind"] == "synthetic_fixture"


def test_cli_audit_returns_failure_but_keeps_reviewable_json(tmp_path, monkeypatch):
    from comma_post_training.data_pipeline.demo import create_demo
    from comma_post_training.driving.cli import main

    manifest = create_demo(tmp_path / "data")
    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "comma-driving",
            "audit-dataset",
            "--manifest",
            str(manifest),
            "--stage",
            "world",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    assert json.loads(output.read_text())["ready"] is False


def test_adaptation_presets_have_valid_contracts():
    from comma_post_training.engine.config import load_config
    from comma_post_training.simulator.config import load_closed_loop_config
    from comma_post_training.simulator.official_runner import load_official_config

    directory = Path(__file__).resolve().parents[1] / "configs/adaptation"
    for name in ("student-distill", "student-sft", "official-residual"):
        config = load_config(directory / (name + ".toml"))
        if name == "student-distill":
            assert config.data.require_official_teacher and config.loss.kd_weight > 0
    for algorithm in ("dagger", "td3"):
        student = load_closed_loop_config(directory / f"student-{algorithm}.toml")
        official = load_official_config(directory / f"official-{algorithm}.toml")
        assert student.algorithm == official.algorithm == algorithm
        if algorithm == "td3":
            assert student.actor_warmup_updates == official.actor_warmup_updates == 500
    for split in ("val", "test"):
        suite = load_suite(directory / f"recovery-{split}.json")
        assert suite["split"] == split and len(suite["cases"]) == 15

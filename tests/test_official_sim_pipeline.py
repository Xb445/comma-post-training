"""Official-shaped synthetic ONNX integration; never substitutes for GPU/road validation."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from test_official_teacher import make_fixture

from comma_post_training.data_pipeline.episodes import Catalogue
from comma_post_training.engine.checkpoints import file_sha256, load_checkpoint
from comma_post_training.engine.config import DataConfig, ModelConfig, RunConfig, TrainingConfig
from comma_post_training.engine.trainer import train
from comma_post_training.simulator import official_runner
from comma_post_training.simulator.official_runner import (
    OfficialConfig,
    evaluate_official_closed_loop,
    load_official_config,
    train_official_closed_loop,
)
from comma_post_training.simulator.recorder import record_simulation
from comma_post_training.simulator.recording import RoadEnvironment, teacher_rgb_profile
from comma_post_training.teacher.cache import cache_teacher
from comma_post_training.teacher.evaluation import RouteMetrics, gate_residual
from comma_post_training.teacher.policy import OfficialTeacherPolicy
from comma_post_training.teacher.predict import predict_adapted
from comma_post_training.teacher.runtime import TeacherSession


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp("official-sim")
    graph = root / "synthetic.onnx"
    identity = make_fixture(graph, official_layout=True)

    def factory():
        return TeacherSession(graph, identity=identity, threads=2)

    record_simulation(root / "recorded", duration_s=6.4)
    manifest = root / "recorded/manifest.json"
    catalogue = Catalogue(manifest)
    source = {ref.metadata: file_sha256(ref.metadata) for ref in catalogue.episodes}
    asset_hash = file_sha256(graph)
    cached_report = cache_teacher(manifest, factory(), root / "cache")
    config = RunConfig(
        "official_residual",
        DataConfig(root / "cache/manifest.json", history=1, image_size=(256, 512)),
        TrainingConfig(root / "sft", max_steps=6, batch_size=4, val_every=3),
        ModelConfig(hidden_size=16),
    )
    trained = train(config)
    checkpoint = root / "sft/best.pt"
    return {
        "root": root,
        "graph": graph,
        "factory": factory,
        "manifest": manifest,
        "cache": cached_report,
        "source": source,
        "asset_hash": asset_hash,
        "trained": trained,
        "checkpoint": checkpoint,
        "spec": load_checkpoint(checkpoint)["model_spec"],
    }


def test_simulator_to_cache_to_frozen_residual_to_per_route_evaluation(pipeline, tmp_path):
    data = pipeline
    assert data["cache"]["teacher_identity"]["kind"] == "synthetic_fixture"
    assert set(data["cache"]["valid_frames"].values()) == {7}
    assert all(file_sha256(path) == digest for path, digest in data["source"].items())
    assert file_sha256(data["graph"]) == data["asset_hash"]
    cached = Catalogue(data["root"] / "cache/manifest.json")
    assert data["asset_hash"][:16] in cached.revision
    saved = load_checkpoint(data["checkpoint"])
    assert all(key == "residual_scale" or key.startswith("head.") for key in saved["model_state"])
    assert np.isfinite(data["trained"]["final_validation"]["loss"])
    for split in ("val", "test"):
        report = predict_adapted(
            data["manifest"], data["checkpoint"], data["factory"](), tmp_path / split, split=split
        )
        assert report["frames"] == report["expert_frames"] == 7
        assert len(report["per_route"]) == 1
        for route in report["per_route"].values():
            assert len(route["teacher_mae_mps2"]) == len(route["adapted_mae_mps2"]) == 2
            assert np.isfinite(route["adapted_jerk_rms_mps3"]).all()
        if split == "val":
            gate = gate_residual(report)
            assert gate["checkpoint_sha256"] == file_sha256(data["checkpoint"])
            assert gate["dataset_kind"] == "synthetic"
        else:
            with pytest.raises(ValueError, match="validation"):
                gate_residual(report)


def test_online_context_is_once_per_observation_and_resets_every_episode(pipeline):
    spec = pipeline["spec"]
    session = pipeline["factory"]()
    policy = OfficialTeacherPolicy(session, spec, teacher_rgb=teacher_rgb_profile())
    env = RoadEnvironment(spec=spec, split="train")
    obs = env.reset(seed=31)
    policy.reset()
    first = policy.observe(obs, observation_id=0)
    assert policy.observe(obs, observation_id=0) is first
    for _ in range(3):
        torch.testing.assert_close(policy.act(), first.official_action, rtol=0, atol=0)
    assert session.steps == policy.teacher_calls == 1
    assert first.features.shape == (1, 516)
    assert not first.valid
    with pytest.raises(ValueError, match="sequence"):
        policy.observe(obs, observation_id=2)
    for i in range(1, 27):
        obs = env.step(env.expert_action()).observation
        context = policy.observe(obs, observation_id=i)
        assert context.valid == (i >= 25)
    assert policy.teacher_calls == 27
    with torch.no_grad():
        policy.model.head[-1].bias.fill_(100)
        delta = policy.act(context) - context.official_action
    assert (delta.abs() <= 0.5 + 1e-6).all()
    env.reset(seed=31)
    policy.reset()
    replay = policy.observe(env.reset(seed=31), observation_id=0)
    torch.testing.assert_close(replay.official_action, first.official_action, rtol=0, atol=0)
    assert not replay.valid
    obs = env.step(env.expert_action()).observation
    obs["dt"].fill_(0.05)
    with pytest.raises(ValueError, match="5 Hz"):
        policy.observe(obs, observation_id=1)


def online_config(pipeline, output, algorithm="dagger"):
    return OfficialConfig(
        algorithm,
        pipeline["graph"],
        output,
        init_checkpoint=pipeline["checkpoint"],
        offline_manifest=pipeline["root"] / "cache/manifest.json",
        engineering_only=True,
        steps=70,
        episode_limit=34,
        env_kwargs={"horizon": 34},
        eval_episodes=1,
        batch_size=2,
        warmup_steps=2,
        capacity=8,
        offline_fraction=0.5,
    )


@pytest.mark.parametrize("algorithm", ["dagger", "td3"])
def test_live_learning_provenance_reset_and_checkpoint_reload(pipeline, tmp_path, algorithm):
    config = online_config(pipeline, tmp_path / algorithm, algorithm)
    report = train_official_closed_loop(config, teacher_factory=pipeline["factory"])
    assert report["updates"] > 0 and report["replay_size"] == config.capacity
    assert report["gate2"]["bypassed"] and not report["gate2"]["passed"]
    events = [
        json.loads(row) for row in (config.output_dir / "events.jsonl").read_text().splitlines()
    ]
    for episode in {row["episode"] for row in events}:
        rows = [row for row in events if row["episode"] == episode]
        assert [row["teacher_frame"] for row in rows] == list(range(len(rows)))
        assert all(not row["teacher_warmed"] for row in rows[:25])
    assert any(row["commanded_action"] != row["achieved_action"] for row in events)
    final = torch.load(config.output_dir / "final.pt", weights_only=True)
    learning = torch.load(config.output_dir / "learning_state.pt", weights_only=True)
    replay = learning["replay"]
    by_step = {row["step"]: row for row in events}
    for i, step in enumerate(replay["env_step"].tolist()):
        row = by_step[step]
        assert replay["behavior_policy_update"][i] == row["behavior_policy_update"]
        if algorithm == "dagger":
            np.testing.assert_allclose(replay["label"][i], row["expert_action"])
        else:
            assert replay["features"].shape[1] == 516
            np.testing.assert_allclose(replay["action"][i], row["commanded_action"])
            if step + 1 in by_step and by_step[step + 1]["episode"] == row["episode"]:
                np.testing.assert_allclose(
                    replay["next_base"][i], by_step[step + 1]["anchor_action"]
                )
    if algorithm == "td3":
        initial = load_checkpoint(pipeline["checkpoint"])
        for key, value in initial["model_state"].items():
            torch.testing.assert_close(final["model_state"][key], value, rtol=0, atol=0)
        assert "td3_actor_state" in final
        assert any(row.get("actor_updated") for row in events)
        assert replay["truncated"].any() and not replay["terminated"].all()
    evaluated = evaluate_official_closed_loop(
        config, config.output_dir / "final.pt", split="val", teacher_factory=pipeline["factory"]
    )
    assert evaluated["episodes"] == report["final_validation"]["episodes"]
    assert file_sha256(pipeline["graph"]) == pipeline["asset_hash"]


def test_gate_rejects_missing_failed_or_different_checkpoint(pipeline, tmp_path):
    config = replace(online_config(pipeline, tmp_path / "no-gate"), engineering_only=False)
    with pytest.raises(ValueError, match="passing Gate"):
        train_official_closed_loop(config, teacher_factory=pipeline["factory"])
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps({"format": "comma-gate2-v1", "passed": False, "split": "val"}))
    config = replace(config, gate2_report=gate_path)
    with pytest.raises(ValueError, match="must pass"):
        train_official_closed_loop(config, teacher_factory=pipeline["factory"])
    gate_path.write_text(json.dumps({"format": "comma-gate2-v1", "passed": True, "split": "val"}))
    with pytest.raises(ValueError, match="checkpoint differs"):
        train_official_closed_loop(config, teacher_factory=pipeline["factory"])
    assert not config.output_dir.exists()


def test_online_rejects_training_validation_scenario_leak(pipeline, tmp_path, monkeypatch):
    class Leaky(RoadEnvironment):
        def describe(self):
            return {**super().describe(), "scenario_pool": "same-pool"}

    monkeypatch.setattr(
        official_runner,
        "load_environment",
        lambda factory, spec, split, kwargs: Leaky(spec=spec, split=split, **kwargs),
    )
    config = online_config(pipeline, tmp_path / "leak")
    with pytest.raises(ValueError, match="overlap"):
        train_official_closed_loop(config, teacher_factory=pipeline["factory"])
    assert not config.output_dir.exists()


@pytest.mark.parametrize("name", ["official_dagger.toml", "official_td3.toml"])
def test_shipped_online_configs_require_gate_and_preserve_paths(name):
    root = Path(__file__).resolve().parents[1]
    cfg = load_official_config(root / "configs/training" / name)
    assert not cfg.engineering_only
    assert cfg.teacher_model == root / "models/big_driving_supercombo.onnx"
    assert cfg.gate2_report == root / "outputs/official-residual-sim-val/gate2.json"


def validation_report():
    return {
        "split": "val",
        "teacher_identity": {"sha256": "example"},
        "teacher_asset_verified": True,
        "checkpoint_sha256": "checkpoint",
        "per_route": {
            "road-a": {
                "expert_frames": 12,
                "teacher_mae_mps2": [1, 2],
                "adapted_mae_mps2": [0.8, 2],
            },
            "road-b": {"expert_frames": 12, "teacher_mae_mps2": [1, 2], "adapted_mae_mps2": [1, 2]},
        },
    }


def test_gate_checks_each_axis_route_labels_and_measured_improvement():
    report = validation_report()
    assert gate_residual(report)["passed"]
    report["per_route"]["road-b"]["adapted_mae_mps2"] = [1.01, 1]
    assert not gate_residual(report)["passed"]
    report["per_route"]["road-b"]["adapted_mae_mps2"] = [1, 2]
    report["per_route"]["road-a"]["adapted_mae_mps2"] = [1, 2]
    assert not gate_residual(report)["passed"]
    report["per_route"]["road-a"]["expert_frames"] = 0
    assert not gate_residual(report)["passed"]
    for tolerance in (float("nan"), -1):
        with pytest.raises(ValueError, match="tolerances"):
            gate_residual(report, tolerance_mps2=tolerance)


def test_jerk_uses_elapsed_time_without_cross_episode_or_invalid_gap_differences():
    metrics = RouteMetrics()
    for episode, frame, time, action in (("a", 25, 5.0, 0), ("a", 26, 5.2, 0.4), ("b", 25, 5, 10)):
        metrics.add(
            {
                "episode_id": episode,
                "frame": frame,
                "timestamp_s": time,
                "teacher_action": [action, 0],
                "adapted_action": [action, 0],
            },
            [0.5, 0.5],
        )
    np.testing.assert_allclose(metrics.report()["teacher_jerk_rms_mps3"], [2, 0])
    assert metrics.report()["jerk_elapsed_s"] == pytest.approx(0.2)

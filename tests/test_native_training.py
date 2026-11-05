import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.utils.data import default_collate

from comma_post_training.data_pipeline.demo import create_demo
from comma_post_training.data_pipeline.episodes import Catalogue
from comma_post_training.engine.checkpoints import load_checkpoint
from comma_post_training.engine.config import (
    DataConfig,
    LossConfig,
    ModelConfig,
    RunConfig,
    TrainingConfig,
    WorldConfig,
    load_config,
)
from comma_post_training.engine.tasks import build_model
from comma_post_training.engine.trainer import evaluate, make_windows, select_device, train
from comma_post_training.simulator.config import ClosedLoopConfig
from comma_post_training.simulator.latent import LatentDrivingEnvironment
from comma_post_training.simulator.runner import evaluate_closed_loop, train_closed_loop
from comma_post_training.world_model.cache import cache_latents
from comma_post_training.world_model.flow import LatentFlow
from comma_post_training.world_model.rollout import predict_rollout


def configuration(root, manifest, task="policy", steps=8):
    return RunConfig(
        task=task,
        data=DataConfig(manifest=manifest, history=3, image_size=(16, 32)),
        model=ModelConfig(
            hidden_size=16, vision_channels=4, latent_channels=2, transformer_layers=1, heads=2
        ),
        training=TrainingConfig(
            output_dir=root, max_steps=steps, batch_size=4, val_every=4, learning_rate=0.003
        ),
        world=WorldConfig(anchor_offsets=(2, 3), plan_steps=3),
    )


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("native-pipeline")
    manifest = create_demo(root / "data", frames=20)
    policy_cfg = configuration(root / "policy", manifest)
    policy = train(policy_cfg)
    compressor_cfg = configuration(root / "compressor", manifest, "compressor")
    compressor = train(compressor_cfg)
    cache = cache_latents(manifest, root / "compressor" / "last.pt", root / "latents", batch_size=7)
    world_cfg = configuration(root / "world", Path(cache["manifest"]), "world_model")
    world = train(world_cfg)
    return root, manifest, policy_cfg, compressor_cfg, world_cfg, policy, compressor, world


def test_native_models_train_with_independent_validation_and_reload(pipeline):
    root, _, _, _, _, policy, compressor, world = pipeline
    for task, report in (("policy", policy), ("compressor", compressor), ("world", world)):
        assert report["final_validation"]["loss"] < report["initial_validation"]["loss"]
        assert report["final_validation"]["full_split"]
        checkpoint = root / task / "last.pt"
        heldout = evaluate(checkpoint)
        assert heldout["split"] == "test"
        assert heldout["loss"] >= 0
        assert heldout["full_split"]
        best = load_checkpoint(root / task / "best.pt")
        assert best["validation"]["loss"] <= min(
            report["initial_validation"]["loss"], report["final_validation"]["loss"]
        )
        assert load_checkpoint(root / task / "initial.pt")["step"] == 0


def test_cache_keeps_source_metadata_and_encoder_identity(pipeline):
    root, manifest, _, compressor_cfg, world_cfg, *_ = pipeline
    original = Catalogue(manifest)
    derived = Catalogue(world_cfg.data.manifest)
    checkpoint = load_checkpoint(root / "compressor" / "last.pt")
    model = build_model(checkpoint["model_spec"])
    model.load_state_dict(checkpoint["model_state"])
    for before, after in zip(original.episodes, derived.episodes, strict=True):
        assert "latents" not in json.loads(before.metadata.read_text())["arrays"]
        assert (
            json.loads(after.metadata.read_text())["encoder_fingerprint"]
            == model.encoder_fingerprint()
        )
    cfg = replace(
        compressor_cfg,
        model=replace(
            compressor_cfg.model,
            init_checkpoint=root / "compressor" / "last.pt",
            freeze_encoder=True,
        ),
        training=replace(compressor_cfg.training, output_dir=root / "decoder-adapt", max_steps=2),
    )
    train(cfg)
    adapted = load_checkpoint(root / "decoder-adapt" / "last.pt")
    assert adapted["encoder_fingerprint"] == checkpoint["encoder_fingerprint"]
    assert any(
        not torch.equal(adapted["model_state"][key], value)
        for key, value in checkpoint["model_state"].items()
        if key.startswith("decoder.")
    )


@pytest.mark.parametrize("task", ["policy", "world_model"])
def test_cpu_resume_matches_uninterrupted_training_exactly(pipeline, tmp_path, task):
    _, manifest, _, _, world_cfg, *_ = pipeline
    cfg = configuration(
        tmp_path / "full",
        world_cfg.data.manifest if task == "world_model" else manifest,
        task,
        steps=6,
    )
    train(cfg)
    partial = replace(
        cfg, training=replace(cfg.training, output_dir=tmp_path / "partial", max_steps=3)
    )
    train(partial)
    resumed = replace(cfg, training=replace(cfg.training, output_dir=tmp_path / "partial"))
    train(resumed, resume=tmp_path / "partial" / "last.pt")
    full = load_checkpoint(tmp_path / "full" / "last.pt")
    continuation = load_checkpoint(tmp_path / "partial" / "last.pt")
    for key in full["model_state"]:
        torch.testing.assert_close(
            full["model_state"][key], continuation["model_state"][key], rtol=0, atol=0
        )
    assert full["validation"] == continuation["validation"]
    with pytest.raises(ValueError, match="older than"):
        train(resumed, resume=tmp_path / "partial" / "initial.pt")
    changed = replace(
        resumed,
        training=replace(
            resumed.training, output_dir=tmp_path / "changed", learning_rate=0.002, max_steps=7
        ),
    )
    with pytest.raises(ValueError, match="resume contract changed"):
        train(changed, resume=tmp_path / "partial" / "last.pt")
    assert not (tmp_path / "changed").exists()


def test_plan_head_cannot_read_noised_target_or_target_motion(pipeline):
    root, _, _, _, cfg, *_ = pipeline
    checkpoint = load_checkpoint(root / "world" / "last.pt")
    model = build_model(checkpoint["model_spec"]).eval()
    model.load_state_dict(checkpoint["model_state"])
    dataset = make_windows(cfg, Catalogue(cfg.data.manifest), "test")
    batch = default_collate([dataset[0], dataset[1]])
    assert batch["frame_times"].shape[-1] == cfg.data.history + len(cfg.world.anchor_offsets) + 1
    assert batch["frame_times"][0, cfg.data.history] > batch["frame_times"][0, -1]
    with torch.no_grad():
        first = model(batch["target_latent"], torch.zeros(2), batch)
        changed = model(
            torch.randn_like(batch["target_latent"]) * 100,
            torch.ones(2),
            {**batch, "motion": batch["motion"] * 20},
        )
    torch.testing.assert_close(first["plan"], changed["plan"], rtol=0, atol=0)
    assert not torch.equal(first["velocity"], changed["velocity"])


def test_autoregressive_rollout_marks_privileged_motion_and_anchors(pipeline, tmp_path):
    root, *_ = pipeline
    result = predict_rollout(
        root / "world" / "last.pt", tmp_path / "rollout.pt", frames=3, sample_steps=2
    )
    saved = torch.load(tmp_path / "rollout.pt", weights_only=True)
    assert len(result["latent_mse_per_frame"]) == 3
    assert saved["latents"].shape[0] == 3
    assert saved["uses_ground_truth_future_anchors"]
    assert saved["motion_source"] == "held_out_dataset_relative_poses"
    assert saved["dataset_kind"] == "synthetic"


def test_cpu_inference_can_override_checkpoint_training_device(pipeline, tmp_path):
    root, *_ = pipeline
    saved = load_checkpoint(root / "policy" / "last.pt")
    saved["config"]["training"]["device"] = "cuda"
    torch.save(saved, tmp_path / "trained-on-gpu.pt")
    assert evaluate(tmp_path / "trained-on-gpu.pt", device="cpu")["metrics"]["expert_mse"] >= 0
    world = load_checkpoint(root / "world" / "last.pt")
    world["config"]["training"]["device"] = "cuda"
    torch.save(world, tmp_path / "world-trained-on-gpu.pt")
    assert (
        predict_rollout(
            tmp_path / "world-trained-on-gpu.pt",
            tmp_path / "cpu-rollout.pt",
            frames=1,
            sample_steps=1,
            device="cpu",
        )["frames"]
        == 1
    )


def test_rollout_defaults_to_configured_sampler_steps(pipeline, tmp_path, monkeypatch):
    root, *_ = pipeline
    saved = load_checkpoint(root / "world" / "last.pt")
    saved["config"]["world"]["sampling_steps"] = 1
    torch.save(saved, tmp_path / "one-step-sampler.pt")
    observed = []
    original = LatentFlow.sample

    def spy(self, batch, *, steps, seed):
        observed.append(steps)
        return original(self, batch, steps=steps, seed=seed)

    monkeypatch.setattr(LatentFlow, "sample", spy)
    predict_rollout(tmp_path / "one-step-sampler.pt", tmp_path / "sample.pt", frames=1)
    assert observed == [1]


def test_missing_active_training_labels_fail_and_test_gold_only_evaluates(tmp_path):
    manifest = create_demo(tmp_path / "data", frames=12)
    for ref in Catalogue(manifest).episodes:
        document = json.loads(ref.metadata.read_text())
        del document["arrays"]["teacher_actions"]
        ref.metadata.write_text(json.dumps(document))
    cfg = configuration(tmp_path / "missing", manifest, steps=2)
    with pytest.raises(ValueError, match="teacher"):
        train(replace(cfg, loss=LossConfig(bc_weight=1, kd_weight=1)))
    assert not cfg.training.output_dir.exists()
    # Train KD with teacher labels, then evaluate on a separate gold-only version.
    teacher_manifest = create_demo(tmp_path / "teacher", frames=12)
    cfg = replace(
        cfg,
        data=replace(cfg.data, manifest=teacher_manifest),
        loss=LossConfig(bc_weight=1, kd_weight=1),
    )
    train(cfg)
    result = evaluate(cfg.training.output_dir / "last.pt", manifest=manifest)
    assert result["loss"] is None
    assert result["metrics"]["expert_mse"] >= 0
    assert result["counts"]["teacher_mse"] == 0


@pytest.mark.parametrize("algorithm", ["dagger", "td3"])
def test_actual_closed_loop_updates_and_reloads_policy(pipeline, tmp_path, algorithm):
    root, *_ = pipeline
    cfg = ClosedLoopConfig(
        algorithm=algorithm,
        policy_checkpoint=root / "policy" / "last.pt",
        output_dir=tmp_path / algorithm,
        steps=16,
        batch_size=4,
        warmup_steps=4,
        eval_episodes=1,
        episode_limit=10,
        env_kwargs={"horizon": 8},
    )
    result = train_closed_loop(cfg)
    assert result["updates"] == 13
    assert result["environment"]["kind"] == "synthetic"
    assert (
        result["initial_validation"]["episodes"][0]["seed"]
        == result["final_validation"]["episodes"][0]["seed"]
    )
    tested = evaluate_closed_loop(cfg, cfg.output_dir / "final.pt")
    assert tested["split"] == "test"
    assert tested["episodes"][0]["seed"] != result["final_validation"]["episodes"][0]["seed"]
    exported = torch.load(cfg.output_dir / "final.pt", weights_only=True)
    original = load_checkpoint(cfg.policy_checkpoint)
    for key, value in original["model_state"].items():
        if algorithm == "td3" or key.startswith("encoder."):
            torch.testing.assert_close(value, exported["model_state"][key], rtol=0, atol=0)
    state = torch.load(cfg.output_dir / "learning_state.pt", weights_only=True)
    if algorithm == "td3":
        assert state["learner"]["updates"] == 13
        assert not torch.equal(state["replay"]["action"], state["replay"]["achieved_action"])
        assert state["replay"]["truncated"].any()
        assert any(
            tensor.count_nonzero()
            for key, tensor in exported["residual_state"].items()
            if key.startswith("net.4")
        )
    else:
        assert evaluate(cfg.output_dir / "final.pt")["metrics"]["expert_mse"] >= 0
        assert any(
            not torch.equal(value, exported["model_state"][key])
            for key, value in original["model_state"].items()
            if not key.startswith("encoder.")
        )


def test_native_configs_resolve_paths_and_reject_unknown_options(tmp_path):
    path = tmp_path / "train.toml"
    path.write_text(
        'task="policy"\n[data]\nmanifest="data/manifest.json"\n[training]\noutput_dir="output"\n'
    )
    assert load_config(path).data.manifest == tmp_path / "data" / "manifest.json"
    path.write_text(path.read_text() + "max_step=4\n")
    with pytest.raises(ValueError, match="unknown training"):
        load_config(path)
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match="CUDA requested"):
            select_device("cuda")


def test_learned_environment_uses_generated_history_and_action_derived_motion(pipeline):
    root, _, cfg, *_ = pipeline
    policy = load_checkpoint(root / "policy" / "last.pt")
    env = LatentDrivingEnvironment(
        spec=policy["model_spec"],
        split="test",
        world_checkpoint=root / "world" / "last.pt",
        compressor_checkpoint=root / "compressor" / "last.pt",
        horizon=3,
        sampling_steps=2,
    )
    initial = env.reset(seed=7)
    motions = []
    hook = env.world.register_forward_pre_hook(
        lambda module, args: motions.append(args[2]["motion"].clone())
    )
    assert env.expert_action().shape == (2,)
    assert motions[-1].count_nonzero() == 0
    first = env.step([1, 0.5])
    generated_history = env.history.clone()
    first_motion = motions[-1].clone()
    assert first_motion[0, 1] > 0
    assert first_motion[0, 5] > 0
    assert not torch.equal(initial["images"][-1], first.observation["images"][-1])
    env.reset(seed=7)
    opposite = env.step([-1, -0.5])
    assert motions[-1][0, 1] < 0
    assert not torch.equal(first_motion, motions[-1])
    assert not torch.equal(generated_history[-1], env.history[-1])
    assert not torch.equal(first.observation["images"][-1], opposite.observation["images"][-1])
    for _ in range(2):
        last = env.step([0, 0])
    assert last.truncated
    hook.remove()
    assert env.describe()["uses_recorded_future_anchors"]
    assert env.describe()["split"] == "test"
    env.close()

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from comma_post_training.data_pipeline.episodes import Catalogue, Episode
from comma_post_training.simulator import recorder
from comma_post_training.simulator.recorder import record_simulation
from comma_post_training.simulator.recording import (
    INTRINSICS,
    RoadEnvironment,
    recording_spec,
)


@pytest.fixture(autouse=True)
def limited_threads():
    torch.set_num_threads(2)


def environment(split="train", **kwargs):
    return RoadEnvironment(spec=recording_spec(), split=split, **kwargs)


def test_reset_and_independent_camera_projection_are_reproducible():
    env = environment()
    first = env.reset(seed=42)
    identity = env.describe()["scenario_id"]
    sequence = [env.step(env.expert_action()).observation for _ in range(3)]
    replay = env.reset(seed=42)
    for key in first:
        torch.testing.assert_close(first[key], replay[key], rtol=0, atol=0)
    for expected in sequence:
        actual = env.step(env.expert_action()).observation
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert env.describe()["scenario_id"] == identity
    assert first["images"].shape == (1, 2, 3, 256, 512)
    # The narrow/wide horizons follow their own intrinsic principal points.
    for camera, name in enumerate(("front_narrow", "front_wide")):
        cy = INTRINSICS[name][1][2]
        rgb = first["images"][0, camera, :, :, 200].T.numpy()
        sky = np.array([115, 170, 219]) / 255
        np.testing.assert_allclose(rgb[int(cy) - 1], sky, atol=1e-7)
        assert not np.allclose(rgb[int(cy) + 2], sky)
    assert not torch.equal(first["images"][0, 0], first["images"][0, 1])
    other = environment("val")
    other.reset(seed=42)
    assert other.describe()["scenario_id"] != identity
    assert other.describe()["scenario_parameters"] != env.describe()["scenario_parameters"]
    assert other.describe()["scenario_pool"] != env.describe()["scenario_pool"]


def test_actions_change_reached_observations_and_live_expert_labels():
    env = environment()
    env.reset(seed=7)
    straight = env.step(np.zeros(2)).observation
    env.reset(seed=7)
    command = np.array([2.0, 1.0], dtype=np.float32)
    turning = env.step(command)
    np.testing.assert_allclose(turning.achieved_action, 0.8 * command)
    np.testing.assert_array_equal(turning.observation["past_actions"][0], turning.achieved_action)
    assert not torch.equal(straight["images"], turning.observation["images"])
    disturbed_expert = env.expert_action()
    env.reset(seed=7)
    env.step(np.zeros(2))
    assert not np.allclose(disturbed_expert, env.expert_action())
    with pytest.raises(ValueError, match="commands"):
        env.step([4, 0])


def test_recorder_contract_causality_final_observation_and_diagnostics(tmp_path):
    root = tmp_path / "episodes"
    report = record_simulation(root, duration_s=0.8)
    catalogue = Catalogue(root / "manifest.json")
    assert report["dataset_kind"] == "synthetic"
    assert report["pilot_ready"] is False
    assert [ref.split for ref in catalogue.episodes] == ["train", "train", "val", "test"]
    assert len({ref.route_id for ref in catalogue.episodes}) == 4
    for ref in catalogue.episodes:
        ep = Episode(ref)
        assert ep.array("images").shape == (4, 2, 256, 512, 3)
        assert ep.array("images").dtype == np.uint8
        np.testing.assert_allclose(ep.array("timestamps"), np.arange(4) / 5)
        assert ep.metadata["teacher_rgb"]["calibration_id"]
        assert not ep.array("terminated").any()
        assert ep.array("truncated").tolist() == [False, False, False, True]
        simulation = ep.metadata["simulation"]
        env = environment(ref.split)
        env.reset(seed=simulation["seed"])
        for i in range(4):
            np.testing.assert_allclose(ep.array("poses")[i], env.current_pose())
            np.testing.assert_array_equal(ep.array("expert_actions")[i], env.expert_action())
            transition = env.step(env.expert_action())
            np.testing.assert_array_equal(
                ep.array("executed_actions")[i], transition.achieved_action
            )
        final = np.load(ref.metadata.parent / simulation["final_observation"])
        np.testing.assert_allclose(final["timestamp_s"], 0.8)
        np.testing.assert_allclose(final["pose"], env.current_pose())
        np.testing.assert_array_equal(final["past_actions"], transition.achieved_action)
        assert (ref.metadata.parent / "camera-preview.png").is_file()
        assert "Lateral acceleration" in (ref.metadata.parent / "diagnostics.svg").read_text()
    before = (root / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="empty"):
        record_simulation(root, duration_s=0.8)
    assert before == (root / "manifest.json").read_bytes()
    assert json.loads((root / "recording-report.json").read_text())["quality"]


@pytest.mark.parametrize("failure", ["early_done", "fast_input", "wrong_history", "leak"])
def test_invalid_backend_is_rejected_atomically(tmp_path, monkeypatch, failure):
    class Broken(RoadEnvironment):
        def step(self, action):
            transition = super().step(action)
            if failure == "early_done":
                return replace(transition, truncated=True)
            if failure == "fast_input":
                transition.observation["dt"].fill_(0.05)
            if failure == "wrong_history":
                transition.observation["past_actions"].fill_(99)
            return transition

        def describe(self):
            value = super().describe()
            if failure == "leak":
                value["scenario_id"] = "same-scenario"
            return value

    monkeypatch.setattr(
        recorder,
        "load_environment",
        lambda factory, spec, split, kwargs: Broken(spec=spec, split=split),
    )
    destination = tmp_path / "bad"
    with pytest.raises(ValueError):
        record_simulation(destination, duration_s=0.8)
    assert not destination.exists()

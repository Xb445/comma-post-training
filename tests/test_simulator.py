from dataclasses import replace

import numpy as np
import pytest
import torch

from comma_post_training.simulator.api import check_observation
from comma_post_training.simulator.config import ClosedLoopConfig
from comma_post_training.simulator.reference import ReferenceEnvironment
from comma_post_training.simulator.td3 import TD3, td_target


def test_reference_actions_change_observation_and_report_achieved_acceleration():
    spec = {
        "history": 3,
        "cameras": 1,
        "image_size": [8, 16],
        "state_dim": 2,
        "state_features": ["speed_mps", "yaw_rate_radps"],
    }
    env = ReferenceEnvironment(spec=spec, split="train", horizon=4)
    initial = check_observation(env.reset(seed=2), spec)
    first = env.step(np.array([1, 1]))
    env.reset(seed=2)
    other = env.step(np.array([-1, -1]))
    assert not torch.equal(first.observation["images"], other.observation["images"])
    assert not torch.equal(first.observation["state"], other.observation["state"])
    np.testing.assert_allclose(first.achieved_action, [0.7, 0.7])
    np.testing.assert_allclose(first.observation["past_actions"][-1], first.achieved_action)
    assert initial["dt"].count_nonzero() == 0
    with pytest.raises(ValueError, match="RGB"):
        check_observation({**initial, "images": initial["images"] + 2}, spec)


def test_td3_target_bootstraps_time_limits_but_not_true_terminals():
    rewards = torch.tensor([1.0, 1.0, 1.0])
    terminals = torch.tensor([True, False, False])
    # Rows 2 and 3 may be truncated or ongoing; both have valid next observations.
    torch.testing.assert_close(
        td_target(rewards, terminals, torch.tensor([10.0, 10.0, 10.0]), 0.9),
        torch.tensor([1.0, 10.0, 10.0]),
    )


def test_td3_delays_actor_updates_and_uses_next_base_in_targets(tmp_path):
    cfg = ClosedLoopConfig(
        "td3",
        tmp_path / "base",
        tmp_path / "output",
        steps=8,
        batch_size=2,
        warmup_steps=2,
        target_noise=0,
        tau=0.5,
        policy_delay=2,
    )
    agent = TD3(4, cfg, torch.device("cpu"))
    batch = {
        "features": torch.randn(2, 4),
        "base": torch.zeros(2, 2),
        "action": torch.zeros(2, 2),
        "reward": torch.ones(2),
        "next_features": torch.randn(2, 4),
        "next_base": torch.tensor([[1.0, -1.0], [2.0, -2.0]]),
        "terminated": torch.tensor([False, True]),
    }
    captured = []
    handle = agent.critic_target.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[1].clone())
    )
    actor_before = {key: value.clone() for key, value in agent.actor.state_dict().items()}
    first = agent.update(batch)
    assert not first["actor_updated"]
    for key, value in agent.actor.state_dict().items():
        torch.testing.assert_close(value, actor_before[key], rtol=0, atol=0)
    torch.testing.assert_close(captured[0], batch["next_base"])
    second = agent.update(batch)
    handle.remove()
    assert second["actor_updated"]
    for key, value in agent.actor.state_dict().items():
        torch.testing.assert_close(
            agent.actor_target.state_dict()[key], actor_before[key].lerp(value, cfg.tau)
        )
    actions = agent.action(torch.randn(20, 4), torch.zeros(20, 2))
    assert (actions.abs() <= torch.tensor(cfg.residual_scale) + 1e-6).all()
    with pytest.raises(ValueError, match="residual scales"):
        replace(cfg, residual_scale=(-1, 1))

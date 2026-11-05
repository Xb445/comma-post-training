"""Residual TD3 on frozen causal policy features and final commanded actions."""

import copy

import torch
from torch import nn
from torch.nn import functional as F


class ResidualActor(nn.Module):
    def __init__(self, features: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features + 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, base):
        return torch.tanh(self.net(torch.cat((features, base), -1)))


class TwinCritic(nn.Module):
    def __init__(self, features: int, hidden: int = 64):
        super().__init__()

        def network():
            return nn.Sequential(
                nn.Linear(features + 2, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        self.q1, self.q2 = network(), network()

    def forward(self, features, action):
        values = torch.cat((features, action), -1)
        return self.q1(values).squeeze(-1), self.q2(values).squeeze(-1)


def td_target(reward, terminated, next_q, gamma):
    # A time limit still bootstraps from the final observation; true terminals do not.
    return reward + gamma * (~terminated).to(reward.dtype) * next_q


class TD3:
    def __init__(self, features: int, config, device: torch.device):
        self.config, self.device = config, device
        self.actor = ResidualActor(features).to(device)
        self.critic = TwinCritic(features).to(device)
        self.actor_target = copy.deepcopy(self.actor).requires_grad_(False)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        self.low = torch.tensor(config.action_low, device=device)
        self.high = torch.tensor(config.action_high, device=device)
        self.scale = torch.tensor(config.residual_scale, device=device)
        self.updates = 0

    def clamp(self, action):
        return torch.maximum(self.low, torch.minimum(self.high, action))

    def action(self, features, base, *, target=False):
        actor = self.actor_target if target else self.actor
        return self.clamp(base + self.scale * actor(features, base))

    def update(self, batch: dict) -> dict:
        cfg = self.config
        batch = {key: value.to(self.device) for key, value in batch.items()}
        features, base = batch["features"], batch["base"]
        with torch.no_grad():
            # Noise is in acceleration units after residual scaling; next base is mandatory.
            noise = (torch.randn_like(base) * cfg.target_noise).clamp(
                -cfg.noise_clip, cfg.noise_clip
            )
            next_action = self.clamp(
                self.action(batch["next_features"], batch["next_base"], target=True) + noise
            )
            qa, qb = self.critic_target(batch["next_features"], next_action)
            target = td_target(
                batch["reward"], batch["terminated"], torch.minimum(qa, qb), cfg.gamma
            )
        qa, qb = self.critic(features, batch["action"])
        critic_loss = F.mse_loss(qa, target) + F.mse_loss(qb, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10, error_if_nonfinite=True)
        self.critic_optimizer.step()
        self.updates += 1
        report = {"critic_loss": float(critic_loss.detach()), "actor_updated": False}
        if self.updates % cfg.policy_delay == 0:
            self.critic.requires_grad_(False)
            action = self.action(features, base)
            actor_loss = -self.critic(features, action)[0].mean()
            actor_loss = actor_loss + cfg.residual_penalty * (action - base).square().mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10, error_if_nonfinite=True)
            self.actor_optimizer.step()
            self.critic.requires_grad_(True)
            with torch.no_grad():
                for target_model, model in (
                    (self.actor_target, self.actor),
                    (self.critic_target, self.critic),
                ):
                    for target_parameter, parameter in zip(
                        target_model.parameters(), model.parameters(), strict=True
                    ):
                        target_parameter.lerp_(parameter, cfg.tau)
            report.update(actor_loss=float(actor_loss.detach()), actor_updated=True)
        if any(
            not torch.isfinite(torch.tensor(value))
            for key, value in report.items()
            if key.endswith("loss")
        ):
            raise RuntimeError("nonfinite TD3 update")
        return report

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "updates": self.updates,
        }

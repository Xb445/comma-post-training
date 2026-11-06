"""Train a bounded correction over frozen official actions/features, never a replacement base."""

import torch
from torch import nn


class OfficialResidualPolicy(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        inputs = spec["teacher_feature_dim"] + spec["state_dim"] + 4
        hidden = spec["hidden_size"]
        self.head = nn.Sequential(
            nn.LayerNorm(inputs),
            nn.Linear(inputs, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer(
            "residual_scale", torch.tensor(spec["residual_scale"], dtype=torch.float32)
        )

    def forward(self, batch):
        base = batch["teacher"].detach()
        context = torch.cat(
            (
                batch["teacher_features"].detach(),
                batch["state"][:, -1],
                batch["past_actions"][:, -1],
                base,
            ),
            dim=-1,
        )
        return base + self.residual_scale * torch.tanh(self.head(context))

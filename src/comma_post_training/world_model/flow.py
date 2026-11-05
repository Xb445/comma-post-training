import math

import torch
from torch import nn


class LatentFlow(nn.Module):
    """Frame-token conditional flow with an independently supervised XY plan head."""

    def __init__(self, spec: dict):
        super().__init__()
        self.spec = spec
        hidden, width = spec["hidden_size"], math.prod(spec["latent_shape"])
        self.input_projection = nn.Linear(width, hidden)
        self.frame_time = nn.Linear(1, hidden)
        self.motion = nn.Linear(6, hidden)
        self.flow_time = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        layer = nn.TransformerEncoderLayer(
            hidden,
            spec["heads"],
            hidden * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            spec["transformer_layers"],
            enable_nested_tensor=False,
        )
        self.velocity = nn.Linear(hidden, width)
        self.plan = nn.Sequential(
            nn.Linear(hidden + spec["plan_steps"], hidden),
            nn.SiLU(),
            nn.Linear(hidden, spec["plan_steps"] * 2),
        )

    def forward(self, noisy: torch.Tensor, tau: torch.Tensor, batch: dict) -> dict:
        known = torch.cat((batch["history_latents"], batch["anchor_latents"]), dim=1)
        b, frames = known.shape[:2]
        tokens = self.input_projection(known.reshape(b, frames, -1))
        target = self.input_projection(noisy.flatten(1))
        target = target + self.flow_time(tau[:, None]) + self.motion(batch["motion"])
        tokens = torch.cat((tokens, target[:, None]), dim=1)
        tokens = tokens + self.frame_time(batch["frame_times"][:, :, None])
        # Known condition tokens may not read the noised target at any layer.
        mask = torch.zeros(frames + 1, frames + 1, dtype=torch.bool, device=tokens.device)
        mask[:-1, -1] = True
        hidden = self.transformer(tokens, mask=mask)
        plan_features = hidden[:, self.spec["history"] - 1]
        plan = self.plan(torch.cat((plan_features, batch["plan_times"]), dim=-1))
        return {
            "velocity": self.velocity(hidden[:, -1]).reshape(noisy.shape),
            "plan": plan.reshape(b, self.spec["plan_steps"], 2),
        }

    @torch.no_grad()
    def sample(self, batch: dict, *, steps: int = 8, seed: int = 0) -> dict:
        if steps < 1:
            raise ValueError("sampling steps must be positive")
        device = batch["history_latents"].device
        generator = torch.Generator(device=device).manual_seed(seed)
        shape = (len(batch["history_latents"]), *self.spec["latent_shape"])
        latent = torch.randn(shape, generator=generator, device=device)
        for step in range(steps):
            time = torch.full((len(latent),), step / steps, device=device)
            output = self(latent, time, batch)
            latent = latent + output["velocity"] / steps
        output = self(latent, torch.ones(len(latent), device=device), batch)
        return {"latent": latent, "plan": output["plan"]}

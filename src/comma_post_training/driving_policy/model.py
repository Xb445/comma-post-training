import torch
from torch import nn


class DrivingPolicy(nn.Module):
    def __init__(self, spec: dict):
        super().__init__()
        self.spec = spec
        width, hidden = spec["vision_channels"], spec["hidden_size"]
        pool = tuple(spec.get("vision_pool", (1, 1)))
        self.encoder = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1),
            nn.GroupNorm(1, width),
            nn.SiLU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            nn.GroupNorm(1, width * 2),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(pool),
            nn.Flatten(),
        )
        fused = width * 2 * pool[0] * pool[1] * spec["cameras"] + spec["state_dim"] + 3
        self.fusion = nn.Sequential(nn.Linear(fused, hidden), nn.SiLU())
        self.temporal = nn.GRU(hidden, hidden, batch_first=True)
        self.action_head = nn.Linear(hidden, 2)

    def visual_features(self, batch: dict) -> torch.Tensor:
        images = batch["images"]
        b, t, views, channels, height, width = images.shape
        if views != self.spec["cameras"] or channels != 3:
            raise ValueError("policy images must match the declared cameras and RGB channels")
        return self.encoder(images.reshape(b * t * views, 3, height, width)).reshape(b, t, -1)

    def features_from_encoded(self, batch: dict) -> torch.Tensor:
        fused = torch.cat(
            (batch["vision"], batch["state"], batch["past_actions"], batch["dt"]), dim=-1
        )
        temporal, _ = self.temporal(self.fusion(fused))
        return temporal[:, -1]

    def features(self, batch: dict) -> torch.Tensor:
        return self.features_from_encoded({**batch, "vision": self.visual_features(batch)})

    def forward(self, batch: dict) -> torch.Tensor:
        return self.action_head(self.features(batch))

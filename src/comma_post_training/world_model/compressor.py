import hashlib
import json

import torch
from torch import nn


class Compressor(nn.Module):
    def __init__(self, spec: dict):
        super().__init__()
        self.spec = spec
        channels, width, latent = (
            3 * spec["cameras"],
            spec["vision_channels"],
            spec["latent_channels"],
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, width, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, latent, 4, stride=2, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent, width, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(width, channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        b, cameras, channels, height, width = images.shape
        if cameras != self.spec["cameras"] or channels != 3:
            raise ValueError("compressor expects the checkpoint's camera order and RGB inputs")
        return self.encoder(images.reshape(b, cameras * channels, height, width))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(images))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        result = self.decoder(latent)
        return result.reshape(len(latent), self.spec["cameras"], 3, *self.spec["image_size"])

    def encoder_fingerprint(self) -> str:
        identity = {
            key: self.spec[key]
            for key in (
                "kind",
                "cameras",
                "camera_names",
                "image_size",
                "vision_channels",
                "latent_channels",
            )
        }
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
        for name, value in sorted(self.encoder.state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

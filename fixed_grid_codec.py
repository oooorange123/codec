"""Fixed-grid, variable semantic-length feature JSCC for the v2 experiment."""

import math

import torch
import torch.nn.functional as F
from torch import nn


Q_VALUES = (2, 4, 8, 16, 32)
GRID = 64
CHANNELS = 64


class FixedGridCodec(nn.Module):
    """Generate q real codes per grid cell; transmit one complex use per code."""

    semantic_k_is_real = True

    def __init__(self):
        super().__init__()
        self.project = nn.ModuleList(nn.Sequential(
            nn.Conv2d(256, 16, 1), nn.GELU(), nn.Conv2d(16, 16, 3, padding=1))
            for _ in range(4))
        self.encoder = nn.Sequential(nn.Conv2d(64, CHANNELS, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(CHANNELS, CHANNELS, 3, padding=1))
        self.enc_query = nn.Sequential(nn.Linear(2, CHANNELS), nn.GELU(),
                                       nn.Linear(CHANNELS, CHANNELS))
        self.to_semantic = nn.Linear(CHANNELS, 1)
        self.channel_encoder = nn.Conv1d(1, 2, 3, padding=1)
        self.channel_decoder = nn.Conv1d(2, 1, 3, padding=1)
        self.dec_query = nn.Sequential(nn.Linear(2, CHANNELS), nn.GELU(),
                                       nn.Linear(CHANNELS, CHANNELS))
        self.decoder = nn.Sequential(nn.Conv2d(CHANNELS, CHANNELS, 3, padding=1),
                                     nn.GELU(), nn.Conv2d(CHANNELS, CHANNELS, 3, padding=1),
                                     nn.GELU())
        self.expand = nn.ModuleList(nn.Sequential(
            nn.Conv2d(CHANNELS, CHANNELS, 3, padding=1), nn.GELU(),
            nn.Conv2d(CHANNELS, 256, 3, padding=1)) for _ in range(4))

    @staticmethod
    def query_coordinates(q, reference):
        index = torch.arange(q, device=reference.device, dtype=reference.dtype)
        return torch.stack(((index + 0.5) / q, torch.full_like(index, q / 32)), -1)

    @staticmethod
    def check_k(k):
        q, rem = divmod(k, GRID * GRID)
        if rem or q not in Q_VALUES:
            raise ValueError(f"K_sem must equal 4096*q, q in {Q_VALUES}")
        return q

    def encode(self, features, k):
        q = self.check_k(k)
        if len(features) != 4 or any(x.ndim != 4 or x.shape[1] != 256 for x in features):
            raise ValueError("expected four [B,256,H,W] projected detector maps")
        maps = []
        for projection, feature in zip(self.project, features):
            spatial = projection(feature)
            if max(spatial.shape[-2:]) > GRID:
                spatial = F.interpolate(spatial, size=(GRID, GRID), mode="area")
            else:
                spatial = F.interpolate(spatial, size=(GRID, GRID), mode="bilinear",
                                        align_corners=False)
            maps.append(spatial)
        trunk = self.encoder(torch.cat(maps, 1)).permute(0, 2, 3, 1)
        queries = self.enc_query(self.query_coordinates(q, trunk))
        z = self.to_semantic(torch.tanh(trunk.unsqueeze(-2) + queries)).squeeze(-1)
        batch = z.shape[0]
        paired = self.channel_encoder(z.reshape(-1, 1, q)).transpose(1, 2)
        wire = paired.reshape(batch, k, 2)
        power = wire.square().sum(-1).mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
        return z, wire / power.unsqueeze(-1)

    def decode(self, received, shapes, k):
        q = self.check_k(k)
        if received.ndim != 3 or received.shape[1:] != (k, 2) or len(shapes) != 4:
            raise ValueError("wrong wire or target geometry")
        batch = received.shape[0]
        scalar = self.channel_decoder(received.reshape(-1, q, 2).transpose(1, 2))
        scalar = scalar.reshape(batch, GRID * GRID, q)
        queries = self.dec_query(self.query_coordinates(q, received))
        spatial = torch.einsum("bnq,qc->bnc", scalar, queries) / math.sqrt(q)
        spatial = self.decoder(spatial.transpose(1, 2).reshape(batch, CHANNELS, GRID, GRID))
        return [head(F.interpolate(spatial, size=shape, mode="bilinear", align_corners=False))
                for head, shape in zip(self.expand, shapes)]

    def forward(self, features, k, snr_db, noise=False):
        z, wire = self.encode(features, k)
        sigma = math.sqrt(1 / (2 * 10 ** (snr_db / 10)))
        received = wire + torch.randn_like(wire) * sigma if noise else wire
        restored = self.decode(received, [x.shape[-2:] for x in features], k)
        return restored, wire, received


def selftest():
    torch.manual_seed(7)
    codec = FixedGridCodec()
    features = [torch.randn(1, 256, h, w) for h, w in ((8, 12), (4, 6), (2, 3), (1, 2))]
    count = sum(p.numel() for p in codec.parameters())
    for q in Q_VALUES:
        k = GRID * GRID * q
        z, wire = codec.encode(features, k)
        restored = codec.decode(wire, [x.shape[-2:] for x in features], k)
        assert z.shape == (1, GRID, GRID, q) and wire.shape == (1, k, 2)
        assert all(a.shape == b.shape for a, b in zip(restored, features))
        assert abs(wire.square().sum(-1).mean().item() - 1) < 1e-5
        assert count == sum(p.numel() for p in codec.parameters())
    sum(x.square().mean() for x in restored).backward()
    assert codec.to_semantic.weight.grad is not None
    assert codec.to_semantic.weight.grad.abs().sum() > 0
    assert codec.channel_encoder.weight.grad.abs().sum() > 0
    assert codec.channel_decoder.weight.grad.abs().sum() > 0
    print({"parameter_count": count, "K_sem": [GRID * GRID * q for q in Q_VALUES],
           "N_ch": [GRID * GRID * q for q in Q_VALUES]})


if __name__ == "__main__":
    selftest()

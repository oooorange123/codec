"""The stage B codec with a rectangular grid and exact conditional K."""

import math

import torch
import torch.nn.functional as F
from torch import nn

K_VALUES = (131072, 262144, 524288)


class VariableGridCodec(nn.Module):
    semantic_k_is_real = True

    def __init__(self, grid):
        super().__init__()
        self.grid = tuple(grid)
        if len(self.grid) != 2 or min(self.grid) < 1 or any(k % math.prod(self.grid) for k in K_VALUES):
            raise ValueError("each K must be divisible by grid area")
        self.project = nn.ModuleList(nn.Sequential(
            nn.Conv2d(256, 16, 1), nn.GELU(), nn.Conv2d(16, 16, 3, padding=1))
            for _ in range(4))
        self.encoder = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(64, 64, 3, padding=1))
        self.enc_query = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 64))
        self.to_semantic = nn.Linear(64, 1)
        self.channel_encoder = nn.Conv1d(1, 2, 3, padding=1)
        self.channel_decoder = nn.Conv1d(2, 1, 3, padding=1)
        self.dec_query = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 64))
        self.decoder = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(64, 64, 3, padding=1), nn.GELU())
        self.expand = nn.ModuleList(nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 256, 3, padding=1)) for _ in range(4))

    @staticmethod
    def query_coordinates(q, reference):
        index = torch.arange(q, device=reference.device, dtype=reference.dtype)
        return torch.stack(((index + 0.5) / q, torch.full_like(index, q / 32)), -1)

    def check_k(self, k):
        if k not in K_VALUES:
            raise ValueError(f"K must be one of {K_VALUES}")
        return k // math.prod(self.grid)

    def encode(self, features, k):
        q = self.check_k(k)
        if len(features) != 4 or any(x.ndim != 4 or x.shape[1] != 256 for x in features):
            raise ValueError("expected four [B,256,H,W] projected detector maps")
        maps = []
        for projection, feature in zip(self.project, features):
            spatial = projection(feature)
            mode = "area" if any(a > b for a, b in zip(spatial.shape[-2:], self.grid)) else "bilinear"
            kwargs = {} if mode == "area" else {"align_corners": False}
            maps.append(F.interpolate(spatial, size=self.grid, mode=mode, **kwargs))
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
        scalar = scalar.reshape(batch, math.prod(self.grid), q)
        queries = self.dec_query(self.query_coordinates(q, received))
        spatial = torch.einsum("bnq,qc->bnc", scalar, queries) / math.sqrt(q)
        spatial = self.decoder(spatial.transpose(1, 2).reshape(batch, 64, *self.grid))
        return [head(F.interpolate(spatial, size=shape, mode="bilinear", align_corners=False))
                for head, shape in zip(self.expand, shapes)]

    def forward(self, features, k, snr_db, noise=False):
        z, wire = self.encode(features, k)
        sigma = math.sqrt(1 / (2 * 10 ** (snr_db / 10)))
        received = wire + torch.randn_like(wire) * sigma if noise else wire
        return self.decode(received, [x.shape[-2:] for x in features], k), wire, received


def selftest():
    torch.manual_seed(7)
    for grid in ((128, 128), (128, 256)):
        model = VariableGridCodec(grid)
        features = [torch.randn(1, 256, h, w) for h, w in ((8, 12), (4, 6), (2, 3), (1, 2))]
        assert sum(p.numel() for p in model.parameters()) == 920784
        for k in K_VALUES:
            z, wire = model.encode(features, k)
            restored = model.decode(wire, [x.shape[-2:] for x in features], k)
            assert z.shape == (1, *grid, k // math.prod(grid))
            assert wire.shape == (1, k, 2)
            assert all(a.shape == b.shape for a, b in zip(restored, features))
            assert abs(wire.square().sum(-1).mean().item() - 1) < 1e-5
        sum(x.square().mean() for x in restored).backward()
        assert model.to_semantic.weight.grad is not None


if __name__ == "__main__":
    selftest()

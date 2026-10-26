"""Pure PyTorch K-symbol projected-feature codec and channel."""
import math
import torch
import torch.nn.functional as F
from torch import nn

K_VALUES = (32, 64, 128, 256)
HIGH_K_VALUES = (512, 1024, 2048)
SNR_VALUES = (-5, 0, 5, 10, 15, 20)


class SemanticCodec(nn.Module):
    """Shared prefix codec. The only feature-bearing receiver input is [B,K,2]."""
    def __init__(self, grids=((8, 8),) * 4, channels=32, receiver_scale=False):
        super().__init__()
        self.grids = grids
        self.channels = channels
        self.receiver_scale = receiver_scale
        width = sum(channels * h * w for h, w in grids)
        self.reduce = nn.ModuleList([nn.Conv2d(256, channels, 1) for _ in range(4)])
        self.encoder = nn.Linear(width, 512)
        self.enc_condition = nn.Linear(2, 512)
        self.decoder = nn.Linear(512, width)
        self.dec_condition = nn.Linear(2, width)
        self.expand = nn.ModuleList([nn.Conv2d(channels, 256, 3, padding=1) for _ in range(4)])

    def forward(self, features, k, snr_db, noise=True):
        assert k in K_VALUES and snr_db in SNR_VALUES
        assert len(features) == 4 and all(x.shape[1] == 256 for x in features)
        batch = features[0].shape[0]
        condition = features[0].new_tensor([k / 256, snr_db / 20]).expand(batch, 2)
        pooled = [F.adaptive_avg_pool2d(conv(x), grid).flatten(1)
                  for conv, x, grid in zip(self.reduce, features, self.grids)]
        full = self.encoder(torch.cat(pooled, 1)) + self.enc_condition(condition)
        wire = full[:, :2 * k].reshape(batch, k, 2)
        power = wire.square().sum(-1).mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
        wire = wire / power.unsqueeze(-1)  # mean complex-symbol power = 1 per image.
        sigma = math.sqrt(1 / (2 * 10 ** (snr_db / 10)))
        received = wire + torch.randn_like(wire) * sigma if noise else wire
        restored = self.decode(received, [x.shape[-2:] for x in features], k, snr_db)
        return restored, wire, received

    def decode(self, received, shapes, k, snr_db):
        assert received.shape[1:] == (k, 2) and len(shapes) == 4
        batch = received.shape[0]
        condition = received.new_tensor([k / 256, snr_db / 20]).expand(batch, 2)
        padded = F.pad(received.flatten(1) / (math.sqrt(k) if self.receiver_scale else 1),
                       (0, 512 - 2 * k))
        decoded = self.decoder(padded) + self.dec_condition(condition)
        blocks = decoded.split([self.channels * h * w for h, w in self.grids], dim=1)
        blocks = [part.reshape(batch, self.channels, h, w)
                  for part, (h, w) in zip(blocks, self.grids)]
        return [conv(F.interpolate(block, size=shape,
                                   mode="bilinear", align_corners=False))
                for conv, block, shape in zip(self.expand, blocks, shapes)]


class SpatialCodec(SemanticCodec):
    """Keep a high-resolution first feature grid within the same K-symbol wire."""
    def __init__(self):
        super().__init__(grids=((64, 96), (32, 48), (16, 24), (8, 12)),
                         channels=8, receiver_scale=True)
        nn.init.zeros_(self.dec_condition.weight)
        nn.init.zeros_(self.dec_condition.bias)


class DirectKCodec(nn.Module):
    """Direct K-real-symbol image code with K complex channel uses."""

    def __init__(self, width=64, k_values=K_VALUES):
        super().__init__()
        self.width = width
        self.k_values = tuple(k_values)
        self.project = nn.ModuleList(nn.Conv2d(256, width, 1) for _ in range(4))
        self.coordinate = nn.Linear(8, width)
        self.level = nn.Parameter(torch.randn(4, width) * 0.02)
        self.encoder_queries = nn.ParameterDict({str(k): nn.Parameter(torch.randn(k, width))
                                                 for k in self.k_values})
        self.snr_condition = nn.Linear(1, width)
        self.to_semantic = nn.Linear(width, 1)
        self.channel_encoder = nn.ModuleDict({str(k): nn.Linear(k, 2 * k) for k in self.k_values})
        self.channel_decoder = nn.ModuleDict({str(k): nn.Linear(2 * k, k) for k in self.k_values})
        self.decoder_slots = nn.ParameterDict({str(k): nn.Parameter(torch.randn(k, width))
                                               for k in self.k_values})
        self.decoder_snr_condition = nn.Linear(1, width)
        self.from_semantic = nn.Linear(1, width)
        self.decoder_key = nn.Linear(width, width)
        self.decoder_value = nn.Linear(width, width)
        self.spatial_refine = nn.Identity()
        self.expand = nn.ModuleList(nn.Conv2d(width, 256, 1) for _ in range(4))

    def _coordinates(self, height, width, reference):
        y = torch.linspace(-1, 1, height, device=reference.device, dtype=reference.dtype)
        x = torch.linspace(-1, 1, width, device=reference.device, dtype=reference.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy, (math.pi * xx).sin(), (math.pi * xx).cos(),
                            (math.pi * yy).sin(), (math.pi * yy).cos(),
                            (4 * math.pi * xx).sin(), (4 * math.pi * yy).sin()), -1).reshape(-1, 8)

    def _positions(self, shape, level, reference):
        return (self.coordinate(self._coordinates(*shape, reference)) + self.level[level]).to(reference.dtype)

    def _decoder_positions(self, shape, level, reference):
        return self._positions(shape, level, reference)

    def encode_semantic(self, features, k, snr_db, masks=None):
        if k not in self.k_values or snr_db not in SNR_VALUES or len(features) != 4 or any(x.shape[1] != 256 for x in features):
            raise ValueError("DirectKCodec expects four 256-channel features and a supported K")
        if masks is not None and (len(masks) != 4 or any(m.shape != x.shape[:1] + x.shape[-2:]
                                                        for m, x in zip(masks, features))):
            raise ValueError("feature masks must match all four spatial features")
        batch = features[0].shape[0]
        tokens, valid = [], []
        for index, (projection, feature) in enumerate(zip(self.project, features)):
            position = self._positions(feature.shape[-2:], index, feature)
            tokens.append(projection(feature).flatten(2).transpose(1, 2) + position.unsqueeze(0))
            valid.append(torch.ones(batch, feature.shape[-2] * feature.shape[-1],
                                    device=feature.device, dtype=torch.bool)
                         if masks is None else ~masks[index].flatten(1))
        tokens = torch.cat(tokens, dim=1)
        valid = torch.cat(valid, dim=1)
        if not valid.any(dim=1).all():
            raise ValueError("at least one sample has no valid feature positions")
        snr = features[0].new_tensor([[snr_db / 20]])
        queries = (self.encoder_queries[str(k)] + self.snr_condition(snr)).to(tokens.dtype)
        queries = queries.unsqueeze(0).expand(batch, -1, -1)
        attended = F.scaled_dot_product_attention(queries.unsqueeze(1), tokens.unsqueeze(1),
                                                   tokens.unsqueeze(1),
                                                   attn_mask=valid[:, None, None, :]).squeeze(1)
        return self.to_semantic(attended).squeeze(-1)

    def encode_channel(self, semantic, k):
        if semantic.ndim != 2 or semantic.shape[1] != k or k not in self.k_values:
            raise ValueError("semantic code must have shape [B,K]")
        wire = self.channel_encoder[str(k)](semantic).reshape(semantic.shape[0], k, 2)
        power = wire.square().sum(-1).mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
        return wire / power.unsqueeze(-1)

    def decode(self, received, shapes, k, snr_db):
        if k not in self.k_values or snr_db not in SNR_VALUES or received.ndim != 3 or received.shape[1:] != (k, 2) or len(shapes) != 4:
            raise ValueError("received wire must have shape [B,K,2] and four target shapes")
        batch = received.shape[0]
        semantic = self._decode_semantic(received, k)
        snr = received.new_tensor([[snr_db / 20]])
        slots = (self.from_semantic(semantic.unsqueeze(-1)) +
                 self.decoder_slots[str(k)].unsqueeze(0) + self.decoder_snr_condition(snr).unsqueeze(1))
        key = self.decoder_key(slots).unsqueeze(1)
        value = self.decoder_value(slots).unsqueeze(1)
        restored = []
        for index, (shape, expand) in enumerate(zip(shapes, self.expand)):
            height, width = shape
            query = self._decoder_positions((height, width), index, received).to(key.dtype).unsqueeze(0).expand(batch, -1, -1)
            decoded = torch.cat([F.scaled_dot_product_attention(part.unsqueeze(1), key, value).squeeze(1)
                                 for part in query.split(2048, dim=1)], dim=1)
            decoded = (decoded + query).transpose(1, 2).reshape(batch, self.width, height, width)
            restored.append(expand(self.spatial_refine(decoded)))
        return restored

    def _decode_semantic(self, received, k):
        return self.channel_decoder[str(k)](received.flatten(1))

    def forward(self, features, k, snr_db, noise=True, masks=None):
        if snr_db not in SNR_VALUES:
            raise ValueError("unsupported SNR")
        semantic = self.encode_semantic(features, k, snr_db, masks)
        wire = self.encode_channel(semantic, k)
        sigma = math.sqrt(1 / (2 * 10 ** (snr_db / 10)))
        received = wire + torch.randn_like(wire) * sigma if noise else wire
        return self.decode(received, [x.shape[-2:] for x in features], k, snr_db), wire, received


class ResidualSpatialRefine(nn.Sequential):
    def forward(self, value):
        return value + super().forward(value)


class DirectKCodecV2(DirectKCodec):
    """Direct K codec with higher-frequency coordinates and local receiver refinement."""

    def __init__(self, width=64, k_values=K_VALUES):
        super().__init__(width, k_values)
        self.decoder_coordinate = nn.Linear(26, width)
        self.decoder_level = nn.Parameter(self.level.detach().clone())
        self.spatial_refine = ResidualSpatialRefine(
            nn.Conv2d(width, width, 3, padding=1, groups=width),
            nn.GELU(),
            nn.Conv2d(width, width, 1),
        )

    def _decoder_positions(self, shape, level, reference):
        return (self.decoder_coordinate(self._decoder_coordinates(*shape, reference)) +
                self.decoder_level[level]).to(reference.dtype)

    def _decoder_coordinates(self, height, width, reference):
        y = torch.linspace(-1, 1, height, device=reference.device, dtype=reference.dtype)
        x = torch.linspace(-1, 1, width, device=reference.device, dtype=reference.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        components = [xx, yy]
        for frequency in (1, 2, 4, 8, 16, 32):
            phase = math.pi * frequency
            components.extend(((phase * xx).sin(), (phase * xx).cos(),
                               (phase * yy).sin(), (phase * yy).cos()))
        return torch.stack(components, -1).reshape(-1, 26)


class DirectKCodecV3(DirectKCodecV2):
    """A wider receiver with 1024 hidden slots for every K; wire stays [B,K,2]."""

    def __init__(self, width=64, k_values=K_VALUES):
        super().__init__(width, k_values)
        self.slot_expand = nn.ModuleDict({str(k): nn.Linear(k, max(1024, k)) for k in self.k_values})
        self.decoder_slots = nn.ParameterDict({str(k): nn.Parameter(torch.randn(max(1024, k), width))
                                               for k in self.k_values})

    def _decode_semantic(self, received, k):
        return self.slot_expand[str(k)](super()._decode_semantic(received, k))


class DirectKCodecHighK(DirectKCodecV3):
    """Separate diagnostic budget: fresh heads for K=512/1024/2048."""

    def __init__(self, width=64):
        super().__init__(width, HIGH_K_VALUES)

def feature_loss(capture):
    terms = []
    for original, restored, mask in zip(capture["original"], capture["restored"], capture["masks"]):
        valid = (~mask).unsqueeze(1)
        numerator = ((restored - original).square() * valid).sum()
        denominator = (original.detach().square() * valid).sum().clamp_min(1e-8)
        terms.append(numerator / denominator)
    return torch.stack(terms).mean()

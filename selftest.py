"""Small runnable channel and no-bypass check; run on server after compiling ops."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor" / "Deformable-DETR"))
import torch

from codec import K_VALUES, HIGH_K_VALUES, SNR_VALUES, SemanticCodec, SpatialCodec, DirectKCodec, DirectKCodecV2, DirectKCodecV3, DirectKCodecHighK


def run():
    torch.manual_seed(2026)
    model = SemanticCodec()
    features = [torch.randn(2, 256, h, w) for h, w in ((16, 20), (8, 10), (4, 5), (2, 3))]
    shapes = [x.shape[-2:] for x in features]
    for k in K_VALUES:
        for snr in SNR_VALUES:
            model.zero_grad(set_to_none=True)
            restored, wire, received = model(features, k, snr, noise=False)
            assert wire.shape == (2, k, 2)
            assert torch.equal(wire, received)
            assert all(a.shape == b.shape for a, b in zip(restored, features))
            assert torch.allclose(wire.square().sum(-1).mean(-1), torch.ones(2), atol=1e-5)
            zeroed = model.decode(torch.zeros_like(wire), shapes, k, snr)
            assert sum((a - b).square().mean().item() for a, b in zip(restored, zeroed)) > 1e-7
            sum(x.square().mean() for x in restored).backward()
            assert model.encoder.weight.grad is not None
            assert torch.isfinite(model.encoder.weight.grad).all()
            assert model.encoder.weight.grad.abs().sum() > 0
    with torch.no_grad():
        # More symbols reduce Monte Carlo variation for the 20 dB AWGN check.
        many = [torch.randn(16, 256, h, w) for h, w in ((8, 8), (4, 4), (2, 2), (1, 1))]
        _, wire, received = model(many, 256, 20)
        measured = 10 * math.log10((wire.square().sum(-1).mean() / (received - wire).square().sum(-1).mean()).item())
        assert abs(measured - 20) < 1.0, measured
    print("PASS: legacy 24 K/SNR paths, [B,K,2], unit complex power, no-noise, gradients, zero-wire, AWGN")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    spatial = SpatialCodec().to(device)
    spatial_features = [torch.randn(1, 256, h, w, device=device)
                        for h, w in ((64, 96), (32, 48), (16, 24), (8, 12))]
    for k in K_VALUES:
        spatial.zero_grad(set_to_none=True)
        restored, wire, received = spatial(spatial_features, k, 10, noise=False)
        assert wire.shape == (1, k, 2) and torch.equal(wire, received)
        assert all(a.shape == b.shape for a, b in zip(restored, spatial_features))
        assert torch.allclose(wire.square().sum(-1).mean(-1), torch.ones(1, device=device), atol=1e-5)
        zeroed = spatial.decode(torch.zeros_like(wire), [x.shape[-2:] for x in spatial_features], k, 10)
        assert sum((a - b).square().mean().item() for a, b in zip(restored, zeroed)) > 1e-7
        sum(x.square().mean() for x in restored).backward()
        assert spatial.encoder.weight.grad is not None
        assert torch.isfinite(spatial.encoder.weight.grad).all()
        assert spatial.encoder.weight.grad.abs().sum() > 0
    print("PASS: spatial 4 K paths, shapes, power, gradients, zero-wire")

    direct = DirectKCodec(width=32)
    direct_features = [torch.randn(2, 256, h, w) for h, w in ((8, 10), (4, 5), (2, 3), (1, 2))]
    direct_masks = [torch.zeros(2, h, w, dtype=torch.bool) for h, w in ((8, 10), (4, 5), (2, 3), (1, 2))]
    direct_masks[0][0, -1] = True
    for k in K_VALUES:
        direct.zero_grad(set_to_none=True)
        semantic = direct.encode_semantic(direct_features, k, 10, direct_masks)
        assert semantic.shape == (2, k) and not torch.allclose(semantic[0], semantic[1])
        if k > 32:
            assert not torch.allclose(semantic[:, :32], direct.encode_semantic(direct_features, 32, 10, direct_masks))
        restored, wire, received = direct(direct_features, k, 10, noise=False, masks=direct_masks)
        assert wire.shape == (2, k, 2) and torch.equal(wire, received)
        assert not torch.allclose(wire[0], wire[1])
        assert sum((x[0] - x[1]).abs().mean().item() for x in restored) > 1e-6
        assert len(wire[0].detach().float().numpy().tobytes()) == 8 * k
        assert all(a.shape == b.shape for a, b in zip(restored, direct_features))
        assert torch.allclose(wire.square().sum(-1).mean(-1), torch.ones(2), atol=1e-5)
        zeroed = direct.decode(torch.zeros_like(wire), [x.shape[-2:] for x in direct_features], k, 10)
        assert sum((a - b).square().mean().item() for a, b in zip(restored, zeroed)) > 1e-7
        sum(x.square().mean() for x in restored).backward()
        for parameter in (direct.encoder_queries[str(k)], direct.channel_encoder[str(k)].weight,
                          direct.channel_decoder[str(k)].weight, direct.decoder_slots[str(k)],
                          direct.project[0].weight, direct.snr_condition.weight,
                          direct.decoder_snr_condition.weight):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
        different_snr = direct.encode_semantic(direct_features, k, 20, direct_masks)
        assert not torch.allclose(semantic, different_snr)
        different_reconstruction = direct.decode(wire, [x.shape[-2:] for x in direct_features], k, 20)
        assert sum((a - b).abs().mean().item() for a, b in zip(restored, different_reconstruction)) > 1e-6
    with torch.no_grad():
        _, wire, received = direct(direct_features, 256, 20, noise=True, masks=direct_masks)
        assert not torch.equal(wire, received)
    print("PASS: direct K-real code, 4 K heads, image dependence, wire power, gradients, zero-wire, AWGN")

    torch.manual_seed(123)
    sender_v1 = DirectKCodec(width=32)
    torch.manual_seed(123)
    v2 = DirectKCodecV2(width=32)
    assert v2.coordinate.in_features == 8
    assert v2.decoder_coordinate.in_features == 26
    for k in K_VALUES:
        with torch.no_grad():
            original = sender_v1.encode_semantic(direct_features, k, 10, direct_masks)
            upgraded = v2.encode_semantic(direct_features, k, 10, direct_masks)
            assert torch.equal(original, upgraded), "v2 changed the v1 sender"
            assert torch.equal(sender_v1.encode_channel(original, k), v2.encode_channel(upgraded, k))
        v2.zero_grad(set_to_none=True)
        semantic = v2.encode_semantic(direct_features, k, 10, direct_masks)
        restored, wire, received = v2(direct_features, k, 10, noise=False, masks=direct_masks)
        assert semantic.shape == (2, k) and wire.shape == (2, k, 2)
        assert torch.equal(wire, received)
        assert all(a.shape == b.shape for a, b in zip(restored, direct_features))
        assert torch.allclose(wire.square().sum(-1).mean(-1), torch.ones(2), atol=1e-5)
        zeroed = v2.decode(torch.zeros_like(wire), [x.shape[-2:] for x in direct_features], k, 10)
        assert sum((a - b).square().mean().item() for a, b in zip(restored, zeroed)) > 1e-7
        sum(x.square().mean() for x in restored).backward()
        for parameter in (v2.encoder_queries[str(k)], v2.coordinate.weight,
                          v2.decoder_coordinate.weight, v2.decoder_level,
                          v2.spatial_refine[0].weight, v2.spatial_refine[2].weight):
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    try:
        v2.load_state_dict(direct.state_dict())
        raise AssertionError("v1 weights were accepted as v2")
    except RuntimeError:
        pass
    print("PASS: direct K v2 4 K heads, high-frequency receiver, local gradients, zero-wire, v1 incompatibility")

    torch.manual_seed(123)
    v3 = DirectKCodecV3(width=32)
    for k in K_VALUES:
        with torch.no_grad():
            assert torch.equal(sender_v1.encode_semantic(direct_features, k, 10, direct_masks),
                               v3.encode_semantic(direct_features, k, 10, direct_masks))
        v3.zero_grad(set_to_none=True)
        restored, wire, _ = v3(direct_features, k, 10, noise=False, masks=direct_masks)
        assert wire.shape == (2, k, 2) and v3.decoder_slots[str(k)].shape == (1024, 32)
        assert all(a.shape == b.shape for a, b in zip(restored, direct_features))
        assert torch.allclose(wire.square().sum(-1).mean(-1), torch.ones(2), atol=1e-5)
        sum(x.square().mean() for x in restored).backward()
        assert v3.slot_expand[str(k)].weight.grad.abs().sum() > 0
    try:
        v3.load_state_dict(v2.state_dict())
        raise AssertionError("v2 weights were accepted as v3")
    except RuntimeError:
        pass
    print("PASS: direct K v3 4 K heads, 1024 receiver slots, sender unchanged, gradients")

    high = DirectKCodecHighK(width=16)
    tiny = [torch.randn(1, 256, h, w) for h, w in ((2, 3), (2, 2), (1, 2), (1, 1))]
    for k in HIGH_K_VALUES:
        high.zero_grad(set_to_none=True)
        semantic = high.encode_semantic(tiny, k, 10)
        restored, wire, _ = high(tiny, k, 10, noise=False)
        assert semantic.shape == (1, k) and wire.shape == (1, k, 2)
        assert high.decoder_slots[str(k)].shape == (max(1024, k), 16)
        assert len(wire[0].detach().float().numpy().tobytes()) == 8 * k
        assert torch.allclose(wire.square().sum(-1).mean(-1), torch.ones(1), atol=1e-5)
        sum(x.square().mean() for x in restored).backward()
        assert high.encoder_queries[str(k)].grad.abs().sum() > 0
        assert high.slot_expand[str(k)].weight.grad.abs().sum() > 0
    print("PASS: high-K diagnostic 512/1024/2048 fresh heads, budget, gradients")


if __name__ == "__main__":
    run()

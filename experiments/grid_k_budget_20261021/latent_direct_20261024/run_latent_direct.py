"""Stage B: pass the K=524288 semantic latent directly to its decoder."""

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pg_no_bottleneck_20261023"))
import run_pg as pg  # noqa: E402

base, run, stage = pg.base, pg.run, pg.stage
OUT = HERE / "run_v2"
K_REF = 524288
Q = 16
PG_REFERENCE = HERE.parent / "pg_no_bottleneck_20261023/run_v2"
FULL_REFERENCE = HERE.parent / "high_k_sweep_20261023/run_v2"
original_validate_predictions = pg.validate_predictions


class LatentDirect(nn.Module):
    """Keep the warmstarted codec except its two channel-mapping convolutions."""

    def __init__(self, source):
        super().__init__()
        if tuple(source.grid) != (128, 256) or source.check_k(K_REF) != Q:
            raise RuntimeError("K=524288 does not map to 16 latent scalars per grid cell")
        for name in ("project", "encoder", "enc_query", "to_semantic",
                     "dec_query", "decoder", "expand"):
            setattr(self, name, getattr(source, name))
        self.grid = tuple(source.grid)
        self.audit_values = False
        self.encoder_latent = None
        self.decoder_latent = None
        self.last_latent_shape = None
        self.last_bypass_exact = False
        if len(self.project) != 4 or len(self.expand) != 4:
            raise RuntimeError("four-level codec interface changed")

    def forward(self, features, k=0, snr_db=None, noise=False):
        if k != 0 or noise or len(features) != 4 or any(x.ndim != 4 or x.shape[1] != 256 for x in features):
            raise ValueError("latent-direct requires four detector maps and no actual wire")
        maps = []
        for projection, feature in zip(self.project, features):
            spatial = projection(feature)
            mode = "area" if any(a > b for a, b in zip(spatial.shape[-2:], self.grid)) else "bilinear"
            kwargs = {} if mode == "area" else {"align_corners": False}
            maps.append(F.interpolate(spatial, size=self.grid, mode=mode, **kwargs))
        trunk = self.encoder(torch.cat(maps, 1)).permute(0, 2, 3, 1)
        queries = self.enc_query(base.VariableGridCodec.query_coordinates(Q, trunk))
        z = self.to_semantic(torch.tanh(trunk.unsqueeze(-2) + queries)).squeeze(-1)
        if z.shape != (features[0].shape[0], *self.grid, Q) or z.dtype != torch.float32:
            raise RuntimeError("semantic latent shape or dtype changed")
        received_latent = z
        self.last_bypass_exact = received_latent is z
        self.last_latent_shape = list(z.shape)
        if not self.last_bypass_exact:
            raise RuntimeError("latent values changed across direct bypass")
        if self.audit_values:
            self.encoder_latent, self.decoder_latent = z.detach().clone(), received_latent.detach().clone()
        dec_queries = self.dec_query(base.VariableGridCodec.query_coordinates(Q, received_latent))
        scalar = received_latent.reshape(z.shape[0], math.prod(self.grid), Q)
        spatial = torch.einsum("bnq,qc->bnc", scalar, dec_queries) / math.sqrt(Q)
        spatial = self.decoder(spatial.transpose(1, 2).reshape(z.shape[0], 64, *self.grid))
        restored = [head(F.interpolate(spatial, size=x.shape[-2:], mode="bilinear", align_corners=False))
                    for head, x in zip(self.expand, features)]
        empty = z.new_empty((z.shape[0], 0, 2))
        return restored, empty, empty


def protocol():
    files = {"run_latent_direct.py": Path(__file__), "run_pg.py": Path(pg.__file__),
             "run_high_k.py": Path(base.__file__),
             "variable_grid_codec.py": HERE.parent / "variable_grid_codec.py",
             "run_server.py": base.PROJECT / "run_server.py",
             "eval_official.m": base.PROJECT / "eval_official.m",
             "eval_size_supplemental.m": base.PROJECT / "eval_size_supplemental.m"}
    return {"stage": "B-semantic-latent-direct", "seed": 101, "epochs": 30,
            "grid": [128, 256], "reference_K": K_REF, "q": Q,
            "latent_real_elements_per_image": K_REF, "actual_wire_symbols": 0,
            "initialization": str(base.WARMSTART), "warmstart_sha256": base.WARMSTART_SHA,
            "full_codec_best_checkpoint": str(FULL_REFERENCE / "epoch_24/codec.pt"),
            "full_codec_best_checkpoint_sha256": stage.sha256(FULL_REFERENCE / "epoch_24/codec.pt"),
            "full_codec_decision_sha256": stage.sha256(FULL_REFERENCE / "decision.json"),
            "detector": str(base.DETECTOR), "detector_sha256": base.DETECTOR_SHA,
            "manifest_sha256": base.EXPECTED_MANIFEST, "train_images": 6467, "val_images": 546,
            "optimizer": "AdamW", "lr": 2e-4, "weight_decay": 1e-4,
            "lr_schedule": "constant", "batch_size": 2, "amp": False,
            "loss": "detector + 10*valid-region relative feature MSE",
            "noise": False, "selection": "maximum val546 AP, first epoch on tie",
            "full_codec_training": "three K losses averaged per batch",
            "this_training": "fixed reference K, one loss per batch",
            "code_sha256": {name: stage.sha256(path) for name, path in files.items()}}


def preflight():
    if OUT.exists():
        raise RuntimeError("latent-direct output exists; refusing to overwrite")
    base.checked_inputs()
    decision = json.loads((FULL_REFERENCE / "decision.json").read_text())
    best = decision["best"][str(K_REF)]
    if best["epoch"] != 24 or abs(best["AP"] - 16.199) > 0.01:
        raise RuntimeError("formal run_v2 high-K decision differs from the reference")
    full_checkpoint = run.load_file(FULL_REFERENCE / "epoch_24/codec.pt")
    if (json.loads((FULL_REFERENCE / "protocol.json").read_text()) != base.protocol() or
            full_checkpoint["epoch"] != 24 or full_checkpoint["seed"] != 101 or
            tuple(full_checkpoint["grid"]) != (128, 256) or
            tuple(full_checkpoint["K_complex"]) != base.K_VALUES or
            full_checkpoint["detector_sha256"] != base.DETECTOR_SHA):
        raise RuntimeError("formal run_v2 checkpoint protocol mismatch")
    direct = json.loads((PG_REFERENCE / "direct/summary.json").read_text())
    pg_protocol = json.loads((PG_REFERENCE / "protocol.json").read_text())
    if not direct["matches_historic"] or pg_protocol["detector_sha256"] != base.DETECTOR_SHA or pg_protocol["manifest_sha256"] != base.EXPECTED_MANIFEST:
        raise RuntimeError("PG direct reference does not match locked detector or data")
    args, detector, criterion, model = pg.checked_models(0)
    _, train = base.dataset("train")
    samples, targets = next(iter(run.loader(train, 2, 0, 101)))
    samples, targets = samples.to("cuda"), run.move_targets(targets, "cuda")
    model.train()
    model.audit_values = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    with run.linked_forward(detector, model, 0, 10, noise=False) as capture:
        output = detector(samples)
    z, incoming = model.encoder_latent, model.decoder_latent
    if not (model.last_bypass_exact and torch.equal(z, incoming) and z.dtype == incoming.dtype == torch.float32):
        raise RuntimeError("latent-direct changed values or precision")
    source = base.VariableGridCodec((128, 256)).cuda().eval()
    source.load_state_dict(run.load_file(base.WARMSTART)["codec"], strict=True)
    with torch.no_grad():
        reference_z, reference_wire = source.encode(capture["original"], K_REF)
    if not torch.equal(z, reference_z) or reference_wire.shape != (2, K_REF, 2):
        raise RuntimeError("bypass latent differs from the original encoder or K mapping")
    det = run.weighted_loss(criterion(output, targets), criterion)
    feat = run.feature_loss(capture)
    loss = det + 10 * feat
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite latent-direct loss")
    loss.backward()
    grads = {name: float(p.grad.abs().sum()) if p.grad is not None else 0.0
             for name, p in model.named_parameters()}
    if any(v <= 0 for v in grads.values()) or any(p.grad is not None for p in detector.parameters()):
        raise RuntimeError("retained codec gradient missing or detector unfrozen")
    optimizer.step()
    changes = {name: float((p.detach() - before[name]).abs().sum())
               for name, p in model.named_parameters()}
    if any(v <= 0 for v in changes.values()):
        raise RuntimeError("retained codec parameter did not update")
    shapes = {"detector_input": list(samples.tensors.shape),
              "original": [list(x.shape) for x in capture["original"]],
              "restored": [list(x.shape) for x in capture["restored"]],
              "latent": list(z.shape), "reference_wire": list(reference_wire.shape),
              "actual_wire": list(capture["wire"].shape)}
    if shapes["original"] != shapes["restored"] or shapes["actual_wire"] != [2, 0, 2]:
        raise RuntimeError("detector feature interface or actual wire differs")
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "direct").mkdir()
    direct["reused_from"] = str(PG_REFERENCE / "direct/summary.json")
    (OUT / "direct/summary.json").write_text(json.dumps(direct, indent=2) + "\n")
    (OUT / "protocol.json").write_text(json.dumps(protocol(), indent=2) + "\n")
    (OUT / "preflight.json").write_text(json.dumps({"shapes": shapes,
        "latent_dtype": str(z.dtype), "latent_min": float(z.min()), "latent_max": float(z.max()),
        "latent_mean": float(z.mean()), "latent_std": float(z.std()),
        "latent_value_equal": True, "latent_matches_reference_encoder": True,
        "retained_grad_abs_sum": grads, "retained_parameter_change_abs_sum": changes,
        "detector_frozen": True, "det_loss": det.item(), "feature_loss": feat.item(),
        "total_loss": loss.item(), "quantization_in_original": False,
        "bypassed": ["channel_encoder", "power_normalization", "channel_interface", "channel_decoder"]}, indent=2) + "\n")
    print(json.dumps({"preflight": "passed", "shapes": shapes, "latent_min": float(z.min()),
                      "latent_max": float(z.max()), "loss": loss.item()}), flush=True)


def validate_predictions(folder, val_rows, model):
    records = original_validate_predictions(folder, val_rows, model)
    if model.last_latent_shape != [1, 128, 256, Q] or not model.last_bypass_exact:
        raise RuntimeError("validation latent-direct identity or shape failed")
    for row in records:
        row.update(reference_K=K_REF, latent_shape=[128, 256, Q],
                   latent_real_elements=K_REF, latent_dtype="float32", latent_bypass_exact=True)
    (folder / "diagnostics.jsonl").write_text("\n".join(json.dumps(row, allow_nan=False) for row in records) + "\n")
    return records


def report():
    pg.checked_protocol()
    rows = pg.write_metrics()
    if len(rows) != 30 or not (OUT / "train_complete.json").exists() or not (OUT / "eval_complete.json").exists():
        raise RuntimeError("latent-direct results incomplete")
    best = max(rows, key=lambda row: row["AP"])
    final = rows[-1]
    refs = [("直通", None, pg.REFERENCE),
            ("PG 最佳", 25, json.loads((PG_REFERENCE / "best.json").read_text())["best"]),
            ("完整 codec K=524288", 24, json.loads((FULL_REFERENCE / "decision.json").read_text())["best"][str(K_REF)]),
            ("潜变量直传最佳", best["epoch"], best), ("潜变量直传最终", 30, final)]
    lines = ["# 语义潜变量直传诊断（seed101）", "",
             "参考 K=524288 对应 q=16，编码器潜变量为每图 524288 个 float32 实数；本路径逐元素直传该潜变量，实际复符号传输数 N_symbol=0。无噪声，且原 codec 也未加噪声。", "",
             "## 审计与预检", "",
             "仅旁路 channel_encoder、复符号功率归一化、无噪声信道接口、channel_decoder。project、encoder、enc_query、to_semantic、dec_query、decoder、expand 均沿用原 warmstart 并训练；原模型未实现量化。解码器所需的标量形状与 z 一致，无新增层或维数操作。", "",
             "接口、float32 数值范围、原值与原编码器输出逐元素相等、单 batch 损失/反向、所有保留模块梯度与参数变化、冻结 detector 均记录于 run_v2/preflight.json。正式高 K 参考仅使用 high_k_sweep_20261023/run_v2/；其第 24 轮检查点、协议与 decision.json 在预检时核对。本实验从相同原始 warmstart 开始，不从已训练的第 24 轮续训。", "",
             "配置和代码/数据/权重哈希见 run_v2/protocol.json；实际命令见 run_v2/*_command.txt；逐轮指标见 run_v2/metrics.csv 和 metrics.json；逐图记录见各 epoch_*/predictions/diagnostics.jsonl。检查点沿用复用训练器的文件名 pg.pt，但内容是潜变量直传模型。", "",
             "## val546 指标", "",
             "| 路径 | 轮次 | AP | AP50 | AP75 | AP_small |", "|---|---:|---:|---:|---:|---:|",
             *[f'| {name} | {epoch if epoch else "—"} | {m["AP"]:.3f} | {m["AP50"]:.3f} | {m["AP75"]:.3f} | {m["AP_small"]:.3f} |'
               for name, epoch, m in refs], "",
             f'最佳潜变量直传比完整 codec 高 {best["AP"] - refs[2][2]["AP"]:.3f} AP 点，比 PG 低 {refs[1][2]["AP"] - best["AP"]:.3f} 点。',
             f'最终轮 AP={final["AP"]:.3f}，与最佳轮相差 {best["AP"] - final["AP"]:.3f} 点；按 val546 AP 最大、并列取最早轮选模。', "",
             "该对照没有实际无线传输，和完整 codec 不是同码率比较；它不能单独归因于 channel_encoder、功率归一化或 channel_decoder 中的某一项，也不能证明噪声影响或完整 codec 已通过阶段 B。完整 codec 为三 K 共同训练，本对照固定一个参考 K，优化目标暴露方式不同；单 seed 仅作诊断。几何 shape/mask 仍旁路提供。未用 test-dev/test，未启动阶段 C。", ""]
    (HERE / "latent_direct_report.md").write_text("\n".join(lines))
    (OUT / "best.json").write_text(json.dumps({"best": best, "final": final}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("preflight", "train", "evaluate", "report"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and args.role != "train":
        parser.error("--resume is only for training")
    pg.HERE, pg.OUT, pg.PG = HERE, OUT, LatentDirect
    pg.protocol, pg.validate_predictions = protocol, validate_predictions
    try:
        if args.role == "preflight":
            preflight()
        elif args.role == "train":
            pg.train(args.resume)
        elif args.role == "evaluate":
            pg.evaluate()
        else:
            report()
    except Exception:
        if args.role == "preflight":
            (HERE / "preflight_failed.txt").write_text(traceback.format_exc())
        if args.role in ("train", "evaluate") and OUT.exists():
            (OUT / f"{args.role}_failed.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()

"""Stage B: compare a frozen semantic latent with raw detector features using the same task probe."""

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "frozen_encoder_decoder_20261027"))
import run_frozen_decoder as fd  # noqa: E402

ld, pg, base, run, stage = fd.ld, fd.pg, fd.base, fd.run, fd.stage
source_checked_models = fd.checked_models
GRID = (128, 256)
ARMS = ("latent", "raw")
EPOCHS = 30


class TaskProbe(nn.Module):
    """A four-level residual probe; both arms train only these same new layers."""

    def __init__(self, source, arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        self.grid = GRID
        if arm == "latent":
            for name in fd.ENCODER:
                setattr(self, name, getattr(source, name))
                getattr(self, name).requires_grad_(False)
        self.probe = nn.ModuleList(nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 256, 3, padding=1)) for _ in range(4))
        for head in self.probe:
            nn.init.normal_(head[-1].weight, std=1e-4)
            nn.init.zeros_(head[-1].bias)
        self.last_latent_shape = None
        self.last_probe_input_shapes = None

    def train(self, mode=True):
        super().train(mode)
        if self.arm == "latent":
            for name in fd.ENCODER:
                getattr(self, name).eval()
        return self

    def forward(self, features, k=0, snr_db=None, noise=False):
        if k != 0 or noise or len(features) != 4 or any(x.ndim != 4 or x.shape[1] != 256 for x in features):
            raise ValueError("probe needs four projected detector levels, no channel")
        if self.arm == "latent":
            maps = []
            for projection, feature in zip(self.project, features):
                spatial = projection(feature)
                mode = "area" if any(a > b for a, b in zip(spatial.shape[-2:], GRID)) else "bilinear"
                kwargs = {} if mode == "area" else {"align_corners": False}
                maps.append(F.interpolate(spatial, size=GRID, mode=mode, **kwargs))
            trunk = self.encoder(torch.cat(maps, 1)).permute(0, 2, 3, 1)
            queries = self.enc_query(base.VariableGridCodec.query_coordinates(16, trunk))
            z = self.to_semantic(torch.tanh(trunk.unsqueeze(-2) + queries)).squeeze(-1)
            if z.shape != (features[0].shape[0], *GRID, 16):
                raise RuntimeError("frozen encoder latent shape changed")
            self.last_latent_shape = list(z.shape)
            # Tiling is injective: the trainable probe sees all 16 latent values at every grid cell.
            grid = z.permute(0, 3, 1, 2).repeat_interleave(16, dim=1)
            inputs = [F.interpolate(grid, size=x.shape[-2:], mode="bilinear", align_corners=False)
                      for x in features]
        else:
            self.last_latent_shape = None
            inputs = features
        self.last_probe_input_shapes = [list(x.shape) for x in inputs]
        restored = [x + head(x) for x, head in zip(inputs, self.probe)]
        empty = inputs[0].new_empty((features[0].shape[0], 0, 2))
        return restored, empty, empty


def output(arm):
    return HERE / "run" / arm


def protocol(arm):
    if arm not in ARMS or stage.sha256(fd.CHECKPOINT) != fd.SOURCE_SHA:
        raise RuntimeError("arm or locked high-K checkpoint changed")
    files = {"run_latent_probe.py": Path(__file__),
             "run_frozen_decoder.py": Path(fd.__file__),
             "run_latent_direct.py": Path(ld.__file__),
             "run_pg.py": Path(pg.__file__),
             "run_high_k.py": Path(base.__file__),
             "variable_grid_codec.py": ROOT / "variable_grid_codec.py",
             "run_server.py": base.PROJECT / "run_server.py",
             "eval_official.m": base.PROJECT / "eval_official.m",
             "eval_size_supplemental.m": base.PROJECT / "eval_size_supplemental.m"}
    return {"stage": "B-latent-task-probe", "arm": arm, "seed": 101,
            "epochs": EPOCHS, "grid": list(GRID), "reference_K": 524288,
            "latent_shape": [128, 256, 16] if arm == "latent" else None,
            "actual_transmission": False, "N_symbol": 0, "noise": False,
            "source_checkpoint": str(fd.CHECKPOINT), "source_sha256": fd.SOURCE_SHA,
            "detector": str(base.DETECTOR), "detector_sha256": base.DETECTOR_SHA,
            "manifest_sha256": base.EXPECTED_MANIFEST, "train_images": 6467,
            "val_images": 546, "batch_size": 2, "amp": False,
            "optimizer": "AdamW", "lr": 2e-4, "weight_decay": 1e-4,
            "lr_schedule": "constant", "loss": "detector + 10*valid-region relative feature MSE",
            "selection": "maximum val546 AP, first epoch on tie",
            "trainable_modules": ["probe.0", "probe.1", "probe.2", "probe.3"],
            "probe_architecture": "per-level residual 256->64->256 3x3 convolutions",
            "latent_fixed_input_map": "[B,128,256,16] -> [B,256,128,256] by repeat_interleave(16), then bilinear to each locked level shape" if arm == "latent" else None,
            "code_sha256": {name: stage.sha256(path) for name, path in files.items()}}


def checked_models(workers, arm):
    args, detector, criterion, source = source_checked_models(workers)
    model = TaskProbe(source, arm).cuda()
    if (any(p.requires_grad for p in detector.parameters()) or
            {name.split(".")[0] for name, p in model.named_parameters() if p.requires_grad} != {"probe"}):
        raise RuntimeError("trainable parameter boundary changed")
    return args, detector, criterion, model


def preflight(arm):
    out = output(arm)
    if out.exists():
        raise RuntimeError("probe output already exists")
    args, detector, criterion, model = checked_models(0, arm)
    _, data = base.dataset("train")
    samples, targets = next(iter(run.loader(data, 2, 0, 101)))
    samples, targets = samples.to("cuda"), run.move_targets(targets, "cuda")
    model.train()
    optimizer = torch.optim.AdamW(fd.trainable(model), lr=2e-4, weight_decay=1e-4)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    with run.linked_forward(detector, model, 0, 10, noise=False) as capture:
        result = detector(samples)
    det = run.weighted_loss(criterion(result, targets), criterion)
    feat = run.feature_loss(capture)
    loss = det + 10 * feat
    original = [list(x.shape) for x in capture["original"]]
    restored = [list(x.shape) for x in capture["restored"]]
    if (not torch.isfinite(loss) or original != restored or
            list(capture["wire"].shape) != [2, 0, 2] or
            (arm == "latent" and model.last_latent_shape != [2, 128, 256, 16])):
        raise RuntimeError("probe forward, loss or detector feature interface failed")
    if arm == "raw" and any(not torch.allclose(a, b, atol=0.01, rtol=0.01)
                            for a, b in zip(capture["original"], capture["restored"])):
        raise RuntimeError("raw probe no longer initializes near direct features")
    loss.backward()
    gradients = {name: float(p.grad.abs().sum()) if p.grad is not None else 0.0
                 for name, p in model.named_parameters()}
    if (any(v <= 0 for n, v in gradients.items() if n.startswith("probe.")) or
            any(v != 0 for n, v in gradients.items() if not n.startswith("probe.")) or
            any(p.grad is not None for p in detector.parameters())):
        raise RuntimeError("probe/encoder/detector gradient boundary failed")
    optimizer.step()
    updates = {name: float((p.detach() - before[name]).abs().sum())
               for name, p in model.named_parameters()}
    if (any(v <= 0 for n, v in updates.items() if n.startswith("probe.")) or
            any(v != 0 for n, v in updates.items() if not n.startswith("probe."))):
        raise RuntimeError("only probe parameters may update")
    out.mkdir(parents=True, exist_ok=False)
    (out / "protocol.json").write_text(json.dumps(protocol(arm), indent=2) + "\n")
    (out / "preflight.json").write_text(json.dumps({"passed": True, "arm": arm,
        "original_shapes": original, "probe_input_shapes": model.last_probe_input_shapes,
        "restored_shapes": restored, "latent_shape": model.last_latent_shape,
        "loss": loss.item(), "det_loss": det.item(), "feature_loss": feat.item(),
        "gradient_abs_sum": gradients, "parameter_update_abs_sum": updates,
        "detector_frozen": True, "actual_transmission": False, "N_symbol": 0}, indent=2) + "\n")
    print(json.dumps({"preflight": "passed", "arm": arm,
                      "loss": loss.item(), "latent_shape": model.last_latent_shape}), flush=True)


def validate_predictions(folder, val_rows, model):
    expected = {Path(row["image_path"]).stem for row in val_rows}
    if {path.stem for path in folder.glob("*.txt")} != expected:
        raise RuntimeError("prediction names differ from val546")
    records = [json.loads(line) for line in (folder / "diagnostics.jsonl").read_text().splitlines()]
    if (len(records) != 546 or {record["image"] for record in records} != expected or
            any(record["N_symbol"] != 0 or record["wire_bytes"] != 0 or
                record["noise"] or record["K_sem"] is not None or
                len(record["level_shapes"]) != 4 for record in records)):
        raise RuntimeError("probe per-image wire or geometry differs")
    for record in records:
        record.update(arm=model.arm, actual_transmission=False,
                      latent_shape=[128, 256, 16] if model.arm == "latent" else None)
    (folder / "diagnostics.jsonl").write_text("\n".join(json.dumps(r, allow_nan=False) for r in records) + "\n")
    return records


def evaluate(arm):
    out = output(arm)
    if json.loads((out / "protocol.json").read_text()) != protocol(arm):
        raise RuntimeError("probe protocol or code changed")
    args, detector, criterion, model = checked_models(4, arm)
    rows, val = base.dataset("val")
    (out / "eval_command.txt").write_text(" ".join(sys.argv) + "\n")
    for epoch in range(1, EPOCHS + 1):
        checkpoint = out / f"epoch_{epoch:02d}/codec.pt"
        while not checkpoint.exists():
            if (out / "train_failed.txt").exists():
                raise RuntimeError("probe trainer failed")
            time.sleep(20)
        folder = checkpoint.parent
        if (folder / "summary.json").exists():
            continue
        state = run.load_file(checkpoint)
        if state["epoch"] != epoch or state["protocol"] != protocol(arm):
            raise RuntimeError("probe checkpoint protocol changed")
        model.load_state_dict(state["codec"], strict=True)
        started = time.monotonic()
        result = pg.evaluate_path(detector, criterion, model, val, rows, args, folder)
        info = state["train_info"]
        summary = {"epoch": epoch, "arm": arm, **result,
                   "train_loss": info["mean_loss"], "train_det_loss": info["mean_det_loss"],
                   "train_feat_loss": info["mean_feat_loss"], "learning_rate": info["learning_rate"],
                   "train_seconds": info["train_seconds"], "eval_seconds": time.monotonic() - started,
                   "checkpoint": str(checkpoint), "latent_shape": [128, 256, 16] if arm == "latent" else None,
                   "actual_transmission": False, "N_symbol": 0}
        (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        pg.write_metrics()
        print(json.dumps({"eval_epoch": epoch, "arm": arm, "AP": summary["AP"]}), flush=True)
    (out / "eval_complete.json").write_text(json.dumps({"epochs": EPOCHS, "rows": len(pg.write_metrics())}) + "\n")


def report():
    rows = {}
    for arm in ARMS:
        out = output(arm)
        if not (out / "train_complete.json").exists() or not (out / "eval_complete.json").exists():
            raise RuntimeError(f"{arm} incomplete")
        values = json.loads((out / "metrics.json").read_text())
        if len(values) != EPOCHS:
            raise RuntimeError(f"{arm} has {len(values)} validation rows")
        rows[arm] = {"best": max(values, key=lambda x: x["AP"]), "final": values[-1]}
    refs = {"direct": pg.REFERENCE,
            "PG": json.loads((ld.PG_REFERENCE / "best.json").read_text())["best"],
            "full_codec": json.loads((fd.SOURCE / "decision.json").read_text())["best"]["524288"],
            "frozen_E_D": json.loads((fd.OUT / "best.json").read_text())["best"]}
    (HERE / "probe_decision.json").write_text(json.dumps({"arms": rows, "references": refs}, indent=2) + "\n")
    comparisons = [("直通", "—", refs["direct"]), ("PG", 25, refs["PG"]),
                   ("完整 codec K=524288", 24, refs["full_codec"]),
                   ("冻结 E、训练 D", refs["frozen_E_D"]["epoch"], refs["frozen_E_D"])]
    comparisons += [(f"{arm} probe 最佳", rows[arm]["best"]["epoch"], rows[arm]["best"])
                    for arm in ARMS]
    comparisons += [(f"{arm} probe 最终", 30, rows[arm]["final"]) for arm in ARMS]
    raw_ap, latent_ap = rows["raw"]["best"]["AP"], rows["latent"]["best"]["AP"]
    if raw_ap < refs["PG"]["AP"] - 1:
        interpretation = "原始特征 probe 也明显低于 PG；探针有效性不足，不能归因于编码器。"
    elif latent_ap < raw_ap - 1:
        interpretation = "原始特征 probe 有效而潜变量 probe 明显较低，支持潜变量或语义编码结构的信息损失；探针表达能力及优化差异仍是替代解释。"
    elif latent_ap >= refs["PG"]["AP"] - 1:
        interpretation = "潜变量 probe 接近 PG；潜变量保留了可供当前探针检测的信息，现有解码器映射或优化更可疑。"
    else:
        interpretation = "两组差值不足以稳定定位，仍无法区分编码器与探针/解码器因素。"
    lines = ["# 冻结语义编码器：潜变量任务探针诊断", "",
             "两组均只训练相同结构的新 probe，原检测器冻结；latent 组冻结高 K 第24轮 E，raw 组输入原始四级投影特征。probe 为每级残差 256→64→256 的 3×3 卷积。latent 经无损通道重复与固定插值进入 probe，仅将四级目标形状作为几何信息提供。", "",
             "两组 seed101、train6467/val546、30轮、AdamW 2e-4/1e-4、batch2、检测损失+10倍有效区域相对特征 MSE；逐轮以同一官方 val546 AP 选模。协议/哈希、预检、命令、逐轮与逐图记录见 run/latent 和 run/raw。", "",
             "| 路径 | 轮次 | AP | AP50 | AP75 | AP_small |", "|---|---:|---:|---:|---:|---:|",
             *[f'| {name} | {epoch} | {m["AP"]:.3f} | {m["AP50"]:.3f} | {m["AP75"]:.3f} | {m["AP_small"]:.3f} |'
               for name, epoch, m in comparisons], "",
             interpretation, "",
             "两组 probe 不发送复符号，N_symbol=0，不是同 K 通信方案。几何 shape/mask 由接口旁路给出；未加噪声、未用 test-dev/test、未启动阶段 C。单 seed 仅提供诊断证据，不能证明普遍因果。", ""]
    (HERE / "probe_report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("preflight", "train", "evaluate", "report"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (args.role != "report" and not args.arm) or (args.role == "report" and args.arm):
        parser.error("--arm is required except for report")
    if args.resume and args.role != "train":
        parser.error("--resume applies only to train")
    if args.role != "report":
        out = output(args.arm)
        pg.OUT, pg.protocol, pg.validate_predictions = out, lambda: protocol(args.arm), validate_predictions
        fd.OUT, fd.protocol = out, lambda: protocol(args.arm)
        fd.checked_models = lambda workers: checked_models(workers, args.arm)
    try:
        if args.role == "preflight":
            preflight(args.arm)
        elif args.role == "train":
            fd.train(args.resume)
        elif args.role == "evaluate":
            evaluate(args.arm)
        else:
            report()
    except Exception:
        if args.role == "preflight":
            (HERE / f"preflight_{args.arm}_failed.txt").write_text(traceback.format_exc())
        elif args.role in ("train", "evaluate") and output(args.arm).exists():
            (output(args.arm) / f"{args.role}_failed.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()

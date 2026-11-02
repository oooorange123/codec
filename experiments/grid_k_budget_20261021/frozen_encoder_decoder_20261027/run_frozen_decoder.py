"""Stage B diagnostic: freeze high-K semantic encoder; retrain its direct-latent decoder."""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "latent_direct_20261024"))
import run_latent_direct as ld  # noqa: E402

pg, base, run, stage = ld.pg, ld.base, ld.run, ld.stage
OUT = HERE / "run"
SOURCE = ROOT / "high_k_sweep_20261023/run_v2"
CHECKPOINT = SOURCE / "epoch_24/codec.pt"
SOURCE_SHA = "140abdfd34af625090b3fea5671c63af195c1bf5fdbbdc50eabaa88787224604"
ENCODER = ("project", "encoder", "enc_query", "to_semantic")
DECODER = ("dec_query", "decoder", "expand")
EPOCHS = 30


class FrozenEncoderDecoder(ld.LatentDirect):
    def __init__(self, source):
        super().__init__(source)
        for name in ENCODER:
            getattr(self, name).requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        for name in ENCODER:
            getattr(self, name).eval()
        return self


def protocol():
    high = json.loads((SOURCE / "protocol.json").read_text())
    if high != base.protocol() or stage.sha256(CHECKPOINT) != SOURCE_SHA:
        raise RuntimeError("high-K protocol or checkpoint hash differs")
    files = {"run_frozen_decoder.py": Path(__file__),
             "run_latent_direct.py": Path(ld.__file__),
             "run_pg.py": Path(pg.__file__),
             "run_high_k.py": Path(base.__file__),
             "variable_grid_codec.py": ROOT / "variable_grid_codec.py",
             "run_server.py": base.PROJECT / "run_server.py",
             "eval_official.m": base.PROJECT / "eval_official.m",
             "eval_size_supplemental.m": base.PROJECT / "eval_size_supplemental.m"}
    return {"stage": "B-frozen-encoder-direct-latent-decoder", "seed": 101,
            "epochs": EPOCHS, "grid": [128, 256], "reference_K": ld.K_REF, "q": ld.Q,
            "actual_transmission": False, "N_symbol": 0, "noise": False,
            "encoder": list(ENCODER), "decoder": list(DECODER),
            "initialization": str(CHECKPOINT), "initialization_sha256": SOURCE_SHA,
            "detector": str(base.DETECTOR), "detector_sha256": base.DETECTOR_SHA,
            "manifest_sha256": base.EXPECTED_MANIFEST,
            "train_images": 6467, "val_images": 546, "batch_size": 2,
            "optimizer": "AdamW", "lr": 2e-4, "weight_decay": 1e-4,
            "lr_schedule": "constant", "amp": False,
            "loss": "detector + 10*valid-region relative feature MSE",
            "selection": "maximum val546 AP, first epoch on tie",
            "code_sha256": {name: stage.sha256(path) for name, path in files.items()}}


def checked_models(workers):
    base.checked_inputs()
    if stage.sha256(CHECKPOINT) != SOURCE_SHA:
        raise RuntimeError("high-K checkpoint hash changed")
    state = run.load_file(CHECKPOINT)
    if (state["epoch"] != 24 or state["seed"] != 101 or
            tuple(state["grid"]) != (128, 256) or
            tuple(state["K_complex"]) != base.K_VALUES or
            state["detector_sha256"] != base.DETECTOR_SHA):
        raise RuntimeError("high-K checkpoint metadata differs")
    run.seed_all(101)
    args = base.model_args(workers)
    detector, criterion, source = base.build_models(args)
    source.load_state_dict(state["codec"], strict=True)
    model = FrozenEncoderDecoder(source).cuda()
    if ({name for name, p in model.named_parameters() if p.requires_grad} !=
            {name for name, _ in model.named_parameters() if name.split(".")[0] in DECODER} or
            any(p.requires_grad for p in detector.parameters())):
        raise RuntimeError("trainable parameters or detector freeze differ")
    return args, detector, criterion, model


def trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


def preflight():
    if OUT.exists():
        raise RuntimeError("output already exists")
    args, detector, criterion, model = checked_models(0)
    _, data = base.dataset("train")
    samples, targets = next(iter(run.loader(data, 2, 0, 101)))
    samples, targets = samples.to("cuda"), run.move_targets(targets, "cuda")
    model.train()
    model.audit_values = True
    optimizer = torch.optim.AdamW(trainable(model), lr=2e-4, weight_decay=1e-4)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    with run.linked_forward(detector, model, 0, 10, noise=False) as capture:
        output = detector(samples)
    det = run.weighted_loss(criterion(output, targets), criterion)
    feat = run.feature_loss(capture)
    loss = det + 10 * feat
    if (not torch.isfinite(loss) or
            not torch.equal(model.encoder_latent, model.decoder_latent) or
            model.last_latent_shape != [2, 128, 256, 16] or
            list(capture["wire"].shape) != [2, 0, 2] or
            [list(x.shape) for x in capture["original"]] !=
            [list(x.shape) for x in capture["restored"]]):
        raise RuntimeError("forward, latent identity, loss, or feature interface failed")
    loss.backward()
    gradients = {name: float(p.grad.abs().sum()) if p.grad is not None else 0.0
                 for name, p in model.named_parameters()}
    if (any(value <= 0 for name, value in gradients.items() if name.split(".")[0] in DECODER) or
            any(value != 0 for name, value in gradients.items() if name.split(".")[0] in ENCODER) or
            any(p.grad is not None for p in detector.parameters())):
        raise RuntimeError("decoder/encoder/detector gradient boundary failed")
    optimizer.step()
    updates = {name: float((p.detach() - before[name]).abs().sum())
               for name, p in model.named_parameters()}
    if (any(value <= 0 for name, value in updates.items() if name.split(".")[0] in DECODER) or
            any(value != 0 for name, value in updates.items() if name.split(".")[0] in ENCODER)):
        raise RuntimeError("decoder-only parameter update failed")
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "protocol.json").write_text(json.dumps(protocol(), indent=2) + "\n")
    direct = json.loads((ld.PG_REFERENCE / "direct/summary.json").read_text())
    if not direct["matches_historic"]:
        raise RuntimeError("locked direct detector retest did not match")
    (OUT / "direct").mkdir()
    (OUT / "direct/summary.json").write_text(json.dumps(direct, indent=2) + "\n")
    (OUT / "preflight.json").write_text(json.dumps({
        "passed": True, "source_checkpoint": str(CHECKPOINT), "source_sha256": SOURCE_SHA,
        "latent_shape": model.last_latent_shape, "latent_direct_exact": True,
        "original_feature_shapes": [list(x.shape) for x in capture["original"]],
        "restored_feature_shapes": [list(x.shape) for x in capture["restored"]],
        "det_loss": det.item(), "feature_loss": feat.item(), "total_loss": loss.item(),
        "gradient_abs_sum": gradients, "parameter_update_abs_sum": updates,
        "detector_frozen": True, "actual_transmission": False, "N_symbol": 0
    }, indent=2) + "\n")
    print(json.dumps({"preflight": "passed", "latent_shape": model.last_latent_shape,
                      "loss": loss.item()}), flush=True)


def train_epoch(detector, criterion, model, optimizer, data, args, epoch, step):
    detector.eval()
    model.train()
    totals = {"loss": 0.0, "det": 0.0, "feature": 0.0}
    seen_grad = {name: False for name, p in model.named_parameters() if p.requires_grad}
    images = batches = 0
    for samples, targets in run.loader(data, 2, 6, 101 + epoch):
        samples, targets = samples.to("cuda"), run.move_targets(targets, "cuda")
        optimizer.zero_grad(set_to_none=True)
        with run.linked_forward(detector, model, 0, 10, noise=False) as capture:
            output = detector(samples)
        det = run.weighted_loss(criterion(output, targets), criterion)
        feat = run.feature_loss(capture)
        loss = det + 10 * feat
        if not torch.isfinite(loss) or capture["wire"].shape != (len(targets), 0, 2):
            raise RuntimeError("nonfinite loss or nonempty wire")
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                seen_grad[name] |= p.grad is not None and bool(torch.any(p.grad != 0))
            elif p.grad is not None:
                raise RuntimeError("frozen encoder received gradient")
        norm = torch.nn.utils.clip_grad_norm_(trainable(model), args.clip_max_norm)
        if not torch.isfinite(norm) or norm <= 0 or any(p.grad is not None for p in detector.parameters()):
            raise RuntimeError("invalid gradients or detector unfrozen")
        optimizer.step()
        for key, value in (("loss", loss), ("det", det), ("feature", feat)):
            totals[key] += value.item()
        images += len(targets)
        batches += 1
        step += 1
    if images != 6467 or not all(seen_grad.values()):
        raise RuntimeError("image count or decoder gradients incomplete")
    return step, {"epoch": epoch + 1, "images": images, "batches": batches,
                  "optimizer_steps": step, "learning_rate": optimizer.param_groups[0]["lr"],
                  "mean_loss": totals["loss"] / batches,
                  "mean_det_loss": totals["det"] / batches,
                  "mean_feat_loss": totals["feature"] / batches,
                  "decoder_gradients_nonzero": True}


def train(resume):
    pg.checked_protocol()
    args, detector, criterion, model = checked_models(6)
    _, data = base.dataset("train")
    optimizer = torch.optim.AdamW(trainable(model), lr=2e-4, weight_decay=1e-4)
    existing = [e for e in range(1, EPOCHS + 1) if (OUT / f"epoch_{e:02d}/codec.pt").exists()]
    if existing != list(range(1, len(existing) + 1)) or bool(existing) != bool(resume):
        raise RuntimeError("resume requires contiguous checkpoints and explicit --resume")
    step = 0
    if existing:
        state = run.load_file(OUT / f"epoch_{existing[-1]:02d}/codec.pt")
        if state["protocol"] != protocol() or state["epoch"] != existing[-1]:
            raise RuntimeError("resume checkpoint protocol mismatch")
        model.load_state_dict(state["codec"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        run.restore_rng(state["rng"])
        step = state["step"]
    (OUT / "train_command.txt").write_text(" ".join(sys.argv) + "\n")
    for epoch in range(len(existing), EPOCHS):
        started = time.monotonic()
        step, info = train_epoch(detector, criterion, model, optimizer, data, args, epoch, step)
        info["train_seconds"] = time.monotonic() - started
        folder = OUT / f"epoch_{epoch + 1:02d}"
        folder.mkdir(exist_ok=False)
        checkpoint = folder / "codec.pt"
        base.save_atomic(checkpoint, {"codec": model.state_dict(),
            "optimizer": optimizer.state_dict(), "rng": run.rng_state(),
            "epoch": epoch + 1, "step": step, "protocol": protocol(), "train_info": info})
        (folder / "train_summary.json").write_text(json.dumps({**info, "checkpoint": str(checkpoint)}, indent=2) + "\n")
        print(json.dumps({"train": info, "checkpoint": str(checkpoint)}), flush=True)
    (OUT / "train_complete.json").write_text(json.dumps({"epochs": EPOCHS, "step": step}) + "\n")


def evaluate():
    pg.checked_protocol()
    args, detector, criterion, model = checked_models(4)
    rows, val = base.dataset("val")
    (OUT / "eval_command.txt").write_text(" ".join(sys.argv) + "\n")
    for epoch in range(1, EPOCHS + 1):
        checkpoint = OUT / f"epoch_{epoch:02d}/codec.pt"
        while not checkpoint.exists():
            if (OUT / "train_failed.txt").exists():
                raise RuntimeError("trainer failed")
            time.sleep(20)
        folder = checkpoint.parent
        if (folder / "summary.json").exists():
            continue
        state = run.load_file(checkpoint)
        if state["epoch"] != epoch or state["protocol"] != protocol():
            raise RuntimeError("evaluation checkpoint protocol mismatch")
        model.load_state_dict(state["codec"], strict=True)
        started = time.monotonic()
        result = pg.evaluate_path(detector, criterion, model, val, rows, args, folder)
        info = state["train_info"]
        summary = {"epoch": epoch, **result,
                   "train_loss": info["mean_loss"], "train_det_loss": info["mean_det_loss"],
                   "train_feat_loss": info["mean_feat_loss"], "learning_rate": info["learning_rate"],
                   "train_seconds": info["train_seconds"], "eval_seconds": time.monotonic() - started,
                   "checkpoint": str(checkpoint), "latent_shape": [128, 256, 16],
                   "actual_transmission": False, "N_symbol": 0}
        (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        pg.write_metrics()
        print(json.dumps({"eval_epoch": epoch, "AP": summary["AP"]}), flush=True)
    (OUT / "eval_complete.json").write_text(json.dumps({"epochs": EPOCHS, "rows": len(pg.write_metrics())}) + "\n")


def report():
    pg.checked_protocol()
    rows = pg.write_metrics()
    if len(rows) != EPOCHS or not all((OUT / name).exists() for name in ("train_complete.json", "eval_complete.json")):
        raise RuntimeError("results incomplete")
    best, final = max(rows, key=lambda x: x["AP"]), rows[-1]
    latent_rows = json.loads((ld.OUT / "metrics.json").read_text())
    latent_best = max(latent_rows, key=lambda x: x["AP"])
    pg_best = json.loads((ld.PG_REFERENCE / "best.json").read_text())["best"]
    full = json.loads((SOURCE / "decision.json").read_text())["best"][str(ld.K_REF)]
    refs = [("直通", "—", pg.REFERENCE), ("PG", 25, pg_best),
            ("完整 codec K=524288", 24, full),
            ("潜变量直传", latent_best["epoch"], latent_best),
            ("冻结 E、训练 D 最佳", best["epoch"], best),
            ("冻结 E、训练 D 最终", 30, final)]
    lines = ["# 冻结语义编码器、训练解码器诊断", "",
             "seed101；高 K 正式第 24 轮检查点初始化 E 与 D。E 与检测器完全冻结；潜变量 z 逐元素直传 D；无实际复符号传输，N_symbol=0、noise=false。", "",
             "## 协议与预检", "",
             "哈希、命令、数据清单见 run/protocol.json、run/*_command.txt；单批次形状、梯度和参数变化见 run/preflight.json。按 val546 AP 最大选模，并列取最早轮。", "",
             "## 同一 val546 结果", "",
             "| 路径 | 轮次 | AP | AP50 | AP75 | AP_small |",
             "|---|---:|---:|---:|---:|---:|",
             *[f'| {name} | {epoch} | {m["AP"]:.3f} | {m["AP50"]:.3f} | {m["AP75"]:.3f} | {m["AP_small"]:.3f} |'
               for name, epoch, m in refs], "",
             f'实验 A 最佳比完整 codec 高 {best["AP"] - full["AP"]:.3f} AP，比 PG 低 {pg_best["AP"] - best["AP"]:.3f} AP。',
             "PG 和直通均无同 K 复符号传输，只作诊断参照。实验 A 也无实际无线传输，不能据此判定完整通信 codec 通过。", "",
             "原完整 codec 每 batch 同时训练三档 K；本诊断固定一个参考容量且从已训练的第 24 轮开始，训练暴露和初始化不同。几何 shape/mask 仍旁路提供。单 seed 不能证明普遍因果。", "",
             "逐轮 CSV/JSON、每轮检查点和逐图 diagnostics.jsonl 均在 run/。未使用 test-dev/test，未启动阶段 C。", ""]
    (HERE / "frozen_decoder_report.md").write_text("\n".join(lines))
    (OUT / "best.json").write_text(json.dumps({"best": best, "final": final}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("preflight", "train", "evaluate", "report"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    pg.HERE, pg.OUT, pg.protocol = HERE, OUT, protocol
    pg.checked_models, pg.validate_predictions = checked_models, ld.validate_predictions
    if args.resume and args.role != "train":
        parser.error("--resume applies only to training")
    try:
        globals()[args.role](args.resume) if args.role == "train" else globals()[args.role]()
    except Exception:
        if args.role == "preflight":
            (HERE / "preflight_failed.txt").write_text(traceback.format_exc())
        if args.role in ("train", "evaluate") and OUT.exists():
            (OUT / f"{args.role}_failed.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()

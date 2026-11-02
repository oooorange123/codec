"""Stage B PG diagnostic: transmit the complete 64-channel grid without a codec."""

import argparse
import csv
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "high_k_sweep_20261023"))
import run_high_k as base  # noqa: E402

run, stage = base.run, base.stage
OUT = HERE / "run_v2"
GRID = (128, 256)
GRID_REAL = 64 * GRID[0] * GRID[1]
EPOCHS = 30
REFERENCE = {"AP": 18.200899376730714, "AP50": 36.047034585951884,
             "AP75": 16.382283898929305, "AP_small": 7.151946437013246}


class PG(nn.Module):
    """Reuse only the warmstarted project and expand modules, with no wire."""

    def __init__(self, source):
        super().__init__()
        self.project, self.expand = source.project, source.expand
        self.grid = tuple(source.grid)
        if self.grid != GRID or len(self.project) != 4 or len(self.expand) != 4:
            raise RuntimeError("PG module count or grid differs from locked codec")
        for project, expand in zip(self.project, self.expand):
            if (project[0].in_channels, project[-1].out_channels,
                    expand[0].in_channels, expand[-1].out_channels) != (256, 16, 64, 256):
                raise RuntimeError("project/expand interface differs from audited 256→16, 64→256")
        self.last_grid_shape = None
        self.last_project_shapes = None

    def forward(self, features, k=0, snr_db=None, noise=False):
        if k != 0 or noise or len(features) != 4 or any(x.shape[1] != 256 for x in features):
            raise ValueError("PG requires four 256-channel maps and no channel")
        maps = []
        projected = []
        for project, feature in zip(self.project, features):
            spatial = project(feature)
            projected.append(list(spatial.shape))
            mode = "area" if any(a > b for a, b in zip(spatial.shape[-2:], self.grid)) else "bilinear"
            kwargs = {} if mode == "area" else {"align_corners": False}
            maps.append(F.interpolate(spatial, size=self.grid, mode=mode, **kwargs))
        grid = torch.cat(maps, 1)
        self.last_grid_shape = list(grid.shape)
        self.last_project_shapes = projected
        restored = [head(F.interpolate(grid, size=x.shape[-2:], mode="bilinear", align_corners=False))
                    for head, x in zip(self.expand, features)]
        empty = grid.new_empty((grid.shape[0], 0, 2))
        return restored, empty, empty


def checked_models(workers):
    base.checked_inputs()
    run.seed_all(101)
    args = base.model_args(workers)
    detector, criterion, source = base.build_models(args)
    pg = PG(source).cuda()
    if any(p.requires_grad for p in detector.parameters()):
        raise RuntimeError("detector is not frozen")
    return args, detector, criterion, pg


def protocol():
    files = {"run_pg.py": Path(__file__), "run_high_k.py": Path(base.__file__),
             "variable_grid_codec.py": HERE.parent / "variable_grid_codec.py",
             "run_server.py": base.PROJECT / "run_server.py",
             "modeling.py": base.PROJECT / "modeling.py",
             "eval_official.m": base.PROJECT / "eval_official.m",
             "eval_size_supplemental.m": base.PROJECT / "eval_size_supplemental.m"}
    return {"stage": "B-PG-no-semantic-bottleneck", "seed": 101, "epochs": EPOCHS,
            "grid": list(GRID), "grid_real_elements": GRID_REAL, "actual_wire_symbols": 0,
            "initialization": str(base.WARMSTART), "warmstart_sha256": base.WARMSTART_SHA,
            "detector": str(base.DETECTOR), "detector_sha256": base.DETECTOR_SHA,
            "manifest_sha256": base.EXPECTED_MANIFEST, "train_images": 6467, "val_images": 546,
            "optimizer": "AdamW", "lr": 2e-4, "weight_decay": 1e-4,
            "lr_schedule": "constant", "batch_size": 2, "amp": False,
            "loss": "detector + 10*valid-region relative feature MSE",
            "noise": False, "validation_selection": "maximum val546 AP, first epoch on tie",
            "code_sha256": {name: stage.sha256(path) for name, path in files.items()}}


def checked_protocol():
    if json.loads((OUT / "protocol.json").read_text()) != protocol():
        raise RuntimeError("PG protocol or code changed")


def preflight():
    if OUT.exists():
        raise RuntimeError("PG output already exists; refusing to overwrite")
    args, detector, criterion, pg = checked_models(0)
    _, train = base.dataset("train")
    samples, targets = next(iter(run.loader(train, 2, 0, 101)))
    samples = samples.to("cuda")
    targets = run.move_targets(targets, "cuda")
    pg.train()
    optimizer = torch.optim.AdamW(pg.parameters(), lr=2e-4, weight_decay=1e-4)
    before = {name: p.detach().clone() for name, p in pg.named_parameters()}
    with run.linked_forward(detector, pg, 0, 10, noise=False) as capture:
        output = detector(samples)
    det = run.weighted_loss(criterion(output, targets), criterion)
    feat = run.feature_loss(capture)
    loss = det + 10 * feat
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite PG preflight loss")
    loss.backward()
    gradients = {name: float(p.grad.abs().sum()) if p.grad is not None else 0.0
                 for name, p in pg.named_parameters()}
    if any(value <= 0 for value in gradients.values()) or any(p.grad is not None for p in detector.parameters()):
        raise RuntimeError("PG gradient missing or detector received gradients")
    optimizer.step()
    changes = {name: float((p.detach() - before[name]).abs().sum())
               for name, p in pg.named_parameters()}
    if any(value <= 0 for value in changes.values()):
        raise RuntimeError("PG parameter did not update")
    shapes = {"detector_input": list(samples.tensors.shape),
              "project_input": [list(x.shape) for x in capture["original"]],
              "project_output_native": pg.last_project_shapes,
              "project_output_grid": [[x.shape[0], 16, *pg.grid] for x in capture["original"]],
              "grid": pg.last_grid_shape,
              "expand_output": [list(x.shape) for x in capture["restored"]],
              "wire": list(capture["wire"].shape)}
    if (shapes["grid"] != [2, 64, *GRID] or shapes["wire"] != [2, 0, 2] or
            shapes["project_input"] != shapes["expand_output"]):
        raise RuntimeError(f"PG tensor shape mismatch: {shapes}")
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "protocol.json").write_text(json.dumps(protocol(), indent=2) + "\n")
    (OUT / "preflight.json").write_text(json.dumps({"shapes": shapes, "det_loss": det.item(),
        "feature_loss": feat.item(), "total_loss": loss.item(), "grad_abs_sum": gradients,
        "parameter_change_abs_sum": changes, "detector_frozen": True,
        "semantic_modules_used": False}, indent=2) + "\n")
    print(json.dumps({"preflight": "passed", "shapes": shapes, "loss": loss.item()}), flush=True)


def validate_predictions(folder, val_rows, pg):
    expected = {Path(row["image_path"]).stem for row in val_rows}
    if {p.stem for p in folder.glob("*.txt")} != expected:
        raise RuntimeError("PG prediction names differ from val546")
    lines = []
    for line in (folder / "diagnostics.jsonl").read_text().splitlines():
        record = json.loads(line)
        if (record["N_symbol"] != 0 or record["wire_bytes"] != 0 or
                record["noise"] or record["K_sem"] is not None or
                len(record["level_shapes"]) != 4):
            raise RuntimeError("PG exported a nonzero wire or wrong feature levels")
        record.update(K=None, SNR_dB=None, complex_power=None, actual_transmission=False,
                      grid_real_elements=GRID_REAL)
        lines.append(json.dumps(record, allow_nan=False))
    if len(lines) != 546:
        raise RuntimeError("PG per-image diagnostics incomplete")
    (folder / "diagnostics.jsonl").write_text("\n".join(lines) + "\n")
    return [json.loads(line) for line in lines]


def evaluate_path(detector, criterion, codec, val, rows, args, folder):
    predictions = folder / "predictions"
    run.export_predictions(detector, codec, val, predictions, args, 0 if codec else None,
                           10 if codec else None, False, save_wire=False,
                           criterion=criterion if codec else None, lambda_feat=10.0)
    expected = {Path(row["image_path"]).stem for row in rows}
    if {p.stem for p in predictions.glob("*.txt")} != expected:
        raise RuntimeError("prediction names differ from val546")
    records = validate_predictions(predictions, rows, codec) if codec else None
    eval_dir = base.AUDIT / "stageB_seed101/validation_data"
    with ThreadPoolExecutor(max_workers=2) as pool:
        official = pool.submit(run.evaluate, predictions, eval_dir, args)
        small = pool.submit(stage.size_metrics, predictions, eval_dir, args.eval_bin)
        scores, sizes = official.result(), small.result()
    return {"AP": scores["AP"], "AP50": scores["AP50"], "AP75": scores["AP75"],
            "AP_small": sizes["AP_small_supplemental"],
            "validation_loss": json.loads((predictions / "validation_loss.json").read_text())["mean_loss"]
                               if codec else None,
            "mean_relative_mse_per_level": [sum(row["relative_mse"][i] for row in records) / 546
                                            for i in range(4)] if codec else None,
            "official": scores, "size_supplemental": sizes}


def direct():
    checked_protocol()
    if not (OUT / "preflight.json").exists() or (OUT / "direct/summary.json").exists():
        raise RuntimeError("preflight absent or direct retest already exists")
    args, detector, criterion, _ = checked_models(4)
    rows, val = base.dataset("val")
    folder = OUT / "direct"
    folder.mkdir(exist_ok=False)
    result = evaluate_path(detector, criterion, None, val, rows, args, folder)
    result["historic_difference"] = {name: result[name] - value for name, value in REFERENCE.items()}
    result["matches_historic"] = all(abs(x) <= 0.02 for x in result["historic_difference"].values())
    (folder / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["matches_historic"]:
        raise RuntimeError("direct detector AP differs from locked reference; training blocked")
    print(json.dumps({"direct_retest": result["historic_difference"]}), flush=True)


def train_epoch(detector, criterion, pg, optimizer, data, args, epoch, step):
    detector.eval()
    pg.train()
    totals = {"loss": 0.0, "det": 0.0, "feature": 0.0}
    gradients = {name: False for name, _ in pg.named_parameters()}
    images = batches = 0
    for samples, targets in run.loader(data, 2, 6, 101 + epoch):
        samples = samples.to("cuda")
        targets = run.move_targets(targets, "cuda")
        optimizer.zero_grad(set_to_none=True)
        with run.linked_forward(detector, pg, 0, 10, noise=False) as capture:
            output = detector(samples)
        det = run.weighted_loss(criterion(output, targets), criterion)
        feat = run.feature_loss(capture)
        loss = det + 10 * feat
        if not torch.isfinite(loss) or capture["wire"].shape != (len(targets), 0, 2):
            raise RuntimeError("nonfinite PG loss or nonempty wire")
        loss.backward()
        for name, p in pg.named_parameters():
            if not gradients[name]:
                gradients[name] = p.grad is not None and bool(torch.any(p.grad != 0))
        norm = torch.nn.utils.clip_grad_norm_(pg.parameters(), args.clip_max_norm)
        if not torch.isfinite(norm) or norm <= 0 or any(p.grad is not None for p in detector.parameters()):
            raise RuntimeError("PG gradients invalid or detector unfrozen")
        optimizer.step()
        for name, value in (("loss", loss), ("det", det), ("feature", feat)):
            totals[name] += value.item()
        images += len(targets)
        batches += 1
        step += 1
    if images != 6467 or not all(gradients.values()):
        raise RuntimeError("PG epoch image count or module gradients incomplete")
    return step, {"epoch": epoch + 1, "images": images, "batches": batches,
                  "optimizer_steps": step, "learning_rate": optimizer.param_groups[0]["lr"],
                  "mean_loss": totals["loss"] / batches, "mean_det_loss": totals["det"] / batches,
                  "mean_feat_loss": totals["feature"] / batches,
                  "all_project_expand_gradients_nonzero": True}


def train(resume):
    checked_protocol()
    if not json.loads((OUT / "direct/summary.json").read_text())["matches_historic"]:
        raise RuntimeError("direct retest did not match reference")
    args, detector, criterion, pg = checked_models(6)
    _, data = base.dataset("train")
    optimizer = torch.optim.AdamW(pg.parameters(), lr=2e-4, weight_decay=1e-4)
    existing = [e for e in range(1, EPOCHS + 1) if (OUT / f"epoch_{e:02d}/pg.pt").exists()]
    if existing != list(range(1, len(existing) + 1)) or bool(existing) != bool(resume):
        raise RuntimeError("resume requires contiguous PG checkpoints and explicit --resume")
    step = 0
    if existing:
        state = run.load_file(OUT / f"epoch_{existing[-1]:02d}/pg.pt")
        if state["protocol"] != protocol() or state["epoch"] != existing[-1]:
            raise RuntimeError("PG resume checkpoint protocol mismatch")
        pg.load_state_dict(state["pg"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        run.restore_rng(state["rng"])
        step = state["step"]
    (OUT / "train_command.txt").write_text(" ".join(sys.argv) + "\n")
    for epoch in range(len(existing), EPOCHS):
        started = time.monotonic()
        step, info = train_epoch(detector, criterion, pg, optimizer, data, args, epoch, step)
        info["train_seconds"] = time.monotonic() - started
        folder = OUT / f"epoch_{epoch + 1:02d}"
        folder.mkdir(exist_ok=False)
        checkpoint = folder / "pg.pt"
        base.save_atomic(checkpoint, {"pg": pg.state_dict(), "optimizer": optimizer.state_dict(),
                                     "rng": run.rng_state(), "epoch": epoch + 1, "step": step,
                                     "protocol": protocol(), "train_info": info})
        (folder / "train_summary.json").write_text(json.dumps({**info, "checkpoint": str(checkpoint)}, indent=2) + "\n")
        print(json.dumps({"train": info, "checkpoint": str(checkpoint)}), flush=True)
    (OUT / "train_complete.json").write_text(json.dumps({"epochs": EPOCHS, "step": step}) + "\n")


def write_metrics():
    fields = ("epoch", "AP", "AP50", "AP75", "AP_small", "train_loss", "train_det_loss",
              "train_feat_loss", "validation_loss", "learning_rate", "train_seconds",
              "eval_seconds", "checkpoint")
    rows = []
    for epoch in range(1, EPOCHS + 1):
        path = OUT / f"epoch_{epoch:02d}/summary.json"
        if path.exists():
            row = json.loads(path.read_text())
            rows.append({key: row[key] for key in fields})
    (OUT / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (OUT / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def evaluate():
    checked_protocol()
    args, detector, criterion, pg = checked_models(4)
    rows, val = base.dataset("val")
    (OUT / "eval_command.txt").write_text(" ".join(sys.argv) + "\n")
    for epoch in range(1, EPOCHS + 1):
        checkpoint = OUT / f"epoch_{epoch:02d}/pg.pt"
        while not checkpoint.exists():
            if (OUT / "train_failed.txt").exists():
                raise RuntimeError("PG trainer failed")
            time.sleep(20)
        folder = checkpoint.parent
        if (folder / "summary.json").exists():
            continue
        state = run.load_file(checkpoint)
        if state["epoch"] != epoch or state["protocol"] != protocol():
            raise RuntimeError("PG evaluation checkpoint changed")
        pg.load_state_dict(state["pg"], strict=True)
        started = time.monotonic()
        result = evaluate_path(detector, criterion, pg, val, rows, args, folder)
        info = state["train_info"]
        summary = {"epoch": epoch, **result, "train_loss": info["mean_loss"],
                   "train_det_loss": info["mean_det_loss"], "train_feat_loss": info["mean_feat_loss"],
                   "learning_rate": info["learning_rate"], "train_seconds": info["train_seconds"],
                   "eval_seconds": time.monotonic() - started, "checkpoint": str(checkpoint),
                   "grid_real_elements_per_image": GRID_REAL, "actual_wire_symbols_per_image": 0}
        (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        write_metrics()
        print(json.dumps({"eval_epoch": epoch, "AP": summary["AP"],
                          "AP_small": summary["AP_small"]}), flush=True)
    (OUT / "eval_complete.json").write_text(json.dumps({"epochs": EPOCHS, "rows": len(write_metrics())}) + "\n")


def report():
    checked_protocol()
    rows = write_metrics()
    if len(rows) != EPOCHS or not (OUT / "train_complete.json").exists():
        raise RuntimeError("PG training or evaluation incomplete")
    best = max(rows, key=lambda row: row["AP"])
    final = rows[-1]
    high = json.loads((base.OUT / "decision.json").read_text())["best"]
    lines = ["# PG 无语义瓶颈诊断：seed101", "",
             "PG 直接交付 64×128×256=2,097,152 个 float32 实数的完整特征网格（按内存表示为 8,388,608 字节/图）；没有生成或发送复信道符号。它是诊断路径，不是可部署通信方案，也不与任何 K 等码率。", "",
             "## 协议与预检", "",
             "同一冻结 detector、train6467/val546、seed101、warmstart、AdamW 2e-4/1e-4、batch 2、FP32、30 轮；检测损失 + 10×有效区域相对特征 MSE；每轮按同一官方评估器和补充 AP_small 评估。最佳检查点以 val AP 选择。", "",
             "`project_i`: 256→16；四级各自上采样至 128×256 后按原顺序拼接成 64 通道；完整网格分别缩放到四级原尺寸，再由对应 `expand_i`: 64→256。语义编码器、符号生成、信道映射和语义解码器均未参与。", "",
             "预检细节、形状、梯度和参数变化见 `run/preflight.json`；代码/数据/检测器哈希见 `run/protocol.json`；命令见 `run/*_command.txt`。", "",
             "## 验证结果", "",
             "| 路径 | 轮次 | AP | AP50 | AP75 | AP_small |", "|---|---:|---:|---:|---:|---:|",
             f'| 直通复测 | — | {json.loads((OUT / "direct/summary.json").read_text())["AP"]:.3f} | {json.loads((OUT / "direct/summary.json").read_text())["AP50"]:.3f} | {json.loads((OUT / "direct/summary.json").read_text())["AP75"]:.3f} | {json.loads((OUT / "direct/summary.json").read_text())["AP_small"]:.3f} |',
             f'| PG 最佳 | {best["epoch"]} | {best["AP"]:.3f} | {best["AP50"]:.3f} | {best["AP75"]:.3f} | {best["AP_small"]:.3f} |',
             f'| PG 最终 | {final["epoch"]} | {final["AP"]:.3f} | {final["AP50"]:.3f} | {final["AP75"]:.3f} | {final["AP_small"]:.3f} |']
    for k in (131072, 262144, 524288):
        item = high[str(k)]
        lines.append(f'| 完整 codec K={k} | {item["epoch"]} | {item["AP"]:.3f} | {item["AP50"]:.3f} | {item["AP75"]:.3f} | {item["AP_small"]:.3f} |')
    direct_ap = json.loads((OUT / "direct/summary.json").read_text())["AP"]
    high_ap = max(high[str(k)]["AP"] for k in (131072, 262144, 524288))
    lines += ["", f'PG 最佳比最高的完整 codec 高 {best["AP"] - high_ap:.3f} AP 点，比直通低 {direct_ap - best["AP"]:.3f} AP 点。',
              "若 PG 明显高于完整 codec，损失更可能位于 PG 旁路的语义编码/信道映射/解码路径，但本实验无法细分。若 PG 仍低于直通，project、网格或 expand 仍可能造成损失。单 seed 只提供诊断证据。", "",
              "几何 shape/mask 由接口旁路提供；未使用 test-dev/test、未加噪声、未启动阶段 C。逐轮 CSV/JSON、逐图诊断和最佳/最终检查点均在 `run/`。", ""]
    (HERE / "pg_report.md").write_text("\n".join(lines))
    (OUT / "best.json").write_text(json.dumps({"best": best, "final": final}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("preflight", "direct", "train", "evaluate", "report"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and args.role != "train":
        parser.error("--resume applies only to training")
    try:
        globals()[args.role]() if args.role != "train" else train(args.resume)
    except Exception:
        if args.role in ("train", "evaluate") and OUT.exists():
            (OUT / f"{args.role}_failed.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()

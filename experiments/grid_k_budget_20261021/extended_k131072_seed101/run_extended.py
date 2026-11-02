"""Train and evaluate one fixed-K codec for the planned 30 epochs."""

import argparse
import contextlib
import csv
import io
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
GRID_ROOT = HERE.parent
PROJECT = HERE.parents[2]
sys.path[:0] = [str(PROJECT), str(GRID_ROOT)]
import run_server as run  # noqa: E402
import run_v2_stage_b as stage  # noqa: E402
from variable_grid_codec import VariableGridCodec  # noqa: E402

AUDIT = PROJECT / "experiments/codec_k_diagnosis/v2_20261019"
DETECTOR = PROJECT / "runs/det_seed42/best.pt"
WARMSTART = AUDIT / "stageB_seed101/best.pt"
OLD_PILOT = GRID_ROOT / "results/pilot_128x256_seed101"
K = 131072
EPOCHS = 30
GRID = (128, 256)
SHAPES = ((94, 167), (47, 84), (24, 42), (12, 21))
DETECTOR_SHA = "3f85d3f8a70bcebf673f59b6eabdb5407da3e560fba34aeb01385a6b0aae4bc5"
WARMSTART_SHA = "6b74d443714c5741f25c3263731c6809caae5c51c4d19f27e5f0f1f8170eaebb"
FIELDS = ("epoch", "AP", "AP50", "AP75", "AP_small_supplemental",
          "train_loss", "train_det_loss", "train_feat_loss", "validation_loss",
          "learning_rate", "train_seconds", "eval_seconds", "checkpoint")


def save_atomic(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_metrics(out):
    rows = []
    for epoch in range(1, EPOCHS + 1):
        path = out / f"epoch_{epoch:02d}/summary.json"
        if path.exists():
            item = json.loads(path.read_text())
            rows.append({key: item[key] for key in FIELDS})
    (out / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def evaluate_epoch(detector, criterion, codec, val, val_rows, args, out, epoch,
                   train_info, train_seconds, checkpoint):
    started = time.monotonic()
    folder = out / f"epoch_{epoch:02d}"
    predictions = folder / "predictions"
    run.export_predictions(detector, codec, val, predictions, args, K, 10, False,
                           save_wire=False, criterion=criterion, lambda_feat=10.0)
    records = [json.loads(line) for line in (predictions / "diagnostics.jsonl").read_text().splitlines()]
    expected = {Path(row["image_path"]).stem for row in val_rows}
    if (len(records) != 546 or {item["image"] for item in records} != expected or
            {path.stem for path in predictions.glob("*.txt")} != expected or
            any(item["K_sem"] != K or item["N_symbol"] != K or
                item["wire_bytes"] != 8 * K or item["noise"] or
                abs(item["complex_power"] - 1) > 1e-5 or
                tuple(map(tuple, item["level_shapes"])) != SHAPES for item in records)):
        raise RuntimeError(f"epoch {epoch}: invalid val list, actual wire, power or feature shapes")
    eval_dir = AUDIT / "stageB_seed101/validation_data"
    with ThreadPoolExecutor(max_workers=2) as pool:
        official = pool.submit(run.evaluate, predictions, eval_dir, args)
        small = pool.submit(stage.size_metrics, predictions, eval_dir, args.eval_bin)
        scores, size = official.result(), small.result()
    validation_loss = json.loads((predictions / "validation_loss.json").read_text())["mean_loss"]
    info = {
        "epoch": epoch, "seed": 101, "K_complex": K, "images": len(records),
        "AP": scores["AP"], "AP50": scores["AP50"], "AP75": scores["AP75"],
        "AP_small_supplemental": size["AP_small_supplemental"],
        "train_loss": train_info["mean_loss"], "train_det_loss": train_info["mean_det_loss"],
        "train_feat_loss": train_info["mean_feat_loss"], "validation_loss": validation_loss,
        "learning_rate": 2e-4, "train_seconds": train_seconds,
        "eval_seconds": time.monotonic() - started, "checkpoint": str(checkpoint),
        "mean_complex_power": sum(item["complex_power"] for item in records) / len(records),
        "mean_relative_mse_per_level": [sum(item["relative_mse"][i] for item in records) / len(records)
                                        for i in range(4)],
        "feature_shapes": SHAPES, "actual_complex_symbols_per_image": K,
        "official": scores, "size_supplemental": size,
    }
    (folder / "summary.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if stage.sha256(DETECTOR) != DETECTOR_SHA or stage.sha256(WARMSTART) != WARMSTART_SHA:
        parser.error("locked detector or original initialization changed")
    old = json.loads((OLD_PILOT / "protocol.json").read_text())
    if (old["grid"] != list(GRID) or old["seed"] != 101 or
            old["K_complex"] != [131072, 262144, 524288] or old["epochs"] != 5):
        parser.error("original five-epoch pilot protocol changed")
    if args.resume:
        if not args.out.is_dir():
            parser.error("resume directory does not exist")
    else:
        args.out.mkdir(parents=True, exist_ok=False)
    model_args = argparse.ArgumentParser(parents=[run.get_args_parser()]).parse_args([])
    model_args.dataset_file = "coco"
    model_args.device = "cuda"
    model_args.seed = 101
    model_args.epochs = EPOCHS
    model_args.batch_size = 2
    model_args.num_workers = 6
    model_args.lr = 2e-4
    model_args.weight_decay = 1e-4
    model_args.lambda_feat = 10.0
    model_args.train_k = K
    model_args.train_snr = 10
    model_args.no_noise = True
    model_args.amp = False
    model_args.eval_bin = str(PROJECT / "runtime/octave_eval.sh")
    model_args.eval_backend = "octave"
    model_args.ablate_fraction = 0
    model_args.ablate_mode = "random"
    model_args.renormalize = False
    hashes = {name: stage.sha256(path) for name, path in {
        "run_extended.py": Path(__file__), "run_server.py": PROJECT / "run_server.py",
        "variable_grid_codec.py": GRID_ROOT / "variable_grid_codec.py",
        "visdrone_data.py": PROJECT / "visdrone_data.py",
        "run_v2_stage_b.py": PROJECT / "run_v2_stage_b.py",
        "eval_official.m": PROJECT / "eval_official.m",
        "eval_size_supplemental.m": PROJECT / "eval_size_supplemental.m",
    }.items()}
    protocol = {"experiment": "fixed-K training duration", "seed": 101, "epochs": EPOCHS,
                "grid": GRID, "K_complex": K, "train_images": 6467, "val_images": 546,
                "detector_sha256": DETECTOR_SHA, "warmstart_sha256": WARMSTART_SHA,
                "initialization": str(WARMSTART), "optimizer": "AdamW", "lr": 2e-4,
                "lr_schedule": "constant; no scheduler state", "weight_decay": 1e-4,
                "batch_size": 2, "workers": 6, "amp": False,
                "loss": "detector + 10*valid-region relative feature MSE",
                "noise": False, "mean_complex_symbol_power": 1,
                "augmentation": "VisDrone uses pinned Deformable-DETR train transforms; val transforms for val546",
                "old_pilot_training_K": old["K_complex"],
                "old_pilot_not_matched_fixed_K": True,
                "train_manifest_sha256": stage.sha256(AUDIT / "train_manifest.csv"),
                "val_manifest_sha256": stage.sha256(AUDIT / "val_manifest.csv"),
                "code_sha256": hashes}
    protocol_path = args.out / "protocol.json"
    if args.resume:
        if json.loads(protocol_path.read_text()) != protocol:
            parser.error("resume protocol or code differs")
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
        (args.out / "command.txt").write_text(" ".join(sys.argv) + "\n")
    run.seed_all(101)
    state = run.load_file(DETECTOR)
    if not state.get("gate_pass") or state.get("attention_backend") != run.ATTENTION_BACKEND:
        raise RuntimeError("detector gate or attention backend changed")
    detector, criterion = run.build_detector(model_args)
    detector.load_state_dict(state["model"], strict=True)
    detector.to("cuda").eval().requires_grad_(False)
    criterion.to("cuda").eval()
    codec = VariableGridCodec(GRID).to("cuda")
    codec.load_state_dict(run.load_file(WARMSTART)["codec"], strict=True)
    optimizer = torch.optim.AdamW(codec.parameters(), lr=2e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    train_rows = stage.manifest_rows(AUDIT, "train")
    val_rows = stage.manifest_rows(AUDIT, "val")
    train = stage.select_dataset(run.VisDrone(PROJECT / "../../data", "train"), train_rows)
    val = stage.select_dataset(run.VisDrone(PROJECT / "../../data", "val"), val_rows)
    assert len(train) == 6467 and len(val) == 546
    assert sum(p.numel() for p in codec.parameters()) == 920784
    assert len(list((AUDIT / "stageB_seed101/validation_data/images").glob("*.jpg"))) == 546
    existing = [epoch for epoch in range(1, EPOCHS + 1)
                if (args.out / f"epoch_{epoch:02d}/codec.pt").exists()]
    if existing and existing != list(range(1, existing[-1] + 1)):
        raise RuntimeError("noncontiguous training checkpoints; refusing mixed trajectories")
    step = 0
    for epoch in range(EPOCHS):
        folder = args.out / f"epoch_{epoch + 1:02d}"
        folder.mkdir(exist_ok=True)
        checkpoint = folder / "codec.pt"
        if checkpoint.exists():
            state = run.load_file(checkpoint)
            if (state["epoch"] != epoch or state["seed"] != 101 or state["K_complex"] != K or
                    tuple(state["grid"]) != GRID or state["detector_sha256"] != DETECTOR_SHA):
                raise RuntimeError(f"checkpoint protocol mismatch at epoch {epoch + 1}")
            codec.load_state_dict(state["codec"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            run.restore_rng(state["rng"])
            step = state["step"]
            train_info, train_seconds = state["train_info"], state["train_seconds"]
        else:
            started = time.monotonic()
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                step = run.train_epoch(detector, criterion, codec, optimizer, scaler,
                                       train, model_args, epoch, step)
            train_seconds = time.monotonic() - started
            train_info = json.loads(captured.getvalue().splitlines()[-1])
            if (train_info["K_SNR_counts"] != {f"{K}/10": 6467} or
                    any(p.grad is not None for p in detector.parameters()) or
                    optimizer.param_groups[0]["lr"] != 2e-4):
                raise RuntimeError("K, detector freeze or constant LR changed")
            save_atomic(checkpoint, {"codec": codec.state_dict(),
                                     "optimizer": optimizer.state_dict(), "rng": run.rng_state(),
                                     "epoch": epoch, "step": step, "seed": 101,
                                     "grid": GRID, "K_complex": K,
                                     "detector_sha256": DETECTOR_SHA,
                                     "train_info": train_info, "train_seconds": train_seconds})
            print(json.dumps({"epoch": epoch + 1, "train": train_info,
                              "train_seconds": train_seconds}), flush=True)
        training_rng = run.rng_state()
        summary = folder / "summary.json"
        if not summary.exists():
            info = evaluate_epoch(detector, criterion, codec, val, val_rows, model_args,
                                  args.out, epoch + 1, train_info, train_seconds, checkpoint)
            print(json.dumps({"epoch": epoch + 1, "AP": info["AP"],
                              "AP_small": info["AP_small_supplemental"],
                              "validation_loss": info["validation_loss"]}), flush=True)
        run.restore_rng(training_rng)  # Keep validation from changing the next training epoch.
        write_metrics(args.out)
    rows = write_metrics(args.out)
    if len(rows) != EPOCHS:
        raise RuntimeError("incomplete epoch metrics")
    best = max(rows, key=lambda item: item["AP"])
    shutil.copy2(best["checkpoint"], args.out / "best.pt")
    (args.out / "best.json").write_text(json.dumps(best, indent=2) + "\n")
    print(json.dumps({"complete": True, "best_epoch": best["epoch"],
                      "best_AP": best["AP"]}), flush=True)


if __name__ == "__main__":
    main()

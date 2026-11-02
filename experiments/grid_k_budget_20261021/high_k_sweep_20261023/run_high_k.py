"""Balanced 30-epoch conditional-K training and independent val546 evaluation."""

import argparse
import csv
import json
import os
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
GRID_ROOT = HERE.parent
PROJECT = HERE.parents[2]
sys.path[:0] = [str(PROJECT), str(GRID_ROOT), str(GRID_ROOT / "extended_k131072_seed101")]
import run_server as run  # noqa: E402
import run_v2_stage_b as stage  # noqa: E402
from run_extended import save_atomic  # noqa: E402
from variable_grid_codec import K_VALUES, VariableGridCodec  # noqa: E402

OUT = HERE / "run_v2"
AUDIT = PROJECT / "experiments/codec_k_diagnosis/v2_20261019"
DETECTOR = PROJECT / "runs/det_seed42/best.pt"
WARMSTART = AUDIT / "stageB_seed101/best.pt"
GRID = (128, 256)
EPOCHS = 30
SHAPES = ((94, 167), (47, 84), (24, 42), (12, 21))
DETECTOR_SHA = "3f85d3f8a70bcebf673f59b6eabdb5407da3e560fba34aeb01385a6b0aae4bc5"
WARMSTART_SHA = "6b74d443714c5741f25c3263731c6809caae5c51c4d19f27e5f0f1f8170eaebb"
EXPECTED_MANIFEST = {"train": "840670f4b4f4b5f32117844978ad1213b3bee899f136fbe611116b5c48c512e0",
                     "val": "3af48a18c687df3ebd41973731ded6f3eac1e2e43ccb965d3dc4bbb5fd7351c3"}


def model_args(workers):
    args = argparse.ArgumentParser(parents=[run.get_args_parser()]).parse_args([])
    args.dataset_file = "coco"
    args.device = "cuda"
    args.seed = 101
    args.epochs = EPOCHS
    args.batch_size = 2
    args.num_workers = workers
    args.lr = 2e-4
    args.weight_decay = 1e-4
    args.lambda_feat = 10.0
    args.train_snr = 10
    args.no_noise = True
    args.amp = False
    args.eval_bin = str(PROJECT / "runtime/octave_eval.sh")
    args.eval_backend = "octave"
    args.ablate_fraction = 0
    args.ablate_mode = "random"
    args.renormalize = False
    return args


def protocol():
    files = {"run_high_k.py": Path(__file__),
             "run_server.py": PROJECT / "run_server.py",
             "variable_grid_codec.py": GRID_ROOT / "variable_grid_codec.py",
             "modeling.py": PROJECT / "modeling.py",
             "visdrone_data.py": PROJECT / "visdrone_data.py",
             "run_v2_stage_b.py": PROJECT / "run_v2_stage_b.py"}
    return {"stage": "B-directed-high-K", "seed": 101, "epochs": EPOCHS,
            "grid": list(GRID), "K_complex": list(K_VALUES),
            "q_per_grid_cell": {str(k): k // (GRID[0] * GRID[1]) for k in K_VALUES},
            "training": "every batch covers all K; mean of three losses; one optimizer step",
            "train_images": 6467, "val_images": 546, "batch_size": 2,
            "optimizer": "AdamW", "lr": 2e-4, "weight_decay": 1e-4,
            "lr_schedule": "constant", "amp": False,
            "loss": "detector + 10*valid-region relative feature MSE",
            "noise": False, "SNR_dB_metadata_only": 10,
            "mean_complex_symbol_power": 1, "detector_sha256": DETECTOR_SHA,
            "warmstart_sha256": WARMSTART_SHA,
            "manifest_sha256": EXPECTED_MANIFEST,
            "code_sha256": {name: stage.sha256(path) for name, path in files.items()}}


def checked_inputs():
    if stage.sha256(DETECTOR) != DETECTOR_SHA or stage.sha256(WARMSTART) != WARMSTART_SHA:
        raise RuntimeError("locked detector or original codec initialization changed")
    for split, expected in EXPECTED_MANIFEST.items():
        if stage.sha256(AUDIT / f"{split}_manifest.csv") != expected:
            raise RuntimeError(f"{split} manifest changed")


def build_models(args):
    state = run.load_file(DETECTOR)
    if not state.get("gate_pass") or state.get("attention_backend") != run.ATTENTION_BACKEND:
        raise RuntimeError("frozen detector protocol changed")
    detector, criterion = run.build_detector(args)
    detector.load_state_dict(state["model"], strict=True)
    detector.cuda().eval().requires_grad_(False)
    criterion.cuda().eval()
    codec = VariableGridCodec(GRID).cuda().eval()
    codec.load_state_dict(run.load_file(WARMSTART)["codec"], strict=True)
    if sum(p.numel() for p in codec.parameters()) != 920784:
        raise RuntimeError("codec structure or parameter count changed")
    return detector, criterion, codec


def verify_mapping(codec):
    features = [torch.zeros(1, 256, h, w, device="cuda")
                for h, w in ((8, 12), (4, 6), (2, 3), (1, 2))]
    mapping = {}
    with torch.inference_mode():
        for k in K_VALUES:
            z, wire = codec.encode(features, k)
            q = k // (GRID[0] * GRID[1])
            if codec.check_k(k) != q or z.shape != (1, *GRID, q) or wire.shape != (1, k, 2):
                raise RuntimeError(f"actual semantic/wire mapping differs for K={k}")
            power = wire.square().sum(-1).mean().item()
            if abs(power - 1) > 1e-5:
                raise RuntimeError(f"complex power differs for K={k}: {power}")
            mapping[str(k)] = {"q": q, "latent_shape": list(z.shape),
                               "wire_shape": list(wire.shape), "mean_complex_power": power}
    return mapping


def dataset(split):
    rows = stage.manifest_rows(AUDIT, split)
    selected = stage.select_dataset(run.VisDrone(PROJECT / "../../data", split), rows)
    if len(selected) != (6467 if split == "train" else 546):
        raise RuntimeError(f"locked {split} image count changed")
    return rows, selected


def train_epoch_balanced(detector, criterion, codec, optimizer, data, args, epoch, step):
    detector.eval()
    codec.train()
    counts = Counter()
    sums = {k: {"loss": 0.0, "det": 0.0, "feat": 0.0} for k in K_VALUES}
    batches = 0
    for samples, targets in run.loader(data, 2, args.num_workers, args.seed + epoch):
        samples = samples.to("cuda")
        targets = run.move_targets(targets, "cuda")
        optimizer.zero_grad(set_to_none=True)
        for k in K_VALUES:
            with run.linked_forward(detector, codec, k, 10, noise=False) as capture:
                output = detector(samples)
            det = run.weighted_loss(criterion(output, targets), criterion)
            feat = run.feature_loss(capture)
            loss = det + 10 * feat
            wire = capture["wire"]
            if wire.shape != (len(targets), k, 2) or not torch.isfinite(loss):
                raise RuntimeError(f"invalid wire or loss: epoch={epoch + 1} K={k}")
            (loss / len(K_VALUES)).backward()
            counts[k] += len(targets)
            sums[k]["loss"] += loss.item()
            sums[k]["det"] += det.item()
            sums[k]["feat"] += feat.item()
        norm = torch.nn.utils.clip_grad_norm_(codec.parameters(), args.clip_max_norm)
        if not torch.isfinite(norm) or norm <= 0 or any(p.grad is not None for p in detector.parameters()):
            raise RuntimeError(f"invalid gradients or unfrozen detector: epoch={epoch + 1}")
        optimizer.step()
        batches += 1
        step += 1
    if counts != Counter({k: len(data) for k in K_VALUES}):
        raise RuntimeError(f"unbalanced actual K exposure: {counts}")
    return step, {"epoch": epoch + 1, "batches_per_K": batches,
                  "images_per_K": dict(counts), "optimizer_steps": step,
                  "mean_loss_per_K": {str(k): {name: value / batches for name, value in sums[k].items()}
                                      for k in K_VALUES},
                  "learning_rate": optimizer.param_groups[0]["lr"]}


def write_train_metrics():
    rows = []
    for epoch in range(1, EPOCHS + 1):
        path = OUT / f"epoch_{epoch:02d}/train_summary.json"
        if not path.exists():
            break
        rows.append(json.loads(path.read_text()))
    (OUT / "train_metrics.json").write_text(json.dumps(rows, indent=2) + "\n")


def train(resume):
    checked_inputs()
    current_protocol = protocol()
    if resume:
        if json.loads((OUT / "protocol.json").read_text()) != current_protocol:
            raise RuntimeError("resume code or protocol changed")
    else:
        OUT.mkdir(parents=True, exist_ok=False)
        (OUT / "protocol.json").write_text(json.dumps(current_protocol, indent=2) + "\n")
        (OUT / "train_command.txt").write_text(" ".join(sys.argv) + "\n")
    run.seed_all(101)
    args = model_args(6)
    detector, criterion, codec = build_models(args)
    mapping = verify_mapping(codec)
    if resume:
        if json.loads((OUT / "mapping_check.json").read_text()) != mapping:
            raise RuntimeError("K mapping changed on resume")
    else:
        (OUT / "mapping_check.json").write_text(json.dumps(mapping, indent=2) + "\n")
    _, data = dataset("train")
    optimizer = torch.optim.AdamW(codec.parameters(), lr=2e-4, weight_decay=1e-4)
    existing = [epoch for epoch in range(1, EPOCHS + 1)
                if (OUT / f"epoch_{epoch:02d}/codec.pt").exists()]
    if existing and (not resume or existing != list(range(1, existing[-1] + 1))):
        raise RuntimeError("existing checkpoints require explicit contiguous resume")
    step = 0
    if existing:
        state = run.load_file(OUT / f"epoch_{existing[-1]:02d}/codec.pt")
        codec.load_state_dict(state["codec"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        run.restore_rng(state["rng"])
        step = state["step"]
    for epoch in range(len(existing), EPOCHS):
        started = time.monotonic()
        step, info = train_epoch_balanced(detector, criterion, codec, optimizer,
                                          data, args, epoch, step)
        seconds = time.monotonic() - started
        folder = OUT / f"epoch_{epoch + 1:02d}"
        folder.mkdir(exist_ok=False)
        checkpoint = folder / "codec.pt"
        save_atomic(checkpoint, {"codec": codec.state_dict(),
                                 "optimizer": optimizer.state_dict(), "rng": run.rng_state(),
                                 "epoch": epoch + 1, "step": step, "seed": 101,
                                 "grid": GRID, "K_complex": K_VALUES,
                                 "detector_sha256": DETECTOR_SHA,
                                 "train_info": info, "train_seconds": seconds})
        (folder / "train_summary.json").write_text(json.dumps({
            "epoch": epoch + 1, "checkpoint": str(checkpoint),
            "optimizer_steps": step, "train_seconds": seconds,
            "batches_per_K": info["batches_per_K"],
            "images_per_K": info["images_per_K"],
            "mean_loss_per_K": info["mean_loss_per_K"]}, indent=2) + "\n")
        write_train_metrics()
        print(json.dumps({"train": info, "seconds": seconds, "checkpoint": str(checkpoint)}), flush=True)
    (OUT / "train_complete.json").write_text(json.dumps({"epochs": EPOCHS, "step": step}) + "\n")


def write_eval_metrics():
    rows = []
    fields = ("epoch", "K_complex", "q", "AP", "AP50", "AP75", "AP_small",
              "train_loss", "train_det_loss", "train_feat_loss", "validation_loss",
              "learning_rate", "train_seconds", "eval_seconds", "batches_per_K",
              "images_per_K_cumulative", "mean_complex_power", "checkpoint")
    for epoch in range(1, EPOCHS + 1):
        for k in K_VALUES:
            path = OUT / f"epoch_{epoch:02d}/K_{k}/summary.json"
            if path.exists():
                item = json.loads(path.read_text())
                rows.append({field: item[field] for field in fields})
    (OUT / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (OUT / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def evaluate():
    while not (OUT / "protocol.json").exists():
        time.sleep(10)
    checked_inputs()
    if json.loads((OUT / "protocol.json").read_text()) != protocol():
        raise RuntimeError("evaluation protocol or code changed")
    (OUT / "eval_command.txt").write_text(" ".join(sys.argv) + "\n")
    run.seed_all(101)
    args = model_args(4)
    detector, criterion, codec = build_models(args)
    val_rows, val = dataset("val")
    expected = {Path(row["image_path"]).stem for row in val_rows}
    eval_dir = AUDIT / "stageB_seed101/validation_data"
    for epoch in range(1, EPOCHS + 1):
        checkpoint = OUT / f"epoch_{epoch:02d}/codec.pt"
        while not checkpoint.exists():
            if (OUT / "train_failed.txt").exists():
                raise RuntimeError("trainer failed; inspect train_failed.txt")
            time.sleep(20)
        state = run.load_file(checkpoint)
        if (state["epoch"] != epoch or state["seed"] != 101 or
                tuple(state["K_complex"]) != K_VALUES or tuple(state["grid"]) != GRID or
                state["detector_sha256"] != DETECTOR_SHA):
            raise RuntimeError(f"checkpoint protocol mismatch: epoch {epoch}")
        codec.load_state_dict(state["codec"], strict=True)
        futures = []
        with ThreadPoolExecutor(max_workers=6) as pool:
            for k in K_VALUES:
                folder = OUT / f"epoch_{epoch:02d}/K_{k}"
                if (folder / "summary.json").exists():
                    continue
                started = time.monotonic()
                wire_shapes = []
                original_forward = codec.forward

                def traced_forward(*arguments, **keywords):
                    restored, wire, received = original_forward(*arguments, **keywords)
                    wire_shapes.append(list(wire.shape))
                    return restored, wire, received

                codec.forward = traced_forward
                try:
                    predictions = folder / "predictions"
                    run.export_predictions(detector, codec, val, predictions, args, k, 10, False,
                                           save_wire=False, criterion=criterion, lambda_feat=10.0)
                finally:
                    codec.forward = original_forward
                records = [json.loads(line) for line in (predictions / "diagnostics.jsonl").read_text().splitlines()]
                if (len(records) != 546 or len(wire_shapes) != 546 or
                        {record["image"] for record in records} != expected or
                        {path.stem for path in predictions.glob("*.txt")} != expected or
                        any(shape != [1, k, 2] or record["K_sem"] != k or
                            record["N_symbol"] != k or record["noise"] or
                            abs(record["complex_power"] - 1) > 1e-5 or
                            tuple(map(tuple, record["level_shapes"])) != SHAPES
                            for record, shape in zip(records, wire_shapes))):
                    raise RuntimeError(f"invalid actual wire, power, shapes or val list at epoch={epoch} K={k}")
                with (folder / "per_image.jsonl").open("w") as handle:
                    for record, shape in zip(records, wire_shapes):
                        handle.write(json.dumps({**record, "wire_shape": shape}) + "\n")
                official = pool.submit(run.evaluate, predictions, eval_dir, args)
                small = pool.submit(stage.size_metrics, predictions, eval_dir, args.eval_bin)
                futures.append((k, folder, started, records, official, small))
            for k, folder, started, records, official, small in futures:
                scores, size = official.result(), small.result()
                train_losses = state["train_info"]["mean_loss_per_K"][str(k)]
                summary = {"epoch": epoch, "K_complex": k, "q": k // (GRID[0] * GRID[1]),
                           "AP": scores["AP"], "AP50": scores["AP50"], "AP75": scores["AP75"],
                           "AP_small": size["AP_small_supplemental"],
                           "train_loss": train_losses["loss"], "train_det_loss": train_losses["det"],
                           "train_feat_loss": train_losses["feat"],
                           "validation_loss": json.loads((folder / "predictions/validation_loss.json").read_text())["mean_loss"],
                           "learning_rate": state["train_info"]["learning_rate"],
                           "train_seconds": state["train_seconds"], "eval_seconds": time.monotonic() - started,
                           "batches_per_K": state["train_info"]["batches_per_K"],
                           "images_per_K_cumulative": epoch * 6467,
                           "mean_complex_power": sum(r["complex_power"] for r in records) / len(records),
                           "mean_relative_mse_per_level": [sum(r["relative_mse"][i] for r in records) / len(records)
                                                           for i in range(4)],
                           "wire_shape": [1, k, 2], "actual_K_each_image": k,
                           "images": len(records), "feature_shapes": SHAPES,
                           "noise": False, "SNR_dB_metadata_only": 10,
                           "checkpoint": str(checkpoint), "official": scores,
                           "small_supplemental": size}
                (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                write_eval_metrics()
                print(json.dumps({"epoch": epoch, "K": k, "AP": summary["AP"],
                                  "AP_small": summary["AP_small"]}), flush=True)
    rows = write_eval_metrics()
    if len(rows) != EPOCHS * len(K_VALUES):
        raise RuntimeError("evaluation incomplete")
    (OUT / "eval_complete.json").write_text(json.dumps({"epochs": EPOCHS, "rows": len(rows)}) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("smoke", "train", "evaluate"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and args.role != "train":
        parser.error("--resume is only for training")
    try:
        if args.role == "smoke":
            checked_inputs()
            run.seed_all(101)
            smoke_args = model_args(0)
            detector, criterion, codec = build_models(smoke_args)
            _, data = dataset("train")
            subset = torch.utils.data.Subset(data, [0, 1])
            optimizer = torch.optim.AdamW(codec.parameters(), lr=2e-4, weight_decay=1e-4)
            _, info = train_epoch_balanced(detector, criterion, codec, optimizer,
                                           subset, smoke_args, 0, 0)
            print(json.dumps({"smoke": info, "mapping": verify_mapping(codec)}), flush=True)
        elif args.role == "train":
            train(args.resume)
        else:
            evaluate()
    except Exception:
        if args.role == "train" and OUT.exists():
            (OUT / "train_failed.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()

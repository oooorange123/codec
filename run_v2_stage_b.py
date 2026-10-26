"""Train one pre-registered high-K seed; later stages require the B gate."""

import argparse
import contextlib
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch.utils.data import Subset

PROJECT = Path(__file__).resolve().parent
if not (PROJECT / "run_server.py").exists():
    PROJECT = PROJECT.parent / "visdrone_semantic_v2"
sys.path.insert(0, str(PROJECT))
import run_server as run  # noqa: E402
from fixed_grid_codec import FixedGridCodec, GRID  # noqa: E402
from prepare_v2 import DETECTOR_SHA, sha256  # noqa: E402

K_SEM = 131072
SEEDS = (101, 202, 303)


def manifest_rows(audit, split):
    path = audit / f"{split.replace('-', '')}_manifest.csv"
    expected = dict(line.split(None, 1)[::-1] for line in
                    (audit / "manifest_hashes.txt").read_text().splitlines())
    if sha256(path) != expected[path.name]:
        raise ValueError(f"locked {split} manifest hash changed")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        image, annotation = Path(row["image_path"]), Path(row["annotation_path"])
        if sha256(image) != row["image_sha256"] or sha256(annotation) != row["annotation_sha256"]:
            raise ValueError(f"data changed after manifest lock: {image}")
    return rows


def select_dataset(dataset, rows):
    indices = {path.stem: i for i, path in enumerate(dataset.images)}
    names = [Path(row["image_path"]).stem for row in rows]
    if len(names) != len(set(names)) or not set(names).issubset(indices):
        raise ValueError("manifest does not match dataset")
    return Subset(dataset, [indices[name] for name in names])


def make_eval_folder(rows, folder):
    for subdir, key in (("images", "image_path"), ("annotations", "annotation_path")):
        target = folder / subdir
        target.mkdir(parents=True, exist_ok=False)
        for row in rows:
            source = Path(row[key])
            destination = target / source.name
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)


def size_metrics(predictions, data_dir, eval_bin):
    quote = lambda path: "'" + str(path.resolve()).replace("'", "''") + "'"
    expression = (f"addpath({quote(PROJECT)}); m=eval_size_supplemental("
                  f"{quote(data_dir)},{quote(predictions)}); "
                  "fprintf('JSON_SIZE:%s\\n',jsonencode(m));")
    done = subprocess.run([eval_bin, "--quiet", "--eval", expression], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    (predictions / "size_eval.log").write_text(done.stdout)
    line = next((s for s in done.stdout.splitlines() if s.startswith("JSON_SIZE:")), None)
    if line is None:
        raise RuntimeError("supplemental area AP missing; see size_eval.log")
    return json.loads(line.split(":", 1)[1])


def save_checkpoint(path, codec, optimizer, epoch, step, score, args):
    payload = {"arch": "FixedGridCodec-v2", "codec": codec.state_dict(),
               "optimizer": optimizer.state_dict(), "epoch": epoch, "step": step,
               "val_ap": score, "seed": args.seed, "K_sem": K_SEM, "N_ch": K_SEM,
               "detector_sha256": DETECTOR_SHA,
               "manifest_hashes": (args.audit / "manifest_hashes.txt").read_text()}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(parents=[run.get_args_parser()], description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--eval-bin", required=True)
    parser.add_argument("--smoke", action="store_true", help="16 train/8 val images, one epoch; no B gate")
    parser.set_defaults(seed=101, epochs=30, batch_size=2, num_workers=4,
                        lr=2e-4, lambda_feat=10.0)
    args = parser.parse_args()
    if args.seed not in SEEDS or args.epochs < 1 or args.batch_size < 1:
        parser.error(f"seed must be {SEEDS}; epochs and batch size must be positive")
    if sha256(args.detector) != DETECTOR_SHA or not args.audit.is_dir():
        parser.error("frozen detector or preflight audit is missing")
    args.dataset_file = "coco"
    args.eval_backend = "octave"
    args.train_k = K_SEM
    args.train_snr = 10  # No noise: this number is metadata, not a 10 dB result.
    args.no_noise = True
    args.amp = False  # FP16 failed the first real detector-gradient smoke at step 0.
    args.ablate_fraction = 0
    args.ablate_mode = "random"
    args.renormalize = False
    args.out.mkdir(parents=True, exist_ok=False)
    run.seed_all(args.seed)
    detector_state = run.load_file(args.detector)
    if not detector_state.get("gate_pass") or detector_state.get("attention_backend") != run.ATTENTION_BACKEND:
        raise ValueError("detector protocol gate failed")
    detector, criterion = run.build_detector(args)
    detector.load_state_dict(detector_state["model"], strict=True)
    detector.to(args.device).eval().requires_grad_(False)
    criterion.to(args.device).eval()
    assert not any(p.requires_grad for p in detector.parameters())
    codec = FixedGridCodec().to(args.device)
    optimizer = torch.optim.AdamW(codec.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    train_rows = manifest_rows(args.audit, "train")
    val_rows = manifest_rows(args.audit, "val")
    if args.smoke:
        train_rows, val_rows = train_rows[:16], val_rows[:8]
        args.epochs = 1
    train = select_dataset(run.VisDrone(args.data_root, "train"), train_rows)
    val = select_dataset(run.VisDrone(args.data_root, "val"), val_rows)
    eval_dir = args.out / "validation_data"
    make_eval_folder(val_rows, eval_dir)
    protocol = {"stage": "B-smoke" if args.smoke else "B", "seed": args.seed,
                "epochs": args.epochs, "train_images": len(train), "val_images": len(val),
                "batch_size": args.batch_size, "optimizer": "AdamW", "lr": args.lr,
                "lr_schedule": "constant",
                "weight_decay": args.weight_decay, "lambda_feat": args.lambda_feat,
                "grid": GRID, "C": 64, "K_sem": K_SEM, "N_ch": K_SEM, "noise": False,
                "parameters": sum(p.numel() for p in codec.parameters()),
                "detector_sha256": DETECTOR_SHA,
                "manifest_hashes": (args.audit / "manifest_hashes.txt").read_text()}
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (args.out / "commands.log").write_text(" ".join(sys.argv) + "\n")
    step, best_ap = 0, -1.0
    with (args.out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("epoch", "seed", "split", "K_sem", "N_ch", "AP", "AP50", "AP75",
                         "AP_small_supplemental", "AP_medium_supplemental", "AP_large_supplemental",
                         "train_loss", "train_det_loss", "train_feat_loss", "validation_loss", "epoch_seconds"))
        for epoch in range(args.epochs):
            started = time.time()
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                step = run.train_epoch(detector, criterion, codec, optimizer, scaler,
                                       train, args, epoch, step)
            print(captured.getvalue(), end="", flush=True)
            train_info = json.loads(captured.getvalue().splitlines()[-1])
            predictions = args.out / "predictions" / f"epoch_{epoch + 1:02d}"
            run.export_predictions(detector, codec, val, predictions, args, K_SEM, 10,
                                   False, save_wire=False, criterion=criterion,
                                   lambda_feat=args.lambda_feat)
            with ThreadPoolExecutor(max_workers=2) as pool:
                official_future = pool.submit(run.evaluate, predictions, eval_dir, args)
                size_future = pool.submit(size_metrics, predictions, eval_dir, args.eval_bin)
                official, size = official_future.result(), size_future.result()
            val_loss = json.loads((predictions / "validation_loss.json").read_text())["mean_loss"]
            elapsed = time.time() - started
            writer.writerow((epoch + 1, args.seed, "val", K_SEM, K_SEM,
                             official["AP"], official["AP50"], official["AP75"],
                             size["AP_small_supplemental"], size["AP_medium_supplemental"],
                             size["AP_large_supplemental"], train_info["mean_loss"],
                             train_info["mean_det_loss"], train_info["mean_feat_loss"],
                             val_loss, elapsed))
            handle.flush()
            save_checkpoint(args.out / "last.pt", codec, optimizer, epoch, step, official["AP"], args)
            if official["AP"] > best_ap:
                best_ap = official["AP"]
                save_checkpoint(args.out / "best.pt", codec, optimizer, epoch, step, best_ap, args)
            print(json.dumps({"epoch": epoch + 1, "seed": args.seed, "step": step,
                              "official": official, "size": size, "validation_loss": val_loss,
                              "epoch_seconds": elapsed}), flush=True)
    print(json.dumps({"stage": protocol["stage"], "best_val_ap": best_ap,
                      "run": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()

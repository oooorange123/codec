"""Lock the v2 data, detector, and wire protocol before any training."""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

from fixed_grid_codec import CHANNELS, GRID, Q_VALUES, FixedGridCodec


EXCLUDE = {
    "train": {"0000239_06450_d_0000017", "0000239_06950_d_0000018",
              "9999950_00000_d_0000079", "9999950_00000_d_10000080"},
    "val": {"0000022_00000_d_0000004", "0000023_00000_d_0000008"},
    "test-dev": set(),
}
COUNTS = {"train": 6467, "val": 546, "test-dev": 1610}
DETECTOR_SHA = "3f85d3f8a70bcebf673f59b6eabdb5407da3e560fba34aeb01385a6b0aae4bc5"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.detector) != DETECTOR_SHA:
        raise ValueError("detector checkpoint differs from the approved fixed checkpoint")
    args.out.mkdir(parents=True, exist_ok=False)
    for split, expected in COUNTS.items():
        folder = args.data_root / f"VisDrone2019-DET-{split}"
        images = sorted((folder / "images").glob("*.jpg"))
        if len(images) - len(EXCLUDE[split]) != expected:
            raise ValueError(f"unexpected {split} count: {len(images)}")
        if not EXCLUDE[split].issubset({p.stem for p in images}):
            raise ValueError(f"excluded duplicate name missing in {split}")
        with (args.out / f"{split.replace('-', '')}_manifest.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("image_path", "split", "annotation_path", "image_sha256", "annotation_sha256"))
            for image in images:
                if image.stem in EXCLUDE[split]:
                    continue
                label = folder / "annotations" / f"{image.stem}.txt"
                if not label.is_file():
                    raise FileNotFoundError(label)
                writer.writerow((str(image.resolve()), split, str(label.resolve()),
                                 sha256(image), sha256(label)))
        print(f"locked {split}: {expected}", flush=True)
    manifests = sorted(args.out.glob("*_manifest.csv"))
    (args.out / "manifest_hashes.txt").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in manifests))
    (args.out / "checkpoint_hashes.txt").write_text(f"{DETECTOR_SHA}  {args.detector.resolve()}\n")
    code = [Path(__file__), Path(__file__).with_name("fixed_grid_codec.py"),
            Path(__file__).with_name("eval_size_supplemental.m")]
    (args.out / "code_hashes.txt").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in code))
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "N/A: project directory is not a Git repository; see code_hashes.txt"
    (args.out / "git_commit.txt").write_text(commit + "\n")
    environment = {"python": sys.version, "platform": platform.platform(),
                   "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                   "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    (args.out / "environment.txt").write_text(json.dumps(environment, indent=2) + "\n")
    model = FixedGridCodec()
    parameters = sum(p.numel() for p in model.parameters())
    lines = ["# V2 preflight audit", "", f"Detector SHA256: `{DETECTOR_SHA}`.",
             "Dataset: train 6467, val 546, test-dev 1610; duplicate conflict images excluded from train/val manifests.",
             "Official AP/AP50/AP75 remain from pinned VisDrone evaluator. Area AP is a separately labeled supplemental metric.",
             "Test-dev has already influenced detector retention and is exploratory for this fixed detector.",
             "", f"Codec: four [B,256,H_i,W_i] projected maps → G={GRID} grid, C={CHANNELS} trunk.",
             "Encoder generates q real scalars at each grid position from shared position/q queries; it does not truncate a maximum-length vector.",
             "Shared 1→2 channel convolution maps each semantic scalar to one complex I/Q use; receiver uses only I/Q and target geometry.",
             "Mean complex-symbol power is 1 per image. Clean channel in stage B; no AWGN.",
             "Per-image source map shape and mask are given to the frozen detector at the receiver; geometry payload is not included in N_ch.",
             "No source feature values, RGB pixels, labels, or detection predictions bypass the wire.",
             "Float32 I/Q bytes are simulator storage, not a quantized radio bit rate.",
             "The 5,351,936-real reference is one sample's feature-element count, not an information-dimension floor or measured wireless bypass.",
             "", f"Trainable parameter count (all K): {parameters}.", "",
             "| q | K_sem (real scalars/image) | N_ch (complex uses/image) | z shape | wire shape |",
             "|---:|---:|---:|---|---|"]
    for q in Q_VALUES:
        k = GRID * GRID * q
        lines.append(f"| {q} | {k} | {k} | [B,64,64,{q}] | [B,{k},2] |")
    (args.out / "config_audit.md").write_text("\n".join(lines) + "\n")
    (args.out / "commands.log").write_text(" ".join(sys.argv) + "\n")
    print(f"preflight complete: {args.out}; parameters={parameters}", flush=True)


if __name__ == "__main__":
    main()

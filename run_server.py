"""Server entry point for official detector fine-tuning and semantic feature codec."""
import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
ATTENTION_BACKEND = os.environ.setdefault("VISDRONE_ATTN_BACKEND", "reference")
if ATTENTION_BACKEND not in ("reference", "cuda"):
    raise ValueError("VISDRONE_ATTN_BACKEND must be reference or cuda")
VENDOR = HERE / "vendor" / "Deformable-DETR"
sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(VENDOR / "models" / "ops"))
import compat_torchvision  # noqa: E402
from main import get_args_parser  # noqa: E402
from util.misc import collate_fn  # noqa: E402
from util import box_ops  # noqa: E402
from visdrone_data import VisDrone  # noqa: E402
from modeling import (K_VALUES, SNR_VALUES, SemanticCodec, build_detector,
                      feature_loss, linked_forward, load_coco)  # noqa: E402
from codec import SpatialCodec, DirectKCodec, DirectKCodecV2, DirectKCodecV3, DirectKCodecHighK, HIGH_K_VALUES  # noqa: E402


def load_file(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch supported by the pinned repository
        return torch.load(path, map_location="cpu")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def codec_train_protocol(args):
    return {"k": args.train_k, "snr": args.train_snr, "noise": not args.no_noise,
            "val_images": args.val_images, "lambda_feat": args.lambda_feat,
            "batch_size": args.batch_size, "amp": args.amp,
            "lr": args.lr, "weight_decay": args.weight_decay,
            "lr_drop": args.lr_drop, "clip_max_norm": args.clip_max_norm,
            "num_workers": args.num_workers, "seed": args.seed, "fraction": args.fraction}


def codec_protocol(codec):
    if isinstance(codec, DirectKCodecHighK):
        return "direct-k-real-v3-high-budget"
    if isinstance(codec, DirectKCodecV3):
        return "direct-k-real-v3"
    if isinstance(codec, DirectKCodecV2):
        return "direct-k-real-v2"
    return "direct-k-real-v1" if isinstance(codec, DirectKCodec) else None


def save_checkpoint(path, detector, codec, optimizer, scheduler, scaler, epoch, step, extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": detector.state_dict(), "codec": codec.state_dict() if codec else None,
             "codec_arch": type(codec).__name__ if codec else None,
             "codec_protocol": codec_protocol(codec),
             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "scaler": scaler.state_dict(), "epoch": epoch, "global_step": step,
             "sampler": {"next_epoch": epoch + 1, "offset": 0},
             "rng": rng_state(), "attention_backend": ATTENTION_BACKEND, **extra}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def resume_checkpoint(path, detector, codec, optimizer, scheduler, scaler):
    state = load_file(path)
    if state.get("attention_backend") != ATTENTION_BACKEND:
        raise RuntimeError("checkpoint attention backend differs from this run")
    detector.load_state_dict(state["model"])
    if codec:
        if state.get("codec_arch", "SemanticCodec") != type(codec).__name__:
            raise RuntimeError("checkpoint codec architecture differs from this run")
        if codec_protocol(codec) and state.get("codec_protocol") != codec_protocol(codec):
            raise RuntimeError("checkpoint codec protocol differs from this run")
        codec.load_state_dict(state["codec"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    restore_rng(state["rng"])
    assert state["sampler"] == {"next_epoch": state["epoch"] + 1, "offset": 0}
    observed_steps = [float(item["step"]) for item in optimizer.state.values() if "step" in item]
    if state["global_step"] and (not observed_steps or max(observed_steps) != state["global_step"]):
        raise RuntimeError("optimizer moments/step are not continuous with global_step")
    print(f"resume epoch={state['epoch'] + 1} optimizer_step={state['global_step']} "
          f"lr={[g['lr'] for g in optimizer.param_groups]}", flush=True)
    return state


def move_targets(targets, device):
    return [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()}
            for t in targets]


def loader(dataset, batch_size, workers, seed=None):
    generator = torch.Generator().manual_seed(seed if seed is not None else 2026)
    return DataLoader(dataset, batch_size=batch_size, shuffle=seed is not None,
                      drop_last=False, num_workers=workers,
                      collate_fn=collate_fn, pin_memory=True, generator=generator)


def optimizer_for_detector(model, args):
    def named(group):
        return [p for n, p in model.named_parameters() if p.requires_grad and group(n)]
    backbone = lambda n: any(s in n for s in args.lr_backbone_names)
    linear = lambda n: any(s in n for s in args.lr_linear_proj_names)
    groups = [
        {"params": named(lambda n: not backbone(n) and not linear(n)), "lr": args.lr},
        {"params": named(backbone), "lr": args.lr_backbone},
        {"params": named(linear), "lr": args.lr * args.lr_linear_proj_mult},
    ]
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def weighted_loss(losses, criterion):
    return sum(value * criterion.weight_dict[name] for name, value in losses.items()
               if name in criterion.weight_dict)


def train_epoch(detector, criterion, codec, optimizer, scaler, dataset, args, epoch, step):
    detector.train(codec is None)
    if codec:
        detector.eval()  # Frozen BatchNorm/dropout, but gradients flow to the decoder.
        codec.train()
    covered = Counter()
    total = det_total = feat_total = 0.0
    processed = batches = 0
    batches_loader = loader(dataset, args.batch_size, args.num_workers, args.seed + epoch)
    for samples, targets in batches_loader:
        samples = samples.to(args.device)
        targets = move_targets(targets, args.device)
        optimizer.zero_grad(set_to_none=True)
        k, snr = ((args.train_k or random.choice(K_VALUES)),
                  (args.train_snr if args.train_snr is not None else random.choice(SNR_VALUES))) if codec else (None, None)
        with torch.cuda.amp.autocast(enabled=args.amp):
            if codec:
                with linked_forward(detector, codec, k, snr, not args.no_noise) as capture:
                    output = detector(samples)
                det_loss = weighted_loss(criterion(output, targets), criterion)
                feat = feature_loss(capture)
                loss = det_loss + args.lambda_feat * feat
                assert capture["wire"].shape == (len(targets), k, 2)
                covered[(k, snr)] += len(targets)
            else:
                det_loss = weighted_loss(criterion(detector(samples), targets), criterion)
                feat = None
                loss = det_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at epoch={epoch}, step={step}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        trainable = codec.parameters() if codec else detector.parameters()
        norm = torch.nn.utils.clip_grad_norm_(trainable, args.clip_max_norm)
        if not torch.isfinite(norm) or norm <= 0:
            raise FloatingPointError(f"nonfinite or zero gradient at epoch={epoch}, step={step}, "
                                     f"norm={norm.item()}, loss={loss.item()}")
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
        det_total += det_loss.item()
        feat_total += feat.item() if feat is not None else 0.0
        processed += len(targets)
        batches += 1
        step += 1
    assert processed == len(dataset), (processed, len(dataset))
    print(json.dumps({"epoch": epoch, "steps": step, "processed_images": processed, "mean_loss": total / batches,
                      "mean_det_loss": det_total / batches, "mean_feat_loss": feat_total / batches if codec else None,
                      "K_SNR_counts": {f"{k}/{s}": n for (k, s), n in sorted(covered.items())},
                      "N_symbol_equals_K_sem": (isinstance(codec, DirectKCodec) or
                                                getattr(codec, "semantic_k_is_real", False)) if codec else None,
                      "noise": not args.no_noise if codec else None}), flush=True)
    return step


@torch.no_grad()
def export_predictions(detector, codec, dataset, directory, args, k=None, snr=None, noise=True,
                       save_wire=True, criterion=None, lambda_feat=0.0):
    directory.mkdir(parents=True, exist_ok=True)
    detector.eval()
    if codec:
        codec.eval()
        (directory / "diagnostics.jsonl").write_text("")
    loss_sum = 0.0
    loss_count = 0
    for samples, targets in loader(dataset, 1, args.num_workers):
        samples = samples.to(args.device)
        if codec:
            with linked_forward(detector, codec, k, snr, noise,
                                args.ablate_fraction, args.ablate_mode, args.renormalize) as capture:
                output = detector(samples)
            assert capture["wire"].shape == (1, k, 2)
            name = targets[0]["name"]
            payload = capture["wire"][0].detach().float().cpu().numpy().tobytes()
            assert len(payload) == 8 * k  # K complex float32 I/Q symbols.
            if save_wire:
                (directory / (name + ".iqf32")).write_bytes(payload)
            distortion = []
            cosine = []
            for source, restored, mask in zip(capture["original"], capture["restored"], capture["masks"]):
                valid = (~mask).unsqueeze(1)
                a = (source * valid).flatten()
                b = (restored * valid).flatten()
                distortion.append(((a - b).square().sum() / a.square().sum().clamp_min(1e-8)).item())
                cosine.append(torch.nn.functional.cosine_similarity(a, b, dim=0).item())
            record = {"image": name, "K": k,
                      "K_sem": k if isinstance(codec, DirectKCodec) or
                      getattr(codec, "semantic_k_is_real", False) else None,
                      "N_symbol": k, "SNR_dB": snr, "noise": noise,
                      "wire_bytes": len(payload), "complex_power": capture["wire"].square().sum(-1).mean().item(),
                      "level_shapes": [list(x.shape[-2:]) for x in capture["original"]],
                      "relative_mse": distortion, "cosine": cosine,
                      "ablate_fraction": args.ablate_fraction, "ablate_mode": args.ablate_mode,
                      "renormalize": args.renormalize}
            with (directory / "diagnostics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        else:
            output = detector(samples)
        if criterion is not None:
            losses = weighted_loss(criterion(output, move_targets(targets, args.device)), criterion)
            if codec:
                losses = losses + lambda_feat * feature_loss(capture)
            loss_sum += losses.item()
            loss_count += len(targets)
        prob = output["pred_logits"].sigmoid()
        score, index = prob.flatten(1).topk(min(500, prob.shape[1] * prob.shape[2]), dim=1)
        query, category = index // 10, index % 10
        boxes = box_ops.box_cxcywh_to_xyxy(output["pred_boxes"])
        boxes = torch.gather(boxes, 1, query.unsqueeze(-1).expand(-1, -1, 4))[0]
        height, width = targets[0]["orig_size"].tolist()
        boxes = boxes * boxes.new_tensor([width, height, width, height])
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, height)
        rows = []
        for box, confidence, label in zip(boxes.tolist(), score[0].tolist(), category[0].tolist()):
            x1, y1, x2, y2 = box
            if x2 > x1 and y2 > y1:
                rows.append(f"{x1:.4f},{y1:.4f},{x2-x1:.4f},{y2-y1:.4f},{confidence:.8f},{label+1},-1,-1")
        (directory / (targets[0]["name"] + ".txt")).write_text("\n".join(rows) + "\n")
    if criterion is not None:
        (directory / "validation_loss.json").write_text(json.dumps(
            {"mean_loss": loss_sum / loss_count, "images": loss_count}) + "\n")
    return directory


def evaluate(directory, data_dir, args):
    if not args.eval_bin:
        raise ValueError("--eval-bin must point to MATLAB or Octave for official evaluation")
    quote = lambda value: "'" + str(value).replace("\\", "/").replace("'", "''") + "'"
    with tempfile.TemporaryDirectory(prefix="visdrone_eval_") as temporary:
        results_dir = directory
        files = list(directory.glob("*.txt"))
        if any(path.stat().st_size == 0 for path in files):
            results_dir = Path(temporary)
            for source in files:
                target = results_dir / source.name
                if source.stat().st_size:
                    shutil.copyfile(source, target)
                else:
                    # Official load() rejects empty files; category 0 is excluded from AP classes 1..10.
                    target.write_text("1,1,1,1,0,0,-1,-1\n")
        expression = (f"addpath({quote(HERE)}); m=eval_official({quote(data_dir)},{quote(results_dir)}); "
                      "fprintf('JSON_METRICS:%s\\n',jsonencode(m));")
        command = ([args.eval_bin, "-batch", expression] if args.eval_backend == "matlab" else
                   [args.eval_bin, "--quiet", "--eval", expression])
        done = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, check=True)
    (directory / "official_eval.log").write_text(done.stdout)
    line = next((s for s in done.stdout.splitlines() if s.startswith("JSON_METRICS:")), None)
    if line is None:
        raise RuntimeError(f"official evaluator returned no metrics; see {directory / 'official_eval.log'}")
    metrics = json.loads(line.split(":", 1)[1])
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def val200(dataset, out, limit=200):
    indices = sorted(random.Random(2026).sample(range(len(dataset)), min(limit, len(dataset))))
    selected = Subset(dataset, indices)
    directory = out / f"val{limit}"
    image_dir, annotation_dir = directory / "images", directory / "annotations"
    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    for index in indices:
        path = dataset.images[index]
        for source, destination in ((path, image_dir / path.name),
                                    (dataset.folder / "annotations" / (path.stem + ".txt"), annotation_dir / (path.stem + ".txt"))):
            if not destination.exists():
                shutil.copy2(source, destination)
    return selected, directory


def main():
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    parser.add_argument("command", choices=("train-detector", "train-codec", "export", "selftest"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=HERE / "runs")
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--detector", type=Path)
    parser.add_argument("--codec-checkpoint", type=Path)
    parser.add_argument("--codec-arch", choices=("legacy", "spatial", "direct-k", "direct-k-v2", "direct-k-v3", "direct-k-high"))
    parser.add_argument("--eval-bin")
    parser.add_argument("--eval-backend", choices=("matlab", "octave"), default="octave")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--lambda-feat", type=float, default=0.1)
    parser.add_argument("--k", type=int, choices=K_VALUES + HIGH_K_VALUES, default=256)
    parser.add_argument("--snr", type=int, choices=SNR_VALUES, default=10)
    parser.add_argument("--train-k", type=int, choices=K_VALUES + HIGH_K_VALUES)
    parser.add_argument("--train-snr", type=int, choices=SNR_VALUES)
    parser.add_argument("--val-images", type=int, default=200)
    parser.add_argument("--no-noise", action="store_true")
    parser.add_argument("--ablate-fraction", type=float, choices=(0, 0.25, 0.5, 0.75, 1.0), default=0)
    parser.add_argument("--ablate-mode", choices=("random", "high", "low"), default="random")
    parser.add_argument("--renormalize", action="store_true")
    args = parser.parse_args()
    args.dataset_file = "coco"  # Build official one-stage model, then replace only class head.
    if (args.with_box_refine or args.two_stage or args.dilation or not args.aux_loss
            or args.num_feature_levels != 4 or args.backbone != "resnet50"):
        parser.error("main method requires R50, four levels, one-stage, no DC5/refinement, auxiliary loss")
    if args.command == "selftest":
        from selftest import run
        run()
        return
    if args.command.startswith("train") and not args.eval_bin:
        parser.error("training requires --eval-bin for official validation")
    if args.ablate_fraction and (args.command != "export" or args.k != 256):
        parser.error("symbol ablation is an export-only K=256 diagnostic")
    if args.command == "train-codec" and args.epochs > 30:
        parser.error("codec main training is capped at 30 epochs by the plan")
    if args.command == "export" and not args.codec_checkpoint:
        parser.error("export requires --codec-checkpoint")
    if args.val_images < 1 or args.val_images > 548:
        parser.error("--val-images must be in [1,548]")
    if (args.train_k is not None or args.train_snr is not None or args.val_images != 200) and args.command != "train-codec":
        parser.error("--train-k, --train-snr and --val-images require train-codec")
    if args.no_noise and args.command not in ("train-codec", "export"):
        parser.error("--no-noise requires train-codec or export")
    source = args.resume if args.command == "train-codec" else args.codec_checkpoint if args.command == "export" else None
    if source:
        source_state = load_file(source)
        saved_arch = source_state.get("codec_arch", "SemanticCodec")
        selected = {"SemanticCodec": "legacy", "SpatialCodec": "spatial",
                    "DirectKCodec": "direct-k", "DirectKCodecV2": "direct-k-v2",
                    "DirectKCodecV3": "direct-k-v3", "DirectKCodecHighK": "direct-k-high"}.get(saved_arch)
        if selected is None or (args.codec_arch and args.codec_arch != selected):
            parser.error("checkpoint codec architecture does not match --codec-arch")
        if (args.command == "export" and selected == "direct-k-high" and
                source_state.get("train_protocol", {}).get("k") != args.k):
            parser.error("high-K export requires the K trained in this checkpoint")
        args.codec_arch = selected
    elif args.command == "train-codec":
        args.codec_arch = args.codec_arch or "spatial"
    if args.command in ("train-codec", "export"):
        if args.codec_arch == "direct-k-high":
            selected_k = args.train_k if args.command == "train-codec" else args.k
            if selected_k not in HIGH_K_VALUES:
                parser.error("direct-k-high requires a fixed K from 512, 1024, 2048")
        elif (args.train_k if args.command == "train-codec" else args.k) in HIGH_K_VALUES:
            parser.error("K>256 requires --codec-arch direct-k-high")
    args.out.mkdir(parents=True, exist_ok=True)
    environment = {"args": vars(args), "torch": torch.__version__,
                   "attention_backend": ATTENTION_BACKEND, "torch_cuda": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   "official_detector_commit": "11169a60c33333af00a4849f1808023eba96a931",
                   "official_evaluator_commit": "005445782213e20cb91bc50a597db3dd949e749a"}
    config_name = f"config_resume_{os.getpid()}.json" if args.resume else "config.json"
    (args.out / config_name).write_text(json.dumps(environment, indent=2, default=str) + "\n")
    seed_all(args.seed)
    detector, criterion = build_detector(args)
    detector.to(args.device)
    criterion.to(args.device)
    codec = None
    detector_hash = None
    pretrained_hash = sha256(args.pretrained) if args.pretrained else None
    if args.command == "train-detector":
        if not args.pretrained and not args.resume:
            parser.error("detector fine-tuning requires --pretrained COCO checkpoint or --resume")
        if args.pretrained and not args.resume:
            print("loaded COCO tensors:", load_coco(detector, args.pretrained))
        optimizer = optimizer_for_detector(detector, args)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    else:
        if not args.detector:
            parser.error("codec/export requires --detector checkpoint")
        detector_state = load_file(args.detector)
        if not detector_state.get("gate_pass"):
            parser.error("detector checkpoint has not passed official AP/AP50 gate")
        if detector_state.get("attention_backend") != ATTENTION_BACKEND:
            parser.error("detector checkpoint attention backend differs from this run")
        detector.load_state_dict(detector_state["model"])
        detector_hash = sha256(args.detector)
        for parameter in detector.parameters():
            parameter.requires_grad_(False)
        codec = {"spatial": SpatialCodec, "legacy": SemanticCodec,
                 "direct-k": DirectKCodec, "direct-k-v2": DirectKCodecV2,
                 "direct-k-v3": DirectKCodecV3,
                 "direct-k-high": DirectKCodecHighK}[args.codec_arch]().to(args.device)
        optimizer = torch.optim.AdamW(codec.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    start_epoch = step = 0
    best_ap = -1.0
    if args.resume:
        state = resume_checkpoint(args.resume, detector, codec, optimizer, scheduler, scaler)
        start_epoch, step = state["epoch"] + 1, state["global_step"]
        best_ap = state.get("best_ap", -1.0)
        assert state["seed"] == args.seed and state["fraction"] == args.fraction
        if codec:
            assert state["detector_sha256"] == detector_hash
            if isinstance(codec, DirectKCodec) and state.get("train_protocol") != codec_train_protocol(args):
                parser.error("DirectKCodec training protocol differs from checkpoint")
    val = VisDrone(args.data_root, "val")
    if args.command == "export":
        state = load_file(args.codec_checkpoint)
        assert state["detector_sha256"] == detector_hash
        assert state.get("attention_backend") == ATTENTION_BACKEND
        if codec_protocol(codec) and state.get("codec_protocol") != codec_protocol(codec):
            parser.error("DirectKCodec checkpoint protocol differs from this run")
        codec.load_state_dict(state["codec"])
        directory = args.out / (f"K{args.k}_SNR{args.snr}_noise{not args.no_noise}"
                                f"_drop{args.ablate_fraction}_{args.ablate_mode}_renorm{args.renormalize}")
        export_predictions(detector, codec, val, directory, args, args.k, args.snr, not args.no_noise)
        print(json.dumps(evaluate(directory, val.folder, args), indent=2))
        return
    train = VisDrone(args.data_root, "train")
    if args.fraction != 1:
        if not 0 < args.fraction < 1:
            parser.error("--fraction must be in (0,1]")
        count = round(len(train) * args.fraction)
        train = Subset(train, sorted(random.Random(2026).sample(range(len(train)), count)))
    if codec:
        eval_dataset, eval_dir = val200(val, args.out, args.val_images)
    else:
        eval_dataset, eval_dir = val, val.folder
    def finish_epoch(epoch, step):
        nonlocal best_ap
        predictions = args.out / "predictions" / f"epoch_{epoch+1}"
        export_predictions(detector, codec, eval_dataset, predictions, args,
                           (args.train_k or 256) if codec else None,
                           (args.train_snr if args.train_snr is not None else 10) if codec else None,
                           not args.no_noise)
        metrics = evaluate(predictions, eval_dir, args)
        ap, ap50 = metrics["AP"], metrics["AP50"]
        gate_pass = not codec and args.fraction == 1 and ap >= 1.0 and ap50 >= 5.0
        eligible = codec is not None or gate_pass
        extra = {"best_ap": max(best_ap, ap) if eligible else best_ap, "gate_pass": gate_pass,
                 "detector_sha256": detector_hash, "train_images": len(train),
                 "seed": args.seed, "fraction": args.fraction,
                 "pretrained_sha256": pretrained_hash if not codec else None,
                 "train_protocol": codec_train_protocol(args) if isinstance(codec, DirectKCodec) else None,
                 "official_metrics": metrics, "evaluation_pending": False}
        save_checkpoint(args.out / "last.pt", detector, codec, optimizer, scheduler, scaler,
                        epoch, step, extra)
        if eligible and ap > best_ap:
            best_ap = ap
            save_checkpoint(args.out / "best.pt", detector, codec, optimizer, scheduler, scaler,
                            epoch, step, extra)
        print(json.dumps({"epoch": epoch, "official": metrics, "gate_pass": gate_pass}), flush=True)

    if args.resume and state.get("evaluation_pending"):
        finish_epoch(state["epoch"], step)
    for epoch in range(start_epoch, args.epochs):
        step = train_epoch(detector, criterion, codec, optimizer, scaler, train, args, epoch, step)
        scheduler.step()
        save_checkpoint(args.out / "last.pt", detector, codec, optimizer, scheduler, scaler,
                        epoch, step, {"best_ap": best_ap, "gate_pass": False,
                                      "detector_sha256": detector_hash, "train_images": len(train),
                                      "seed": args.seed, "fraction": args.fraction,
                                      "pretrained_sha256": pretrained_hash if not codec else None,
                                      "train_protocol": codec_train_protocol(args) if isinstance(codec, DirectKCodec) else None,
                                      "evaluation_pending": True})
        finish_epoch(epoch, step)


if __name__ == "__main__":
    main()

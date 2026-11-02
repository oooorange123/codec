"""Final Stage B diagnostic: matched probes without input-to-output shortcuts."""

import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "latent_probe_20261028"))
import run_latent_probe as prior  # noqa: E402

fd, pg, base, run, stage = prior.fd, prior.pg, prior.base, prior.run, prior.stage
from visdrone_data import make_coco_transforms  # noqa: E402

original_protocol = prior.protocol
original_export = run.export_predictions
prior.HERE = HERE


class MatchedProbe(nn.Module):
    def __init__(self, source, arm):
        super().__init__()
        if arm not in prior.ARMS:
            raise ValueError(arm)
        self.arm = arm
        if arm == "latent":
            for name in fd.ENCODER:
                setattr(self, name, getattr(source, name))
                getattr(self, name).requires_grad_(False)
        # Construct the common body first so seed 101 initializes it identically in both arms.
        self.probe = nn.ModuleList(nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.GELU(),
            nn.Conv2d(128, 256, 3, padding=1)) for _ in range(4))
        self.adapter = nn.ModuleList(nn.Conv2d(16, 256, 1) for _ in range(4)) if arm == "latent" else None
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
            raise ValueError("four projected levels and zero wire required")
        if self.arm == "latent":
            maps = []
            for projection, feature in zip(self.project, features):
                spatial = projection(feature)
                mode = "area" if any(a > b for a, b in zip(spatial.shape[-2:], prior.GRID)) else "bilinear"
                maps.append(F.interpolate(spatial, size=prior.GRID, mode=mode,
                                          **({} if mode == "area" else {"align_corners": False})))
            trunk = self.encoder(torch.cat(maps, 1)).permute(0, 2, 3, 1)
            queries = self.enc_query(base.VariableGridCodec.query_coordinates(16, trunk))
            z = self.to_semantic(torch.tanh(trunk.unsqueeze(-2) + queries)).squeeze(-1)
            if list(z.shape) != [features[0].shape[0], 128, 256, 16]:
                raise RuntimeError("frozen encoder latent shape changed")
            self.last_latent_shape = list(z.shape)
            latent = z.permute(0, 3, 1, 2)
            inputs = [adapt(F.interpolate(latent, size=feature.shape[-2:], mode="bilinear", align_corners=False))
                      for adapt, feature in zip(self.adapter, features)]
        else:
            self.last_latent_shape = None
            inputs = features
        self.last_probe_input_shapes = [list(x.shape) for x in inputs]
        restored = [head(x) for head, x in zip(self.probe, inputs)]
        empty = restored[0].new_empty((features[0].shape[0], 0, 2))
        return restored, empty, empty


def protocol(arm):
    record = original_protocol(arm)
    record.update(stage="B-final-matched-probe", probe_architecture="per-level 256->128->256 convolutions, no skip",
                  latent_fixed_input_map="[B,128,256,16] bilinear to each level then learned 16->256 adapter" if arm == "latent" else None,
                  trainable_modules=["probe", "adapter"] if arm == "latent" else ["probe"],
                  train_AP="official evaluator on exact locked train6467 with deterministic val transform",
                  evaluation_code="undefined zero-wire complex_power recorded as null")
    record["code_sha256"]["run_matched_probe.py"] = stage.sha256(Path(__file__))
    return record


def checked_models(workers, arm):
    args, detector, criterion, source = prior.source_checked_models(workers)
    model = MatchedProbe(source, arm).cuda()
    trainable = {name.split(".")[0] for name, p in model.named_parameters() if p.requires_grad}
    if trainable != ({"probe", "adapter"} if arm == "latent" else {"probe"}):
        raise RuntimeError("trainable module boundary changed")
    if any(p.requires_grad for p in detector.parameters()):
        raise RuntimeError("detector not frozen")
    return args, detector, criterion, model


def preflight(arm):
    out = prior.output(arm)
    if out.exists():
        raise RuntimeError("arm output already exists")
    args, detector, criterion, model = checked_models(0, arm)
    _, dataset = base.dataset("train")
    samples, targets = next(iter(run.loader(dataset, 2, 0, 101)))
    samples, targets = samples.to("cuda"), run.move_targets(targets, "cuda")
    model.train()
    optimizer = torch.optim.AdamW(fd.trainable(model), lr=2e-4, weight_decay=1e-4)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    with run.linked_forward(detector, model, 0, 10, noise=False) as capture:
        outputs = detector(samples)
    det = run.weighted_loss(criterion(outputs, targets), criterion)
    feat = run.feature_loss(capture)
    loss = det + 10 * feat
    original = [list(x.shape) for x in capture["original"]]
    restored = [list(x.shape) for x in capture["restored"]]
    if (not torch.isfinite(loss) or original != restored or
            list(capture["wire"].shape) != [2, 0, 2] or
            (arm == "latent" and model.last_latent_shape != [2, 128, 256, 16]) or
            all(torch.allclose(a, b, atol=1e-6, rtol=1e-6) for a, b in zip(capture["original"], capture["restored"]))):
        raise RuntimeError("forward, shape, loss, or no-shortcut preflight failed")
    loss.backward()
    gradients = {n: float(p.grad.abs().sum()) if p.grad is not None else 0.0 for n, p in model.named_parameters()}
    if any((v <= 0 if model.get_parameter(n).requires_grad else v != 0) for n, v in gradients.items()):
        raise RuntimeError("probe/encoder gradient boundary failed")
    if any(p.grad is not None for p in detector.parameters()):
        raise RuntimeError("detector received gradients")
    optimizer.step()
    updates = {n: float((p.detach() - before[n]).abs().sum()) for n, p in model.named_parameters()}
    if any((v <= 0 if model.get_parameter(n).requires_grad else v != 0) for n, v in updates.items()):
        raise RuntimeError("probe/encoder update boundary failed")
    counts = {"probe": sum(p.numel() for p in model.probe.parameters()),
              "adapter": sum(p.numel() for p in model.adapter.parameters()) if model.adapter else 0}
    out.mkdir(parents=True, exist_ok=False)
    (out / "protocol.json").write_text(json.dumps(protocol(arm), indent=2) + "\n")
    (out / "preflight.json").write_text(json.dumps({"passed": True, "arm": arm,
        "input_shapes": original, "probe_input_shapes": model.last_probe_input_shapes,
        "output_shapes": restored, "latent_shape": model.last_latent_shape,
        "loss": loss.item(), "det_loss": det.item(), "feature_loss": feat.item(),
        "parameter_count": counts, "gradient_abs_sum": gradients,
        "parameter_update_abs_sum": updates, "detector_frozen": True,
        "no_input_output_shortcut": True, "N_symbol": 0}, indent=2) + "\n")
    print(json.dumps({"preflight": "passed", "arm": arm, "parameter_count": counts}), flush=True)


def export_and_clean(*args, **kwargs):
    dataset, folder = args[2], Path(args[3])
    diagnostics = folder / "diagnostics.jsonl"
    expected = len(dataset)
    if folder.exists():
        if (len(list(folder.glob("*.txt"))) != expected or
                not diagnostics.exists() or len(diagnostics.read_text().splitlines()) != expected or
                not (folder / "validation_loss.json").exists()):
            raise RuntimeError("partial predictions exist; inspect before continuing")
    else:
        original_export(*args, **kwargs)
    original = folder / "diagnostics_with_nan.jsonl"
    if original.exists():
        records = [json.loads(line) for line in diagnostics.read_text().splitlines()]
        if len(records) != expected or any(r["N_symbol"] != 0 or r["complex_power"] is not None for r in records):
            raise RuntimeError("cleaned diagnostics changed")
        return
    shutil.copy2(diagnostics, original)
    records = [json.loads(line) for line in diagnostics.read_text().splitlines()]
    if len(records) != expected:
        raise RuntimeError("prediction count changed")
    for record in records:
        if record["N_symbol"] != 0 or record["wire_bytes"] != 0 or not math.isnan(record["complex_power"]):
            raise RuntimeError("unexpected wire diagnostics")
        record["complex_power"] = None
        record["actual_transmission"] = False
    diagnostics.write_text("\n".join(json.dumps(r, allow_nan=False) for r in records) + "\n")


def evaluate(arm):
    out = prior.output(arm)
    if json.loads((out / "protocol.json").read_text()) != protocol(arm):
        raise RuntimeError("protocol or code changed")
    args, detector, criterion, model = checked_models(4, arm)
    val_rows, val = base.dataset("val")
    train_rows, _ = base.dataset("train")
    train_base = run.VisDrone(base.PROJECT / "../../data", "train")
    train_base.transform = make_coco_transforms("val")
    train_eval = stage.select_dataset(train_base, train_rows)
    train_gt = out / "train_eval_data"
    if not train_gt.exists():
        stage.make_eval_folder(train_rows, train_gt)
    if (len(list((train_gt / "images").glob("*.jpg"))) != 6467 or
            len(list((train_gt / "annotations").glob("*.txt"))) != 6467):
        raise RuntimeError("train official AP folder not exact train6467")
    (out / "eval_command.txt").write_text(" ".join(sys.argv) + "\n")
    for epoch in range(1, prior.EPOCHS + 1):
        checkpoint = out / f"epoch_{epoch:02d}/codec.pt"
        while not checkpoint.exists():
            if (out / "train_failed.txt").exists():
                raise RuntimeError("trainer failed")
            time.sleep(20)
        folder = checkpoint.parent
        if (folder / "summary.json").exists():
            continue
        state = run.load_file(checkpoint)
        if state["epoch"] != epoch or state["protocol"] != protocol(arm):
            raise RuntimeError("checkpoint protocol changed")
        model.load_state_dict(state["codec"], strict=True)
        started = time.monotonic()
        validation = pg.evaluate_path(detector, criterion, model, val, val_rows, args, folder)
        train_predictions = folder / "train_predictions"
        run.export_predictions(detector, model, train_eval, train_predictions, args, 0, 10, False,
                               save_wire=False, criterion=criterion, lambda_feat=10.0)
        expected_names = {Path(row["image_path"]).stem for row in train_rows}
        if {p.stem for p in train_predictions.glob("*.txt")} != expected_names:
            raise RuntimeError("train AP predictions do not match locked train6467")
        train_scores = run.evaluate(train_predictions, train_gt, args)
        info = state["train_info"]
        summary = {"epoch": epoch, "arm": arm, **validation,
                   "train_AP": train_scores["AP"], "train_AP50": train_scores["AP50"],
                   "train_AP75": train_scores["AP75"],
                   "train_official": train_scores, "train_eval_images": 6467,
                   "train_eval_loss": json.loads((train_predictions / "validation_loss.json").read_text())["mean_loss"],
                   "train_loss": info["mean_loss"], "train_det_loss": info["mean_det_loss"],
                   "train_feat_loss": info["mean_feat_loss"], "learning_rate": info["learning_rate"],
                   "train_seconds": info["train_seconds"], "eval_seconds": time.monotonic() - started,
                   "checkpoint": str(checkpoint), "latent_shape": [128, 256, 16] if arm == "latent" else None,
                   "actual_transmission": False, "N_symbol": 0}
        (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        write_metrics(arm)
        print(json.dumps({"eval_epoch": epoch, "arm": arm,
                          "val_AP": summary["AP"], "train_AP": summary["train_AP"]}), flush=True)
    (out / "eval_complete.json").write_text(json.dumps({"epochs": prior.EPOCHS,
                                                        "rows": len(write_metrics(arm))}) + "\n")


def write_metrics(arm):
    out = prior.output(arm)
    fields = ("epoch", "AP", "AP50", "AP75", "AP_small", "train_AP", "train_AP50", "train_AP75",
              "train_loss", "train_det_loss", "train_feat_loss", "validation_loss", "train_eval_loss",
              "learning_rate", "train_seconds", "eval_seconds", "checkpoint")
    rows = []
    for epoch in range(1, prior.EPOCHS + 1):
        path = out / f"epoch_{epoch:02d}/summary.json"
        if path.exists():
            record = json.loads(path.read_text())
            rows.append({key: record[key] for key in fields})
    (out / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def report():
    data = {}
    for arm in prior.ARMS:
        out = prior.output(arm)
        if not (out / "train_complete.json").exists() or not (out / "eval_complete.json").exists():
            raise RuntimeError(f"{arm} incomplete")
        rows = write_metrics(arm)
        if len(rows) != 30:
            raise RuntimeError(f"{arm} missing epochs")
        data[arm] = {"best": max(rows, key=lambda x: x["AP"]), "final": rows[-1],
                     "parameters": json.loads((out / "preflight.json").read_text())["parameter_count"]}
    refs = {"direct": pg.REFERENCE,
            "PG": json.loads((fd.ld.PG_REFERENCE / "best.json").read_text())["best"],
            "full_codec": json.loads((fd.SOURCE / "decision.json").read_text())["best"]["524288"],
            "frozen_E_D": json.loads((fd.OUT / "best.json").read_text())["best"]}
    (HERE / "decision.json").write_text(json.dumps({"arms": data, "references": refs}, indent=2) + "\n")
    raw, latent = data["raw"]["best"], data["latent"]["best"]
    lines = ["# 匹配探针：阶段 B 最后一项定位诊断", "",
             "两组 seed101、train6467/val546、30轮、AdamW 2e-4/1e-4、batch2、FP32、检测损失+10×有效区域相对特征 MSE。每级探针主体均为 256→128→256 的两层 3×3 卷积，输出只来自可训练映射，无输入输出恒等或残差捷径。latent 组额外使用每级可训练 16→256 输入适配器；两组主体以相同 seed 初始化。", "",
             "潜变量 z 来自冻结的高 K 第24轮编码器，检测器也冻结。两组均以 val546 官方 AP 最大选择最佳轮，平局取最早。train AP 使用锁定 train6467 清单构成的独立官方评估目录及确定性 val 预处理；训练更新仍使用原随机增强。", "",
             "| 路径 | 轮次 | val AP | AP50 | AP75 | AP_small | train AP | 训练损失 | 验证损失 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
             *[f'| {name} | {m["epoch"]} | {m["AP"]:.3f} | {m["AP50"]:.3f} | {m["AP75"]:.3f} | {m["AP_small"]:.3f} | {m["train_AP"]:.3f} | {m["train_loss"]:.4f} | {m["validation_loss"]:.4f} |'
               for name, m in (("raw 最佳", raw), ("raw 最终", data["raw"]["final"]),
                               ("latent 最佳", latent), ("latent 最终", data["latent"]["final"]))], "",
             f'raw 与 latent 最佳 AP 差：{raw["AP"]-latent["AP"]:.3f}。', "",
             "| 既有参考 | AP | AP50 | AP75 | AP_small |", "|---|---:|---:|---:|---:|",
             *[f'| {name} | {m["AP"]:.3f} | {m["AP50"]:.3f} | {m["AP75"]:.3f} | {m["AP_small"]:.3f} |'
               for name, m in refs.items()], "",
             "逐轮训练/验证曲线、是否收敛、梯度与冻结记录、参数量、代码/权重/清单哈希和命令见各组 run 目录。两组均无复符号传输，N_symbol=0；固定几何信息旁路提供；单 seed 仅作为无噪声诊断。"]
    if raw["AP"] < refs["PG"]["AP"] - 1:
        lines += ["", "原始特征组也表现较差，探针有效性不足，不能区分编码器和解码器问题。"]
    elif latent["train_AP"] > raw["train_AP"] - 1 and latent["AP"] < raw["AP"] - 1:
        lines += ["", "潜变量训练表现接近而验证表现较差，优先解释为泛化问题。"]
    elif latent["AP"] < raw["AP"] - 1 and latent["train_AP"] < raw["train_AP"] - 1:
        lines += ["", "当前冻结编码器产生的 z，在本探针能力与训练预算下不能满足现有检测任务要求；不构成信息不可逆丢失的数学证明。"]
    else:
        lines += ["", "潜变量组接近原始特征组；z 可被该探针转成检测器可用特征，原解码器或训练方式更可疑。"]
    lines += ["", "本次为最后一项定位实验，停止后续 codec 诊断，不启动新实验。", ""]
    (HERE / "matched_probe_report.md").write_text("\n".join(lines))


prior.TaskProbe = MatchedProbe
prior.protocol = protocol
prior.checked_models = checked_models
prior.preflight = preflight
prior.evaluate = evaluate
prior.report = report
run.export_predictions = export_and_clean

if __name__ == "__main__":
    prior.main()

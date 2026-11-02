"""Evaluate only the locked matched-probe candidate checkpoints on val546."""

import argparse
import csv
import json
import tempfile
from pathlib import Path

import run_matched_probe as matched

EPOCHS = (5, 10, 15, 20, 25, 30)
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "targeted_eval"


def evaluate(arm):
    source = matched.prior.output(arm)
    if json.loads((source / "protocol.json").read_text()) != matched.protocol(arm):
        raise RuntimeError("locked protocol or code changed")
    args, detector, criterion, model = matched.checked_models(4, arm)
    rows, val = matched.base.dataset("val")
    if len(rows) != 546:
        raise RuntimeError("val manifest changed")
    for epoch in EPOCHS:
        destination = OUT / arm / f"epoch_{epoch:02d}"
        summary = destination / "summary.json"
        if summary.exists():
            record = json.loads(summary.read_text())
            if record["epoch"] != epoch or record["arm"] != arm or record["val_images"] != 546:
                raise RuntimeError(f"invalid existing result: {summary}")
            print(f"reused {arm} epoch {epoch}", flush=True)
            continue
        prior_summary = source / f"epoch_{epoch:02d}/summary.json"
        if prior_summary.exists():
            record = json.loads(prior_summary.read_text())
            if record["epoch"] != epoch or record["arm"] != arm or not all(k in record for k in ("AP", "AP50", "AP75", "AP_small")):
                raise RuntimeError(f"invalid existing result: {prior_summary}")
            destination.mkdir(parents=True, exist_ok=True)
            summary.write_text(json.dumps({"epoch": epoch, "arm": arm, "source": "original_complete", "val_images": 546,
                                            **{k: record[k] for k in ("AP", "AP50", "AP75", "AP_small", "validation_loss", "checkpoint")}}, indent=2) + "\n")
            print(f"reused original {arm} epoch {epoch}", flush=True)
            continue
        checkpoint = source / f"epoch_{epoch:02d}/codec.pt"
        state = matched.run.load_file(checkpoint)
        if state["epoch"] != epoch or state["protocol"] != matched.protocol(arm):
            raise RuntimeError(f"invalid checkpoint: {checkpoint}")
        model.load_state_dict(state["codec"], strict=True)
        with tempfile.TemporaryDirectory(prefix=f"{arm}_{epoch:02d}_", dir=OUT) as temporary:
            scores = matched.pg.evaluate_path(detector, criterion, model, val, rows, args, Path(temporary))
        destination.mkdir(parents=True, exist_ok=True)
        record = {"epoch": epoch, "arm": arm, "source": "targeted_val_only", "val_images": 546,
                  "checkpoint": str(checkpoint), "checkpoint_sha256": matched.stage.sha256(checkpoint),
                  "actual_transmission": False, "N_symbol": 0, **scores,
                  "train_loss": state["train_info"]["mean_loss"],
                  "train_det_loss": state["train_info"]["mean_det_loss"],
                  "train_feat_loss": state["train_info"]["mean_feat_loss"],
                  "learning_rate": state["train_info"]["learning_rate"]}
        summary.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({"arm": arm, "epoch": epoch, "AP": record["AP"]}), flush=True)
    (OUT / arm / "complete.json").write_text(json.dumps({"candidate_epochs": EPOCHS}) + "\n")


def report():
    results = {}
    for arm in ("raw", "latent"):
        records = []
        for epoch in EPOCHS:
            path = OUT / arm / f"epoch_{epoch:02d}/summary.json"
            if not path.exists():
                raise RuntimeError(f"missing {path}")
            records.append(json.loads(path.read_text()))
        results[arm] = records
        with (OUT / arm / "metrics.csv").open("w", newline="") as handle:
            fields = ("epoch", "AP", "AP50", "AP75", "AP_small", "validation_loss", "checkpoint", "source")
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row[key] for key in fields} for row in records)
    best = {arm: max(records, key=lambda row: row["AP"]) for arm, records in results.items()}
    final = {arm: records[-1] for arm, records in results.items()}
    lines = ["# 匹配探针：六个候选检查点定点验证", "",
             "原始逐轮评估在各组完成第 1–3 轮后安全停止；第 4 轮的训练集 AP 未完成，文件原样保留，未计为完整结果。",
             "30 轮训练检查点均保留。仅从第 5、10、15、20、25、30 轮选择最佳，平局取较早轮次。",
             "使用原官方 val546 评估函数；官方评估所需的逐图预测写入临时目录，评估后清理。不计算训练集 AP。", "",
             "| 组别 | 轮次 | val AP | AP50 | AP75 | AP_small | 来源 |", "|---|---:|---:|---:|---:|---:|---|"]
    for arm, records in results.items():
        for row in records:
            lines.append(f'| {arm} | {row["epoch"]} | {row["AP"]:.3f} | {row["AP50"]:.3f} | {row["AP75"]:.3f} | {row["AP_small"]:.3f} | {row["source"]} |')
    lines += ["", f'最佳候选：raw 第 {best["raw"]["epoch"]} 轮 AP {best["raw"]["AP"]:.3f}；latent 第 {best["latent"]["epoch"]} 轮 AP {best["latent"]["AP"]:.3f}。',
              f'最终第 30 轮：raw AP {final["raw"]["AP"]:.3f}；latent AP {final["latent"]["AP"]:.3f}。',
              f'最佳候选 AP 差（raw−latent）{best["raw"]["AP"]-best["latent"]["AP"]:.3f}；最终轮 AP 差 {final["raw"]["AP"]-final["latent"]["AP"]:.3f}。',
              "", "本次只评估 val546；训练集 AP 仅有原逐轮流程已完成的第 1–3 轮，不足以判断最佳候选的训练集收敛或训练/验证差异。",
              "这是冻结编码器与检测器、无复符号传输的单 seed 诊断，不构成完整通信 codec 验收。", ""]
    (OUT / "matched_probe_targeted_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("evaluate", "report"))
    parser.add_argument("--arm", choices=("raw", "latent"))
    opts = parser.parse_args()
    if opts.command == "evaluate":
        if opts.arm is None:
            parser.error("--arm required")
        OUT.mkdir(exist_ok=True)
        evaluate(opts.arm)
    else:
        report()

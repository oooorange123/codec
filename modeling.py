"""Pinned Deformable-DETR adapter and K-symbol semantic feature link."""
import math
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models import build_model
from models.deformable_detr import SetCriterion
from util import box_ops
from util.misc import NestedTensor, accuracy


from codec import K_VALUES, SNR_VALUES, SemanticCodec, DirectKCodec, feature_loss


def build_detector(args):
    assert args.backbone == "resnet50" and args.num_feature_levels == 4
    assert not args.two_stage and not args.with_box_refine and not args.masks
    model, base_criterion, _ = build_model(args)
    head = nn.Linear(args.hidden_dim, 10)
    nn.init.constant_(head.bias, -math.log(99))
    model.class_embed = nn.ModuleList([head for _ in range(args.dec_layers)])
    criterion = IgnoreCriterion(10, base_criterion.matcher, base_criterion.weight_dict,
                                base_criterion.losses, base_criterion.focal_alpha)
    return model, criterion


def load_coco(model, path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    incoming = checkpoint.get("model", checkpoint)
    current = model.state_dict()
    usable = {k: v for k, v in incoming.items()
              if k in current and current[k].shape == v.shape and not k.startswith("class_embed.")}
    missing, unexpected = model.load_state_dict(usable, strict=False)
    assert not unexpected
    assert all(k.startswith("class_embed.") for k in missing), missing
    return len(usable)


def _ignore_query_mask(pred_boxes, targets, indices):
    """Official toolkit's ignored-area/detection-area >= 0.5 rule on transformed boxes.

    Its ignored-region union and integral-image corner calculation are repeated
    here; category 11 is retained as ignore during training (the official
    evaluator itself spatially filters category 0 only).
    """
    output = np.zeros(pred_boxes.shape[:2], dtype=np.bool_)
    with torch.no_grad():
        for b, target in enumerate(targets):
            ignored = target["ignore_boxes"]
            if not len(ignored):
                continue
            height, width = (int(x) for x in target["size"])
            ignored = box_ops.box_cxcywh_to_xyxy(ignored).detach().cpu().numpy()
            canvas = np.zeros((height, width), dtype=np.int8)
            for x1, y1, x2, y2 in ignored:
                x = max(1, math.floor(x1 * width + 0.5))
                y = max(1, math.floor(y1 * height + 0.5))
                w = max(1, math.floor((x2 - x1) * width + 0.5))
                h = max(1, math.floor((y2 - y1) * height + 0.5))
                canvas[y - 1:min(height, y + h), x - 1:min(width, x + w)] = 1
            integral = canvas.cumsum(0).cumsum(1)
            pred = box_ops.box_cxcywh_to_xyxy(pred_boxes[b]).detach().cpu().numpy()
            matched = set(indices[b][0].tolist())
            for q, (x1, y1, x2, y2) in enumerate(pred):
                if q in matched:
                    continue
                x = min(width, max(1, math.floor(x1 * width + 0.5)))
                y = min(height, max(1, math.floor(y1 * height + 0.5)))
                w = max(1, math.floor((x2 - x1) * width + 0.5))
                h = max(1, math.floor((y2 - y1) * height + 0.5))
                x2i, y2i = min(width, x + w), min(height, y + h)
                tl, tr = integral[y - 1, x - 1], integral[y - 1, x2i - 1]
                bl, br = integral[y2i - 1, x - 1], integral[y2i - 1, x2i - 1]
                output[b, q] = (tl + br - tr - bl) / (w * h) >= 0.5
    return torch.as_tensor(output, device=pred_boxes.device)


class IgnoreCriterion(SetCriterion):
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        matched_labels = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, indices)])
        labels = torch.full(logits.shape[:2], self.num_classes, dtype=torch.long, device=logits.device)
        labels[idx] = matched_labels
        wanted = torch.zeros_like(logits)
        valid = labels < self.num_classes
        wanted[valid, labels[valid]] = 1
        prob = logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(logits, wanted, reduction="none")
        pt = prob * wanted + (1 - prob) * (1 - wanted)
        loss = ce * (1 - pt).pow(2)
        if self.focal_alpha >= 0:
            loss = loss * (self.focal_alpha * wanted + (1 - self.focal_alpha) * (1 - wanted))
        ignored = _ignore_query_mask(outputs["pred_boxes"], targets, indices)
        loss = loss.masked_fill(ignored.unsqueeze(-1), 0)
        result = {"loss_ce": loss.mean(1).sum() / num_boxes * logits.shape[1]}
        if log:
            result["class_error"] = 100 - accuracy(logits[idx], matched_labels)[0] if len(matched_labels) else logits.new_tensor(0)
        return result


@contextmanager
def linked_forward(model, codec, k, snr_db, noise=True,
                   ablate_fraction=0, ablate_mode="random", renormalize=False):
    capture = {}

    def hook(_, inputs):
        features, masks, _positions, queries = inputs
        # shortcut: masks/shapes are unbudgeted geometry; encode them within K before formal K-only claims.
        # V2 inherits DirectKCodec and receives the same valid-region masks.
        restored, wire, received = (codec(features, k, snr_db, noise, masks=masks)
                                    if isinstance(codec, DirectKCodec) else
                                    codec(features, k, snr_db, noise))
        if ablate_fraction:
            count = round(k * ablate_fraction)
            energy = wire.square().sum(-1)
            order = (torch.rand_like(energy) if ablate_mode == "random" else
                     energy if ablate_mode == "high" else -energy)
            indices = order.topk(count, dim=1).indices
            dropped = torch.zeros_like(energy, dtype=torch.bool).scatter_(1, indices, True)
            altered = wire.masked_fill(dropped.unsqueeze(-1), 0)
            if renormalize:
                power = altered.square().sum(-1).mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
                altered = altered / power.unsqueeze(-1)
            received = altered + (received - wire)  # Same AWGN realization, signal ablated before channel.
            wire = altered
            restored = codec.decode(received, [x.shape[-2:] for x in features], k, snr_db)
        capture["wire"], capture["received"] = wire, received
        capture["original"], capture["restored"] = features, restored
        capture["masks"] = masks
        positions = [model.backbone[1](NestedTensor(x, mask)).to(x.dtype)
                     for x, mask in zip(restored, masks)]
        return (restored, masks, positions, queries)

    handle = model.transformer.register_forward_pre_hook(hook)
    try:
        yield capture
    finally:
        handle.remove()


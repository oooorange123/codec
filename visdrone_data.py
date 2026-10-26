"""VisDrone2019-DET adapter for the pinned Deformable-DETR COCO transforms."""
import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
import compat_torchvision
from datasets.coco import make_coco_transforms


class VisDrone(Dataset):
    def __init__(self, root, split):
        self.folder = Path(root) / f"VisDrone2019-DET-{split}"
        self.images = sorted((self.folder / "images").glob("*.jpg"))
        if not self.images:
            raise FileNotFoundError(self.folder / "images")
        self.transform = make_coco_transforms("train" if split == "train" else "val")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        width, height = image.size
        boxes, labels = [], []
        with (self.folder / "annotations" / (path.stem + ".txt")).open(newline="") as handle:
            for line, row in enumerate(csv.reader(handle), 1):
                if row and row[-1] == "" and len(row) == 9:
                    row = row[:-1]  # 34 train rows in the official archive.
                if not row:
                    continue
                if len(row) != 8:
                    raise ValueError(f"{path.stem}:{line}: expected 8 fields, got {row}")
                x, y, w, h, score, category, _, _ = map(int, row)
                if category not in range(12):
                    raise ValueError(f"{path.stem}:{line}: category {category}")
                if w <= 0 or h <= 0:
                    continue  # Three source boxes; details.json records each raw row.
                if x < 0 or y < 0 or x + w > width or y + h > height:
                    raise ValueError(f"{path.stem}:{line}: box outside image")
                boxes.append([x, y, x + w, y + h])
                labels.append(category - 1 if score == 1 and 1 <= category <= 10 else -1)
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {
            "boxes": boxes, "labels": labels,
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
            "image_id": torch.as_tensor(index),
            "orig_size": torch.as_tensor([height, width]),
            "size": torch.as_tensor([height, width]),
            "name": path.stem,
        }
        image, target = self.transform(image, target)
        good = target["labels"] >= 0
        target["ignore_boxes"] = target["boxes"][~good]
        target["boxes"] = target["boxes"][good]
        target["labels"] = target["labels"][good]
        target["area"] = target["area"][good]
        target["iscrowd"] = target["iscrowd"][good]
        return image, target

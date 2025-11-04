from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import kagglehub

from torchvision import transforms
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import pil_to_tensor

import random
import yaml

import matplotlib.pyplot as plt

from transforms import build_transforms
from transforms import IMAGENET_MEAN, IMAGENET_STD, denormalize

def _list_images(images_dir: Path) -> List[Path]:
    files = []
    for p in images_dir.iterdir():
        files.append(p)
    files.sort()
    return files

def _yolo_to_xyxy(cx: float, cy: float, w: float, h: float, W: int, H: int) -> Tuple[float, float, float, float]:
    x1 = (cx - w / 2.0) * W
    y1 = (cy - h / 2.0) * H
    x2 = (cx + w / 2.0) * W
    y2 = (cy + h / 2.0) * H
    return x1, y1, x2, y2

class TrafficSignsDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        transform=None
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        self.images_dir = self.root / split / "images"
        self.labels_dir = self.root / split / "labels"

        self.images: List[Path] = _list_images(self.images_dir)

        self.class_names: Optional[List[str]] = None
        data_yaml = self.root / "data.yaml"

        with open(data_yaml, "r") as f:
            y = yaml.safe_load(f)
        names = y.get("names", None)
        self.class_names = [str(n) for n in names]

    def __len__(self) -> int:
        return len(self.images)

    def _read_labels(
        self, 
        label_path: Path, 
        img_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        W, H = img_size
        boxes: List[List[float]] = []
        labels: List[int] = []
       
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:5])
                x1, y1, x2, y2 = _yolo_to_xyxy(cx, cy, w, h, W, H)

                x1 = max(0.0, min(x1, W))
                y1 = max(0.0, min(y1, H))
                x2 = max(0.0, min(x2, W))
                y2 = max(0.0, min(y2, H))
                
                boxes.append([x1, y1, x2, y2])
                labels.append(cls)

        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.int64)
        return boxes_t, labels_t

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        img_path = self.images[idx]
        label_path = (self.labels_dir / img_path.name).with_suffix(".txt")

        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        boxes, labels = self._read_labels(label_path, (W, H))

        target: Dict[str, Any] = {
            "boxes": boxes,
            "labels": labels,
        }

        if self.transform is not None:
            img, target = self.transform(img, target)
            if isinstance(img, torch.Tensor):
                H_t, W_t = int(img.shape[-2]), int(img.shape[-1])
            else:
                W_t, H_t = img.size
        else:
            H_t, W_t = H, W

        boxes_out = target["boxes"]

        # Handle case when there are no boxes
        if boxes_out.numel() == 0 or boxes_out.ndim == 1:
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            area = (boxes_out[:, 2] - boxes_out[:, 0]).clamp(min=0) * (boxes_out[:, 3] - boxes_out[:, 1]).clamp(min=0)

        iscrowd = torch.zeros((target["labels"].shape[0],), dtype=torch.int64)
        size = torch.tensor([H_t, W_t], dtype=torch.int64)
        image_id = torch.tensor([idx], dtype=torch.int64)

        target.update({
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
            "size": size,
        })
        target["class_names"] = self.class_names

        return img, target

def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


if __name__ == "__main__":
    src = kagglehub.dataset_download("pkdarabi/cardetection")
    root = Path(src) / "car" 

    train_tfm = build_transforms("train", size=(640, 640), augment=True)
    val_tfm = build_transforms("valid", size=(640, 640), augment=False)
    test_tfm = build_transforms("test",  size=(640, 640), augment=False)

    train_ds = TrafficSignsDataset(root, split="train", transform=train_tfm)
    val_ds = TrafficSignsDataset(root, split="valid", transform=val_tfm)
    test_ds = TrafficSignsDataset(root, split="test",  transform=test_tfm)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,  num_workers=4, collate_fn=detection_collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=4, collate_fn=detection_collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=4, shuffle=False, num_workers=4, collate_fn=detection_collate_fn)
    
    print("samples:", len(train_ds), len(val_ds), len(test_ds))
    print("batches:", len(train_loader), len(val_loader), len(test_loader))

    idx = random.randrange(len(train_ds))
    img, target = train_ds[idx]

    img_vis = denormalize(img, IMAGENET_MEAN, IMAGENET_STD)
    img_t = (img_vis * 255).clamp(0, 255).to(torch.uint8)

    labels_txt = [str(l.item()) for l in target["labels"]]
    vis = draw_bounding_boxes(img_t, target["boxes"], labels=labels_txt, colors="lime", width=2)

    plt.imshow(vis.permute(1, 2, 0))
    plt.axis("off")
    plt.title(f"Image {idx}")
    plt.show()
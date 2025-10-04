import math
import random

from PIL import Image
from kagglehub import dataset_download
from numpy import add
import torch
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

class ComposeDet:
    def __init__(self, transforms_list):
        self.transforms = transforms_list

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target

class LetterboxDet:
    def __init__(self, size=(640, 640), fill=(114, 114, 114), interp=Image.BILINEAR):
        self.size = size
        self.fill = fill
        self.interp = interp

    def __call__(self, img, target):
        W, H = img.size
        out_w, out_h = self.size
        scale = min(out_w / W, out_h / H)
        new_w = int(round(W * scale))
        new_h = int(round(H * scale))

        img_r = img.resize((new_w, new_h), self.interp)
        canvas = Image.new("RGB", (out_w, out_h), self.fill)
        dx = (out_w - new_w) // 2
        dy = (out_h - new_h) // 2
        canvas.paste(img_r, (dx, dy))

        boxes = target["boxes"]
        if boxes.numel() > 0:
            boxes = boxes * torch.tensor([scale, scale, scale, scale], dtype=boxes.dtype)
            boxes[:, [0, 2]] += dx
            boxes[:, [1, 3]] += dy
        target["boxes"] = boxes
        target["size"] = torch.tensor([out_h, out_w], dtype=torch.int64)
        return canvas, target

class RandomHFlipDet:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            W, H = img.size
            boxes = target["boxes"]
            if boxes.numel() > 0:
                x1 = boxes[:, 0].clone()
                x2 = boxes[:, 2].clone()
                boxes[:, 0] = W - x2
                boxes[:, 2] = W - x1
            target["boxes"] = boxes
        return img, target

class ColorJitterDet:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02):
        self.t = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, img, target):
        return self.t(img), target

class ToTensorNormalizeDet:
    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize(mean, std)

    def __call__(self, img, target):
        img = self.to_tensor(img)
        img = self.norm(img)
        return img, target

def build_transforms(split: str, size=(640, 640), augment: bool = True):
    ts = [LetterboxDet(size=size)]
    if split == "train" and augment:
        ts += [
            RandomHFlipDet(p=0.5),
            ColorJitterDet(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        ]
    ts += [ToTensorNormalizeDet()]
    return ComposeDet(ts)

def denormalize(img_t: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    mean_t = torch.tensor(mean, device=img_t.device)[:, None, None]
    std_t = torch.tensor(std, device=img_t.device)[:, None, None]
    return (img_t * std_t + mean_t).clamp(0.0, 1.0)

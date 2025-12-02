import os
import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
from rfdetr import RFDETRNano
from rfdetr.util import misc

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils import get_device, get_kaggle_dataset_root


# Патч для nested_tensor_from_tensor_list щоб працювати з градієнтами
def _patched_nested_tensor_from_tensor_list(tensor_list):
    from rfdetr.util.misc import NestedTensor
    
    if len(tensor_list) == 1:
        img = tensor_list[0]
        if img.dim() == 3:
            img = img.unsqueeze(0)
        mask = torch.zeros((img.shape[0], img.shape[2], img.shape[3]), dtype=torch.bool, device=img.device)
        return NestedTensor(img, mask)
    
    max_size = tuple(max(s) for s in zip(*[img.shape for img in tensor_list]))
    batch_shape = (len(tensor_list),) + max_size
    b, c, h, w = batch_shape
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device
    tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
    mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
    for img, pad_img, m in zip(tensor_list, tensor, mask):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        m[: img.shape[1], :img.shape[2]] = False
    
    return NestedTensor(tensor, mask)

import rfdetr.models.lwdetr as lwdetr_module
lwdetr_module.nested_tensor_from_tensor_list = _patched_nested_tensor_from_tensor_list

CLASS_NAMES = [
    "Green Light",
    "Red Light",
    "Speed Limit 10",
    "Speed Limit 100",
    "Speed Limit 110",
    "Speed Limit 120",
    "Speed Limit 20",
    "Speed Limit 30",
    "Speed Limit 40",
    "Speed Limit 50",
    "Speed Limit 60",
    "Speed Limit 70",
    "Speed Limit 80",
    "Speed Limit 90",
    "Stop",
]


class GuidedBackpropActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, activation_type='relu'):
        if activation_type == 'silu':
            output = input * torch.sigmoid(input)
        elif activation_type == 'gelu':
            output = F.gelu(input)
        else:
            output = input.clamp(min=0)
        
        ctx.save_for_backward(output)
        ctx.activation_type = activation_type
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        positive_activation = (output > 0).type_as(grad_output)
        positive_grad = (grad_output > 0).type_as(grad_output)
        guided_grad = positive_activation * positive_grad * grad_output
        return guided_grad, None


class RFDetrGuidedBackprop:
    def __init__(self, checkpoint_path: Path):
        self.device = get_device()
        self.checkpoint_path = Path(checkpoint_path)
        self.model = self._load_model()
        self.model.model.model.to(self.device)
        self.model.model.device = self.device
        self.model.model.model.eval()
        self.resolution = self.model.model.resolution

        self.mean = torch.tensor(self.model.means, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(self.model.stds, device=self.device).view(1, 3, 1, 1)

        self._replace_activations()

    def _load_model(self):
        with torch.serialization.safe_globals([type(None)]):
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        num_classes = state_dict["class_embed.weight"].shape[0] - 1
        model = RFDETRNano(
            num_classes=num_classes,
            force_no_pretrain=True,
            pretrain_weights=None,
        )
        model.model.model.load_state_dict(state_dict, strict=False)
        return model

    def _replace_activations(self):
        replaced_count = {'relu': 0, 'silu': 0, 'gelu': 0}
        
        for name, module in self.model.model.model.named_modules():
            if isinstance(module, (nn.ReLU, nn.ReLU6)):
                module.forward = lambda x, m=module: GuidedBackpropActivation.apply(x, 'relu')
                replaced_count['relu'] += 1
            elif isinstance(module, nn.SiLU):
                module.forward = lambda x, m=module: GuidedBackpropActivation.apply(x, 'silu')
                replaced_count['silu'] += 1
            elif isinstance(module, nn.GELU):
                module.forward = lambda x, m=module: GuidedBackpropActivation.apply(x, 'gelu')
                replaced_count['gelu'] += 1
        
        print(f"Replaced activations: {replaced_count}")

    def _preprocess(self, img_rgb):
        img_resized = cv2.resize(img_rgb, (self.resolution, self.resolution))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(self.device)
        img_tensor = (img_tensor - self.mean) / self.std
        return img_resized, img_tensor

    def _format_outputs(self, outputs):
        if isinstance(outputs, dict):
            return outputs
        
        if isinstance(outputs, (list, tuple)):
            outputs = list(outputs)
            formatted = {}
            if len(outputs) > 0:
                formatted["pred_boxes"] = outputs[0]
            if len(outputs) > 1:
                formatted["pred_logits"] = outputs[1]
            return formatted
        
        return outputs

    def _find_target_query(self, outputs):
        logits = outputs["pred_logits"][0]
        boxes = outputs["pred_boxes"][0]
        
        probs = logits.sigmoid()
        scores_per_query, labels_per_query = probs.max(dim=-1)
        sorted_scores, sorted_idx = torch.sort(scores_per_query, descending=True)
        
        # Використовуємо топ-K queries з порогом
        confident_mask = sorted_scores > 0.1  # Знижений поріг з 0.3 до 0.1
        top_indices = sorted_idx[confident_mask][:10]  # Топ-10 queries
        if top_indices.numel() == 0:
            top_indices = sorted_idx[:3]  # Мінімум 3 queries
        
        best_query_idx = int(sorted_idx[0])
        best_label = int(labels_per_query[best_query_idx])
        best_score = scores_per_query[best_query_idx].item()
        best_box_norm = boxes[best_query_idx]
        
        target_logit = logits[top_indices, labels_per_query[top_indices]].sum()
        
        return target_logit, best_query_idx, best_label, best_score, best_box_norm

    def _normalize_gradients(self, grads, percentile_min=2, percentile_max=98):
        vmin, vmax = np.percentile(grads, percentile_min), np.percentile(grads, percentile_max)
        if vmax > vmin:
            return np.clip((grads - vmin) / (vmax - vmin), 0, 1)
        return np.zeros_like(grads)

    def _box_cxcywh_to_xyxy(self, box):
        cx, cy, w, h = box
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return np.array([x1, y1, x2, y2])

    def _scale_box(self, box, width, height):
        scaled = np.array([
            box[0] * width,
            box[1] * height,
            box[2] * width,
            box[3] * height,
        ])
        scaled[[0, 2]] = np.clip(scaled[[0, 2]], 0, width - 1)
        scaled[[1, 3]] = np.clip(scaled[[1, 3]], 0, height - 1)
        return scaled.astype(int)

    def generate(self, img_path):
        img_path = Path(img_path)
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]

        img_resized, img_tensor = self._preprocess(img_rgb)
        img_tensor = img_tensor.requires_grad_(True)

        self.model.model.model.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(True):
            raw_outputs = self.model.model.model(img_tensor)

        outputs = self._format_outputs(raw_outputs)
        target_logit, best_query_idx, best_label, best_score, best_box_norm = self._find_target_query(outputs)

        target_logit.backward()

        gradients = img_tensor.grad.data.cpu().numpy()[0]

        guided_rgb = np.maximum(gradients, 0)
        guided_rgb_norm = np.stack([
            self._normalize_gradients(guided_rgb[c]) 
            for c in range(3)
        ]).transpose(1, 2, 0)

        guided_gray = np.sqrt(np.sum(gradients ** 2, axis=0))
        guided_gray_norm = self._normalize_gradients(guided_gray)

        best_box_xyxy = self._box_cxcywh_to_xyxy(best_box_norm.detach().cpu().numpy())
        best_box_pixels = self._scale_box(best_box_xyxy, self.resolution, self.resolution)

        self.model.model.model.eval()
        with torch.no_grad():
            target_sizes = torch.tensor([[orig_h, orig_w]], device=self.device)
            detections = self.model.model.postprocess(outputs, target_sizes)[0]

        return {
            "original": img_rgb,
            "resized": img_resized,
            "guided_gray": guided_gray_norm,
            "guided_rgb": guided_rgb_norm,
            "best_box": best_box_pixels,
            "best_score": best_score,
            "best_label": best_label,
            "detections": detections,
        }

    def _create_heatmap(self, grayscale_img):
        heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_img), cv2.COLORMAP_JET)
        return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    def _draw_predictions(self, img_rgb, detections, max_boxes=10):
        vis = img_rgb.copy()
        boxes = detections["boxes"].cpu().numpy()
        scores = detections["scores"].cpu().numpy()
        labels = detections["labels"].cpu().numpy().astype(int)
        
        for score, label, box in zip(scores[:max_boxes], labels[:max_boxes], boxes[:max_boxes]):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_name = CLASS_NAMES[label] if 0 <= label < len(CLASS_NAMES) else f"cls {label}"
            text = f"{label_name} {score:.2f}"
            cv2.putText(
                vis,
                text,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        return vis

    def visualize(self, img_path, save_path):
        result = self.generate(img_path)

        img_resized = result["resized"]
        heatmap_gray = self._create_heatmap(result["guided_gray"])
        guided_rgb_vis = np.uint8(255 * result["guided_rgb"])

        prediction_img = self._draw_predictions(result["original"], result["detections"])
        prediction_img = cv2.resize(prediction_img, (self.resolution, self.resolution))

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        panels = [
            ("Original", img_resized),
            ("Guided Backprop (Grayscale)", heatmap_gray),
            ("Guided Backprop (RGB)", guided_rgb_vis),
            ("Predictions", prediction_img),
        ]

        for ax, (title, img) in zip(axes, panels):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis("off")

        plt.tight_layout()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved to {save_path}")


if __name__ == "__main__":
    random.seed(42)
    checkpoint = project_root / "lab3/rfdetr/rfdetr-train/checkpoint_best_total.pth"
    guided_bp = RFDetrGuidedBackprop(checkpoint_path=checkpoint)

    test_dir = get_kaggle_dataset_root() / "test/images"
    test_images = random.sample(list(test_dir.glob("*.jpg")), 10)

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    for idx, img_path in enumerate(test_images):
        print(f"Processing {img_path.name}...")
        guided_bp.visualize(img_path, output_dir / f"guided_backprop_rfdetr_{idx}.png")

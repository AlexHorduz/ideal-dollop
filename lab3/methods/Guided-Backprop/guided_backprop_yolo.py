import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils import get_device, get_kaggle_dataset_root


class GuidedBackpropActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, is_silu=False):
        if is_silu:
            output = input * torch.sigmoid(input)
        else:
            output = input.clamp(min=0)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        positive_activation = (output > 0).type_as(grad_output)
        positive_grad = (grad_output > 0).type_as(grad_output)
        return positive_activation * positive_grad * grad_output, None


class YoloGuidedBackprop:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.device = get_device()
        self.model.model.to(self.device)
        self.model.model.eval()
        self._replace_activations()

    def _replace_activations(self):
        for module in self.model.model.modules():
            if isinstance(module, (nn.ReLU, nn.ReLU6)):
                module.forward = lambda x, m=module: GuidedBackpropActivation.apply(x, False)
            elif isinstance(module, nn.SiLU):
                module.forward = lambda x, m=module: GuidedBackpropActivation.apply(x, True)

    def _normalize_gradients(self, grads, percentile_min=2, percentile_max=98):
        vmin, vmax = np.percentile(grads, percentile_min), np.percentile(grads, percentile_max)
        if vmax > vmin:
            return np.clip((grads - vmin) / (vmax - vmin), 0, 1)
        return np.zeros_like(grads)

    def _find_target_score(self, output):
        def collect_tensors(obj):
            if isinstance(obj, torch.Tensor):
                return [obj]
            if isinstance(obj, (list, tuple)):
                return [t for item in obj for t in collect_tensors(item)]
            return []

        max_score = 0
        target = None

        for pred in collect_tensors(output):
            if pred is None or len(pred.shape) < 2:
                continue

            try:
                if pred.shape[-1] > 5:
                    objectness = pred[..., 4]
                    class_probs, _ = pred[..., 5:].max(dim=-1)
                    scores = objectness * class_probs
                    current_max = scores.max()
                else:
                    current_max = pred.abs().max()

                if current_max > max_score:
                    max_score = current_max
                    target = current_max
            except (IndexError, RuntimeError):
                raise ValueError("Unexpected tensor shape encountered in output.")

        return target if target is not None else collect_tensors(output)[0].abs().mean()

    def generate(self, img_path):
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(cv2.resize(img_rgb, (640, 640))).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device).requires_grad_(True)
        best_box = None
        results = None
        with torch.no_grad():
            results = self.model(str(img_path), device=self.device, verbose=False)
            if results and results[0].boxes:
                best_box = results[0].boxes.data[results[0].boxes.conf.argmax()].cpu().numpy()

        self.model.model.zero_grad()
        self.model.model.train()
        output = self.model.model(img_tensor)
        self.model.model.eval()
        target = self._find_target_score(output)
        target.backward()

        gradients = img_tensor.grad.data.cpu().numpy()[0]
        guided_rgb = np.maximum(gradients, 0)
        guided_rgb_norm = np.stack([self._normalize_gradients(guided_rgb[c]) for c in range(3)]).transpose(1, 2, 0)

        guided_gray = np.sqrt(np.sum(gradients ** 2, axis=0))
        guided_gray_norm = self._normalize_gradients(guided_gray)

        if best_box is not None:
            orig_h, orig_w = img_rgb.shape[:2]
            x1 = int(best_box[0] * 640 / orig_w)
            y1 = int(best_box[1] * 640 / orig_h)
            x2 = int(best_box[2] * 640 / orig_w)
            y2 = int(best_box[3] * 640 / orig_h)

            x1 = max(0, min(639, x1))
            y1 = max(0, min(639, y1))
            x2 = max(0, min(640, x2))
            y2 = max(0, min(640, y2))

            mask = np.ones_like(guided_gray_norm) * 0.3
            mask[y1:y2, x1:x2] = 1.0

            guided_gray_norm = guided_gray_norm * mask
            guided_rgb_norm = guided_rgb_norm * mask[..., None]

        return img_rgb, guided_gray_norm, guided_rgb_norm, best_box, results

    def _create_heatmap(self, grayscale_img):
        return cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * grayscale_img), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)

    def visualize(self, img_path, save_path):
        img, guided_gray, guided_rgb, best_box, results = self.generate(img_path)
        img_resized = cv2.resize(img, (640, 640))
        overlay = img_resized.copy()

        prediction_img = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        titles = ["Original", "Guided Backprop",
                  "Guided Backprop", "YOLO Prediction"]
        images = [overlay, self._create_heatmap(guided_gray), np.uint8(255 * guided_rgb), prediction_img]

        for ax, title, img_data in zip(axes, titles, images):
            ax.imshow(img_data)
            ax.set_title(title)
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved to {save_path}")

if __name__ == "__main__":
    import random

    random.seed(42)
    model_path = project_root / "lab3/yolo/yolo-train/weights/best.pt"
    guided_bp = YoloGuidedBackprop(model_path)

    test_dir = get_kaggle_dataset_root() / "test/images"
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    test_images = random.sample(list(test_dir.glob("*.jpg")), 10)

    for idx, img_path in enumerate(test_images):
        print(f"Processing {img_path.name}...")
        guided_bp.visualize(img_path, output_dir / f"guided_backprop_yolo_{idx}.png")

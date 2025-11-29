import torch
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import sys
import random

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils import get_kaggle_dataset_root, get_device


class YoloGradCAM:
    def __init__(self, model_path, target_layer_idx=-4):
        self.model = YOLO(model_path)
        self.device = get_device()
        self.model.model.to(self.device)

        self.activations = None
        self.gradients = None
        self.target_layer = self.model.model.model[target_layer_idx]

        self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out if isinstance(out, torch.Tensor) else out[0]
        if self.activations.requires_grad:
            self.activations.register_hook(lambda grad: setattr(self, 'gradients', grad))

    def generate(self, img_path):
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (640, 640))

        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device).requires_grad_(True)

        self.model.model.to(self.device)
        self.model.model.eval()
        self.model.model.zero_grad()
        self.activations = None
        self.gradients = None

        output = self.model.model(img_tensor)

        target = self.activations.mean()
        target.backward()

        grads = self.gradients.cpu().data.numpy()
        acts = self.activations.cpu().data.numpy()

        weights = np.mean(grads, axis=(2, 3), keepdims=True)
        cam = np.sum(weights * acts, axis=1)[0]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (640, 640))

        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return img_rgb, cam

    def visualize(self, img_path, save_path):
        img, heatmap = self.generate(img_path)
        img_resized = cv2.resize(img, (640, 640))

        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        self.model.model.eval()
        with torch.no_grad():
            results = self.model(str(img_path), device=self.device)
        prediction_img = results[0].plot()
        prediction_img = cv2.cvtColor(prediction_img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.title("Original")
        plt.imshow(img_resized)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Grad-CAM")
        plt.imshow(heatmap_color)
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Prediction")
        plt.imshow(prediction_img)
        plt.axis('off')

        plt.tight_layout()

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

        plt.close()


if __name__ == "__main__":
    random.seed(42)
    model_path = project_root / "lab3/yolo/yolo-train/weights/best.pt"
    grad_cam = YoloGradCAM(model_path, target_layer_idx=-4)

    test_dir = get_kaggle_dataset_root() / "test/images"
    all_images = list(test_dir.glob("*.jpg"))
    test_images = random.sample(all_images, 10)

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    for idx, img_path in enumerate(test_images):
        print(f"Processing {img_path.name}...")
        save_path = output_dir / f"gradcam_yolo_{idx}.png"
        grad_cam.visualize(img_path, save_path)

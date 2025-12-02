import os
import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from rfdetr import RFDETRNano

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils import get_device, get_kaggle_dataset_root

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


class RFDetrGradCAM:
    def __init__(self, checkpoint_path: Path, target_stage_idx: int = 0):
        self.device = get_device()
        self.checkpoint_path = Path(checkpoint_path)
        self.model = self._load_model()
        self.model.model.model.to(self.device)
        self.model.model.device = self.device
        self.model.model.model.eval()
        self.resolution = self.model.model.resolution

        self.mean = torch.tensor(self.model.means, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(self.model.stds, device=self.device).view(1, 3, 1, 1)

        self.activations = None
        self.gradients = None

        backbone = self.model.model.model.backbone[0]
        projector = backbone.projector
        self.target_layer = projector.stages[target_stage_idx][0]
        self._register_hooks()

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

    def _register_hooks(self):
        def forward_hook(_, __, output):
            tensor = output if isinstance(output, torch.Tensor) else output[0]
            self.activations = tensor

            def backward_hook(grad):
                self.gradients = grad
                return grad

            tensor.register_hook(backward_hook)

        self.target_layer.register_forward_hook(forward_hook)

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
            if len(outputs) > 2:
                formatted["pred_masks"] = outputs[2]
            return formatted

    def _preprocess(self, img_rgb):
        img_resized = cv2.resize(img_rgb, (self.resolution, self.resolution))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(self.device)
        img_tensor = (img_tensor - self.mean) / self.std
        return img_resized, img_tensor

    def _detach_output(self, output):
        if isinstance(output, torch.Tensor):
            return output.detach()
        if isinstance(output, dict):
            return {k: self._detach_output(v) for k, v in output.items()}
        if isinstance(output, (list, tuple)):
            return type(output)(self._detach_output(v) for v in output)
        return output

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

    def _compute_cam(self):
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        abs_weights = self.gradients.abs().mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        if cam.max() <= 0:
            cam = torch.relu((abs_weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze(0).squeeze(0).detach().cpu().numpy()
        cam = cv2.resize(cam, (self.resolution, self.resolution))
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam

    def _create_heatmap(self, cam):
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    def _overlay_heatmap(self, img_rgb, cam, alpha=0.55):
        heatmap = self._create_heatmap(cam)
        img_f = img_rgb.astype(np.float32)
        heatmap_f = heatmap.astype(np.float32)
        overlay = cv2.addWeighted(img_f, 1 - alpha, heatmap_f, alpha, 0)
        return overlay.astype(np.uint8), heatmap

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

    def generate(self, img_path):
        img_path = Path(img_path)
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]

        img_resized, img_tensor = self._preprocess(img_rgb)
        self.model.model.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None
        with torch.enable_grad():
            raw_outputs = self.model.model.model(img_tensor)
        outputs = self._format_outputs(raw_outputs)

        logits = outputs["pred_logits"][0]
        boxes = outputs["pred_boxes"][0]
        probs = logits.sigmoid()
        scores_per_query, labels_per_query = probs.max(dim=-1)
        sorted_scores, sorted_idx = torch.sort(scores_per_query, descending=True)
        confident_mask = sorted_scores > 0.05
        top_indices = sorted_idx[confident_mask][:10]
        if top_indices.numel() == 0:
            top_indices = sorted_idx[:5]
        if top_indices.numel() == 0:
            top_indices = sorted_idx[:1]

        target_logit = logits[top_indices, labels_per_query[top_indices]].sum()
        target_logit.backward()

        best_query_idx = int(top_indices[0])
        best_label = int(labels_per_query[best_query_idx])
        best_score = scores_per_query[best_query_idx].item()

        cam = self._compute_cam()
        best_box_norm = boxes[best_query_idx].detach().cpu().numpy()
        best_box_xyxy = self._box_cxcywh_to_xyxy(best_box_norm)
        best_box_pixels = self._scale_box(best_box_xyxy, self.resolution, self.resolution)

        with torch.no_grad():
            processed_outputs = self._detach_output(outputs)
            target_sizes = torch.tensor([[orig_h, orig_w]], device=self.device)
            detections = self.model.model.postprocess(processed_outputs, target_sizes)[0]

        return {
            "original": img_rgb,
            "resized": img_resized,
            "cam": cam,
            "best_box": best_box_pixels,
            "best_score": best_score,
            "best_label": best_label,
            "detections": detections,
        }

    def visualize(self, img_path, save_path):
        result = self.generate(img_path)

        img_resized = result["resized"]
        overlay, heatmap = self._overlay_heatmap(img_resized, result["cam"], alpha=0.6)
        cv2.rectangle(overlay, (result["best_box"][0], result["best_box"][1]), (result["best_box"][2], result["best_box"][3]), (255, 255, 255), 2)

        original_img = img_resized
        prediction_img = self._draw_predictions(result["original"], result["detections"])
        prediction_img = cv2.resize(prediction_img, (self.resolution, self.resolution))

        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        panels = [
            ("Original", original_img),
            ("Grad-CAM Heatmap", heatmap),
            ("Prediction", prediction_img),
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


if __name__ == "__main__":
    random.seed(42)
    checkpoint = project_root / "lab3/rfdetr/rfdetr-train/checkpoint_best_total.pth"
    grad_cam = RFDetrGradCAM(checkpoint_path=checkpoint, target_stage_idx=0)

    test_dir = get_kaggle_dataset_root() / "test/images"
    test_images = random.sample(list(test_dir.glob("*.jpg")), 10)

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    for idx, img_path in enumerate(test_images):
        print(f"Processing {img_path.name}...")
        grad_cam.visualize(img_path, output_dir / f"gradcam_rfdetr_{idx}.png")

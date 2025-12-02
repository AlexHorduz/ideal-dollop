import os
import sys
from pathlib import Path

import cv2
import torch
from rfdetr import RFDETRNano
import supervision as sv

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from utils import get_kaggle_dataset_root, get_device

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


def run_rfdetr_inference(img_dir):
    device = get_device()
    weights_path = Path("/Users/programistich/Study/autumn-6/NeuralNetwork/ideal-dollop/lab3/rfdetr/rfdetr-train/checkpoint_best_total.pth")

    img_files = [
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
    ]

    if not img_files:
        print(f"No image files found in {img_dir}.")
        return

    with torch.serialization.safe_globals([type(None)]):
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    state_dict = checkpoint.get("model")
    num_classes = state_dict["class_embed.weight"].shape[0] - 1

    model = RFDETRNano(
        num_classes=num_classes,
        force_no_pretrain=True,
        pretrain_weights=None,
    )
    model.model.model.to(device)
    model.model.device = device
    model.model.model.load_state_dict(state_dict, strict=False)
    model.model.model.eval()

    detections = model.predict([str(p) for p in img_files[:5]], threshold=0.25)
    if not isinstance(detections, list):
        detections = [detections]

    save_dir = Path("runs/predict")
    save_dir.mkdir(parents=True, exist_ok=True)

    for img_path, det in zip(img_files[:5], detections):
        image = cv2.imread(img_path)
        labels = [
            f"{CLASS_NAMES[int(cls_id)]} {score:.2f}"
            for score, cls_id in zip(det.confidence, det.class_id)
        ]
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        annotated = box_annotator.annotate(scene=image.copy(), detections=det)
        annotated = label_annotator.annotate(scene=annotated, detections=det, labels=labels)
        out_path = save_dir / Path(img_path).name
        cv2.imwrite(str(out_path), annotated)

    print(f"Inference completed. Annotated images saved to {save_dir}.")


if __name__ == "__main__":
    dataset_root = get_kaggle_dataset_root()
    test_images_dir = dataset_root / "test/images"
    run_rfdetr_inference(img_dir=test_images_dir)

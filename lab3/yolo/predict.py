from pathlib import Path
from ultralytics import YOLO
import os
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from utils import get_kaggle_dataset_root


def run_tolo_inference(img_dir):
    model = YOLO("yolo-train/weights/best.pt")
    img_files = [
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
    ]

    if not img_files:
        print(f"No image files found in {img_dir}.")
        return

    results = model.predict(
        source=img_files[:5],
        conf=0.25,
        save=True,
        save_txt=True,
    )

    print("Inference completed. Results saved.")


if __name__ == "__main__":
    src_path = get_kaggle_dataset_root()
    test_images_dir = src_path / "test/images"
    run_tolo_inference(img_dir=test_images_dir)

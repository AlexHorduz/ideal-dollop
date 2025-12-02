import os
import json
from pathlib import Path
from PIL import Image

# Your class list
LABELS = [
    "Green Light", "Red Light", "Speed Limit 10", "Speed Limit 100",
    "Speed Limit 110", "Speed Limit 120", "Speed Limit 20", "Speed Limit 30",
    "Speed Limit 40", "Speed Limit 50", "Speed Limit 60", "Speed Limit 70",
    "Speed Limit 80", "Speed Limit 90", "Stop"
]

def convert_split(split, input_root="car", output_root="car_yolo"):
    input_images = Path(input_root) / split / "images"
    input_labels = Path(input_root) / split / "labels"
    output_split = Path(output_root) / split
    output_split.mkdir(parents=True, exist_ok=True)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i, "name": name, "supercategory": "none"} for i, name in enumerate(LABELS)
        ]
    }

    coco = {
        "info": {
            "description": "Car dataset",
            "version": "1.0",
            "year": 2025
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i, "name": name, "supercategory": "none"}
            for i, name in enumerate(LABELS)
        ]
    }

    ann_id = 1
    img_id = 1

    for img_file in sorted(input_images.glob("*.*")):
        if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        # Load image to get shape
        img = Image.open(img_file)
        w, h = img.size

        # Copy image into new dataset
        out_img_path = output_split / img_file.name
        img.save(out_img_path)

        coco["images"].append({
            "id": img_id,
            "file_name": img_file.name,
            "width": w,
            "height": h
        })

        # Load labels
        label_path = input_labels / (img_file.stem + ".txt")
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    cls, xc, yc, bw, bh = map(float, line.strip().split())

                    # Convert YOLO → COCO (absolute, xywh)
                    x = (xc - bw/2) * w
                    y = (yc - bh/2) * h
                    ww = bw * w
                    hh = bh * h

                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": int(cls),
                        "bbox": [x, y, ww, hh],  # COCO xywh
                        "area": ww * hh,
                        "iscrowd": 0
                    })
                    ann_id += 1

        img_id += 1

    # Save COCO annotation file
    with open(output_split / "_annotations.coco.json", "w") as f:
        json.dump(coco, f, indent=2)


def main():
    for split in ["train", "valid", "test"]:
        # If no test split exists, skip
        if not Path("dataset") / split / "images":
            continue
        convert_split(split, input_root="dataset", output_root="dataset_coco")

    print("✔ Conversion complete! RF-DETR dataset saved to dataset_coco/")


if __name__ == "__main__":
    main()

from typing import Dict, List
from collections import defaultdict
import os
import random
from functools import reduce

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def read_labels(folder: str) -> Dict[str, List[Dict]]:
    labels = defaultdict(lambda: [])
    for file_name in os.listdir(folder):
        with open(os.path.join(folder, file_name), "r") as f:
            for line in f:
                annotation = line.strip().split()
                labels[file_name].append({
                    "class_id": int(annotation[0]),
                    "x_center": float(annotation[1]),
                    "y_center": float(annotation[2]),
                    "width": float(annotation[3]),
                    "height": float(annotation[4]),
                })

    return labels

def count_labels(labels: Dict[str, List[Dict]]):
    """
    Counts total number of instances of each of the class_id
    """
    counts = defaultdict(lambda: 0)
    for filename, annotations in labels.items():
        for ann in annotations:
            counts[ann["class_id"]] += 1

    return counts

def count_labels_per_image(labels: Dict[str, List[Dict]]) -> List[int]:
    """
    Count how many labels each image has.
    """
    counts = []
    for filename, annotations in labels.items():
        counts.append(len(annotations))
    return counts

def show_image_grid(
        labels_dict, images_path, labels_names, grid_shape=(3, 3), seed=None
    ):
    """
    Display a grid of randomly selected images with bounding boxes and class names.
    """
    n_images = reduce(lambda a, b: a*b, grid_shape)

    filenames = list(labels_dict.keys())
    if seed is not None:
        random.seed(seed)
    filenames = random.sample(filenames, min(n_images, len(filenames)))
    
    fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=(15, 15))
    axes = axes.flatten()
    
    for ax, fname in zip(axes, filenames):
        img_path = os.path.join(images_path, fname.replace(".txt", ".jpg"))
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        ax.imshow(img)
        ax.set_title(fname, fontsize=8)
        ax.axis("off")
        
        h, w, _ = img.shape
        for ann in labels_dict[fname]:
            xc, yc, bw, bh = (
                ann["x_center"], ann["y_center"], ann["width"], ann["height"]
            )
            x1 = (xc - bw / 2) * w
            y1 = (yc - bh / 2) * h
            rect_w = bw * w
            rect_h = bh * h
            
            rect = Rectangle(
                (x1, y1), rect_w, rect_h, linewidth=2, edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)
            
            class_name = labels_names[ann["class_id"]]
            ax.text(
                x1, y1 - 5, class_name, color="yellow", fontsize=10,
                weight="bold", bbox=dict(facecolor='black', alpha=0.5, pad=1)
            )
    
    for ax in axes[n_images:]:
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()
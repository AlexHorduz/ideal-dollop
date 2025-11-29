import os
import shutil

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def copy_tree(src, dst):
    """Copies all files from src to dst."""
    makedirs(dst)
    if not os.path.exists(src):
        return
    for f in os.listdir(src):
        shutil.copy(os.path.join(src, f), os.path.join(dst, f))

def main():
    # Source directories
    src_root = "car"
    train_img_src = os.path.join(src_root, "train/images")
    train_lbl_src = os.path.join(src_root, "train/labels")
    val_img_src   = os.path.join(src_root, "valid/images")
    val_lbl_src   = os.path.join(src_root, "valid/labels")

    # Output directories
    dst_root = "car_yolo"
    img_train_dst = os.path.join(dst_root, "images/train")
    img_val_dst   = os.path.join(dst_root, "images/val")
    lbl_train_dst = os.path.join(dst_root, "labels/train")
    lbl_val_dst   = os.path.join(dst_root, "labels/val")

    # Create required folders
    for p in [img_train_dst, img_val_dst, lbl_train_dst, lbl_val_dst]:
        makedirs(p)

    # Copy files
    copy_tree(train_img_src, img_train_dst)
    copy_tree(train_lbl_src, lbl_train_dst)
    copy_tree(val_img_src,   img_val_dst)
    copy_tree(val_lbl_src,   lbl_val_dst)

    # Write YAML
    # yaml_path = "car_yolo.yaml"
    # with open(yaml_path, "w") as f:
    #     f.write(
    #         f"path: {dst_root}\n"
    #         f"train: images/train\n"
    #         f"val: images/val\n\n"
    #         f"names:\n"
    #         f"  0: car\n"
    #     )

    print("Dataset prepared successfully!")
    print("YOLO dataset folder: car_yolo/")
    print("YOLO dataset config: car_yolo.yaml")

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
from tqdm import tqdm

from loader import TrafficSignsDataset, detection_collate_fn
from transforms import build_transforms
from models.yolo import YOLOv1
from models.alexnet import AlexNetOD
from eval import compute_map, precision_recall_per_class_with_f1_thresh, box_iou
import utils
import config as cfg


def decode_yolo_output(output, grid_size=7, B=2, num_classes=15, conf_threshold=0.25, img_size=(448, 448)):
    batch_size = output.shape[0]
    output = output.reshape(batch_size, grid_size, grid_size, B * 5 + num_classes)

    predictions = []
    H, W = img_size

    for b in range(batch_size):
        boxes_list = []
        scores_list = []
        labels_list = []

        for i in range(grid_size):
            for j in range(grid_size):
                cell_data = output[b, i, j]
                class_probs = torch.softmax(cell_data[B * 5:], dim=0)

                for k in range(B):
                    bbox_data = cell_data[k * 5:(k + 1) * 5]
                    x, y, w, h, conf_logit = bbox_data

                    confidence = torch.sigmoid(conf_logit)
                    x_center = (j + torch.sigmoid(x)) / grid_size * W
                    y_center = (i + torch.sigmoid(y)) / grid_size * H
                    width = torch.sigmoid(w) * W
                    height = torch.sigmoid(h) * H

                    x1 = torch.clamp(x_center - width / 2, 0, W)
                    y1 = torch.clamp(y_center - height / 2, 0, H)
                    x2 = torch.clamp(x_center + width / 2, 0, W)
                    y2 = torch.clamp(y_center + height / 2, 0, H)

                    class_prob, class_idx = torch.max(class_probs, dim=0)
                    score = confidence * class_prob

                    if score > conf_threshold:
                        boxes_list.append([x1.item(), y1.item(), x2.item(), y2.item()])
                        scores_list.append(score.item())
                        labels_list.append(class_idx.item())

        predictions.append({
            'boxes': np.array(boxes_list) if boxes_list else np.zeros((0, 4)),
            'scores': np.array(scores_list) if scores_list else np.zeros(0),
            'labels': np.array(labels_list) if labels_list else np.zeros(0, dtype=int)
        })

    return predictions


def compute_yolo_loss(output, targets, grid_size=7, B=2, num_classes=15,
                      lambda_box=0.05, lambda_obj=1.0, lambda_cls=0.5, lambda_noobj=0.5):
    device = output.device
    batch_size = output.shape[0]
    output = output.reshape(batch_size, grid_size, grid_size, B * 5 + num_classes)

    total_loss = 0.0
    box_loss = 0.0
    obj_loss = 0.0
    noobj_loss = 0.0
    cls_loss = 0.0

    # Track which cells are responsible for objects
    responsible_mask = torch.zeros(batch_size, grid_size, grid_size, B, device=device, dtype=torch.bool)

    for b in range(batch_size):
        target = targets[b]
        boxes = target['boxes']
        labels = target['labels']
        img_size = target['size']
        H, W = img_size[0].item(), img_size[1].item()

        if len(boxes) == 0:
            continue

        for n in range(len(boxes)):
            box = boxes[n]
            label = labels[n].item()

            x1, y1, x2, y2 = box
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            i = min(int(y_center / H * grid_size), grid_size - 1)
            j = min(int(x_center / W * grid_size), grid_size - 1)

            cell_pred = output[b, i, j]

            # Find which bbox has highest IoU with ground truth (responsible predictor)
            best_iou = 0.0
            best_bbox_idx = 0
            pred_boxes_list = []

            gt_box = torch.tensor([[x1, y1, x2, y2]], device=device, dtype=torch.float32)

            for k in range(B):
                bbox_pred = cell_pred[k * 5:(k + 1) * 5]
                x_pred, y_pred, w_pred, h_pred, conf_pred = bbox_pred

                # Convert predicted offsets to absolute coordinates
                x_center_pred = (j + torch.sigmoid(x_pred)) / grid_size * W
                y_center_pred = (i + torch.sigmoid(y_pred)) / grid_size * H
                width_pred = torch.sigmoid(w_pred) * W
                height_pred = torch.sigmoid(h_pred) * H

                # Create predicted box in [x1, y1, x2, y2] format
                pred_box = torch.stack([
                    x_center_pred - width_pred / 2,
                    y_center_pred - height_pred / 2,
                    x_center_pred + width_pred / 2,
                    y_center_pred + height_pred / 2
                ]).unsqueeze(0)

                pred_boxes_list.append(pred_box)

                # Compute IoU to find responsible predictor
                iou = box_iou(pred_box, gt_box)
                if iou[0, 0] > best_iou:
                    best_iou = iou[0, 0]
                    best_bbox_idx = k

            # Mark this bbox as responsible
            responsible_mask[b, i, j, best_bbox_idx] = True

            # Compute loss for the responsible bbox
            best_bbox_pred = cell_pred[best_bbox_idx * 5:(best_bbox_idx + 1) * 5]
            x_pred, y_pred, w_pred, h_pred, conf_pred = best_bbox_pred

            # Box loss (IoU-based)
            box_loss += (1.0 - best_iou)

            # Objectness loss for responsible predictor (should predict 1)
            obj_loss += nn.functional.binary_cross_entropy_with_logits(
                conf_pred, torch.tensor(1.0, device=device)
            )

            # Classification loss (only once per cell, not per bbox)
            class_logits = cell_pred[B * 5:]
            cls_loss += nn.functional.cross_entropy(
                class_logits.unsqueeze(0),
                torch.tensor([label], device=device, dtype=torch.long)
            )

    # No-object loss: penalize confidence for bboxes not responsible for any object
    for b in range(batch_size):
        for i in range(grid_size):
            for j in range(grid_size):
                cell_pred = output[b, i, j]
                for k in range(B):
                    if not responsible_mask[b, i, j, k]:
                        conf_pred = cell_pred[k * 5 + 4]
                        noobj_loss += nn.functional.binary_cross_entropy_with_logits(
                            conf_pred, torch.tensor(0.0, device=device)
                        )

    num_objects = sum(len(t['boxes']) for t in targets)
    num_grid_cells = batch_size * grid_size * grid_size * B

    if num_objects > 0:
        box_loss = box_loss / num_objects
        obj_loss = obj_loss / num_objects
        cls_loss = cls_loss / num_objects

    if num_grid_cells > 0:
        noobj_loss = noobj_loss / num_grid_cells

    total_loss = lambda_box * box_loss + lambda_obj * obj_loss + lambda_noobj * noobj_loss + lambda_cls * cls_loss

    return {
        'total_loss': total_loss,
        'box_loss': box_loss,
        'obj_loss': obj_loss,
        'noobj_loss': noobj_loss,
        'cls_loss': cls_loss
    }


def train_one_epoch(model, dataloader, optimizer, device, config):
    model.train()
    total_loss = 0.0
    total_box_loss = 0.0
    total_obj_loss = 0.0
    total_noobj_loss = 0.0
    total_cls_loss = 0.0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, (images, targets) in enumerate(pbar):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in t.items()} for t in targets]
        images = torch.stack(images)

        outputs = model(images)
        losses = compute_yolo_loss(
            outputs, targets,
            grid_size=config['grid_size'],
            B=config['B'],
            num_classes=config['num_classes'],
            lambda_box=config['lambda_box'],
            lambda_obj=config['lambda_obj'],
            lambda_cls=config['lambda_cls'],
            lambda_noobj=config.get('lambda_noobj', 0.5)
        )

        loss = losses['total_loss']
        optimizer.zero_grad()
        loss.backward()

        if 'grad_clip_norm' in config:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip_norm'])

        optimizer.step()
        total_loss += loss.item()
        total_box_loss += float(losses['box_loss'])
        total_obj_loss += float(losses['obj_loss'])
        total_noobj_loss += float(losses['noobj_loss'])
        total_cls_loss += float(losses['cls_loss'])

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'box': f"{float(losses['box_loss']):.4f}",
            'obj': f"{float(losses['obj_loss']):.4f}",
            'noobj': f"{float(losses['noobj_loss']):.4f}",
            'cls': f"{float(losses['cls_loss']):.4f}"
        })

    num_batches = max(1, len(dataloader))
    return {
        'total_loss': total_loss / num_batches,
        'box_loss': total_box_loss / num_batches,
        'obj_loss': total_obj_loss / num_batches,
        'noobj_loss': total_noobj_loss / num_batches,
        'cls_loss': total_cls_loss / num_batches
    }


def validate(model, dataloader, device, config):
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    total_box_loss = 0.0
    total_obj_loss = 0.0
    total_noobj_loss = 0.0
    total_cls_loss = 0.0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", leave=False)
        for images, targets in pbar:
            targets_cpu = targets
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in t.items()} for t in targets]
            images = torch.stack(images)

            outputs = model(images)
            losses = compute_yolo_loss(
                outputs, targets,
                grid_size=config['grid_size'],
                B=config['B'],
                num_classes=config['num_classes'],
                lambda_box=config['lambda_box'],
                lambda_obj=config['lambda_obj'],
                lambda_cls=config['lambda_cls'],
                lambda_noobj=config.get('lambda_noobj', 0.5)
            )

            total_loss += float(losses['total_loss'])
            total_box_loss += float(losses['box_loss'])
            total_obj_loss += float(losses['obj_loss'])
            total_noobj_loss += float(losses['noobj_loss'])
            total_cls_loss += float(losses['cls_loss'])

            predictions = decode_yolo_output(
                outputs,
                grid_size=config['grid_size'],
                B=config['B'],
                num_classes=config['num_classes'],
                conf_threshold=cfg.EVAL_CONFIG['conf_threshold'],
                img_size=config['input_size']
            )

            all_predictions.extend(predictions)

            for target in targets_cpu:
                all_targets.append({
                    'boxes': target['boxes'].detach().numpy(),
                    'labels': target['labels'].detach().numpy()
                })

            # Update progress bar with current loss
            pbar.set_postfix({
                'loss': f"{float(losses['total_loss']):.4f}"
            })

    map_results = compute_map(
        all_predictions,
        all_targets,
        num_classes=config['num_classes'],
        iou_thresholds=cfg.EVAL_CONFIG['iou_thresholds']
    )

    pr_results = precision_recall_per_class_with_f1_thresh(
        all_predictions,
        all_targets,
        num_classes=config['num_classes'],
        iou_thr=0.5
    )

    num_batches = max(1, len(dataloader))

    return {
        'loss': total_loss / num_batches,
        'box_loss': total_box_loss / num_batches,
        'obj_loss': total_obj_loss / num_batches,
        'noobj_loss': total_noobj_loss / num_batches,
        'cls_loss': total_cls_loss / num_batches,
        'mAP': map_results['map'],
        'per_class_ap': map_results['per_class_ap'],
        'precision': pr_results['precision'],
        'recall': pr_results['recall'],
        'best_threshold': pr_results['best_threshold']
    }


def train(model_name, config_override=None, debug_samples=None):
    config = cfg.get_config(model_name)
    if config_override:
        config.update(config_override)

    device = utils.get_device()

    root = utils.get_kaggle_dataset_root()

    train_tfm = build_transforms("train", size=config['input_size'], augment=config['augment'])
    val_tfm = build_transforms("valid", size=config['input_size'], augment=False)

    train_ds = TrafficSignsDataset(root, split="train", transform=train_tfm)
    val_ds = TrafficSignsDataset(root, split="valid", transform=val_tfm)

    if debug_samples is not None:
        from torch.utils.data import Subset
        train_indices = list(range(min(debug_samples, len(train_ds))))
        val_indices = list(range(min(debug_samples // 5, len(val_ds))))
        train_ds = Subset(train_ds, train_indices)
        val_ds = Subset(val_ds, val_indices)
        print(f"Debug mode: using {len(train_ds)} training samples and {len(val_ds)} validation samples")

    train_loader = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=cfg.TRAINING_CONFIG['num_workers'],
        pin_memory=cfg.TRAINING_CONFIG['pin_memory'],
        collate_fn=detection_collate_fn
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=cfg.TRAINING_CONFIG['num_workers'],
        pin_memory=cfg.TRAINING_CONFIG['pin_memory'],
        collate_fn=detection_collate_fn
    )

    if model_name.lower() == 'yolo':
        model = YOLOv1(
            grid_size=config['grid_size'],
            B=config['B'],
            num_classes=config['num_classes']
        ).to(device)
    elif model_name.lower() == 'alexnet':
        model = AlexNetOD(
            grid_size=config['grid_size'],
            B=config['B'],
            num_classes=config['num_classes']
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'], momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")

    print(f"Starting training {model_name} for {config['epochs']} epochs...")
    print(f"Config: {config}")

    best_map = 0.0
    best_f1 = 0.0

    epoch_times = []
    start_time = time.time()
    train_history = []
    val_history = []

    epoch_bar = tqdm(range(config['epochs']), desc="Training Progress")
    for epoch in epoch_bar:
        epoch_start_time = time.time()

        epoch_bar.set_description(f"Epoch {epoch + 1}/{config['epochs']}")

        train_results = train_one_epoch(model, train_loader, optimizer, device, config)

        train_history.append({
            'epoch': epoch + 1,
            'total_loss': float(train_results['total_loss']),
            'box_loss': float(train_results['box_loss']),
            'obj_loss': float(train_results['obj_loss']),
            'noobj_loss': float(train_results['noobj_loss']),
            'cls_loss': float(train_results['cls_loss'])
        })

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = config['epochs'] - (epoch + 1)
        eta_seconds = avg_epoch_time * remaining_epochs
        eta_minutes = eta_seconds / 60
        eta_hours = eta_minutes / 60

        # Format ETA
        if eta_hours >= 1:
            eta_str = f"{eta_hours:.1f}h"
        elif eta_minutes >= 1:
            eta_str = f"{eta_minutes:.1f}m"
        else:
            eta_str = f"{eta_seconds:.0f}s"

        # Update progress bar with metrics
        postfix_dict = {
            'loss': f"{train_results['total_loss']:.4f}",
            'epoch_time': f"{epoch_time:.1f}s",
            'ETA': eta_str
        }

        if (epoch + 1) % cfg.TRAINING_CONFIG['eval_every'] == 0:
            val_results = validate(model, val_loader, device, config)

            mean_precision = float(val_results['precision'].mean())
            mean_recall = float(val_results['recall'].mean())
            f1_score = 2 * mean_precision * mean_recall / (mean_precision + mean_recall + 1e-9)

            val_history.append({
                'epoch': epoch + 1,
                'loss': float(val_results['loss']),
                'box_loss': float(val_results['box_loss']),
                'obj_loss': float(val_results['obj_loss']),
                'noobj_loss': float(val_results['noobj_loss']),
                'cls_loss': float(val_results['cls_loss']),
                'mAP': float(val_results['mAP']),
                'precision': mean_precision,
                'recall': mean_recall,
                'f1_score': float(f1_score)
            })

            postfix_dict['mAP'] = f"{val_results['mAP']:.4f}"
            postfix_dict['F1'] = f"{f1_score:.4f}"

            if val_results['mAP'] > best_map:
                best_map = val_results['mAP']
                tqdm.write(f"✓ New best mAP: {best_map:.4f}")

            if f1_score > best_f1:
                best_f1 = f1_score
                tqdm.write(f"✓ New best F1: {best_f1:.4f}")

        epoch_bar.set_postfix(postfix_dict)

    total_time = time.time() - start_time
    print(f"Training completed in {total_time / 60:.1f} minutes ({total_time / 3600:.2f} hours)")

    final_results = validate(model, val_loader, device, config)

    # Calculate F1-score
    mean_precision = float(final_results['precision'].mean())
    mean_recall = float(final_results['recall'].mean())
    f1_score = 2 * mean_precision * mean_recall / (mean_precision + mean_recall + 1e-9)

    # Add F1-score to results
    final_results['f1_score'] = f1_score

    final_summary = {
        'epoch': config['epochs'],
        'loss': float(final_results['loss']),
        'box_loss': float(final_results['box_loss']),
        'obj_loss': float(final_results['obj_loss']),
        'noobj_loss': float(final_results['noobj_loss']),
        'cls_loss': float(final_results['cls_loss']),
        'mAP': float(final_results['mAP']),
        'precision': mean_precision,
        'recall': mean_recall,
        'f1_score': float(f1_score)
    }
    if not val_history or val_history[-1]['epoch'] != final_summary['epoch']:
        val_history.append(final_summary)

    final_results['train_history'] = train_history
    final_results['val_history'] = val_history
    final_results['validation_summary'] = final_summary

    test_results = None
    test_summary = None
    test_split_dir = root / "test"
    if test_split_dir.exists():
        test_tfm = build_transforms("test", size=config['input_size'], augment=False)
        test_ds = TrafficSignsDataset(root, split="test", transform=test_tfm)

        if debug_samples is not None:
            from torch.utils.data import Subset
            num_test_samples = max(1, debug_samples // 5)
            test_indices = list(range(min(num_test_samples, len(test_ds))))
            test_ds = Subset(test_ds, test_indices)

        test_loader = DataLoader(
            test_ds,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=cfg.TRAINING_CONFIG['num_workers'],
            pin_memory=cfg.TRAINING_CONFIG['pin_memory'],
            collate_fn=detection_collate_fn
        )

        test_results = validate(model, test_loader, device, config)

        test_precision = float(test_results['precision'].mean())
        test_recall = float(test_results['recall'].mean())
        test_f1 = 2 * test_precision * test_recall / (test_precision + test_recall + 1e-9)
        test_summary = {
            'loss': float(test_results['loss']),
            'box_loss': float(test_results['box_loss']),
            'obj_loss': float(test_results['obj_loss']),
            'noobj_loss': float(test_results['noobj_loss']),
            'cls_loss': float(test_results['cls_loss']),
            'mAP': float(test_results['mAP']),
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': float(test_f1)
        }
        final_results['test_summary'] = test_summary
    else:
        print("Test split not found. Skipping test evaluation.")

    print(f"\nFinal Results:")
    print(f"mAP@[0.5:0.95]: {final_results['mAP']:.4f}")
    print(f"Mean Precision: {mean_precision:.4f}")
    print(f"Mean Recall: {mean_recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    print(f"Loss: {final_results['loss']:.4f} "
          f"(Box: {final_results['box_loss']:.4f}, Obj: {final_results['obj_loss']:.4f}, "
          f"NoObj: {final_results['noobj_loss']:.4f}, Cls: {final_results['cls_loss']:.4f})")

    if test_summary:
        print(f"\nTest Results:")
        print(f"mAP@[0.5:0.95]: {test_summary['mAP']:.4f}")
        print(f"Mean Precision: {test_summary['precision']:.4f}")
        print(f"Mean Recall: {test_summary['recall']:.4f}")
        print(f"F1-Score: {test_summary['f1_score']:.4f}")
        print(f"Loss: {test_summary['loss']:.4f} "
              f"(Box: {test_summary['box_loss']:.4f}, Obj: {test_summary['obj_loss']:.4f}, "
              f"NoObj: {test_summary['noobj_loss']:.4f}, Cls: {test_summary['cls_loss']:.4f})")

    return final_results


if __name__ == "__main__":
    # print('Testing YOLO...')
    # train('yolo', debug_samples=20)
    print('Testing AlexNet...')
    train('alexnet', debug_samples=20)
    print('All tests passed!')

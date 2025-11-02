"""
Helpers for object detection metrics and losses using PyTorch.

Реалізовано:
- IoU computation (boxes in [x1, y1, x2, y2] форматі)
- Box loss: L_box = 1 - IoU(pred, gt)
- Objectness loss: BCEWithLogits (підтримує logits)
- Classification loss: CrossEntropyLoss (підтримує logits)
- Total loss: L_total = λ_box * L_box + λ_obj * L_obj + λ_cls * L_cls
- mAP computation: AP averaged over IoU thresholds 0.5:0.05:0.95 (AP@[0.5:0.95]) по всім класам
- Precision & Recall per class at IoU=0.5, with per-class confidence threshold selected by max F1

Примітки:
- Метрики очікують формат даних, який зазвичай використовують у detection:
    predictions: list of dicts per image, each dict має ключи:
        'boxes' : (N,4) array-like (x1,y1,x2,y2)
        'scores': (N,) confidences
        'labels': (N,) integer class ids (0..C-1)
    ground_truths: list of dicts per image, each dict має:
        'boxes': (M,4)
        'labels': (M,)
- Для loss-функцій очікується, що предіказання та таргети вирівняні по елементам
  (тобто для кожного предікта є відповідний gt) — якщо потрібне складніше match'ування,
  зробіть matching зверху моделі або використовуйте окремий matching-утиліт.
"""

from typing import List, Dict, Tuple, Optional
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np


# ---------------------------
# Utility: IoU
# ---------------------------
def box_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """
    Compute IoU between two sets of boxes.
    boxes1: (N,4), boxes2: (M,4) in [x1,y1,x2,y2]
    Returns: (N,M) IoU matrix
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

    # intersection
    x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    union = area1[:, None] + area2[None, :] - inter
    iou = inter / (union + eps)
    return iou


# ---------------------------
# Loss components
# ---------------------------
def box_loss_iou(pred_boxes: Tensor, gt_boxes: Tensor, reduction: str = 'mean') -> Tensor:
    """
    L_box = 1 - IoU(pred, gt)
    Assumes pred_boxes and gt_boxes have same shape (N,4)
    """
    if pred_boxes.numel() == 0:
        return torch.tensor(0.0, device=pred_boxes.device, requires_grad=True)
    ious = box_iou(pred_boxes, gt_boxes)  # (N,N) if not aligned - but we expect aligned; handle aligned case
    # if shapes aligned (same N) and matching by index, take diagonal
    if ious.shape[0] == ious.shape[1]:
        iou_diag = torch.diag(ious)
    else:
        # fallback: assume one-to-one in order
        min_n = min(pred_boxes.shape[0], gt_boxes.shape[0])
        iou_diag = ious[:min_n, :min_n].diag()
    loss = 1.0 - iou_diag
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss  # no reduction


def objectness_loss(pred_logits: Tensor, targets: Tensor, reduction: str = 'mean') -> Tensor:
    """
    Binary cross-entropy for objectness.
    pred_logits: logits predicted for objectness, shape (N,)
    targets: 0/1 targets shape (N,)
    Uses BCEWithLogits for numerical stability.
    """
    pred_logits = pred_logits.view(-1)
    targets = targets.view(-1).float()
    loss = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction=reduction)
    return loss


def classification_loss(pred_logits: Tensor, targets: Tensor, reduction: str = 'mean') -> Tensor:
    """
    Multi-class classification loss (CrossEntropy).
    pred_logits: (N, num_classes)
    targets: (N,) int64 class labels in [0..C-1]
    """
    if pred_logits.numel() == 0:
        return torch.tensor(0.0, device=pred_logits.device, requires_grad=True)
    loss = F.cross_entropy(pred_logits, targets.long(), reduction=reduction)
    return loss


def total_loss(pred_boxes: Tensor,
               gt_boxes: Tensor,
               pred_obj_logits: Tensor,
               obj_targets: Tensor,
               pred_cls_logits: Tensor,
               cls_targets: Tensor,
               lambda_box: float = 0.05,
               lambda_obj: float = 1.0,
               lambda_cls: float = 0.5) -> Dict[str, Tensor]:
    """
    Compute all three components and weighted sum.
    Returns dict with 'L_box', 'L_obj', 'L_cls', 'L_total'
    NOTE: pred/gt alignment must be handled by caller (matching).
    """
    device = pred_boxes.device if pred_boxes.numel() else pred_obj_logits.device
    L_box = box_loss_iou(pred_boxes, gt_boxes, reduction='mean') if pred_boxes.numel() else torch.tensor(0.0, device=device)
    L_obj = objectness_loss(pred_obj_logits, obj_targets, reduction='mean') if pred_obj_logits.numel() else torch.tensor(0.0, device=device)
    L_cls = classification_loss(pred_cls_logits, cls_targets, reduction='mean') if pred_cls_logits.numel() else torch.tensor(0.0, device=device)

    L_total = lambda_box * L_box + lambda_obj * L_obj + lambda_cls * L_cls
    return {'L_box': L_box, 'L_obj': L_obj, 'L_cls': L_cls, 'L_total': L_total}


# ---------------------------
# mAP and per-class precision/recall
# ---------------------------

def _gather_all_detections(predictions: List[Dict], num_classes: int) -> List[List[Dict]]:
    """
    For internal use.
    Returns detections_by_class[c] = list of dicts:
      {'image_id': int, 'box': [x1,y1,x2,y2], 'score': float}
    """
    detections_by_class = [[] for _ in range(num_classes)]
    for img_id, pred in enumerate(predictions):
        boxes = np.asarray(pred.get('boxes', []))
        scores = np.asarray(pred.get('scores', [])) if 'scores' in pred else np.ones(len(boxes))
        labels = np.asarray(pred.get('labels', []))
        for b, s, l in zip(boxes, scores, labels):
            if 0 <= int(l) < num_classes:
                detections_by_class[int(l)].append({'image_id': img_id, 'box': b.astype(float), 'score': float(s)})
    return detections_by_class


def _gather_all_gts(ground_truths: List[Dict], num_classes: int) -> List[Dict]:
    """
    Returns gts_by_image: dict image_id -> {class: [boxes]}
    Also counts number of GTs per class.
    """
    gts_by_image = {}
    gt_count_per_class = [0] * num_classes
    for img_id, gt in enumerate(ground_truths):
        boxes = np.asarray(gt.get('boxes', []))
        labels = np.asarray(gt.get('labels', []))
        per_class = {}
        for b, l in zip(boxes, labels):
            l = int(l)
            per_class.setdefault(l, []).append(b.astype(float))
            if 0 <= l < num_classes:
                gt_count_per_class[l] += 1
        gts_by_image[img_id] = per_class
    return gts_by_image, gt_count_per_class


def _compute_ap_from_pr(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Compute Average Precision as area under precision-recall curve.
    Use interpolation by computing precision envelope and integrating using trapezoid.
    """
    # Append sentinel values at ends
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # precision envelope
    for i in range(mpre.size - 1, 0, -1):
        if mpre[i - 1] < mpre[i]:
            mpre[i - 1] = mpre[i]

    # find points where recall changes
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = 0.0
    for i in indices:
        ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
    return float(ap)


def compute_map(predictions: List[Dict],
                ground_truths: List[Dict],
                num_classes: int,
                iou_thresholds: Optional[List[float]] = None) -> Dict:
    """
    Compute mAP@[0.5:0.95] (mean AP over IoU thresholds) and per-class APs.

    predictions, ground_truths: lists per image (see docstring top)
    num_classes: number of classes C

    Returns dict:
      - 'map': scalar mAP over all classes and IoU thresholds
      - 'per_class_ap': np.array shape (C,) - average AP across IoU thresholds for each class
      - 'per_iou_ap': dict mapping iou-> average AP across classes
    """
    if iou_thresholds is None:
        iou_thresholds = [round(x, 2) for x in np.arange(0.5, 0.95 + 1e-9, 0.05)]

    detections_by_class = _gather_all_detections(predictions, num_classes)
    gts_by_image, gt_count_per_class = _gather_all_gts(ground_truths, num_classes)

    per_iou_ap = {}
    per_class_ap_matrix = np.zeros((len(iou_thresholds), num_classes), dtype=float)

    for iou_idx, iou_thr in enumerate(iou_thresholds):
        # compute AP per class at this IoU threshold
        aps = []
        for c in range(num_classes):
            detections = detections_by_class[c]
            # sort detections by score desc
            detections = sorted(detections, key=lambda x: x['score'], reverse=True)
            nd = len(detections)
            tp = np.zeros(nd, dtype=float)
            fp = np.zeros(nd, dtype=float)

            # prepare gt matches tracker: for each image and class, flags
            gt_for_class = {}
            for img_id, per_class in gts_by_image.items():
                boxes = per_class.get(c, [])
                if len(boxes) > 0:
                    gt_for_class[img_id] = {'boxes': np.array(boxes), 'matched': np.zeros(len(boxes), dtype=bool)}
                else:
                    gt_for_class[img_id] = {'boxes': np.zeros((0, 4)), 'matched': np.zeros(0, dtype=bool)}

            for d_idx, det in enumerate(detections):
                img_id = det['image_id']
                box_det = det['box']
                gts = gt_for_class.get(img_id, {'boxes': np.zeros((0, 4)), 'matched': np.zeros(0, dtype=bool)})
                gt_boxes = gts['boxes']
                if gt_boxes.shape[0] == 0:
                    fp[d_idx] = 1.0
                    continue
                # compute IoUs between det and all gts of that class in image
                # convert to tensors for iou util or compute in numpy
                # use numpy implementation:
                ix1 = np.maximum(gt_boxes[:, 0], box_det[0])
                iy1 = np.maximum(gt_boxes[:, 1], box_det[1])
                ix2 = np.minimum(gt_boxes[:, 2], box_det[2])
                iy2 = np.minimum(gt_boxes[:, 3], box_det[3])
                iw = np.maximum(ix2 - ix1, 0.0)
                ih = np.maximum(iy2 - iy1, 0.0)
                inter = iw * ih
                area_det = max((box_det[2] - box_det[0]) * (box_det[3] - box_det[1]), 0.0)
                area_gt = np.maximum((gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1]), 0.0)
                union = area_det + area_gt - inter
                ious = inter / (union + 1e-7)

                # find best IoU
                best_idx = ious.argmax()
                best_iou = ious[best_idx]
                if best_iou >= iou_thr:
                    if not gts['matched'][best_idx]:
                        tp[d_idx] = 1.0
                        gts['matched'][best_idx] = True
                        gt_for_class[img_id]['matched'][best_idx] = True
                    else:
                        # multiple detections for same gt -> false positive (duplicate)
                        fp[d_idx] = 1.0
                else:
                    fp[d_idx] = 1.0

            # compute precision-recall
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            if nd == 0:
                precision = np.array([])
                recall = np.array([])
                ap = 0.0
            else:
                precisions = cum_tp / (cum_tp + cum_fp + 1e-9)
                recalls = cum_tp / (gt_count_per_class[c] + 1e-9)
                # ensure monotonicity and compute AP
                ap = _compute_ap_from_pr(recalls, precisions) if gt_count_per_class[c] > 0 else 0.0

            per_class_ap_matrix[iou_idx, c] = ap
            aps.append(ap)

        # average AP across classes (classes with zero GTs are considered with AP=0)
        per_iou_ap[iou_thr] = float(np.mean(per_class_ap_matrix[iou_idx, :]))
    # mean across IoU thresholds and classes
    per_class_ap = per_class_ap_matrix.mean(axis=0)  # average across IoU thresholds
    mAP = float(per_class_ap.mean())  # average across classes

    return {'map': mAP, 'per_class_ap': per_class_ap, 'per_iou_ap': per_iou_ap, 'per_class_ap_matrix': per_class_ap_matrix}


def precision_recall_per_class_with_f1_thresh(predictions: List[Dict],
                                              ground_truths: List[Dict],
                                              num_classes: int,
                                              iou_thr: float = 0.5,
                                              n_conf_thresholds: int = 100) -> Dict:
    """
    For each class:
      - find confidence threshold that maximizes F1 (scan thresholds)
      - compute Precision and Recall at that threshold
    Returns dict with keys:
      - 'precision': ndarray (num_classes,)
      - 'recall': ndarray (num_classes,)
      - 'best_threshold': ndarray (num_classes,)
      - 'per_class_pr_curve': list of dicts with arrays (optional for inspection)
    """
    detections_by_class = _gather_all_detections(predictions, num_classes)
    gts_by_image, gt_count_per_class = _gather_all_gts(ground_truths, num_classes)

    precisions = np.zeros(num_classes, dtype=float)
    recalls = np.zeros(num_classes, dtype=float)
    best_thresholds = np.zeros(num_classes, dtype=float)
    pr_curves = [None] * num_classes

    for c in range(num_classes):
        dets = detections_by_class[c]
        if len(dets) == 0:
            precisions[c] = 0.0
            recalls[c] = 0.0
            best_thresholds[c] = 0.0
            pr_curves[c] = {'conf': np.array([]), 'precision': np.array([]), 'recall': np.array([])}
            continue

        scores = np.array([d['score'] for d in dets])
        thresholds = np.linspace(scores.min() if scores.size else 0.0, scores.max() if scores.size else 1.0, n_conf_thresholds)
        best_f1 = -1.0
        best_p = 0.0
        best_r = 0.0
        best_t = 0.0

        # pre-build mapping of gts
        gt_for_class_copy = {}
        for img_id, per_class in gts_by_image.items():
            boxes = per_class.get(c, [])
            gt_for_class_copy[img_id] = {'boxes': np.array(boxes), 'matched': np.zeros(len(boxes), dtype=bool)}

        # For efficiency, compute once IoUs per detection against GTs in image when needed.
        # But for simplicity, we re-evaluate per threshold (det count usually manageable)

        p_list = []
        r_list = []
        t_list = []

        for t in thresholds:
            # filter detections by score >= t
            sel = [d for d in dets if d['score'] >= t]
            nd = len(sel)
            tp = 0
            fp = 0

            # fresh copy of matched flags
            gt_for_class = {}
            for k, v in gt_for_class_copy.items():
                gt_for_class[k] = {'boxes': v['boxes'], 'matched': np.zeros_like(v['matched'])}

            for d in sel:
                img_id = d['image_id']
                box_det = d['box']
                gts = gt_for_class.get(img_id, {'boxes': np.zeros((0, 4)), 'matched': np.zeros(0, dtype=bool)})
                gt_boxes = gts['boxes']
                if gt_boxes.shape[0] == 0:
                    fp += 1
                    continue
                # compute IoUs
                ix1 = np.maximum(gt_boxes[:, 0], box_det[0])
                iy1 = np.maximum(gt_boxes[:, 1], box_det[1])
                ix2 = np.minimum(gt_boxes[:, 2], box_det[2])
                iy2 = np.minimum(gt_boxes[:, 3], box_det[3])
                iw = np.maximum(ix2 - ix1, 0.0)
                ih = np.maximum(iy2 - iy1, 0.0)
                inter = iw * ih
                area_det = max((box_det[2] - box_det[0]) * (box_det[3] - box_det[1]), 0.0)
                area_gt = np.maximum((gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1]), 0.0)
                union = area_det + area_gt - inter
                ious = inter / (union + 1e-7)
                best_idx = -1 if ious.size == 0 else int(np.argmax(ious))
                best_iou = 0.0 if ious.size == 0 else float(ious[best_idx])
                if best_iou >= iou_thr and not gts['matched'][best_idx]:
                    tp += 1
                    gts['matched'][best_idx] = True
                    gt_for_class[img_id] = gts
                else:
                    fp += 1

            fn = max(gt_count_per_class[c] - tp, 0)
            prec = tp / (tp + fp + 1e-9) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn + 1e-9) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec + 1e-9) if (prec + rec) > 0 else 0.0

            p_list.append(prec)
            r_list.append(rec)
            t_list.append(t)

            if f1 > best_f1:
                best_f1 = f1
                best_p = prec
                best_r = rec
                best_t = t

        precisions[c] = best_p
        recalls[c] = best_r
        best_thresholds[c] = best_t
        pr_curves[c] = {'conf': np.array(t_list), 'precision': np.array(p_list), 'recall': np.array(r_list)}

    return {'precision': precisions, 'recall': recalls, 'best_threshold': best_thresholds, 'pr_curves': pr_curves}



if __name__ == "__main__":
    # quick small sanity check with dummy data
    # two images, two classes (0,1)
    preds = [
        {
            'boxes': np.array([[10, 10, 20, 20], [30, 30, 50, 50]]),
            'scores': np.array([0.9, 0.6]),
            'labels': np.array([0, 1])
        },
        {
            'boxes': np.array([[12, 12, 22, 22]]),
            'scores': np.array([0.8]),
            'labels': np.array([0])
        }
    ]
    gts = [
        {
            'boxes': np.array([[11, 11, 21, 21]]),
            'labels': np.array([0])
        },
        {
            'boxes': np.array([[30, 30, 50, 50]]),
            'labels': np.array([1])
        }
    ]

    print("Computing mAP...")
    res = compute_map(preds, gts, num_classes=2)
    print("mAP:", res['map'])
    print("per_class_ap:", res['per_class_ap'])
    print("per_iou_ap:", res['per_iou_ap'])

    pr = precision_recall_per_class_with_f1_thresh(preds, gts, num_classes=2, iou_thr=0.5)
    print("Precision per class:", pr['precision'])
    print("Recall per class:", pr['recall'])
    print("Best thresholds per class:", pr['best_threshold'])

    # simple loss example
    import utils
    device = utils.get_device()
    pred_boxes = torch.tensor([[10.0,10.0,20.0,20.0]], device=device)
    gt_boxes = torch.tensor([[11.0,11.0,21.0,21.0]], device=device)
    pred_obj_logits = torch.tensor([0.5], device=device)  # logits
    obj_targets = torch.tensor([1.0], device=device)
    pred_cls_logits = torch.tensor([[2.0, 0.5, -1.0]+[0]*12], device=device)[:, :15]  # shape (1,15) (example)
    pred_cls_logits = torch.randn((1,15), device=device)
    cls_targets = torch.tensor([0], device=device)

    losses = total_loss(pred_boxes, gt_boxes, pred_obj_logits, obj_targets, pred_cls_logits, cls_targets)
    print("Losses:", {k: float(v) for k,v in losses.items()})

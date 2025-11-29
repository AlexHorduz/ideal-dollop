import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2

class RCNN(nn.Module):
    def __init__(self, num_classes=15, pretrained=True):
        super(RCNN, self).__init__()
        self.num_classes = num_classes

        alexnet = models.alexnet(pretrained=pretrained)

        self.features = alexnet.features
        self.avgpool = alexnet.avgpool

        self.fc_features = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Linear(4096, num_classes + 1)
        self.bbox_regressor = nn.Linear(4096, 4 * num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        features = self.fc_features(x)
        class_scores = self.classifier(features)
        bbox_deltas = self.bbox_regressor(features)

        return features, class_scores, bbox_deltas

    def extract_features(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        features = self.fc_features(x)
        return features


class SelectiveSearch:
    def __init__(self):
        self.ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    def get_proposals(self, image, max_proposals=2000):
        self.ss.setBaseImage(image)
        self.ss.switchToSelectiveSearchFast()
        proposals = self.ss.process()
        proposals = proposals[:max_proposals]
        return proposals


class RCNNDetector:
    def __init__(self, model, num_classes=15, device='cpu'):
        self.model = model
        self.num_classes = num_classes
        self.device = device
        self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        self.selective_search = SelectiveSearch()

    def preprocess_region(self, image, bbox):
        x, y, w, h = bbox
        region = image[y:y+h, x:x+w]
        region = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        region_tensor = self.transform(region)
        return region_tensor.unsqueeze(0)

    def detect(self, image, score_threshold=0.5, nms_threshold=0.3):
        proposals = self.selective_search.get_proposals(image)
        all_detections = []

        for bbox in proposals:
            _, _, w, h = bbox

            if w < 10 or h < 10:
                continue

            region_tensor = self.preprocess_region(image, bbox).to(self.device)

            with torch.no_grad():
                _, class_scores, _ = self.model(region_tensor)

            probs = torch.softmax(class_scores, dim=1)[0]
            max_prob, max_class = torch.max(probs[1:], dim=0)

            if max_prob.item() > score_threshold:
                all_detections.append({
                    'bbox': bbox,
                    'class_id': max_class.item(),
                    'score': max_prob.item()
                })

        detections = self.non_max_suppression(all_detections, nms_threshold)
        return detections

    def non_max_suppression(self, detections, iou_threshold=0.3):
        if len(detections) == 0:
            return []

        detections = sorted(detections, key=lambda x: x['score'], reverse=True)

        keep = []
        while len(detections) > 0:
            best = detections.pop(0)
            keep.append(best)
            detections = [d for d in detections
                         if self.compute_iou(best['bbox'], d['bbox']) < iou_threshold]

        return keep

    @staticmethod
    def compute_iou(bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area if union_area > 0 else 0
        return iou


if __name__ == "__main__":
    import time
    import utils

    device = utils.get_device()
    print(f"Using device: {device}")

    model = RCNN(num_classes=15, pretrained=False)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")

    batch_size = 4
    x = torch.randn(batch_size, 3, 227, 227).to(device)

    print("Testing forward pass...")
    start = time.time()
    features, class_scores, bbox_deltas = model(x)
    end = time.time()

    print(f"Time taken: {round(end - start, 3)} seconds")
    print(f"Features shape: {features.shape}")
    print(f"Class scores shape: {class_scores.shape}")
    print(f"Bbox deltas shape: {bbox_deltas.shape}")

    print("Testing feature extraction...")
    extracted_features = model.extract_features(x)
    print(f"Extracted features shape: {extracted_features.shape}")

    print("R-CNN model created successfully!")

import torch
import torch.nn as nn

# AlexNet Object Detection
class AlexNetOD(nn.Module):
    def __init__(self, grid_size=7, B=2, num_classes=15):
        super(AlexNetOD, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # YOLO-style fully connected head
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*6*6, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, grid_size*grid_size*(B*5 + num_classes))
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8

    model = AlexNetOD(num_classes=15).to(device)

    num_of_params = sum([p.numel() for p in model.parameters()])
    print(f"Number of params: {num_of_params}")

    x = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # Warmup call (around 4 sec)
    start = time.time()
    out = model(x)
    end = time.time()
    print(f"Time taken for prediction: {round(end - start, 3)} seconds")

    # All the consecutive calls take ~0.002 sec
    start = time.time()
    out = model(x)
    end = time.time()
    print(f"Time taken for prediction: {round(end - start, 3)} seconds")
    print(out.shape)
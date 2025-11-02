import torch

device = None
# Check if CUDA is available
if torch.cuda.is_available():
    # Use CUDA if available
    device = torch.device("cuda")
# Check if MPS is available
elif torch.backends.mps.is_available():
    # Use MPS if available
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# print(f"Using device: {device}")

def get_device():
    return device


from pathlib import Path

def get_kaggle_dataset_root() -> Path:
    import kagglehub
    downloaded = kagglehub.dataset_download("pkdarabi/cardetection")
    return Path(downloaded) / "car"
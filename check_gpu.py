
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)

    x = torch.rand(1000, 1000, device="cuda")
    y = torch.rand(1000, 1000, device="cuda")

    result = x @ y

    print("GPU computation successful!")
    print("Result device:", result.device)
else:
    print("CUDA GPU not detected.")


from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import DeepfakeDataset


# Dataset path
DATASET_ROOT = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "processed"
    / "real-vs-fake"
    / "Real vs Fake Faces"
    / "8020"
)

# Create datasets
train_dataset = DeepfakeDataset(DATASET_ROOT, split="train")
valid_dataset = DeepfakeDataset(DATASET_ROOT, split="valid")
test_dataset = DeepfakeDataset(DATASET_ROOT, split="test")

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

# Check dataset sizes
print("Dataset sizes:")
print("Train:", len(train_dataset))
print("Validation:", len(valid_dataset))
print("Test:", len(test_dataset))

# Inspect one training sample
image, label = train_dataset[0]

print("\nSingle sample:")
print("Image shape:", image.shape)
print("Label:", label)
print("Image dtype:", image.dtype)

# Inspect one batch
images, labels = next(iter(train_loader))

print("\nFirst training batch:")
print("Batch image shape:", images.shape)
print("Batch labels shape:", labels.shape)
print("Labels:", labels.tolist())

# Verify expected dimensions
assert image.shape == (3, 224, 224)
assert images.shape == (16, 3, 224, 224)
assert len(train_dataset) == 8000
assert len(valid_dataset) == 1000
assert len(test_dataset) == 1000

print("\nAll dataset tests passed!")

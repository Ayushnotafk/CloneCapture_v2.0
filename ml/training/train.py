
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Project root ko Python path mein add karo
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.custom_cnn import CustomCNN
from ml.preprocessing.dataset import DeepfakeDataset


# ---------------- CONFIGURATION ----------------

SEED = 42
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001

DATASET_ROOT = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "processed"
    / "real-vs-fake"
    / "Real vs Fake Faces"
    / "8020"
)

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "custom_cnn_best.pth"


# ---------------- REPRODUCIBILITY ----------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------- DEVICE ----------------

def get_device():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    return device


# ---------------- DATA LOADERS ----------------

def create_data_loaders():

    train_dataset = DeepfakeDataset(
        DATASET_ROOT, split="train"
    )

    valid_dataset = DeepfakeDataset(
        DATASET_ROOT, split="valid"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, valid_loader


# ---------------- METRICS ----------------

def calculate_metrics(all_labels, all_predictions):

    return {
        "accuracy": accuracy_score(
            all_labels, all_predictions
        ),
        "precision": precision_score(
            all_labels, all_predictions,
            zero_division=0
        ),
        "recall": recall_score(
            all_labels, all_predictions,
            zero_division=0
        ),
        "f1": f1_score(
            all_labels, all_predictions,
            zero_division=0
        )
    }


# ---------------- TRAINING ----------------

def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()

    running_loss = 0.0
    all_labels = []
    all_predictions = []

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = calculate_metrics(all_labels, all_predictions)

    return epoch_loss, metrics


# ---------------- VALIDATION ----------------

def validate(model, loader, criterion, device):

    model.eval()

    running_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = calculate_metrics(all_labels, all_predictions)

    return epoch_loss, metrics


# ---------------- MAIN ----------------

def main():

    set_seed(SEED)

    device = get_device()

    train_loader, valid_loader = create_data_loaders()

    print("\nTraining images:", len(train_loader.dataset))
    print("Validation images:", len(valid_loader.dataset))

    model = CustomCNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_f1 = -1.0

    print("\nStarting training...\n")

    for epoch in range(EPOCHS):

        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        valid_loss, valid_metrics = validate(
            model,
            valid_loader,
            criterion,
            device
        )

        print(f"Epoch [{epoch + 1}/{EPOCHS}]")

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_metrics['accuracy']:.4f}"
        )

        print(
            f"Validation Loss: {valid_loss:.4f} | "
            f"Accuracy: {valid_metrics['accuracy']:.4f} | "
            f"Precision: {valid_metrics['precision']:.4f} | "
            f"Recall: {valid_metrics['recall']:.4f} | "
            f"F1: {valid_metrics['f1']:.4f}"
        )

        if valid_metrics["f1"] > best_f1:

            best_f1 = valid_metrics["f1"]

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_f1": best_f1,
                    "seed": SEED
                },
                BEST_MODEL_PATH
            )

            print("Best model checkpoint saved!")

        print("-" * 60)

    print("\nTraining completed!")
    print("Best validation F1:", round(best_f1, 4))
    print("Checkpoint:", BEST_MODEL_PATH)


if __name__ == "__main__":
    main()

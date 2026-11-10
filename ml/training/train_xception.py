
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.xception_model import XceptionModel
from ml.preprocessing.dataset import DeepfakeDataset


# Configuration
BATCH_SIZE = 8
EPOCHS = 15
LEARNING_RATE = 0.0001
SEED = 42

DATASET_ROOT = (
    PROJECT_ROOT / "ml" / "datasets" / "processed"
    / "real-vs-fake" / "Real vs Fake Faces" / "8020"
)

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

BEST_CHECKPOINT = CHECKPOINT_DIR / "xception_best.pth"
LATEST_CHECKPOINT = CHECKPOINT_DIR / "xception_latest.pth"


def main():

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print("Device:", device)
    print("Mixed precision:", use_amp)

    train_dataset = DeepfakeDataset(DATASET_ROOT, split="train")
    valid_dataset = DeepfakeDataset(DATASET_ROOT, split="valid")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=use_amp
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=use_amp
    )

    model = XceptionModel(pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scaler = GradScaler("cuda", enabled=use_amp)

    best_f1 = 0.0

    for epoch in range(EPOCHS):

        # Training
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_labels = []

        for images, labels in train_loader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            train_predictions.extend(predictions.detach().cpu().tolist())
            train_labels.extend(labels.cpu().tolist())

        train_loss /= len(train_dataset)
        train_accuracy = accuracy_score(train_labels, train_predictions)

        # Validation
        model.eval()
        valid_loss = 0.0
        valid_predictions = []
        valid_labels = []

        with torch.no_grad():

            for images, labels in valid_loader:

                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                valid_loss += loss.item() * images.size(0)

                predictions = outputs.argmax(dim=1)
                valid_predictions.extend(predictions.cpu().tolist())
                valid_labels.extend(labels.cpu().tolist())

        valid_loss /= len(valid_dataset)

        valid_accuracy = accuracy_score(valid_labels, valid_predictions)
        valid_precision = precision_score(
            valid_labels, valid_predictions, zero_division=0
        )
        valid_recall = recall_score(
            valid_labels, valid_predictions, zero_division=0
        )
        valid_f1 = f1_score(
            valid_labels, valid_predictions, zero_division=0
        )

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f}")
        print(
            f"Val Loss: {valid_loss:.4f} | Val Acc: {valid_accuracy:.4f} "
            f"| Precision: {valid_precision:.4f} "
            f"| Recall: {valid_recall:.4f} | F1: {valid_f1:.4f}"
        )

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_f1": valid_f1,
            "seed": SEED,
        }

        torch.save(checkpoint, LATEST_CHECKPOINT)

        if valid_f1 > best_f1:
            best_f1 = valid_f1
            torch.save(checkpoint, BEST_CHECKPOINT)
            print("Best Xception checkpoint saved!")

        print("GPU memory allocated:", round(
            torch.cuda.memory_allocated() / 1024**3, 2
        ) if use_amp else "N/A", "GB")

    print("\nXception training completed.")
    print("Best validation F1:", round(best_f1, 4))
    print("Best checkpoint:", BEST_CHECKPOINT)


if __name__ == "__main__":
    main()

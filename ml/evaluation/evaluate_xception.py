
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.xception_model import XceptionModel
from ml.preprocessing.dataset import DeepfakeDataset


# Configuration
BATCH_SIZE = 8

DATASET_ROOT = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "processed"
    / "real-vs-fake"
    / "Real vs Fake Faces"
    / "8020"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "xception_best.pth"
)


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    use_amp = device.type == "cuda"

    print("Evaluation device:", device)

    # Load test dataset
    test_dataset = DeepfakeDataset(
        DATASET_ROOT,
        split="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=use_amp
    )

    print("Test images:", len(test_dataset))

    # Load model and best checkpoint
    model = XceptionModel(
        pretrained=False
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print("Loaded checkpoint from epoch:", checkpoint["epoch"])
    print("Checkpoint validation F1:", checkpoint["validation_f1"])

    model.eval()

    criterion = nn.CrossEntropyLoss()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    total_loss = 0.0

    # Test evaluation
    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp
            ):

                outputs = model(images)
                loss = criterion(outputs, labels)

            probabilities = torch.softmax(
                outputs.float(),
                dim=1
            )

            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

            all_probabilities.extend(
                probabilities[:, 1].cpu().tolist()
            )

    # Calculate metrics
    test_loss = total_loss / len(test_dataset)

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        all_labels,
        all_probabilities
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1]
    )

    # Display results
    print("\n========== XCEPTION TEST RESULTS ==========")

    print(f"Test Loss:  {test_loss:.4f}")
    print(f"Accuracy:   {accuracy:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1 Score:   {f1:.4f}")
    print(f"ROC-AUC:    {auc:.4f}")

    print("\nConfusion Matrix:")
    print("Rows = Actual, Columns = Predicted")
    print("Labels: 0 = Real, 1 = Fake")
    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=["Real", "Fake"],
            zero_division=0
        )
    )

    print("============================================")


if __name__ == "__main__":
    main()


import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.custom_cnn import CustomCNN
from ml.models.xception_model import XceptionModel
from ml.preprocessing.dataset import DeepfakeDataset


# Configuration
BATCH_SIZE = 16

DATASET_ROOT = (
    PROJECT_ROOT / "ml" / "datasets" / "processed"
    / "real-vs-fake" / "Real vs Fake Faces" / "8020"
)

CHECKPOINTS = {
    "Custom CNN": PROJECT_ROOT / "checkpoints" / "custom_cnn_best.pth",
    "Xception": PROJECT_ROOT / "checkpoints" / "xception_best.pth"
}


def evaluate_model(model, loader, device):

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            probabilities = torch.softmax(outputs.float(), dim=1)
            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_probabilities.extend(
                probabilities[:, 1].cpu().tolist()
            )

    return {
        "accuracy": accuracy_score(all_labels, all_predictions),
        "precision": precision_score(
            all_labels, all_predictions, zero_division=0
        ),
        "recall": recall_score(
            all_labels, all_predictions, zero_division=0
        ),
        "f1": f1_score(
            all_labels, all_predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(
            all_labels, all_probabilities
        )
    }


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Evaluation device:", device)

    test_dataset = DeepfakeDataset(
        DATASET_ROOT,
        split="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print("Test images:", len(test_dataset))

    results = {}

    for model_name, checkpoint_path in CHECKPOINTS.items():

        print(f"\nEvaluating {model_name}...")

        if model_name == "Custom CNN":
            model = CustomCNN().to(device)
        else:
            model = XceptionModel(pretrained=False).to(device)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        results[model_name] = evaluate_model(
            model,
            test_loader,
            device
        )

        print("Checkpoint epoch:", checkpoint["epoch"])

    # Display comparison
    print("\n========== MODEL COMPARISON ==========")

    print(
        f"{'Metric':<15}"
        f"{'Custom CNN':>15}"
        f"{'Xception':>15}"
    )

    metrics = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1 Score", "f1"),
        ("ROC-AUC", "roc_auc")
    ]

    for metric_name, metric_key in metrics:

        cnn_score = results["Custom CNN"][metric_key]
        xception_score = results["Xception"][metric_key]

        print(
            f"{metric_name:<15}"
            f"{cnn_score:>15.4f}"
            f"{xception_score:>15.4f}"
        )

    print("======================================")

    # Save results
    output_path = PROJECT_ROOT / "docs" / "model_comparison.txt"

    with open(output_path, "w", encoding="utf-8") as file:

        file.write("CUSTOM CNN VS XCEPTION\n")
        file.write("======================\n\n")

        file.write("Dataset: Real vs Fake Faces\n")
        file.write("Test images: 1000\n\n")

        for metric_name, metric_key in metrics:

            file.write(
                f"{metric_name}: "
                f"Custom CNN = {results['Custom CNN'][metric_key]:.4f}, "
                f"Xception = {results['Xception'][metric_key]:.4f}\n"
            )

    print("\nComparison saved to:", output_path)


if __name__ == "__main__":
    main()

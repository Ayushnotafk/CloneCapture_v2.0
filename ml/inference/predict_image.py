import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from ml.models.xception_model import XceptionModel

def predict_image(image_path, checkpoint_path):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load trained model
    model = XceptionModel(
        num_classes=2,
        pretrained=False
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Same preprocessing used during evaluation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    class_names = ["REAL", "FAKE"]

    predicted_class = torch.argmax(probabilities).item()
    predicted_label = class_names[predicted_class]
    score = probabilities[predicted_class].item() * 100

    print("\n========== CLONE CATCHER ==========")
    print("Image:", image_path)
    print("Device:", device)
    print("----------------------------------")
    print("Prediction:", predicted_label)
    print(f"Model score: {score:.2f}%")
    print(f"Real score: {probabilities[0].item() * 100:.2f}%")
    print(f"Fake score: {probabilities[1].item() * 100:.2f}%")
    print("==================================\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    checkpoint_path = "checkpoints/xception_best.pth"

    if not Path(args.image).is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")

    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    predict_image(args.image, checkpoint_path)

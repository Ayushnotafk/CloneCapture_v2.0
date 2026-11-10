import sys
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Annotated

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.xception_model import XceptionModel


CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "xception_best.pth"
ALLOWED_FORMATS = {"JPEG", "PNG"}
MAX_UPLOAD_SIZE = 10 * 1024 * 1024
CLASS_NAMES = ("REAL", "FAKE")

preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def load_model() -> tuple[torch.nn.Module, torch.device]:
    """Build the existing model and restore its trained checkpoint once."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    try:
        model = XceptionModel(num_classes=2, pretrained=False).to(device)
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
    except (KeyError, RuntimeError, OSError) as exc:
        raise RuntimeError(f"Unable to load model checkpoint: {CHECKPOINT_PATH}") from exc

    return model, device


def preprocess_image(contents: bytes) -> torch.Tensor:
    """Validate an uploaded image and return the model input tensor."""
    try:
        with Image.open(BytesIO(contents)) as image:
            image_format = image.format
            image.verify()
        if image_format not in ALLOWED_FORMATS:
            raise HTTPException(
                status_code=415,
                detail="Unsupported file format. Please upload a JPG, JPEG, or PNG image.",
            )
        with Image.open(BytesIO(contents)) as image:
            return preprocess(image.convert("RGB")).unsqueeze(0)
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded image is corrupted.") from exc


def predict_tensor(model: torch.nn.Module, device: torch.device, image_tensor: torch.Tensor) -> dict:
    """Run inference and map class probabilities to the public response."""
    with torch.inference_mode():
        probabilities = F.softmax(model(image_tensor.to(device)), dim=1)[0]

    real_score = probabilities[0].item() * 100
    fake_score = probabilities[1].item() * 100
    predicted_index = int(torch.argmax(probabilities).item())
    return {
        "prediction": CLASS_NAMES[predicted_index],
        "real_score": round(real_score, 2),
        "fake_score": round(fake_score, 2),
        "model_score": round((real_score, fake_score)[predicted_index], 2),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, device = load_model()
    app.state.model = model
    app.state.device = device
    yield


app = FastAPI(title="Clone Catcher API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    model = getattr(app.state, "model", None)
    device = getattr(app.state, "device", None)
    return {
        "status": "healthy" if model is not None else "starting",
        "model_loaded": model is not None,
        "device": str(device) if device else None,
    }


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File(...)]):
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    image_tensor = preprocess_image(contents)

    model = getattr(app.state, "model", None)
    device = getattr(app.state, "device", None)
    if model is None or device is None:
        raise HTTPException(status_code=503, detail="The model is still loading. Please try again.")
    return predict_tensor(model, device, image_tensor)

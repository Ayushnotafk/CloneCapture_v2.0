# Clone Catcher

An experimental deepfake detection system built from scratch.

## Objectives
- Binary classification of real and manipulated facial media.
- Train a custom CNN from random initialization.
- Benchmark against a pretrained Xception model.
- Support image and short-video analysis.
- Evaluate using reproducible experiments and independent test data.

## Technology Stack
- Python
- PyTorch
- FastAPI
- React + Vite

## Research Principles
- No fabricated metrics.
- No train-test leakage.
- Reproducible experiments.
- Transparent limitations and evaluation.

**Project Status:** Initial setup

## Run the Clone Catcher application

The application uses the existing `checkpoints/xception_best.pth` checkpoint and loads it once
when the FastAPI server starts.

### Backend

From the repository root, install the application dependencies into the project environment:

```powershell
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

The API is available at `http://127.0.0.1:8000`. Check `http://127.0.0.1:8000/health` before
opening the frontend.

### Frontend

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Supported uploads are JPG, JPEG, and PNG images up to 10 MB.
Results are experimental model estimates and are not definitive deepfake verification.
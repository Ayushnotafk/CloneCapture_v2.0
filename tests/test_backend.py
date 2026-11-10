from io import BytesIO

from PIL import Image
from fastapi.testclient import TestClient

from backend.main import app


def image_bytes(image_format: str) -> bytes:
    image = Image.new("RGB", (32, 32), color=(96, 128, 160))
    output = BytesIO()
    image.save(output, format=image_format)
    return output.getvalue()


def test_backend_endpoints():
    with TestClient(app) as client:
        health_response = client.get("/health")
        assert health_response.status_code == 200
        health = health_response.json()
        assert health == {
            "status": "healthy",
            "model_loaded": True,
            "device": health["device"],
        }
        assert health["device"] in {"cpu", "cuda"}

        for image_format, filename, content_type in (
            ("JPEG", "sample.jpg", "image/jpeg"),
            ("PNG", "sample.png", "image/png"),
        ):
            response = client.post(
                "/predict",
                files={"file": (filename, image_bytes(image_format), content_type)},
            )
            assert response.status_code == 200
            result = response.json()
            assert set(result) == {
                "prediction",
                "real_score",
                "fake_score",
                "model_score",
            }
            assert result["prediction"] in {"REAL", "FAKE"}
            assert 0 <= result["real_score"] <= 100
            assert 0 <= result["fake_score"] <= 100
            assert 0 <= result["model_score"] <= 100
            assert abs(result["real_score"] + result["fake_score"] - 100) <= 0.02

        invalid_format = client.post(
            "/predict",
            files={"file": ("sample.gif", image_bytes("GIF"), "image/gif")},
        )
        assert invalid_format.status_code == 415

        corrupted = client.post(
            "/predict",
            files={"file": ("corrupt.jpg", b"not an image", "image/jpeg")},
        )
        assert corrupted.status_code == 400

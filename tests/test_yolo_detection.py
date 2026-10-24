import pytest
import os
import numpy as np
from PIL import Image
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.services.yolo_service import YOLOService
from app.services.document_service import DocumentService


def test_yolo_service_load():
    yolo = YOLOService()
    model = yolo.load_model()
    assert model is not None
    assert yolo.is_loaded


def test_yolo_detection_real_image():
    image_path = os.path.join("my_dataset", "val", "images", "anh1.jpg")
    assert os.path.exists(image_path), f"Sample image {image_path} must exist"

    doc_service = DocumentService()
    with open(image_path, "rb") as f:
        data = f.read()

    res = doc_service.detect_document_from_bytes(data)
    assert res["detected"] is True
    assert res["confidence"] > 0.5
    assert res["bounding_box"] is not None
    assert "width" in res["bounding_box"]
    assert "height" in res["bounding_box"]
    assert len(res["corners"]) == 4
    assert res["model"] == "YOLOv8-Doc"


def test_yolo_detection_non_document_image():
    # Create a non-document image (green background, no document)
    img = Image.new("RGB", (300, 300), color=(34, 139, 34))
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    non_doc_bytes = buf.getvalue()

    doc_service = DocumentService()
    res = doc_service.detect_document_from_bytes(non_doc_bytes)
    assert res["detected"] is False
    assert res["confidence"] == 0.0
    assert res["bounding_box"] is None


@pytest.mark.asyncio
async def test_scan_endpoint_with_yolo_document():
    image_path = os.path.join("my_dataset", "val", "images", "anh1.jpg")
    with open(image_path, "rb") as f:
        data = f.read()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        files = {"file": ("anh1.jpg", data, "image/jpeg")}
        r = await ac.post("/api/v1/scan/", files=files)

    assert r.status_code == 200, r.text
    j = r.json()
    assert j["status"] == "completed"
    assert "document" in j
    assert j["document"]["detected"] is True
    assert j["document"]["confidence"] > 0.5

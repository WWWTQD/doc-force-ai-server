import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from PIL import Image, ImageDraw, ImageFont
import io
import os


def make_sample_document() -> bytes:
    # Create a simple synthetic document: white page with a black rectangle and some text
    w, h = 800, 600
    img = Image.new("RGB", (w, h), color=(200, 200, 200))
    draw = ImageDraw.Draw(img)
    # draw a white rectangle as the document on a gray background
    margin = 60
    doc_box = (margin, margin, w - margin, h - margin)
    draw.rectangle(doc_box, fill=(255, 255, 255))
    # add some text lines
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    y = margin + 20
    for i in range(10):
        draw.text((margin + 20, y), f"Line {i+1}: This is a test document.", fill=(0, 0, 0), font=font)
        y += 30

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_scan_integration():
    data = make_sample_document()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        files = {"file": ("test_document.png", data, "image/png")}
        r = await ac.post("/api/v1/scan/", files=files)

    assert r.status_code == 200, r.text
    j = r.json()
    assert "scan_id" in j
    assert j.get("status") == "completed"
    # basic keys
    assert "quality" in j
    assert "document" in j
    assert "defects" in j
    assert "ocr" in j
    assert "files" in j

    scan_id = j["scan_id"]
    # Check files on disk
    processed_dir = os.path.join("data", "output")
    summary_path = os.path.join(processed_dir, f"{scan_id}.json")

    assert os.path.exists(summary_path)


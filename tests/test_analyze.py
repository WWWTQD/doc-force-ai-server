import pytest
import io
from PIL import Image, ImageDraw
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.services.structure_engine import StructureEngine


def make_sample_image_bytes() -> bytes:
    w, h = 800, 600
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), "TEST BÁO CÁO PBL4", fill=(0, 0, 0))
    draw.text((50, 100), "Noi dung thu nghiem AI service PP-StructureV3.", fill=(0, 0, 0))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def init_ai_engine():
    engine = StructureEngine.get_instance()
    if not engine.initialized:
        engine.initialize(
            device="cpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            use_seal_recognition=False,
            use_table_recognition=True,
            use_formula_recognition=False,
        )


@pytest.mark.asyncio
async def test_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["service"] == "pbl4-ai-service"
    assert "PP-StructureV3" in data["model"]


@pytest.mark.asyncio
async def test_analyze_endpoint_success():
    img_bytes = make_sample_image_bytes()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        files = {"file": ("test_doc.png", img_bytes, "image/png")}
        res = await ac.post("/analyze", files=files)

    assert res.status_code == 200, res.text
    data = res.json()
    assert data["success"] is True
    assert "file_id" in data
    assert data["filename"] == "test_doc.png"
    assert "pages" in data
    assert "document" in data
    assert "processing_time_ms" in data


@pytest.mark.asyncio
async def test_analyze_unsupported_format():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        files = {"file": ("test.exe", b"invalid_binary", "application/octet-stream")}
        res = await ac.post("/analyze", files=files)

    assert res.status_code == 415
    data = res.json()
    assert data["success"] is False
    assert data["error"]["code"] == "UNSUPPORTED_FORMAT"

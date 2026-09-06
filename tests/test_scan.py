import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
import io


@pytest.mark.asyncio
async def test_upload_invalid_file():
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test"
    ) as ac:
        files = {
            "file": (
                "test.txt",
                io.BytesIO(b"not an image"),
                "text/plain"
            )
        }

        r = await ac.post("/api/v1/scan/", files=files)

    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "UNSUPPORTED_FORMAT"
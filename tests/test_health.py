import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app


@pytest.mark.asyncio
async def test_health():
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test"
    ) as ac:
        r = await ac.get("/api/v1/health/")

    assert r.status_code == 200
    j = r.json()
    assert j.get("status") == "ok"
    assert "service" in j
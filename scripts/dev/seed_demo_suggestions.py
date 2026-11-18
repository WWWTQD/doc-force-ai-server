"""Deterministic demo seed script for the LLM_POSTPROCESS mock review flow (Commit 10).

LƯU Ý QUAN TRỌNG:
  Tạm dừng `real_worker.py` (hoặc container `ai-worker`) trước khi chạy script này
  với Backend thật để tránh worker nền claim mất job vừa được tạo.

Ví dụ:
  # Kiểm tra ngoại tuyến không cần Backend:
  python scripts/dev/seed_demo_suggestions.py --dry-run
  python scripts/dev/seed_demo_suggestions.py --dry-run --adversarial

  # Tạo tài liệu demo trên Backend local:
  python scripts/dev/seed_demo_suggestions.py --backend-url http://localhost:8080
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from worker.llm_client import MockLLMClient
from worker.llm_postprocessor import PostprocessConfig, process_page

FIXTURE_PATH = PROJECT_ROOT / "worker" / "mock_fixtures" / "golden_cases.json"

FONT_CANDIDATES = (
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)


def _load_golden_cases() -> dict[str, Any]:
    with FIXTURE_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_font(size: int) -> tuple[Any, str]:
    from PIL import ImageFont

    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size), path
    print(
        "WARNING: Không tìm thấy font chữ Việt TTF hệ thống; dùng ImageFont.load_default() (dấu tiếng Việt có thể hiển thị hạn chế).",
        file=sys.stderr,
    )
    return ImageFont.load_default(), "PIL.ImageFont.load_default()"


def build_demo_entries(fixtures: dict[str, Any]) -> list[dict[str, Any]]:
    """Build synthetic OCR blocks alongside clean Ground-Truth text for image rendering."""
    text_entries = fixtures.get("text_llm", {})
    by_case = {v.get("caseId"): v for v in text_entries.values() if isinstance(v, dict)}

    rows: list[dict[str, Any]] = [
        {
            "caseId": "ANCHOR_HEADER",
            "groundTruth": "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM",
            "rawOcr": "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM",
            "confidence": 0.98,
            "type": "heading",
        },
        {
            "caseId": "ANCHOR_1",
            "groundTruth": "Ủy ban nhân dân thành phố quyết định kiểm tra thực tế tại hiện trường.",
            "rawOcr": "Ủy ban nhân dân thành phố quyết định kiểm tra thực tế tại hiện trường.",
            "confidence": 0.97,
            "type": "paragraph",
        },
        {
            "caseId": "ANCHOR_2",
            "groundTruth": "Căn cứ quy định của pháp luật hiện hành về quản lý và sử dụng tài sản công.",
            "rawOcr": "Căn cứ quy định của pháp luật hiện hành về quản lý và sử dụng tài sản công.",
            "confidence": 0.97,
            "type": "paragraph",
        },
    ]

    conf_map = {"GC1": 0.72, "GC2": 0.78, "GC3": 0.74, "GC4": 0.75, "GC5": 0.76}
    for cid in ("GC1", "GC2", "GC3", "GC4", "GC5"):
        item = by_case.get(cid)
        if not item:
            continue
        rows.append(
            {
                "caseId": cid,
                "groundTruth": item["groundTruth"],
                "rawOcr": item["rawOcr"],
                "confidence": conf_map[cid],
                "type": "paragraph",
            }
        )
    return rows


def render_demo_page_jpeg(entries: list[dict[str, Any]]) -> tuple[bytes, dict[str, Any], str]:
    """Render a crisp 1200x1600 JPEG showing groundTruth text paired with rawOcr layout blocks."""
    from PIL import Image, ImageDraw

    width, height = 1200, 1600
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    title_font, font_name = _load_font(28)
    body_font, _ = _load_font(24)

    blocks: list[dict[str, Any]] = []
    y = 90
    line_step = 95

    for idx, entry in enumerate(entries, start=1):
        font = title_font if entry["type"] == "heading" else body_font
        gt_text = entry["groundTruth"]
        raw_text = entry["rawOcr"]

        draw.text((80, y + 8), gt_text, fill=(20, 22, 28), font=font)
        draw.line([(76, y + 48), (1124, y + 48)], fill=(235, 236, 240), width=1)

        blocks.append(
            {
                "id": f"demo-{idx}-{entry['caseId'].lower()}",
                "type": entry["type"],
                "order": idx,
                "bbox": [76, y, 1124, y + 48],
                "ocrText": raw_text,
                "text": raw_text,
                "confidence": entry["confidence"],
            }
        )
        y += line_step

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    raw_layout = {
        "schemaVersion": 1,
        "pageNumber": 1,
        "width": width,
        "height": height,
        "blocks": blocks,
    }
    return buf.getvalue(), raw_layout, font_name


def _check_gemini_prereqs() -> bool:
    consent = os.getenv("LLM_CLOUD_CONSENT", "").strip().lower() in ("1", "true", "yes", "on")
    has_key = bool(os.getenv("GEMINI_API_KEY", "").strip())
    if consent and has_key:
        return True

    print("Không thể chạy --provider gemini vì thiếu điều kiện tiên quyết:")
    if not consent:
        print("  - LLM_CLOUD_CONSENT chưa được bật ('true')")
    if not has_key:
        print("  - GEMINI_API_KEY đang trống hoặc chưa được thiết lập")
    print()
    print(
        "Cảnh báo quyền riêng tư: Google AI Studio Free Tier có thể sử dụng văn bản/ảnh gửi lên "
        "để cải thiện sản phẩm; dùng dữ liệu demo hoặc nâng cấp Paid Tier."
    )
    print("Thiết lập trên PowerShell trước khi chạy lại:")
    print('  $env:LLM_CLOUD_CONSENT="true"')
    print('  $env:LLM_CLOUD_TIER="free"')
    print('  $env:GEMINI_API_KEY="<your-gemini-api-key>"')
    return False


def _pillow_crop_provider(jpeg_bytes: bytes):
    from PIL import Image

    page_img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    page_w, page_h = page_img.size

    def _crop(block: dict[str, Any]) -> bytes | None:
        bbox = block.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            return None
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        x1 = max(0, min(page_w - 1, x1))
        y1 = max(0, min(page_h - 1, y1))
        x2 = max(x1 + 1, min(page_w, x2))
        y2 = max(y1 + 1, min(page_h, y2))
        cropped = page_img.crop((x1, y1, x2, y2))
        buf = io.BytesIO()
        cropped.save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    return _crop


def run_gemini_postprocess(raw_layout: dict[str, Any], *, jpeg_bytes: bytes) -> dict[str, Any]:
    base_cfg = PostprocessConfig.from_env()
    cfg = PostprocessConfig(
        enabled=True,
        provider="gemini",
        mock_mode="normal",
        text_model=base_cfg.text_model,
        vision_model=base_cfg.vision_model,
        fallback_model=base_cfg.fallback_model,
        fallback_on_unavailable=base_cfg.fallback_on_unavailable,
        thinking_level=base_cfg.thinking_level,
        cloud_consent=True,
        cloud_tier=base_cfg.cloud_tier,
        rpm_limit=base_cfg.rpm_limit,
        daily_request_budget=base_cfg.daily_request_budget,
        request_timeout_seconds=base_cfg.request_timeout_seconds,
        line_conf_threshold=0.90,
        max_candidates_per_page=30,
        max_invalid_token_ratio=0.5,
        reocr_enabled=False,
        max_reocr_lines_per_page=15,
        allow_common_loanwords=True,
        page_deadline_seconds=max(base_cfg.page_deadline_seconds, 75.0),
        max_vlm_crops=5,
    )
    return process_page(
        raw_layout,
        options={"llmPostprocess": True, "localOnly": False, "redactPii": False},
        crop_provider=_pillow_crop_provider(jpeg_bytes),
        config=cfg,
    )


def run_mock_postprocess(raw_layout: dict[str, Any], *, adversarial: bool) -> dict[str, Any]:
    mode = "adversarial" if adversarial else "normal"
    cfg = PostprocessConfig(
        enabled=True,
        provider="mock",
        mock_mode=mode,
        line_conf_threshold=0.90,
        max_candidates_per_page=30,
        max_invalid_token_ratio=0.5,
        reocr_enabled=False,
        max_reocr_lines_per_page=15,
        allow_common_loanwords=True,
        page_deadline_seconds=15.0,
        max_vlm_crops=5,
    )
    client = MockLLMClient(mode=mode)
    return process_page(
        raw_layout,
        options={"llmPostprocess": True, "localOnly": True, "redactPii": False},
        config=cfg,
        llm_client=client,
    )


def seed_backend(
    *,
    backend_url: str,
    frontend_url: str,
    worker_key: str,
    email: str,
    password: str,
    title: str,
    jpeg_bytes: bytes,
    processed_layout: dict[str, Any],
    local_only: bool = True,
    pipeline_version: str = "mock-demo-seed-v1",
) -> str:
    import requests

    base = backend_url.rstrip("/")

    # 1. Register (ignore 409 Conflict if demo user already exists) & Login
    reg_resp = requests.post(
        f"{base}/api/auth/register",
        json={"email": email, "password": password},
        timeout=15,
    )
    if reg_resp.status_code not in (200, 201, 409):
        reg_resp.raise_for_status()

    login_resp = requests.post(
        f"{base}/api/auth/login",
        json={"email": email, "password": password},
        timeout=15,
    )
    login_resp.raise_for_status()
    token = login_resp.json()["accessToken"]
    user_headers = {"Authorization": f"Bearer {token}"}

    # 2. Create Document & upload rendered JPEG page
    doc_resp = requests.post(
        f"{base}/api/documents",
        json={"title": title, "sourceType": "UPLOAD"},
        headers=user_headers,
        timeout=15,
    )
    doc_resp.raise_for_status()
    doc_id = doc_resp.json()["id"]

    up_resp = requests.post(
        f"{base}/api/documents/{doc_id}/uploads",
        files={"file": ("demo_page_1.jpg", jpeg_bytes, "image/jpeg")},
        headers=user_headers,
        timeout=30,
    )
    up_resp.raise_for_status()
    uploaded_pages = up_resp.json()
    if not uploaded_pages:
        raise RuntimeError("Upload did not create any pages.")
    page_id = uploaded_pages[0]["id"]

    # 3. Create OCR job with llmPostprocess=True
    job_resp = requests.post(
        f"{base}/api/documents/{doc_id}/jobs",
        json={
            "options": {"llmPostprocess": True, "localOnly": local_only, "redactPii": False},
            "pipelineVersion": pipeline_version,
        },
        headers=user_headers,
        timeout=15,
    )
    job_resp.raise_for_status()
    created_job_id = job_resp.json()["jobId"]

    # 4. Claim queued job as worker (requires real_worker.py to be paused!)
    worker_headers = {"X-Worker-Key": worker_key}
    claim_resp = requests.post(
        f"{base}/api/worker/jobs/claim",
        json={"workerInstanceId": f"seed-demo-{uuid.uuid4().hex[:6]}"},
        headers=worker_headers,
        timeout=15,
    )
    if claim_resp.status_code == 204:
        raise RuntimeError(
            "Worker claim returned 204 No Content. Pause real_worker.py / ai-worker so it does not steal the job."
        )
    claim_resp.raise_for_status()
    claimed = claim_resp.json()
    if claimed.get("documentId") != doc_id or claimed.get("jobId") != created_job_id:
        raise RuntimeError(
            f"Claimed different job ({claimed.get('jobId')} for doc {claimed.get('documentId')}) "
            f"instead of expected {created_job_id} for doc {doc_id}. Clear stale queued jobs first."
        )

    lease_token = claimed["leaseToken"]
    claimed_pages = claimed.get("pages") or []
    claimed_page_id = claimed_pages[0]["pageId"] if claimed_pages else page_id

    # 5. Upload restored image & submit processed rawLayoutJson
    restored_resp = requests.post(
        f"{base}/api/worker/jobs/{created_job_id}/pages/{claimed_page_id}/restored",
        files={"file": ("restored.jpg", jpeg_bytes, "image/jpeg")},
        headers={**worker_headers, "X-Lease-Token": lease_token},
        timeout=30,
    )
    restored_resp.raise_for_status()
    restored_key = restored_resp.json()["storageKey"]

    complete_resp = requests.post(
        f"{base}/api/worker/jobs/{created_job_id}/results",
        json={
            "leaseToken": lease_token,
            "pageResults": [
                {
                    "pageId": claimed_page_id,
                    "restoredImageKey": restored_key,
                    "rawLayoutJson": processed_layout,
                }
            ],
            "pipelineVersion": pipeline_version,
        },
        headers=worker_headers,
        timeout=15,
    )
    complete_resp.raise_for_status()

    return f"{frontend_url.rstrip('/')}/documents/{doc_id}/review"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed deterministic mock LLM_POSTPROCESS demo page.")
    parser.add_argument("--backend-url", default=os.getenv("BACKEND_URL", "http://localhost:8080"))
    parser.add_argument("--frontend-url", default=os.getenv("FRONTEND_URL", "http://localhost:3000"))
    parser.add_argument("--worker-key", default=os.getenv("WORKER_SECRET_KEY", "worker-secret-local"))
    parser.add_argument("--email", default=os.getenv("DEMO_EMAIL", "demo.postprocess@docforge.local"))
    parser.add_argument("--password", default=os.getenv("DEMO_PASSWORD", "DemoPass123!"))
    parser.add_argument("--title", default="")
    parser.add_argument(
        "--provider",
        choices=("mock", "gemini"),
        default="mock",
        help="Postprocess provider to seed with ('mock' offline or live 'gemini').",
    )
    parser.add_argument("--adversarial", action="store_true", help="Use MockLLMClient(mode='adversarial').")
    parser.add_argument("--dry-run", action="store_true", help="Render & run postprocess locally without calling Backend.")
    args = parser.parse_args(argv)

    if args.provider == "gemini" and not _check_gemini_prereqs():
        return 2

    fixtures = _load_golden_cases()
    entries = build_demo_entries(fixtures)
    jpeg_bytes, raw_layout, font_name = render_demo_page_jpeg(entries)
    if args.provider == "gemini":
        processed_layout = run_gemini_postprocess(raw_layout, jpeg_bytes=jpeg_bytes)
    else:
        processed_layout = run_mock_postprocess(raw_layout, adversarial=args.adversarial)

    summary = {
        "provider": args.provider,
        "mode": "gemini" if args.provider == "gemini" else ("adversarial" if args.adversarial else "normal"),
        "fontUsed": font_name,
        "jpegBytes": len(jpeg_bytes),
        "postprocessMeta": processed_layout.get("postprocessMeta"),
        "blocks": [
            {
                "id": b["id"],
                "isWarning": bool(b.get("isWarning")),
                "postprocess": b.get("postprocess"),
                "suggestions": [
                    {
                        "id": s["id"],
                        "source": s["source"],
                        "provider": s.get("provider"),
                        "risk": s["risk"],
                        "requiresVisualCheck": s["requiresVisualCheck"],
                        "ops": s["ops"],
                    }
                    for s in b.get("suggestions", [])
                ],
            }
            for b in processed_layout.get("blocks", [])
        ],
    }

    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    default_title = (
        "Demo Gemini Postprocess (Golden Cases)"
        if args.provider == "gemini"
        else ("Demo Mock Postprocess (Adversarial)" if args.adversarial else "Demo Mock Postprocess (Golden Cases)")
    )
    doc_title = args.title or default_title
    local_only = args.provider == "mock"
    pipeline_ver = "gemini-demo-seed-v1" if args.provider == "gemini" else "mock-demo-seed-v1"
    review_url = seed_backend(
        backend_url=args.backend_url,
        frontend_url=args.frontend_url,
        worker_key=args.worker_key,
        email=args.email,
        password=args.password,
        title=doc_title,
        jpeg_bytes=jpeg_bytes,
        processed_layout=processed_layout,
        local_only=local_only,
        pipeline_version=pipeline_ver,
    )
    print(f"Seeded successfully. Review Studio URL: {review_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Controlled live smoke & Golden Case 1-8 harness for the Gemini LLM/VLM provider (C11).

LƯU Ý AN TOÀN & CHI PHÍ:
  - Script này CHỈ dùng dữ liệu tổng hợp (synthetic Pillow image + Golden Cases 1-8),
    KHÔNG nhận đường dẫn tệp tùy ý.
  - Phải đặt biến môi trường `LLM_CLOUD_CONSENT=true` và `GEMINI_API_KEY` trước khi chạy.
  - Với Google AI Studio Free Tier, dữ liệu văn bản/ảnh gửi lên có thể được Google sử dụng
    để cải thiện sản phẩm; sử dụng Paid Tier / Pay-as-you-go nếu cần cam kết không huấn luyện.
  - Cấp ngân sách cứng qua `--max-requests` (mặc định 12 request/lần chạy).

Ví dụ (PowerShell):
  $env:LLM_CLOUD_CONSENT="true"
  $env:LLM_CLOUD_TIER="free"
  $env:GEMINI_API_KEY="<your-gemini-api-key>"
  python scripts/dev/gemini_smoke_test.py
  python scripts/dev/gemini_smoke_test.py --models gemini-3.5-flash-lite --golden --max-requests 12
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from worker.gemini_provider import GeminiProvider
from worker.llm_postprocessor import PostprocessConfig, process_page
from worker.llm_resilience import InMemoryLRUCache, LLMResilienceController
from worker.token_guard import TokenLocker

DEFAULT_MODELS = "gemini-3.5-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite"

FONT_CANDIDATES = (
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)


def _is_truthy(raw: str | None) -> bool:
    return (raw or "").strip().lower() in ("1", "true", "yes", "on")


def _print_setup_guide(missing_consent: bool, missing_key: bool) -> None:
    reasons: list[str] = []
    if missing_consent:
        reasons.append("LLM_CLOUD_CONSENT chưa được bật ('true')")
    if missing_key:
        reasons.append("GEMINI_API_KEY đang trống hoặc chưa được thiết lập")

    print("Không thể chạy Gemini smoke test vì thiếu điều kiện tiên quyết:")
    for r in reasons:
        print(f"  - {r}")
    print()
    print("Cảnh báo quyền riêng tư:")
    print(
        "  Google AI Studio Free Tier có thể dùng dữ liệu gửi lên để cải thiện sản phẩm; "
        "hãy dùng tài liệu tổng hợp/demo hoặc nâng lên Paid Tier."
    )
    print()
    print("Hướng dẫn thiết lập trên PowerShell trước khi chạy lại:")
    print('  $env:LLM_CLOUD_CONSENT="true"')
    print('  $env:LLM_CLOUD_TIER="free"   # hoặc "paid" nếu tài khoản có billing')
    print('  $env:GEMINI_API_KEY="<your-gemini-api-key>"')
    print("  .\\.venv311\\Scripts\\python.exe scripts\\dev\\gemini_smoke_test.py")


def _load_font(size: int) -> Any:
    from PIL import ImageFont

    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _render_crop_jpeg(text: str, *, min_width: int = 960, height: int = 72) -> bytes:
    from PIL import Image, ImageDraw

    font = _load_font(26)
    probe = Image.new("RGB", (min_width, height), color=(255, 255, 255))
    probe_draw = ImageDraw.Draw(probe)
    bbox = probe_draw.textbbox((20, 18), text, font=font)
    width = max(min_width, int(bbox[2]) + 40)

    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 18), text, fill=(18, 20, 26), font=font)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _parse_models(raw: str) -> list[str]:
    seen: set[str] = set()
    models: list[str] = []
    for part in raw.split(","):
        name = part.strip()
        if name and name not in seen:
            seen.add(name)
            models.append(name)
    return models


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _check_json_lines(raw_out: Any, expected_line_id: str, required_key: str) -> tuple[bool, str]:
    if not isinstance(raw_out, str) or not raw_out.strip():
        return False, "EMPTY_RESPONSE"
    try:
        data = json.loads(raw_out)
    except json.JSONDecodeError:
        return False, "INVALID_JSON"
    lines = data.get("lines") if isinstance(data, dict) else None
    if not isinstance(lines, list):
        return False, "MISSING_LINES_ARRAY"
    for item in lines:
        if isinstance(item, dict) and item.get("lineId") == expected_line_id and required_key in item:
            return True, ""
    return False, "LINE_ID_MISMATCH"


def run_model_smoke_checks(
    *,
    models: list[str],
    max_requests: int,
    rpm_limit: int,
    timeout_s: float,
    cloud_tier: str,
) -> tuple[list[dict[str, Any]], int]:
    """Run minimal Text-LLM JSON schema test + synthetic Pillow VLM crop test for each model."""
    results: list[dict[str, Any]] = []
    requests_used = 0
    cache = InMemoryLRUCache(max_entries=64)

    smoke_raw = "Thi hn np đon đăng ký mua tài sán công něu trên, k t ngày 06/01/2021"
    smoke_locked, _ = TokenLocker.lock(smoke_raw)
    smoke_payload = {
        "pageNumber": 1,
        "lines": [
            {
                "lineId": "smoke-t1",
                "order": 1,
                "isCandidate": True,
                "text": smoke_locked,
                "invalidSyllables": ["hn", "np", "đon", "sán", "něu", "k", "t"],
            }
        ],
    }
    smoke_crop = _render_crop_jpeg("Công văn số: 20/CV-QLTT")
    vlm_items = [
        {
            "lineId": "smoke-v1",
            "currentText": "Công văn s: 20/CV-",
            "focusTokens": ["s:", "20/CV-"],
            "cropBytes": smoke_crop,
        }
    ]

    for model in models:
        remaining = max_requests - requests_used
        if remaining <= 0:
            results.append(
                {
                    "model": model,
                    "text_ok": "SKIP",
                    "vision_ok": "SKIP",
                    "latency_ms": "0",
                    "tokens": "0/0/0",
                    "note": "MAX_REQUESTS_REACHED",
                }
            )
            continue

        # Dedicated controller per probed model so 404 on one experimental model does not lock out others.
        ctrl = LLMResilienceController(
            rpm_limit=rpm_limit,
            daily_budget=remaining,
            deadline_reserve_seconds=2.0,
        )
        provider = GeminiProvider(
            text_model=model,
            vision_model=model,
            fallback_model=None,
            request_timeout_seconds=timeout_s,
            cloud_consent=True,
            cloud_tier=cloud_tier,
            rpm_limit=rpm_limit,
            daily_budget=remaining,
            controller=ctrl,
            cache=cache,
        )

        notes: list[str] = []
        deadline = time.monotonic() + max(timeout_s + 5.0, 75.0)
        text_raw, text_usage = provider.correct_lines_batch(
            smoke_payload,
            local_only=False,
            deadline_ts=deadline,
        )
        if text_usage.status == "ok":
            text_ok, schema_err = _check_json_lines(text_raw, "smoke-t1", "corrected")
            text_label = "OK" if text_ok else "FAIL"
            if schema_err:
                notes.append(f"text:{schema_err}")
        else:
            text_label = "FAIL"
            notes.append(f"text:{text_usage.error_reason or text_usage.status}")

        vision_label = "SKIP"
        vision_usage = None
        if ctrl.disabled_reason is not None:
            notes.append(f"disabled:{ctrl.disabled_reason}")
        elif (max_requests - (requests_used + ctrl.daily_used)) <= 0:
            notes.append("vision:MAX_REQUESTS_REACHED")
        else:
            vlm_raw, vision_usage = provider.verify_crops_vlm(
                vlm_items,
                local_only=False,
                deadline_ts=time.monotonic() + max(timeout_s + 5.0, 75.0),
            )
            if vision_usage.status == "ok":
                vis_ok, vis_err = _check_json_lines(vlm_raw, "smoke-v1", "transcription")
                vision_label = "OK" if vis_ok else "FAIL"
                if vis_err:
                    notes.append(f"vision:{vis_err}")
            else:
                vision_label = "FAIL"
                notes.append(f"vision:{vision_usage.error_reason or vision_usage.status}")

        requests_used += ctrl.daily_used

        t_in = text_usage.prompt_tokens + (vision_usage.prompt_tokens if vision_usage else 0)
        t_out = text_usage.completion_tokens + (vision_usage.completion_tokens if vision_usage else 0)
        t_th = text_usage.thought_tokens + (vision_usage.thought_tokens if vision_usage else 0)
        lat_str = f"{text_usage.latency_ms}/{vision_usage.latency_ms if vision_usage else 0}ms"

        results.append(
            {
                "model": model,
                "text_ok": text_label,
                "vision_ok": vision_label,
                "latency_ms": lat_str,
                "tokens": f"{t_in}/{t_out}/{t_th}",
                "note": "; ".join(notes) if notes else "—",
            }
        )

    return results, requests_used


def _golden_cases_specs() -> list[dict[str, Any]]:
    """Synthetic Golden Cases 1-8 aligned with tests/test_golden_cases.py."""
    return [
        {
            "gc": "GC1",
            "raw": "Thi hn np đon đăng ký mua tài sán công něu trên, k t ngày 06/01/2021",
            "gt": "Thời hạn nộp đơn đăng ký mua tài sản công nêu trên, kể từ ngày 06/01/2021",
            "conf": 0.95,
        },
        {
            "gc": "GC2",
            "raw": "QLTT ngày 06/01/2021 cùa Đôi QLTT s 4, vè vic xin ý kién phé duyět phuong",
            "gt": "QLTT ngày 06/01/2021 của Đội QLTT số 4, về việc xin ý kiến phê duyệt phương",
            "conf": 0.94,
        },
        {
            "gc": "GC3",
            "raw": "tang vt vi phąm hành chinh tjch thu sung quý nhà nuóc theo Công văn s: 20/CV-",
            "gt": "tang vật vi phạm hành chính tịch thu sung quỹ nhà nước theo Công văn số: 20/CV-QLTT",
            "conf": 0.88,
        },
        {
            "gc": "GC4",
            "raw": "Hài Phòng. ngày th tháng 01 năm 2021",
            "gt": "Hải Phòng, ngày 05 tháng 01 năm 2021",
            "conf": 0.85,
        },
        {
            "gc": "GC5",
            "raw": "ngly .. thíng 01 nim 2021",
            "gt": "ngày .. tháng 01 năm 2021",
            "conf": 0.82,
        },
        {
            "gc": "GC6",
            "raw": (
                "Liên hệ a@doc-forge.vn hoặc https://doc-forge.vn/hd "
                "số 14/2024/HĐ-DOCFORCE ngày 06/01/2021"
            ),
            "gt": (
                "Liên hệ a@doc-forge.vn hoặc https://doc-forge.vn/hd "
                "số 14/2024/HĐ-DOCFORCE ngày 06/01/2021"
            ),
            "conf": 0.85,
        },
        {
            "gc": "GC7",
            "raw": "Công văn 705/cPngey",
            "gt": "Công văn 705/CP-VP",
            "conf": 0.78,
        },
        {
            "gc": "GC8",
            "raw": "tang vật tjch thu nhà nuóc",
            "gt": "tang vật tịch thu nhà nước",
            "conf": 0.96,
        },
    ]


def run_golden_checks(
    *,
    model: str,
    max_requests: int,
    requests_used: int,
    rpm_limit: int,
    timeout_s: float,
    cloud_tier: str,
) -> tuple[list[dict[str, Any]], int]:
    """Run synthetic Golden Cases 1-8 through real process_page + Pillow crops."""
    remaining = max_requests - requests_used
    ctrl = LLMResilienceController(
        rpm_limit=rpm_limit,
        daily_budget=max(0, remaining),
        deadline_reserve_seconds=2.0,
    )
    fallback_model = os.getenv("LLM_FALLBACK_MODEL", "gemini-3.5-flash-lite").strip() or None
    fallback_on_unavail = _is_truthy(os.getenv("LLM_FALLBACK_ON_UNAVAILABLE"))
    provider = GeminiProvider(
        text_model=model,
        vision_model=model,
        fallback_model=fallback_model,
        fallback_on_unavailable=fallback_on_unavail,
        request_timeout_seconds=timeout_s,
        cloud_consent=True,
        cloud_tier=cloud_tier,
        rpm_limit=rpm_limit,
        daily_budget=max(0, remaining),
        controller=ctrl,
        cache=InMemoryLRUCache(max_entries=64),
    )
    cfg = PostprocessConfig(
        enabled=True,
        provider="gemini",
        text_model=model,
        vision_model=model,
        fallback_model=fallback_model,
        fallback_on_unavailable=fallback_on_unavail,
        cloud_consent=True,
        cloud_tier=cloud_tier,
        rpm_limit=rpm_limit,
        daily_request_budget=max(0, remaining),
        request_timeout_seconds=timeout_s,
        line_conf_threshold=0.90,
        max_candidates_per_page=30,
        max_invalid_token_ratio=0.5,
        reocr_enabled=False,
        allow_common_loanwords=True,
        page_deadline_seconds=max(timeout_s + 5.0, 75.0),
        max_vlm_crops=5,
    )

    rows: list[dict[str, Any]] = []
    for spec in _golden_cases_specs():
        if requests_used + ctrl.daily_used >= max_requests:
            rows.append(
                {
                    "gc": spec["gc"],
                    "suggestions": 0,
                    "risk": "—",
                    "blocked": "MAX_REQUESTS_REACHED",
                }
            )
            continue

        crop_bytes = _render_crop_jpeg(spec["gt"])
        blk_id = f"{spec['gc'].lower()}-line-1"
        layout = {
            "schemaVersion": 1,
            "pageNumber": 1,
            "width": 1000,
            "height": 200,
            "pageMargins": {"top": 4.0, "bottom": 4.0, "left": 5.0, "right": 5.0},
            "blocks": [
                {
                    "id": blk_id,
                    "type": "paragraph",
                    "order": 1,
                    "bbox": [20, 40, 960, 112],
                    "ocrText": spec["raw"],
                    "text": spec["raw"],
                    "confidence": spec["conf"],
                }
            ],
        }

        out = process_page(
            layout,
            options={"llmPostprocess": True, "localOnly": False, "redactPii": False},
            crop_provider=lambda _blk, _b=crop_bytes: _b,
            config=cfg,
            llm_client=provider,
        )
        meta = out.get("postprocessMeta") or {}
        sugs = out["blocks"][0].get("suggestions") or []
        risks = sorted({str(s.get("risk") or "") for s in sugs if s.get("risk")})
        rejections = meta.get("rejections") or {}
        blocked_parts = [f"{k}:{v}" for k, v in sorted(rejections.items())]
        if meta.get("skippedReason"):
            blocked_parts.append(f"{meta.get('status')}:{meta['skippedReason']}")

        rows.append(
            {
                "gc": spec["gc"],
                "suggestions": int(meta.get("suggestionsCreated", len(sugs))),
                "risk": ", ".join(risks) if risks else "—",
                "blocked": ", ".join(blocked_parts) if blocked_parts else "—",
                "preview": (sugs[0].get("text") or "—") if sugs else "—",
            }
        )

    return rows, requests_used + ctrl.daily_used


def _print_table(headers: list[str], body: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in body:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _fmt(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    print(_fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in body:
        print(_fmt(row))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Controlled Gemini API smoke test & synthetic Golden Cases 1-8 runner."
    )
    parser.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated Gemini model names to probe "
            f"(default: LLM_MODEL env or {DEFAULT_MODELS})."
        ),
    )
    parser.add_argument(
        "--golden",
        action="store_true",
        help="Also run synthetic Golden Cases 1-8 end-to-end through process_page.",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=12,
        help="Hard cap on total live Gemini API requests per invocation (default: 12).",
    )
    args = parser.parse_args(argv)

    missing_consent = not _is_truthy(os.getenv("LLM_CLOUD_CONSENT"))
    missing_key = not bool(os.getenv("GEMINI_API_KEY", "").strip())
    if missing_consent or missing_key:
        _print_setup_guide(missing_consent, missing_key)
        return 2

    if args.max_requests <= 0:
        print("--max-requests phải > 0.", file=sys.stderr)
        return 2

    raw_models = args.models if args.models is not None else (os.getenv("LLM_MODEL") or DEFAULT_MODELS)
    models = _parse_models(raw_models)
    if not models:
        print("--models phải chứa ít nhất một tên model hợp lệ.", file=sys.stderr)
        return 2

    rpm_limit = _env_int("LLM_RPM_LIMIT", 5)
    timeout_s = _env_float("LLM_REQUEST_TIMEOUT_SECONDS", 10.0)
    cloud_tier = (os.getenv("LLM_CLOUD_TIER") or "free").strip().lower()
    if cloud_tier not in ("free", "paid"):
        cloud_tier = "free"

    model_rows, used = run_model_smoke_checks(
        models=models,
        max_requests=args.max_requests,
        rpm_limit=rpm_limit,
        timeout_s=timeout_s,
        cloud_tier=cloud_tier,
    )
    print("\n=== KẾT QUẢ SMOKE TEST THEO MODEL ===")
    _print_table(
        ["model", "text ok", "vision ok", "độ trễ", "token vào/ra/suy nghĩ", "ghi chú lỗi"],
        [
            [
                r["model"],
                r["text_ok"],
                r["vision_ok"],
                r["latency_ms"],
                r["tokens"],
                r["note"],
            ]
            for r in model_rows
        ],
    )

    if args.golden:
        golden_model = (
            (os.getenv("LLM_TEXT_MODEL") or os.getenv("LLM_MODEL") or "").strip() or models[0]
        )
        gc_rows, used = run_golden_checks(
            model=golden_model,
            max_requests=args.max_requests,
            requests_used=used,
            rpm_limit=rpm_limit,
            timeout_s=timeout_s,
            cloud_tier=cloud_tier,
        )
        print(f"\n=== GOLDEN CASES 1-8 (model={golden_model}) ===")
        _print_table(
            ["GC", "số đề xuất", "rủi ro", "mã bị chặn", "đề xuất"],
            [
                [r["gc"], str(r["suggestions"]), r["risk"], r["blocked"], r.get("preview", "—")]
                for r in gc_rows
            ],
        )

    print(f"\nTổng request thực tế đã gửi: {used}/{args.max_requests}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

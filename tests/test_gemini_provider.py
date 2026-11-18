"""Unit tests for worker.gemini_provider using injected fake SDK clients (no network, no real keys)."""

from __future__ import annotations

import io
import json
import logging
import sys
import time
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from google.genai import errors as genai_errors
from worker.gemini_provider import GeminiProvider, reset_gemini_process_state
from worker.llm_client import build_llm_client
from worker.llm_postprocessor import PostprocessConfig, process_page
from worker.llm_resilience import InMemoryLRUCache, LLMResilienceController

DUMMY_KEY = "TEST_KEY_DO_NOT_USE"


class _FakeClock:
    def __init__(self, start: float = 100.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class _FakeResponse:
    def __init__(
        self,
        text: str | None,
        *,
        in_tok: int = 120,
        out_tok: int = 45,
        thought_tok: int = 10,
        finish_reason: str = "STOP",
        block_reason: str | None = None,
    ) -> None:
        self.text = text
        self.usage_metadata = SimpleNamespace(
            prompt_token_count=in_tok,
            candidates_token_count=out_tok,
            thoughts_token_count=thought_tok,
        )
        self.candidates = [SimpleNamespace(finish_reason=finish_reason)]
        self.prompt_feedback = (
            SimpleNamespace(block_reason=block_reason) if block_reason else None
        )


class _FakeModels:
    def __init__(self, handler: Any) -> None:
        self._handler = handler
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self._handler(model, contents, config, len(self.calls))


class _FakeSDKClient:
    def __init__(self, handler: Any) -> None:
        self.models = _FakeModels(handler)


@pytest.fixture(autouse=True)
def _clean_gemini_state(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_gemini_process_state()
    for var in (
        "GEMINI_API_KEY",
        "LLM_MODEL",
        "LLM_TEXT_MODEL",
        "LLM_VISION_MODEL",
        "LLM_FALLBACK_MODEL",
        "LLM_FALLBACK_ON_UNAVAILABLE",
        "LLM_MAX_VLM_CROPS",
    ):
        monkeypatch.delenv(var, raising=False)


def _jpeg_bytes() -> bytes:
    img = Image.new("RGB", (240, 36), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _make_layout(texts: list[str], confs: list[float] | None = None) -> dict[str, Any]:
    confs = confs or [0.85] * len(texts)
    return {
        "schemaVersion": 1,
        "pageNumber": 1,
        "width": 1000,
        "height": 1400,
        "pageMargins": {"top": 4.0, "bottom": 4.0, "left": 5.0, "right": 5.0},
        "blocks": [
            {
                "id": f"gem-b-{i}",
                "type": "paragraph",
                "order": i,
                "bbox": [50, i * 60, 900, i * 60 + 40],
                "ocrText": txt,
                "text": txt,
                "confidence": conf,
            }
            for i, (txt, conf) in enumerate(zip(texts, confs), start=1)
        ],
    }


def _gem_cfg(**overrides: Any) -> PostprocessConfig:
    base = {
        "enabled": True,
        "provider": "gemini",
        "mock_mode": "normal",
        "line_conf_threshold": 0.90,
        "max_candidates_per_page": 30,
        "max_invalid_token_ratio": 0.5,
        "reocr_enabled": False,
        "max_reocr_lines_per_page": 15,
        "allow_common_loanwords": True,
        "page_deadline_seconds": 15.0,
        "max_vlm_crops": 5,
        "text_model": "gemini-3.5-flash",
        "vision_model": "gemini-3.5-flash",
        "fallback_model": "gemini-3.5-flash-lite",
        "thinking_level": "minimal",
        "rpm_limit": 5,
        "daily_request_budget": 150,
        "cloud_consent": True,
        "cloud_tier": "free",
        "request_timeout_seconds": 10.0,
    }
    base.update(overrides)
    return PostprocessConfig(**base)


def test_gemini_text_and_batched_vlm_success_and_logs_private(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="docforge.worker.gemini")
    crop_jpg = _jpeg_bytes()
    secret_text_line = (
        "Thi hn np đon đăng ký mua tài sản SECRET_SENTINEL_XYZ nêu trên, ngày 06/01/2021"
    )
    vlm_line = "Công văn s: 20/CV-"

    def handler(model: str, contents: Any, config: Any, idx: int) -> Any:
        assert model == "gemini-3.5-flash"
        assert config.temperature == 0.0
        assert config.response_mime_type == "application/json"
        assert config.http_options.timeout >= 500
        assert config.http_options.retry_options.attempts == 1
        assert str(getattr(config.thinking_config.thinking_level, "value", "")).upper() == "MINIMAL"
        if idx == 1:
            assert config.max_output_tokens >= 2048
            return _FakeResponse(
                json.dumps(
                    {
                        "lines": [
                            {
                                "lineId": "gem-b-1",
                                "corrected": (
                                    "Thời hạn nộp đơn đăng ký mua tài sản "
                                    "SECRET_SENTINEL_XYZ nêu trên, ngày ⟦T1⟧"
                                ),
                                "confidence": 0.96,
                                "changes": [
                                    {"from": "hn", "to": "hạn", "reason": "Sửa âm tiết"}
                                ],
                            },
                            {
                                "lineId": "gem-b-2",
                                "corrected": "Công văn số: ⟦T2⟧",
                                "confidence": 0.92,
                                "changes": [],
                            },
                        ]
                    }
                ),
                in_tok=210,
                out_tok=65,
                thought_tok=12,
            )
        assert config.max_output_tokens >= 1024
        # C7: text part "lineId=<id>" directly precedes JPEG image part
        part_pairs = [(p.text, p.inline_data) for p in contents]
        assert any(t == "lineId=gem-b-2" for t, _ in part_pairs)
        idx_tag = next(i for i, (t, _) in enumerate(part_pairs) if t == "lineId=gem-b-2")
        assert part_pairs[idx_tag + 1][1].mime_type == "image/jpeg"
        return _FakeResponse(
            json.dumps(
                {
                    "lines": [
                        {
                            "lineId": "gem-b-2",
                            "transcription": "Công văn số: 20/CV-QLTT",
                            "confidence": 0.93,
                            "uncertain": False,
                            "uncertainSpans": [],
                            "changes": [
                                {"from": "20/CV-", "to": "20/CV-QLTT", "reason": "Đọc đuôi mã"}
                            ],
                        }
                    ]
                }
            ),
            in_tok=340,
            out_tok=48,
            thought_tok=8,
        )

    sdk = _FakeSDKClient(handler)
    provider = GeminiProvider(
        api_key=DUMMY_KEY,
        cloud_consent=True,
        cloud_tier="free",
        sdk_client=sdk,
        controller=LLMResilienceController(rpm_limit=5, daily_budget=150),
        cache=InMemoryLRUCache(256),
    )

    out = process_page(
        _make_layout([secret_text_line, vlm_line], confs=[0.85, 0.85]),
        options={"llmPostprocess": True},
        crop_provider=lambda _blk: crop_jpg,
        config=_gem_cfg(),
        llm_client=provider,
    )

    meta = out["postprocessMeta"]
    assert meta["status"] == "COMPLETED"
    assert meta["provider"] == "gemini"
    assert meta["llmModel"] == "gemini-3.5-flash"
    assert meta["vlmModel"] == "gemini-3.5-flash"
    assert meta["cloudTier"] == "free"
    assert meta["llmCalls"] == 1
    assert meta["vlmCalls"] == 1
    assert meta["tokens"] == {"in": 550, "out": 113, "thought": 20}
    assert meta["rateLimited"] is False
    assert meta["suggestionsCreated"] == 2

    s1 = out["blocks"][0]["suggestions"][0]
    assert s1["provider"] == "gemini"
    assert s1["requiresVisualCheck"] is False
    assert "06/01/2021" in s1["text"]

    s2 = out["blocks"][1]["suggestions"][0]
    assert s2["source"] == "vlm"
    assert s2["risk"] == "high"
    assert s2["requiresVisualCheck"] is True
    assert "20/CV-QLTT" in s2["text"]

    # Security: NO key, NO OCR text in any log line; 1-line summary logged per call + free-tier warning
    assert DUMMY_KEY not in caplog.text
    assert "SECRET_SENTINEL_XYZ" not in caplog.text
    assert "Thi hn np" not in caplog.text
    assert "20/CV-" not in caplog.text
    assert "gemini_call kind=text" in caplog.text
    assert "gemini_call kind=vlm" in caplog.text
    assert caplog.text.count("tier=free") == 1


def test_sdk_isolation_and_lazy_import_contract() -> None:
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "worker"
    for fname in ("llm_client.py", "llm_postprocessor.py", "llm_resilience.py", "token_guard.py"):
        src = (root / fname).read_text(encoding="utf-8")
        assert "google.genai" not in src, f"{fname} must never import google.genai"

    gem_tree = ast.parse((root / "gemini_provider.py").read_text(encoding="utf-8"))
    top_level_imports: list[str] = []
    for node in gem_tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_imports.append(node.module)
    assert not any(m.startswith("google") for m in top_level_imports)


def test_gemini_schema_error_and_safety_finish_reason() -> None:
    # 1. Empty / malformed JSON -> REJECT_SCHEMA_INVALID, 0 suggestions
    sdk_bad_json = _FakeSDKClient(lambda *_a: _FakeResponse("not-valid-json"))
    p_bad = GeminiProvider(
        api_key=DUMMY_KEY,
        cloud_consent=True,
        sdk_client=sdk_bad_json,
        controller=LLMResilienceController(),
        cache=InMemoryLRUCache(256),
    )
    out_bad = process_page(
        _make_layout(["Thi hn np đon ngày 06/01/2021"]),
        options={"llmPostprocess": True},
        config=_gem_cfg(),
        llm_client=p_bad,
    )
    assert out_bad["postprocessMeta"]["status"] == "COMPLETED"
    assert out_bad["postprocessMeta"]["suggestionsCreated"] == 0
    assert out_bad["postprocessMeta"]["rejections"].get("REJECT_SCHEMA_INVALID") == 1

    # 2. Non-STOP finish_reason or prompt_feedback block -> FAILED_SAFE
    sdk_safety = _FakeSDKClient(
        lambda *_a: _FakeResponse('{"lines":[]}', finish_reason="SAFETY")
    )
    p_safe = GeminiProvider(
        api_key=DUMMY_KEY,
        cloud_consent=True,
        sdk_client=sdk_safety,
        controller=LLMResilienceController(),
        cache=InMemoryLRUCache(256),
    )
    out_safe = process_page(
        _make_layout(["Thi hn np đon ngày 06/01/2021"]),
        options={"llmPostprocess": True},
        config=_gem_cfg(),
        llm_client=p_safe,
    )
    assert out_safe["postprocessMeta"]["status"] == "FAILED_SAFE"
    assert out_safe["postprocessMeta"]["suggestionsCreated"] == 0


def test_gemini_429_breaker_401_404_and_consent_privacy_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _make_layout(["tang vt vi phąm s: 20/CV-"])
    crop_jpg = _jpeg_bytes()

    # 1. Consent gate default False -> SKIPPED CLOUD_NOT_CONSENTED (via build_llm_client AND direct)
    monkeypatch.setenv("GEMINI_API_KEY", DUMMY_KEY)
    c_none, r_none = build_llm_client(_gem_cfg(cloud_consent=False), {"llmPostprocess": True})
    assert c_none is None and r_none == "CLOUD_NOT_CONSENTED"
    out_noconsent = process_page(
        layout, options={"llmPostprocess": True}, config=_gem_cfg(cloud_consent=False)
    )
    assert out_noconsent["postprocessMeta"]["status"] == "SKIPPED"
    assert out_noconsent["postprocessMeta"]["skippedReason"] == "CLOUD_NOT_CONSENTED"

    # 2. localOnly + gemini -> SKIPPED LOCAL_ENDPOINT_UNAVAILABLE
    out_local = process_page(
        layout, options={"llmPostprocess": True, "localOnly": True}, config=_gem_cfg(cloud_consent=True)
    )
    assert out_local["postprocessMeta"]["status"] == "SKIPPED"
    assert out_local["postprocessMeta"]["skippedReason"] == "LOCAL_ENDPOINT_UNAVAILABLE"

    # 3. 429 -> RATE_LIMITED + breaker cooldown -> immediate BREAKER_OPEN on next page
    clk = _FakeClock(time.monotonic())
    ctrl_429 = LLMResilienceController(rpm_limit=10, daily_budget=100, clock=clk.time, sleep_fn=clk.sleep)
    sdk_429 = _FakeSDKClient(
        lambda *_a: (_ for _ in ()).throw(
            genai_errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "429"}})
        )
    )
    p_429 = GeminiProvider(
        api_key=DUMMY_KEY, cloud_consent=True, sdk_client=sdk_429, controller=ctrl_429, cache=InMemoryLRUCache(16), clock=clk.time
    )
    out_429 = process_page(layout, options={"llmPostprocess": True}, config=_gem_cfg(), llm_client=p_429)
    assert out_429["postprocessMeta"]["status"] == "SKIPPED"
    assert out_429["postprocessMeta"]["skippedReason"] == "RATE_LIMITED"
    assert out_429["postprocessMeta"]["rateLimited"] is True
    assert len(sdk_429.models.calls) == 1

    out_open = process_page(layout, options={"llmPostprocess": True}, config=_gem_cfg(), llm_client=p_429)
    assert out_open["postprocessMeta"]["status"] == "SKIPPED"
    assert out_open["postprocessMeta"]["skippedReason"] == "BREAKER_OPEN"
    assert len(sdk_429.models.calls) == 1  # Zero network calls during cooldown

    # 4. 401 -> disables provider for process (AUTH_ERROR)
    ctrl_401 = LLMResilienceController()
    sdk_401 = _FakeSDKClient(
        lambda *_a: (_ for _ in ()).throw(
            genai_errors.ClientError(401, {"error": {"code": 401, "status": "UNAUTHENTICATED", "message": "bad key"}})
        )
    )
    p_401 = GeminiProvider(
        api_key=DUMMY_KEY, cloud_consent=True, sdk_client=sdk_401, controller=ctrl_401, cache=InMemoryLRUCache(16)
    )
    out_401_a = process_page(layout, options={"llmPostprocess": True}, config=_gem_cfg(), llm_client=p_401)
    out_401_b = process_page(layout, options={"llmPostprocess": True}, config=_gem_cfg(), llm_client=p_401)
    assert out_401_a["postprocessMeta"]["skippedReason"] == "AUTH_ERROR"
    assert out_401_b["postprocessMeta"]["skippedReason"] == "AUTH_ERROR"
    assert len(sdk_401.models.calls) == 1

    # 5a. 404 on primary (gemini-3.5-flash) auto-falls back to gemini-3.5-flash-lite and remembers it
    models_seen: list[str] = []

    def fb_handler(model: str, _c: Any, _cfg: Any, _idx: int) -> Any:
        models_seen.append(model)
        if model == "gemini-3.5-flash":
            raise genai_errors.ClientError(404, {"error": {"code": 404, "status": "NOT_FOUND", "message": "missing"}})
        return _FakeResponse(
            json.dumps(
                {
                    "lines": [
                        {
                            "lineId": "gem-b-1",
                            "corrected": "tang vật vi phạm số: ⟦T1⟧",
                            "confidence": 0.94,
                            "changes": [],
                        }
                    ]
                }
            )
        )

    ctrl_fb = LLMResilienceController()
    p_fb = GeminiProvider(
        api_key=DUMMY_KEY,
        cloud_consent=True,
        sdk_client=_FakeSDKClient(fb_handler),
        controller=ctrl_fb,
        cache=InMemoryLRUCache(16),
    )
    out_fb1 = process_page(
        _make_layout(["tang vt vi phąm 06/01/2021"]),
        options={"llmPostprocess": True},
        config=_gem_cfg(),
        llm_client=p_fb,
    )
    assert out_fb1["postprocessMeta"]["status"] == "COMPLETED"
    assert out_fb1["postprocessMeta"]["llmModel"] == "gemini-3.5-flash-lite"
    assert models_seen == ["gemini-3.5-flash", "gemini-3.5-flash-lite"]

    out_fb2 = process_page(
        _make_layout(["Thi hn np đon ngày 06/01/2021"]),
        options={"llmPostprocess": True},
        config=_gem_cfg(),
        llm_client=p_fb,
    )
    assert out_fb2["postprocessMeta"]["status"] == "COMPLETED"
    assert models_seen == ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.5-flash-lite"]

    # 5b. 404 on BOTH primary and fallback -> disables provider for process (MODEL_UNAVAILABLE)
    ctrl_404 = LLMResilienceController()
    sdk_404 = _FakeSDKClient(
        lambda *_a: (_ for _ in ()).throw(
            genai_errors.ClientError(404, {"error": {"code": 404, "status": "NOT_FOUND", "message": "model missing"}})
        )
    )
    p_404 = GeminiProvider(
        api_key=DUMMY_KEY, cloud_consent=True, sdk_client=sdk_404, controller=ctrl_404, cache=InMemoryLRUCache(16)
    )
    out_404_a = process_page(layout, options={"llmPostprocess": True}, config=_gem_cfg(), llm_client=p_404)
    out_404_b = process_page(layout, options={"llmPostprocess": True}, config=_gem_cfg(), llm_client=p_404)
    assert out_404_a["postprocessMeta"]["skippedReason"] == "MODEL_UNAVAILABLE"
    assert out_404_b["postprocessMeta"]["skippedReason"] == "MODEL_UNAVAILABLE"
    assert len(sdk_404.models.calls) == 2

    # 6. redactPii=True -> Text-LLM runs, VLM MUST NOT run
    vlm_called = False

    def pii_handler(_m: str, _c: Any, _cfg: Any, idx: int) -> Any:
        nonlocal vlm_called
        if idx > 1:
            vlm_called = True
        return _FakeResponse(
            json.dumps(
                {
                    "lines": [
                        {
                            "lineId": "gem-b-1",
                            "corrected": "tang vật vi phạm số: ⟦T1⟧",
                            "confidence": 0.94,
                            "changes": [],
                        }
                    ]
                }
            )
        )

    p_pii = GeminiProvider(
        api_key=DUMMY_KEY,
        cloud_consent=True,
        sdk_client=_FakeSDKClient(pii_handler),
        controller=LLMResilienceController(),
        cache=InMemoryLRUCache(16),
    )
    out_pii = process_page(
        layout,
        options={"llmPostprocess": True, "redactPii": True},
        crop_provider=lambda _b: crop_jpg,
        config=_gem_cfg(),
        llm_client=p_pii,
    )
    assert vlm_called is False
    assert out_pii["postprocessMeta"]["status"] == "COMPLETED"
    assert out_pii["postprocessMeta"]["skippedReason"] == "VLM_SKIPPED_PII_REDACTION"
    assert out_pii["postprocessMeta"]["vlmCalls"] == 0
    assert out_pii["blocks"][0].get("isWarning") is True


def test_gemini_timeout_retry_deadline_thinking_fallback_and_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _make_layout(["Thi hn np đon ngày 06/01/2021"])

    # 1. Timeout retries ONCE within generous page deadline then returns FAILED_SAFE
    clk1 = _FakeClock(time.monotonic())

    def timeout_always(*_a: Any) -> Any:
        clk1.advance(0.2)
        raise TimeoutError("timed out")

    p_to = GeminiProvider(
        api_key=DUMMY_KEY,
        cloud_consent=True,
        sdk_client=_FakeSDKClient(timeout_always),
        controller=LLMResilienceController(clock=clk1.time, sleep_fn=clk1.sleep),
        cache=InMemoryLRUCache(16),
        clock=clk1.time,
    )
    out_to_retry = process_page(
        layout, options={"llmPostprocess": True}, config=_gem_cfg(page_deadline_seconds=15.0), llm_client=p_to
    )
    assert len(p_to._sdk_client.models.calls) == 2
    assert out_to_retry["postprocessMeta"]["status"] == "FAILED_SAFE"

    # 2. Tight page deadline is respected -> 1 attempt only, TIMEOUT
    sdk_tight = _FakeSDKClient(lambda *_a: (_ for _ in ()).throw(TimeoutError("slow")))
    p_tight = GeminiProvider(
        api_key=DUMMY_KEY,
        cloud_consent=True,
        sdk_client=sdk_tight,
        controller=LLMResilienceController(),
        cache=InMemoryLRUCache(16),
    )
    out_tight = process_page(
        layout, options={"llmPostprocess": True}, config=_gem_cfg(page_deadline_seconds=0.5), llm_client=p_tight
    )
    assert len(sdk_tight.models.calls) == 1
    assert out_tight["postprocessMeta"]["status"] == "TIMEOUT"

    # 3. 400 on thinking_config -> retries once without thinking_config & remembers across pages + cache hit!
    thinking_seen: list[bool] = []

    def thinking_handler(_m: str, _c: Any, config: Any, _idx: int) -> Any:
        has_think = config.thinking_config is not None
        thinking_seen.append(has_think)
        if has_think:
            raise genai_errors.ClientError(
                400,
                {"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": "thinking_config unsupported"}},
            )
        return _FakeResponse(
            json.dumps(
                {
                    "lines": [
                        {
                            "lineId": "gem-b-1",
                            "corrected": "Thời hạn nộp đơn ngày ⟦T1⟧",
                            "confidence": 0.95,
                            "changes": [],
                        }
                    ]
                }
            )
        )

    sdk_think = _FakeSDKClient(thinking_handler)
    ctrl_think = LLMResilienceController()
    cache_think = InMemoryLRUCache(256)
    p_think = GeminiProvider(
        api_key=DUMMY_KEY,
        cloud_consent=True,
        sdk_client=sdk_think,
        controller=ctrl_think,
        cache=cache_think,
    )
    out_t1 = process_page(layout, options={"llmPostprocess": True}, config=_gem_cfg(), llm_client=p_think)
    assert out_t1["postprocessMeta"]["status"] == "COMPLETED"
    assert out_t1["postprocessMeta"]["suggestionsCreated"] == 1
    assert thinking_seen == [True, False]
    assert ctrl_think.thinking_disabled is True

    # Identical page hits cache (0 extra SDK calls)
    out_cached = process_page(layout, options={"llmPostprocess": True}, config=_gem_cfg(), llm_client=p_think)
    assert out_cached["postprocessMeta"]["suggestionsCreated"] == 1
    assert len(thinking_seen) == 2

    # Different text bypasses cache and starts WITHOUT thinking_config immediately
    out_t2 = process_page(
        _make_layout(["QLTT ngày 06/01/2021 cùa Đôi QLTT"]),
        options={"llmPostprocess": True},
        config=_gem_cfg(),
        llm_client=p_think,
    )
    assert out_t2["postprocessMeta"]["status"] == "COMPLETED"
    assert thinking_seen == [True, False, False]

    # Env aliases: LLM_MODEL, LLM_POSTPROCESS_DEFAULT, LLM_FALLBACK_ON_UNAVAILABLE
    monkeypatch.setenv("LLM_MODEL", "gemini-3.5-flash")
    monkeypatch.setenv("LLM_POSTPROCESS_DEFAULT", "true")
    monkeypatch.setenv("LLM_FALLBACK_ON_UNAVAILABLE", "true")
    monkeypatch.delenv("LLM_POSTPROCESS_ENABLED", raising=False)
    monkeypatch.delenv("LLM_TEXT_MODEL", raising=False)
    monkeypatch.delenv("LLM_VISION_MODEL", raising=False)
    cfg_alias = PostprocessConfig.from_env()
    assert cfg_alias.enabled is True
    assert cfg_alias.text_model == "gemini-3.5-flash"
    assert cfg_alias.vision_model == "gemini-3.5-flash"
    assert cfg_alias.fallback_model == "gemini-3.5-flash-lite"
    assert cfg_alias.fallback_on_unavailable is True

    # Opt-in fallback_on_unavailable=True switches primary -> fallback on 504/503/timeout and remembers it
    unavail_models: list[str] = []

    def unavail_handler(model: str, _c: Any, _cfg: Any, idx: int) -> Any:
        unavail_models.append(model)
        if model == "gemini-3.5-flash":
            if idx == 1:
                raise genai_errors.ServerError(
                    504, {"error": {"code": 504, "status": "DEADLINE_EXCEEDED", "message": "Deadline expired"}}
                )
            raise TimeoutError("timed out")
        return _FakeResponse(
            json.dumps(
                {
                    "lines": [
                        {
                            "lineId": "gem-b-1",
                            "corrected": "Thời hạn nộp đơn ngày ⟦T1⟧",
                            "confidence": 0.95,
                            "changes": [],
                        }
                    ]
                }
            )
        )

    p_unavail = GeminiProvider.from_config(
        _gem_cfg(fallback_on_unavailable=True, page_deadline_seconds=25.0),
        api_key=DUMMY_KEY,
        sdk_client=_FakeSDKClient(unavail_handler),
        controller=LLMResilienceController(),
        cache=InMemoryLRUCache(16),
    )
    out_u1 = process_page(
        layout,
        options={"llmPostprocess": True},
        config=_gem_cfg(fallback_on_unavailable=True, page_deadline_seconds=25.0),
        llm_client=p_unavail,
    )
    assert out_u1["postprocessMeta"]["status"] == "COMPLETED"
    assert out_u1["postprocessMeta"]["llmModel"] == "gemini-3.5-flash-lite"
    assert out_u1["postprocessMeta"]["suggestionsCreated"] == 1
    assert unavail_models == ["gemini-3.5-flash", "gemini-3.5-flash-lite"]

    out_u2 = process_page(
        _make_layout(["QLTT ngày 06/01/2021 cùa Đôi QLTT"]),
        options={"llmPostprocess": True},
        config=_gem_cfg(fallback_on_unavailable=True, page_deadline_seconds=25.0),
        llm_client=p_unavail,
    )
    assert out_u2["postprocessMeta"]["status"] == "COMPLETED"
    assert out_u2["postprocessMeta"]["llmModel"] == "gemini-3.5-flash-lite"
    assert unavail_models == ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.5-flash-lite"]

    # Comma-separated fallback chain: primary 503 -> fallback1 503 -> fallback2 200 OK
    chain_models: list[str] = []

    def chain_handler(model: str, _c: Any, _cfg: Any, _idx: int) -> Any:
        chain_models.append(model)
        if model in ("gemini-3.1-flash-lite", "gemini-3.5-flash-lite"):
            raise genai_errors.ServerError(
                503, {"error": {"code": 503, "status": "UNAVAILABLE", "message": "busy"}}
            )
        return _FakeResponse(
            json.dumps(
                {
                    "lines": [
                        {
                            "lineId": "gem-b-1",
                            "corrected": "Thời hạn nộp đơn ngày ⟦T1⟧",
                            "confidence": 0.96,
                            "changes": [],
                        }
                    ]
                }
            )
        )

    p_chain = GeminiProvider.from_config(
        _gem_cfg(
            text_model="gemini-3.1-flash-lite",
            fallback_model="gemini-3.5-flash-lite,gemini-3.5-flash",
            fallback_on_unavailable=True,
            page_deadline_seconds=25.0,
        ),
        api_key=DUMMY_KEY,
        sdk_client=_FakeSDKClient(chain_handler),
        controller=LLMResilienceController(),
        cache=InMemoryLRUCache(16),
    )
    out_chain = process_page(
        layout,
        options={"llmPostprocess": True},
        config=_gem_cfg(
            text_model="gemini-3.1-flash-lite",
            fallback_model="gemini-3.5-flash-lite,gemini-3.5-flash",
            fallback_on_unavailable=True,
            page_deadline_seconds=25.0,
        ),
        llm_client=p_chain,
    )
    assert out_chain["postprocessMeta"]["status"] == "COMPLETED"
    assert out_chain["postprocessMeta"]["llmModel"] == "gemini-3.5-flash"
    assert chain_models == ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.5-flash"]

    # Heavy ML packages must remain unimported
    for mod in ("torch", "paddle", "paddleocr", "vietocr", "ultralytics"):
        assert mod not in sys.modules

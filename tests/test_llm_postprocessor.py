"""Unit tests for worker.llm_client and worker.llm_postprocessor (no network, no heavy ML libs)."""

from __future__ import annotations

import copy
import sys
import time
from typing import Any

from worker.llm_client import LLMUsageLog, MockLLMClient, build_llm_client
from worker.llm_postprocessor import PostprocessConfig, crop_line, prepare_vlm_crop_jpeg, process_page


def _make_layout(texts: list[str], confs: list[float] | None = None) -> dict[str, Any]:
    confs = confs or [0.85] * len(texts)
    blocks = []
    for i, (txt, conf) in enumerate(zip(texts, confs), start=1):
        blocks.append(
            {
                "id": f"b-{i}",
                "type": "paragraph",
                "order": i,
                "bbox": [50, i * 60, 900, i * 60 + 40],
                "ocrText": txt,
                "text": txt,
                "confidence": conf,
            }
        )
    return {
        "schemaVersion": 1,
        "pageNumber": 1,
        "width": 1000,
        "height": 1400,
        "pageMargins": {"top": 4.0, "bottom": 4.0, "left": 5.0, "right": 5.0},
        "blocks": blocks,
    }


def _mock_cfg(**overrides: Any) -> PostprocessConfig:
    base = {
        "enabled": True,
        "provider": "mock",
        "mock_mode": "normal",
        "line_conf_threshold": 0.90,
        "max_candidates_per_page": 30,
        "max_invalid_token_ratio": 0.5,
        "reocr_enabled": False,
        "max_reocr_lines_per_page": 15,
        "allow_common_loanwords": True,
        "page_deadline_seconds": 15.0,
        "max_vlm_crops": 5,
    }
    base.update(overrides)
    return PostprocessConfig(**base)


def test_stage_disabled_adds_no_new_keys_b11() -> None:
    layout = _make_layout(["Thi hn np đon đăng ký mua tài sán công něu trên, k t ngày 06/01/2021"])
    snap = copy.deepcopy(layout)

    # Globally off + option on -> unchanged
    out1 = process_page(layout, options={"llmPostprocess": True}, config=_mock_cfg(enabled=False))
    assert out1 == snap

    # Globally on + option off -> unchanged
    out2 = process_page(layout, options={"llmPostprocess": False}, config=_mock_cfg(enabled=True))
    assert out2 == snap

    # Both absent -> unchanged
    out3 = process_page(layout, options=None, config=_mock_cfg(enabled=False))
    assert out3 == snap


def test_mock_normal_golden_cases_1_to_5_and_7_without_crop_calls() -> None:
    texts = [
        "Thi hn np đon đăng ký mua tài sán công něu trên, k t ngày 06/01/2021",
        "QLTT ngày 06/01/2021 cùa Đôi QLTT s 4, vè vic xin ý kién phé duyět phuong",
        "tang vt vi phąm hành chinh tjch thu sung quý nhà nuóc theo Công văn s: 20/CV-",
        "Hài Phòng. ngày th tháng 01 năm 2021",
        "ngly .. thíng 01 nim 2021",
        "Công văn 1027/",
    ]
    # Note: custom fixture maps GC7 "Số: 1027/" / "1027/"; test exact key "1027/" too
    texts[5] = "1027/"
    layout = _make_layout(texts, confs=[0.95, 0.94, 0.88, 0.85, 0.82, 0.78])
    crop_calls: list[str] = []

    def fail_if_called_crop(blk: dict[str, Any]) -> bytes:
        crop_calls.append(blk["id"])
        return b"jpeg"

    out = process_page(
        layout,
        options={"llmPostprocess": True, "localOnly": True},
        crop_provider=fail_if_called_crop,
        config=_mock_cfg(),
    )

    assert crop_calls == []  # Mock VLM must NOT trigger crop_provider
    meta = out["postprocessMeta"]
    assert meta["status"] == "COMPLETED"
    assert meta["provider"] == "mock"
    assert meta["llmCalls"] == 1
    assert meta["vlmCalls"] == 1
    assert meta["suggestionsCreated"] >= 5

    # Original block text & confidence remain untouched
    for orig, b in zip(texts, out["blocks"]):
        assert b["text"] == orig

    # GC1 & GC2: non-damaged Text-LLM suggestions have requiresVisualCheck=False and NO isWarning (B2)
    b1, b2, b3, b4, b5, b7 = out["blocks"]
    assert b1["postprocess"]["candidate"] is True
    assert not b1.get("isWarning")
    assert b1["suggestions"][0]["provider"] == "mock"
    assert b1["suggestions"][0]["requiresVisualCheck"] is False
    assert "06/01/2021" in b1["suggestions"][0]["text"]
    assert b1["suggestions"][0]["ops"]

    assert not b2.get("isWarning")
    assert b2["suggestions"][0]["provider"] == "mock"
    assert "06/01/2021" in b2["suggestions"][0]["text"]

    # GC3, GC4: mock VLM produces risk="high" + requiresVisualCheck=True + isWarning=True (B2)
    for blk, expected_substr in ((b3, "20/CV-QLTT"), (b4, "05 tháng 01 năm 2021"), (b7, "1027/QĐ-UBND")):
        assert blk.get("isWarning") is True
        sug = blk["suggestions"][0]
        assert sug["source"] == "vlm"
        assert sug["provider"] == "mock"
        assert sug["risk"] == "high"
        assert sug["requiresVisualCheck"] is True
        assert expected_substr in sug["text"]

    # GC5: VLM mock marks '..' uncertain -> isWarning=True + Text-LLM word suggestion kept with requiresVisualCheck=True
    assert b5.get("isWarning") is True
    assert b5["suggestions"][0]["requiresVisualCheck"] is True
    assert "ngày .. tháng 01 năm 2021" == b5["suggestions"][0]["text"]


def test_mock_adversarial_rejects_all_and_records_codes() -> None:
    layout = _make_layout(
        [
            "Thi hn np đon đăng ký mua tài sán công něu trên, k t ngày 06/01/2021",
            "QLTT ngày 06/01/2021 cùa Đôi QLTT s 4, vè vic xin ý kién phé duyět phuong",
        ]
    )
    mock = MockLLMClient(mode="adversarial")
    seen_codes: set[str] = set()
    for _ in range(4):
        out = process_page(
            layout,
            options={"llmPostprocess": True},
            config=_mock_cfg(mock_mode="adversarial"),
            llm_client=mock,
        )
        assert out["postprocessMeta"]["status"] == "COMPLETED"
        assert out["postprocessMeta"]["suggestionsCreated"] == 0
        for blk in out["blocks"]:
            assert not blk.get("suggestions")
        seen_codes.update(out["postprocessMeta"]["rejections"].keys())

    assert "REJECT_PLACEHOLDER_ALTERED" in seen_codes
    assert "REJECT_LINE_COUNT_MISMATCH" in seen_codes
    assert "REJECT_LENGTH_RATIO_OUT_OF_BOUNDS" in seen_codes
    assert "REJECT_SCHEMA_INVALID" in seen_codes


def test_privacy_matrix_exception_and_timeout() -> None:
    layout = _make_layout(
        ["tang vt vi phąm hành chinh tjch thu sung quý nhà nuóc theo Công văn s: 20/CV-"]
    )

    class CloudStub:
        is_cloud = True
        provider_name = "cloud"
        model_name = "cloud-stub"

        def __init__(self, delay: float = 0.0, boom: bool = False) -> None:
            self.delay = delay
            self.boom = boom
            self.llm_calls = 0
            self.vlm_calls = 0

        def correct_lines_batch(self, page_payload: dict[str, Any], *, local_only: bool = False, deadline_ts: float):
            del local_only, deadline_ts
            self.llm_calls += 1
            if self.boom:
                raise RuntimeError("boom")
            if self.delay:
                time.sleep(self.delay)
            lines = page_payload["lines"]
            return (
                {
                    "lines": [
                        {"lineId": ln["lineId"], "corrected": ln["text"], "confidence": 0.9, "changes": []}
                        for ln in lines
                        if ln.get("isCandidate")
                    ]
                },
                LLMUsageLog(provider_type="cloud", model=self.model_name),
            )

        def verify_crops_vlm(self, items: list[dict[str, Any]], *, local_only: bool = False, deadline_ts: float):
            del items, local_only, deadline_ts
            self.vlm_calls += 1
            raise AssertionError("VLM must not be called when redactPii=True or localOnly=True")

    # 1. Cloud + localOnly=True -> SKIPPED, 0 calls
    c1 = CloudStub()
    out_local = process_page(
        layout,
        options={"llmPostprocess": True, "localOnly": True},
        config=_mock_cfg(),
        llm_client=c1,
    )
    assert out_local["postprocessMeta"]["status"] == "SKIPPED"
    assert out_local["postprocessMeta"]["skippedReason"] == "LOCAL_ENDPOINT_UNAVAILABLE"
    assert c1.llm_calls == 0 and c1.vlm_calls == 0

    # 2. Cloud + redactPii=True -> Text-LLM runs, VLM skipped
    c2 = CloudStub()
    out_pii = process_page(
        layout,
        options={"llmPostprocess": True, "redactPii": True},
        config=_mock_cfg(),
        llm_client=c2,
    )
    assert out_pii["postprocessMeta"]["status"] == "COMPLETED"
    assert out_pii["postprocessMeta"]["skippedReason"] == "VLM_SKIPPED_PII_REDACTION"
    assert c2.llm_calls == 1 and c2.vlm_calls == 0
    assert out_pii["blocks"][0].get("isWarning") is True

    # 3. Exception -> FAILED_SAFE
    c3 = CloudStub(boom=True)
    out_err = process_page(
        layout,
        options={"llmPostprocess": True},
        config=_mock_cfg(),
        llm_client=c3,
    )
    assert out_err["postprocessMeta"]["status"] == "FAILED_SAFE"
    assert not out_err["blocks"][0].get("suggestions")

    # 4. Slow client -> TIMEOUT
    c4 = CloudStub(delay=0.15)
    out_to = process_page(
        layout,
        options={"llmPostprocess": True},
        config=_mock_cfg(page_deadline_seconds=0.05),
        llm_client=c4,
    )
    assert out_to["postprocessMeta"]["status"] == "TIMEOUT"

    # 5. Non-mock provider factory -> PROVIDER_NOT_IMPLEMENTED -> SKIPPED
    client_none, reason = build_llm_client(_mock_cfg(provider="openai_compat"), {})
    assert client_none is None and reason == "PROVIDER_NOT_IMPLEMENTED"
    out_unimpl = process_page(
        layout,
        options={"llmPostprocess": True},
        config=_mock_cfg(provider="openai_compat"),
    )
    assert out_unimpl["postprocessMeta"]["status"] == "SKIPPED"
    assert out_unimpl["postprocessMeta"]["skippedReason"] == "PROVIDER_NOT_IMPLEMENTED"

    # Heavy ML packages must remain unimported during text-only tests
    for mod in ("torch", "paddle", "paddleocr", "vietocr", "ultralytics"):
        assert mod not in sys.modules


def test_noise_gate_collision_and_crop_line_synthetic() -> None:
    # B3 placeholder collision skipped from LLM
    col_layout = _make_layout(["Văn bản chứa ⟦T1⟧ và hn np"])
    out_col = process_page(col_layout, options={"llmPostprocess": True}, config=_mock_cfg())
    assert out_col["postprocessMeta"]["rejections"].get("SKIP_PLACEHOLDER_COLLISION") == 1
    assert out_col["postprocessMeta"]["llmCalls"] == 0

    # B7 > 0.5 invalid token ratio -> PAGE_TOO_NOISY_OR_NOT_VIETNAMESE
    noisy_layout = _make_layout(["xxq yyq zzq wwq vvq nnq mmq ppq"])
    out_noisy = process_page(noisy_layout, options={"llmPostprocess": True}, config=_mock_cfg())
    assert out_noisy["postprocessMeta"]["status"] == "SKIPPED"
    assert out_noisy["postprocessMeta"]["skippedReason"] == "PAGE_TOO_NOISY_OR_NOT_VIETNAMESE"

    # crop_line synthetic image test (clamping + max_side downscale)
    import numpy as np

    img = np.zeros((200, 1500, 3), dtype=np.uint8)
    cropped = crop_line(img, [-10, 10, 1600, 120], pad=6, max_side=960)
    assert cropped is not None
    assert max(cropped.shape[:2]) == 960
    assert crop_line(img, [100, 100, 50, 50], pad=0) is None


def test_vlm_batching_cap_metadata_strip_and_lineid_mismatch() -> None:
    import io
    from PIL import Image

    # Oversized PNG with EXIF chunk -> normalized to JPEG <= 1024px, no EXIF
    raw_img = Image.new("RGB", (1400, 120), color=(255, 255, 255))
    exif = Image.Exif()
    exif[0x010E] = "SECRET_CAMERA_TAG"
    raw_buf = io.BytesIO()
    raw_img.save(raw_buf, format="JPEG", exif=exif.tobytes(), quality=95)
    norm_bytes = prepare_vlm_crop_jpeg(raw_buf.getvalue(), max_side=1024, quality=90)
    with Image.open(io.BytesIO(norm_bytes)) as check_img:
        assert check_img.format == "JPEG"
        assert max(check_img.size) == 1024
        assert not check_img.info.get("exif")

    # 6 damaged lines -> batched in <=1 VLM request of <=5 crops
    six_lines = [f"Công văn s: {i}/" for i in range(1, 7)]
    layout_6 = _make_layout(six_lines, confs=[0.85] * 6)
    seen_batches: list[list[dict[str, Any]]] = []

    class VlmBatchStub:
        is_cloud = True
        provider_name = "stub-vlm"
        model_name = "stub-vlm"
        needs_crop_image = True

        def __init__(self, drop_last: bool = False) -> None:
            self.drop_last = drop_last

        def correct_lines_batch(self, page_payload: dict[str, Any], *, local_only: bool = False, deadline_ts: float):
            del local_only, deadline_ts
            return (
                {
                    "lines": [
                        {"lineId": ln["lineId"], "corrected": ln["text"], "confidence": 0.9, "changes": []}
                        for ln in page_payload["lines"]
                        if ln.get("isCandidate")
                    ]
                },
                LLMUsageLog(provider_type="cloud", model=self.model_name),
            )

        def verify_crops_vlm(self, items: list[dict[str, Any]], *, local_only: bool = False, deadline_ts: float):
            del local_only, deadline_ts
            seen_batches.append(items)
            subset = items[:-1] if self.drop_last else items
            return (
                {
                    "lines": [
                        {
                            "lineId": it["lineId"],
                            "transcription": it["currentText"] + "QĐ",
                            "confidence": 0.92,
                            "uncertain": False,
                            "uncertainSpans": [],
                            "changes": [{"from": it["focusTokens"][0], "to": it["focusTokens"][0] + "QĐ", "reason": "read crop"}],
                        }
                        for it in subset
                    ]
                },
                LLMUsageLog(provider_type="cloud", model=self.model_name),
            )

    out_ok = process_page(
        layout_6,
        options={"llmPostprocess": True},
        crop_provider=lambda _blk: raw_buf.getvalue(),
        config=_mock_cfg(max_vlm_crops=10),
        llm_client=VlmBatchStub(drop_last=False),
    )
    assert out_ok["postprocessMeta"]["llmCalls"] == 1
    assert out_ok["postprocessMeta"]["vlmCalls"] == 1
    assert len(seen_batches) == 1
    assert len(seen_batches[0]) == 5
    for sent in seen_batches[0]:
        with Image.open(io.BytesIO(sent["cropBytes"])) as sent_img:
            assert sent_img.format == "JPEG"
            assert max(sent_img.size) <= 1024
            assert not sent_img.info.get("exif")
    assert out_ok["postprocessMeta"]["suggestionsCreated"] == 5

    # Mismatched lineId count in VLM batch -> drops ENTIRE batch with REJECT_LINE_COUNT_MISMATCH
    seen_batches.clear()
    out_mismatch = process_page(
        _make_layout(["Công văn 705/cPngey", "Số: 1027/"], confs=[0.80, 0.80]),
        options={"llmPostprocess": True},
        crop_provider=lambda _blk: raw_buf.getvalue(),
        config=_mock_cfg(),
        llm_client=VlmBatchStub(drop_last=True),
    )
    assert out_mismatch["postprocessMeta"]["vlmCalls"] == 1
    assert out_mismatch["postprocessMeta"]["suggestionsCreated"] == 0
    assert out_mismatch["postprocessMeta"]["rejections"].get("REJECT_LINE_COUNT_MISMATCH") == 1
    assert all(not b.get("suggestions") for b in out_mismatch["blocks"])
    assert all(b.get("isWarning") is True for b in out_mismatch["blocks"])

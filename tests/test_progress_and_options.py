"""Pure unit tests for job option parsing, monotonic stage progress, and Re-OCR wiring (Commit 7)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worker.job_progress import (
    OCR_STAGE_ORDER,
    VALID_HEARTBEAT_STAGES,
    compute_progress,
    parse_job_options,
)
from worker.llm_client import MockLLMClient
from worker.llm_postprocessor import PostprocessConfig, process_page


def test_compute_progress_bounds_and_monotonicity() -> None:
    for stg in OCR_STAGE_ORDER:
        assert stg in VALID_HEARTBEAT_STAGES

    for n_pages in (0, 1, 2, 5, 12):
        prev = -1
        eff_pages = max(1, n_pages)
        for p_idx in range(eff_pages):
            for stg in OCR_STAGE_ORDER:
                val = compute_progress(p_idx, n_pages, stg)
                assert 5 <= val <= 95, f"Out of [5,95]: {val}"
                assert val >= prev, f"Non-monotonic at n={n_pages}, p={p_idx}, s={stg}: {val} < {prev}"
                prev = val

        if eff_pages >= 2:
            assert compute_progress(eff_pages - 1, n_pages, "LLM_POSTPROCESS") > compute_progress(
                0, n_pages, "PREPROCESSING"
            )

    # Clamping edge cases
    assert 5 <= compute_progress(-3, 4, "PREPROCESSING") <= 95
    assert 5 <= compute_progress(99, 4, "LLM_POSTPROCESS") <= 95
    assert compute_progress(0, 4, "UNKNOWN_STAGE") == 5


def test_parse_job_options_only_reads_three_flags_and_ignores_others() -> None:
    assert parse_job_options(None) == {
        "llmPostprocess": False,
        "localOnly": False,
        "redactPii": False,
    }
    assert parse_job_options({"options": None}) == {
        "llmPostprocess": False,
        "localOnly": False,
        "redactPii": False,
    }
    assert parse_job_options({"options": "not-a-dict"}) == {
        "llmPostprocess": False,
        "localOnly": False,
        "redactPii": False,
    }

    parsed = parse_job_options(
        {
            "jobId": "j-1",
            "options": {
                "llmPostprocess": True,
                "localOnly": "true",
                "redactPii": False,
                "deskew": True,
                "restoreMode": "unet_v5",
                "dpi": 300,
            },
        }
    )
    assert set(parsed.keys()) == {"llmPostprocess", "localOnly", "redactPii"}
    assert parsed == {"llmPostprocess": True, "localOnly": True, "redactPii": False}


def test_reocr_evidence_capped_and_never_emits_standalone_suggestion() -> None:
    layout = {
        "schemaVersion": 1,
        "pageNumber": 1,
        "width": 1000,
        "height": 1200,
        "blocks": [
            {
                "id": "b-1",
                "type": "paragraph",
                "order": 1,
                "bbox": [50, 50, 900, 90],
                "ocrText": "Thi hn np đon đăng ký",
                "text": "Thi hn np đon đăng ký",
                "confidence": 0.72,
            },
            {
                "id": "b-2",
                "type": "paragraph",
                "order": 2,
                "bbox": [50, 100, 900, 140],
                "ocrText": "tang vt vi phąm hành chinh",
                "text": "tang vt vi phąm hành chinh",
                "confidence": 0.70,
            },
            {
                "id": "b-3",
                "type": "paragraph",
                "order": 3,
                "bbox": [50, 150, 900, 190],
                "ocrText": "cùa Đôi QLTT s 4",
                "text": "cùa Đôi QLTT s 4",
                "confidence": 0.75,
            },
            {
                "id": "b-4",
                "type": "paragraph",
                "order": 4,
                "bbox": [50, 200, 900, 240],
                "ocrText": "vè vic xin ý kién phé duyět",
                "text": "vè vic xin ý kién phé duyět",
                "confidence": 0.80,
            },
            {
                "id": "b-5",
                "type": "paragraph",
                "order": 5,
                "bbox": [50, 250, 900, 290],
                "ocrText": "Căn cứ quy định của pháp luật hiện hành về xử lý tài sản công.",
                "text": "Căn cứ quy định của pháp luật hiện hành về xử lý tài sản công.",
                "confidence": 0.97,
            },
            {
                "id": "b-6",
                "type": "paragraph",
                "order": 6,
                "bbox": [50, 300, 900, 340],
                "ocrText": "Ủy ban nhân dân thành phố ban hành kế hoạch kiểm tra thực tế.",
                "text": "Ủy ban nhân dân thành phố ban hành kế hoạch kiểm tra thực tế.",
                "confidence": 0.98,
            },
        ],
    }

    received_batches: list[list[str]] = []

    def fake_reocr(reocr_blocks: list[dict]) -> list[str]:
        received_batches.append([b["id"] for b in reocr_blocks])
        # First differs from ocrText, second equals ocrText
        return [
            "Thời hạn nộp đơn đăng ký",
            reocr_blocks[1]["ocrText"] if len(reocr_blocks) > 1 else "",
        ]

    cfg = PostprocessConfig(
        enabled=True,
        provider="mock",
        mock_mode="normal",
        reocr_enabled=True,
        max_reocr_lines_per_page=2,
    )

    # Confirm default MAX_REOCR_LINES_PER_PAGE stays 15 when env not set
    prev_val = os.environ.pop("MAX_REOCR_LINES_PER_PAGE", None)
    try:
        assert PostprocessConfig.from_env().max_reocr_lines_per_page == 15
        assert PostprocessConfig.from_env().reocr_enabled is False
    finally:
        if prev_val is not None:
            os.environ["MAX_REOCR_LINES_PER_PAGE"] = prev_val

    out = process_page(
        layout,
        options={"llmPostprocess": True},
        reocr_provider=fake_reocr,
        config=cfg,
        llm_client=MockLLMClient(mode="normal"),
    )
    assert len(received_batches) == 1
    assert len(received_batches[0]) == 2
    meta = out["postprocessMeta"]
    assert meta["reocrLines"] == 2
    assert meta["reocrDiffs"] == 1
    assert isinstance(meta["reocrLatencyMs"], int) and meta["reocrLatencyMs"] >= 0
    for blk in out["blocks"]:
        for sug in blk.get("suggestions", []):
            assert sug["source"] != "reocr"

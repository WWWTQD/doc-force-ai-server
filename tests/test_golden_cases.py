"""Regression test suite locking Golden Cases 1-12 (no product code edits allowed in P3-1)."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from worker.llm_client import MockLLMClient
from worker.llm_postprocessor import PostprocessConfig, process_page
from worker.token_guard import ReasonCode, TokenGuard, TokenLocker
from worker.vietnamese_normalizer import normalize_vietnamese_text


def _make_layout(texts: list[str], confs: list[float] | None = None) -> dict[str, Any]:
    confs = confs or [0.85] * len(texts)
    blocks = []
    for i, (txt, conf) in enumerate(zip(texts, confs), start=1):
        blocks.append(
            {
                "id": f"gc-b-{i}",
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


def _cfg(mode: str = "normal") -> PostprocessConfig:
    return PostprocessConfig(
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


@pytest.mark.parametrize(
    "gc_id,text,conf,expected",
    [
        (
            "GC1",
            "Thi hn np đon đăng ký mua tài sán công něu trên, k t ngày 06/01/2021",
            0.95,
            {
                "candidate": True,
                "is_warning": False,
                "source": "llm",
                "risk": "medium",
                "requires_visual": False,
                "reason_codes": ["INVALID_SYLLABLE_FIXED"],
                "expected_text": "Thời hạn nộp đơn đăng ký mua tài sản công nêu trên, kể từ ngày 06/01/2021",
                "llm_calls": 1,
                "vlm_calls": 0,
                "invalid_sub": ["hn", "np", "něu", "k", "t"],
            },
        ),
        (
            "GC2",
            "QLTT ngày 06/01/2021 cùa Đôi QLTT s 4, vè vic xin ý kién phé duyět phuong",
            0.94,
            {
                "candidate": True,
                "is_warning": False,
                "source": "llm",
                "risk": "medium",
                "requires_visual": False,
                "reason_codes": ["INVALID_SYLLABLE_FIXED"],
                "expected_text": "QLTT ngày 06/01/2021 của Đội QLTT số 4, về việc xin ý kiến phê duyệt phương",
                "llm_calls": 1,
                "vlm_calls": 0,
                "invalid_sub": ["s", "vic", "kién", "duyět", "phuong"],
                "exempt_token": "QLTT",
            },
        ),
        (
            "GC3",
            "tang vt vi phąm hành chinh tjch thu sung quý nhà nuóc theo Công văn s: 20/CV-",
            0.88,
            {
                "candidate": True,
                "is_warning": True,
                "source": "vlm",
                "risk": "high",
                "requires_visual": True,
                "reason_codes": ["VLM_DAMAGED_TOKEN_VERIFIED"],
                "expected_text": "tang vật vi phạm hành chính tịch thu sung quỹ nhà nước theo Công văn số: 20/CV-QLTT",
                "llm_calls": 1,
                "vlm_calls": 1,
            },
        ),
        (
            "GC4",
            "Hài Phòng. ngày th tháng 01 năm 2021",
            0.85,
            {
                "candidate": True,
                "is_warning": True,
                "source": "vlm",
                "risk": "high",
                "requires_visual": True,
                "reason_codes": ["VLM_DAMAGED_TOKEN_VERIFIED"],
                "expected_text": "Hải Phòng, ngày 05 tháng 01 năm 2021",
                "llm_calls": 1,
                "vlm_calls": 1,
            },
        ),
        (
            "GC5",
            "ngly .. thíng 01 nim 2021",
            0.82,
            {
                "candidate": True,
                "is_warning": True,
                "source": "llm",
                "risk": "medium",
                "requires_visual": True,
                "reason_codes": ["DAMAGED_PROTECTED_TOKEN_UNRESOLVED"],
                "expected_text": "ngày .. tháng 01 năm 2021",
                "llm_calls": 1,
                "vlm_calls": 1,
            },
        ),
        (
            "GC7a",
            "Công văn 705/cPngey",
            0.78,
            {
                "candidate": True,
                "is_warning": True,
                "source": "vlm",
                "risk": "high",
                "requires_visual": True,
                "reason_codes": ["VLM_DAMAGED_TOKEN_VERIFIED"],
                "expected_text": "Công văn 705/CP-VP",
                "llm_calls": 1,
                "vlm_calls": 1,
            },
        ),
        (
            "GC7b",
            "Công văn 19/p",
            0.78,
            {
                "candidate": True,
                "is_warning": True,
                "source": "vlm",
                "risk": "high",
                "requires_visual": True,
                "reason_codes": ["VLM_DAMAGED_TOKEN_VERIFIED"],
                "expected_text": "Công văn 19/QĐ",
                "llm_calls": 1,
                "vlm_calls": 1,
            },
        ),
        (
            "GC7c",
            "Số: 1027/",
            0.78,
            {
                "candidate": True,
                "is_warning": True,
                "source": "vlm",
                "risk": "high",
                "requires_visual": True,
                "reason_codes": ["VLM_DAMAGED_TOKEN_VERIFIED"],
                "expected_text": "Số: 1027/QĐ-UBND",
                "llm_calls": 1,
                "vlm_calls": 1,
            },
        ),
        (
            "GC8",
            "tang vật tjch thu nhà nuóc",
            0.96,
            {
                "candidate": True,
                "is_warning": False,
                "no_suggestion": True,
                "llm_calls": 1,
                "vlm_calls": 0,
                "invalid_exact": ["tjch", "nuóc"],
            },
        ),
        (
            "GC9",
            "Ủy ban nhân dân thành phố Hải Phòng",
            0.97,
            {
                "candidate": False,
                "is_warning": False,
                "no_suggestion": True,
                "llm_calls": 0,
                "vlm_calls": 0,
            },
        ),
    ],
)
def test_golden_cases_normal_mock(
    gc_id: str, text: str, conf: float, expected: dict[str, Any]
) -> None:
    del gc_id
    layout = _make_layout([text], confs=[conf])
    snap = copy.deepcopy(layout)

    out = process_page(
        layout,
        options={"llmPostprocess": True, "localOnly": True},
        config=_cfg("normal"),
        llm_client=MockLLMClient(mode="normal"),
    )

    meta = out["postprocessMeta"]
    blk = out["blocks"][0]
    assert blk["text"] == snap["blocks"][0]["text"]
    assert blk["confidence"] == snap["blocks"][0]["confidence"]
    assert meta["status"] == "COMPLETED"
    assert meta["provider"] == "mock"
    assert meta["llmCalls"] == expected["llm_calls"]
    assert meta["vlmCalls"] == expected["vlm_calls"]

    if not expected["candidate"]:
        assert not blk.get("postprocess")
        assert not blk.get("isWarning")
        assert not blk.get("suggestions")
        return

    assert blk["postprocess"]["candidate"] is True
    for tok in expected.get("invalid_sub", []):
        assert tok in blk["postprocess"]["invalidSyllables"]
    if "invalid_exact" in expected:
        assert blk["postprocess"]["invalidSyllables"] == expected["invalid_exact"]
    if "exempt_token" in expected:
        assert expected["exempt_token"] not in blk["postprocess"]["invalidSyllables"]

    if expected.get("no_suggestion"):
        assert not blk.get("suggestions")
        assert bool(blk.get("isWarning")) is expected["is_warning"]
        return

    assert bool(blk.get("isWarning")) is expected["is_warning"]
    sugs = blk.get("suggestions") or []
    assert len(sugs) == 1
    sug = sugs[0]
    assert sug["source"] == expected["source"]
    assert sug["provider"] == "mock"
    assert sug["risk"] == expected["risk"]
    assert sug["requiresVisualCheck"] is expected["requires_visual"]
    assert sug["reasonCodes"] == expected["reason_codes"]
    assert sug["text"] == expected["expected_text"]
    assert sug["status"] == "pending"
    assert sug["ops"]


def test_golden_case_6_protected_identifiers_and_guard() -> None:
    line = (
        "Liên hệ a@doc-forge.vn hoặc https://doc-forge.vn/hd "
        "số 14/2024/HĐ-DOCFORCE ngày 06/01/2021"
    )
    assert normalize_vietnamese_text(line) == line

    locked, toks = TokenLocker.lock(line)
    tok_texts = [t.text for t in toks]
    assert "a@doc-forge.vn" in tok_texts
    assert "https://doc-forge.vn/hd" in tok_texts
    assert "14/2024/HĐ-DOCFORCE" in tok_texts
    assert "06/01/2021" in tok_texts

    # Identical locked response -> valid (unchanged=True), no spurious error
    v_ok = TokenGuard.check_line(line, locked, toks, locked, 0.98)
    assert v_ok.accepted is False
    assert v_ok.unchanged is True
    assert v_ok.restored_text == line
    assert v_ok.reason_codes == ()

    # Mutating / gluing characters onto ⟦T1⟧ -> rejected deterministically
    glued = locked.replace("⟦T1⟧", "⟦T1⟧vào", 1)
    v_bad = TokenGuard.check_line(line, locked, toks, glued, 0.99)
    assert v_bad.accepted is False
    assert ReasonCode.REJECT_PROTECTED_TOKEN_MUTATED in v_bad.reason_codes


def test_golden_case_9_noisy_page_gate() -> None:
    noisy = _make_layout(["xxq yyq zzq wwq vvq nnq mmq ppq"], confs=[0.50])
    out = process_page(
        noisy,
        options={"llmPostprocess": True},
        config=_cfg("normal"),
        llm_client=MockLLMClient(mode="normal"),
    )
    assert out["postprocessMeta"]["status"] == "SKIPPED"
    assert out["postprocessMeta"]["skippedReason"] == "PAGE_TOO_NOISY_OR_NOT_VIETNAMESE"


@pytest.mark.parametrize(
    "gc_id,adv_step,expected_codes",
    [
        ("GC10", 0, {"REJECT_PLACEHOLDER_ALTERED"}),
        ("GC11", 1, {"REJECT_LINE_COUNT_MISMATCH"}),
        (
            "GC12",
            2,
            {
                "REJECT_LENGTH_RATIO_OUT_OF_BOUNDS",
                "REJECT_WORD_INSERTION_OR_DELETION",
                "REJECT_EXCESSIVE_CHAR_EDIT",
            },
        ),
        ("GC12_schema", 3, {"REJECT_SCHEMA_INVALID"}),
    ],
)
def test_golden_cases_10_to_12_adversarial_mock(
    gc_id: str, adv_step: int, expected_codes: set[str]
) -> None:
    del gc_id
    layout = _make_layout(
        [
            "Thi hn np đon đăng ký mua tài sán công něu trên, k t ngày 06/01/2021",
            "QLTT ngày 06/01/2021 cùa Đôi QLTT s 4, vè vic xin ý kién phé duyět phuong",
        ],
        confs=[0.85, 0.85],
    )
    mock = MockLLMClient(mode="adversarial", adversarial_step=adv_step)
    out = process_page(
        layout,
        options={"llmPostprocess": True},
        config=_cfg("adversarial"),
        llm_client=mock,
    )
    meta = out["postprocessMeta"]
    assert meta["status"] == "COMPLETED"
    assert meta["suggestionsCreated"] == 0
    for blk in out["blocks"]:
        assert not blk.get("suggestions")
    for code in expected_codes:
        assert code in meta["rejections"]

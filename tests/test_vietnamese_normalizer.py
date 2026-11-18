import pytest

from worker.vietnamese_normalizer import normalize_vietnamese_text


@pytest.mark.parametrize(
    "sample",
    [
        "Số: 123/QĐ-UBND ngày 13-08-1945",
        "14/2024/HĐ-DOCFORCE",
        "a@doc-forge.vn",
        "https://doc-forge.vn:8080/tai-lieu/2024-01",
    ],
)
def test_preserves_protected_tokens_and_internal_hyphens(monkeypatch, sample: str):
    monkeypatch.delenv("LEGACY_DOMAIN_RULES", raising=False)
    out = normalize_vietnamese_text(sample)
    assert out == sample
    assert normalize_vietnamese_text(out) == out


def test_bullet_hyphen_normalized_only_at_line_start(monkeypatch):
    monkeypatch.delenv("LEGACY_DOMAIN_RULES", raising=False)
    assert normalize_vietnamese_text("-abc") == "- abc"
    assert normalize_vietnamese_text("-   abc") == "- abc"
    assert normalize_vietnamese_text("-13-08-1945") == "- 13-08-1945"


def test_legacy_domain_rules_gated_at_call_time(monkeypatch):
    sample = "sán xuất và giai cáp theo Mác~Lênin"
    monkeypatch.delenv("LEGACY_DOMAIN_RULES", raising=False)
    assert normalize_vietnamese_text(sample) == sample

    monkeypatch.setenv("LEGACY_DOMAIN_RULES", "false")
    assert normalize_vietnamese_text(sample) == sample

    monkeypatch.setenv("LEGACY_DOMAIN_RULES", "true")
    assert normalize_vietnamese_text(sample) == "sản xuất và giai cấp theo Mác - Lênin"
    assert normalize_vietnamese_text("INTERMANIANCES") == ""

    monkeypatch.setenv("LEGACY_DOMAIN_RULES", "false")
    assert normalize_vietnamese_text("INTERMANIANCES") == "INTERMANIANCES"


@pytest.mark.parametrize(
    "sample",
    [
        "Số: 123/QĐ-UBND ngày 13-08-1945",
        "14/2024/HĐ-DOCFORCE",
        "a@doc-forge.vn",
        "-abc",
        "1.2.Nội dung ( thử ) , kiểm tra .",
        "1, Mục đầu tiên",
        "",
        "   ",
    ],
)
def test_normalize_is_idempotent(monkeypatch, sample: str):
    monkeypatch.delenv("LEGACY_DOMAIN_RULES", raising=False)
    once = normalize_vietnamese_text(sample)
    twice = normalize_vietnamese_text(once)
    assert twice == once

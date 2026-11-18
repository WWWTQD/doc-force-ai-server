import logging
import sys

from worker.ocr_confidence import (
    DEFAULT_FALLBACK_CONFIDENCE,
    clamp_confidence,
    format_conf_summary,
    pair_vietocr_predictions,
    prepare_block_text,
    recognize_with_probs,
    reset_return_prob_warning,
)
from worker.vietnamese_normalizer import normalize_vietnamese_text


class _StubRecognizerWithProb:
    def predict_batch(self, pieces, return_prob=False):
        assert return_prob is True
        return ["a", "b"], [0.8, 0.5]


class _StubRecognizerLegacySignature:
    def __init__(self):
        self.calls = 0

    def predict_batch(self, pieces):
        self.calls += 1
        return [f"line-{i}" for i in range(len(pieces))]


class _StubRecognizerReturnsListOnly:
    def predict_batch(self, pieces, return_prob=False):
        return ["only-text"]


def test_stub_recognizer_preserves_order_and_probs():
    polys = [
        [[0, 0], [10, 0], [10, 10], [0, 10]],
        [[0, 20], [10, 20], [10, 30], [0, 30]],
    ]
    texts, probs = recognize_with_probs(_StubRecognizerWithProb(), ["p1", "p2"])
    paired = pair_vietocr_predictions(polys, texts, probs)

    assert len(paired) == 2
    assert paired[0][0] == polys[0]
    assert paired[0][1] == ("a", 0.8)
    assert paired[1][0] == polys[1]
    assert paired[1][1] == ("b", 0.5)


def test_clamp_confidence_casts_and_bounds():
    assert clamp_confidence(-0.2) == 0.0
    assert clamp_confidence(1.5) == 1.0
    assert clamp_confidence("0.625") == 0.625
    assert clamp_confidence(0) == 0.0
    assert clamp_confidence(1) == 1.0
    assert clamp_confidence(float("nan")) == DEFAULT_FALLBACK_CONFIDENCE
    assert clamp_confidence(float("inf")) == DEFAULT_FALLBACK_CONFIDENCE
    assert clamp_confidence(None) == DEFAULT_FALLBACK_CONFIDENCE
    assert clamp_confidence(True) == DEFAULT_FALLBACK_CONFIDENCE


def test_return_prob_unsupported_logs_warning_once_and_falls_back(caplog):
    reset_return_prob_warning()
    stub = _StubRecognizerLegacySignature()
    test_logger = logging.getLogger("test_ocr_confidence_legacy")

    with caplog.at_level(logging.WARNING, logger="test_ocr_confidence_legacy"):
        texts1, probs1 = recognize_with_probs(stub, ["p1", "p2"], logger=test_logger)
        texts2, probs2 = recognize_with_probs(stub, ["p3"], logger=test_logger)

    assert texts1 == ["line-0", "line-1"]
    assert probs1 == [0.95, 0.95]
    assert texts2 == ["line-0"]
    assert probs2 == [0.95]
    assert stub.calls == 2
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_records) == 1


def test_return_prob_non_tuple_logs_warning_once_and_falls_back(caplog):
    reset_return_prob_warning()
    stub = _StubRecognizerReturnsListOnly()
    test_logger = logging.getLogger("test_ocr_confidence_nontuple")

    with caplog.at_level(logging.WARNING, logger="test_ocr_confidence_nontuple"):
        texts, probs = recognize_with_probs(stub, ["p1"], logger=test_logger)

    assert texts == ["only-text"]
    assert probs == [0.95]
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_prepare_block_text_preserves_raw_ocr_text(monkeypatch):
    monkeypatch.delenv("LEGACY_DOMAIN_RULES", raising=False)

    raw1, norm1 = prepare_block_text("  -abc  ", normalize_vietnamese_text)
    assert raw1 == "-abc"
    assert norm1 == "- abc"

    raw2, norm2 = prepare_block_text("1.2.Nội dung ( thử ) , test .", normalize_vietnamese_text)
    assert raw2 == "1.2.Nội dung ( thử ) , test ."
    assert norm2 == "1.2. Nội dung (thử), test."


def test_format_conf_summary_and_no_heavy_imports():
    assert format_conf_summary([]) == "VietOCR conf: n=0 p5=0.000 p50=0.000 p95=0.000"
    summary = format_conf_summary([0.5, 0.8, 0.95, 1.2])
    assert summary.startswith("VietOCR conf: n=4 p5=")
    assert "p50=" in summary and "p95=" in summary

    for mod in ("torch", "paddle", "paddleocr", "vietocr", "ultralytics"):
        assert mod not in sys.modules

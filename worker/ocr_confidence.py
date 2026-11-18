"""Pure helpers for VietOCR line-probability extraction, clamping, and summary logging."""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Sequence

LOG = logging.getLogger("ocr_engine")

DEFAULT_FALLBACK_CONFIDENCE = 0.95
_warned_no_return_prob = False


def reset_return_prob_warning() -> None:
    """Reset one-time warning flag (for isolated unit tests)."""
    global _warned_no_return_prob
    _warned_no_return_prob = False


def _warn_once(logger: logging.Logger, msg: str, *args: Any) -> None:
    global _warned_no_return_prob
    if not _warned_no_return_prob:
        _warned_no_return_prob = True
        logger.warning(msg, *args)


def clamp_confidence(value: Any, default: float = DEFAULT_FALLBACK_CONFIDENCE) -> float:
    """Convert confidence value to finite float clamped within [0.0, 1.0]."""
    if value is None or isinstance(value, bool):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


def recognize_with_probs(
    recognizer: Any,
    pieces: Sequence[Any],
    logger: logging.Logger | None = None,
) -> tuple[list[str], list[float]]:
    """Call recognizer.predict_batch(..., return_prob=True) with one-time WARNING fallback."""
    active_logger = logger or LOG
    try:
        result = recognizer.predict_batch(pieces, return_prob=True)
    except TypeError as exc:
        err_msg = str(exc)
        if "return_prob" not in err_msg and "keyword argument" not in err_msg and "positional" not in err_msg:
            raise
        _warn_once(
            active_logger,
            "VietOCR recognizer khong ho tro return_prob=True (%s); dung confidence mac dinh %.2f.",
            exc,
            DEFAULT_FALLBACK_CONFIDENCE,
        )
        texts = list(recognizer.predict_batch(pieces))
        return [("" if t is None else str(t)) for t in texts], [DEFAULT_FALLBACK_CONFIDENCE] * len(texts)

    if isinstance(result, tuple) and len(result) == 2:
        raw_texts, raw_probs = result
        texts_list = [("" if t is None else str(t)) for t in raw_texts]
        if raw_probs is not None and not isinstance(raw_probs, (str, bytes)):
            probs_seq = list(raw_probs)
            if len(probs_seq) == len(texts_list):
                return texts_list, [clamp_confidence(p) for p in probs_seq]
        _warn_once(
            active_logger,
            "VietOCR predict_batch(..., return_prob=True) tra ve probs khong hop le; dung confidence mac dinh %.2f.",
            DEFAULT_FALLBACK_CONFIDENCE,
        )
        return texts_list, [DEFAULT_FALLBACK_CONFIDENCE] * len(texts_list)

    _warn_once(
        active_logger,
        "VietOCR predict_batch(..., return_prob=True) khong tra ve (texts, probs); dung confidence mac dinh %.2f.",
        DEFAULT_FALLBACK_CONFIDENCE,
    )
    texts_list = [("" if t is None else str(t)) for t in result]
    return texts_list, [DEFAULT_FALLBACK_CONFIDENCE] * len(texts_list)


def pair_vietocr_predictions(
    polys: Sequence[Any],
    texts: Sequence[Any],
    probs: Sequence[Any] | None = None,
) -> list[list[Any]]:
    """Pair polygons with cleaned OCR text and clamped line probability, preserving input order."""
    out: list[list[Any]] = []
    if probs is None:
        prob_iter: Sequence[Any] = [DEFAULT_FALLBACK_CONFIDENCE] * len(texts)
    else:
        prob_iter = probs

    for poly, text, prob in zip(polys, texts, prob_iter):
        cleaned = ("" if text is None else str(text)).strip()
        if cleaned:
            poly_list = poly.tolist() if hasattr(poly, "tolist") else list(poly)
            out.append([poly_list, (cleaned, clamp_confidence(prob))])
    return out


def prepare_block_text(raw_text: str, normalizer: Callable[[str], str]) -> tuple[str, str]:
    """Return (raw_ocr_text, normalized_text) keeping pre-normalizer ocrText intact."""
    raw_clean = (raw_text or "").strip()
    if not raw_clean:
        return "", ""
    normalized = (normalizer(raw_clean) or "").strip()
    return raw_clean, normalized


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    idx = (q / 100.0) * (len(sorted_vals) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = idx - lo
    return float(sorted_vals[lo]) * (1.0 - frac) + float(sorted_vals[hi]) * frac


def format_conf_summary(confs: Sequence[Any]) -> str:
    """Format one-line page summary: VietOCR conf: n=.. p5=.. p50=.. p95=.."""
    vals = sorted(clamp_confidence(c) for c in confs)
    n = len(vals)
    p5 = _percentile(vals, 5)
    p50 = _percentile(vals, 50)
    p95 = _percentile(vals, 95)
    return f"VietOCR conf: n={n} p5={p5:.3f} p50={p50:.3f} p95={p95:.3f}"

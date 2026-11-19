"""Best-effort LLM/VLM postprocessor orchestrator guarded by 3-layer token protection."""

from __future__ import annotations

import copy
import difflib
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

try:
    from llm_client import LLMClient, build_llm_client
    from token_guard import (
        LockedToken,
        PageTokenCounter,
        ReasonCode,
        TokenGuard,
        TokenLocker,
        compute_ops,
        detect_protected_spans,
        has_placeholder_collision,
        sanitize_reason,
    )
    from vietnamese_syllable_validator import invalid_syllables
except ImportError:
    from worker.llm_client import LLMClient, build_llm_client
    from worker.token_guard import (
        LockedToken,
        PageTokenCounter,
        ReasonCode,
        TokenGuard,
        TokenLocker,
        compute_ops,
        detect_protected_spans,
        has_placeholder_collision,
        sanitize_reason,
    )
    from worker.vietnamese_syllable_validator import invalid_syllables

LOG = logging.getLogger("llm_postprocessor")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class PostprocessConfig:
    enabled: bool = False
    provider: str = "mock"
    mock_mode: str = "normal"
    line_conf_threshold: float = 0.90  # CHƯA HIỆU CHỈNH
    max_candidates_per_page: int = 30
    max_invalid_token_ratio: float = 0.5  # CHƯA HIỆU CHỈNH
    reocr_enabled: bool = False
    max_reocr_lines_per_page: int = 15
    allow_common_loanwords: bool = True
    page_deadline_seconds: float = 90.0
    max_vlm_crops: int = 0
    text_model: str = "gemini-3.5-flash-lite"
    vision_model: str = "gemini-3.5-flash-lite"
    fallback_model: str | None = (
        "gemini-3.1-flash-lite,gemini-3.5-flash,gemini-2.5-flash,gemini-2.5-flash-lite,gemini-2.0-flash"
    )
    fallback_on_unavailable: bool = True
    thinking_level: str = "minimal"
    rpm_limit: int = 5
    daily_request_budget: int = 150
    cloud_consent: bool = False
    cloud_tier: str = "free"
    request_timeout_seconds: float = 28.0
    chunk_trigger_lines: int = 8
    chunk_trigger_chars: int = 900
    chunk_max_lines: int = 7
    chunk_max_chars: int = 1400

    @classmethod
    def from_env(cls) -> PostprocessConfig:
        default_model = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite").strip() or "gemini-3.5-flash-lite"
        text_model = os.getenv("LLM_TEXT_MODEL", default_model).strip() or default_model
        vision_model = os.getenv("LLM_VISION_MODEL", os.getenv("LLM_TEXT_MODEL", default_model)).strip() or default_model
        raw_fb = os.getenv(
            "LLM_FALLBACK_MODEL",
            "gemini-3.1-flash-lite,gemini-3.5-flash,gemini-2.5-flash,gemini-2.5-flash-lite,gemini-2.0-flash",
        ).strip()
        tier = os.getenv("LLM_CLOUD_TIER", "free").strip().lower()
        return cls(
            enabled=_env_bool(
                "LLM_POSTPROCESS_ENABLED",
                _env_bool("ENABLE_LLM_POSTPROCESS", _env_bool("LLM_POSTPROCESS_DEFAULT", False)),
            ),
            provider=os.getenv("LLM_PROVIDER", "mock").strip().lower(),
            mock_mode=os.getenv("LLM_MOCK_MODE", "normal").strip().lower(),
            line_conf_threshold=_env_float("LLM_LINE_CONF_THRESHOLD", 0.90),
            max_candidates_per_page=max(1, _env_int("LLM_MAX_CANDIDATES_PER_PAGE", 30)),
            max_invalid_token_ratio=_env_float("LLM_MAX_INVALID_TOKEN_RATIO", 0.5),
            reocr_enabled=_env_bool("LLM_REOCR_ENABLED", False),
            max_reocr_lines_per_page=max(0, _env_int("MAX_REOCR_LINES_PER_PAGE", 15)),
            allow_common_loanwords=_env_bool("ALLOW_COMMON_LOANWORDS", True),
            page_deadline_seconds=max(0.1, _env_float("LLM_PAGE_DEADLINE_SECONDS", 90.0)),
            max_vlm_crops=max(0, _env_int("LLM_MAX_VLM_CROPS", 0)),
            text_model=text_model,
            vision_model=vision_model,
            fallback_model=raw_fb or None,
            fallback_on_unavailable=_env_bool("LLM_FALLBACK_ON_UNAVAILABLE", True),
            thinking_level=os.getenv("LLM_THINKING_LEVEL", "minimal").strip().lower() or "minimal",
            rpm_limit=_env_int("LLM_RPM_LIMIT", 5),
            daily_request_budget=max(0, _env_int("LLM_DAILY_REQUEST_BUDGET", 150)),
            cloud_consent=_env_bool("LLM_CLOUD_CONSENT", False),
            cloud_tier=tier if tier in ("free", "paid") else "free",
            request_timeout_seconds=max(0.5, _env_float("LLM_REQUEST_TIMEOUT_SECONDS", 28.0)),
            chunk_trigger_lines=max(1, _env_int("LLM_CHUNK_TRIGGER_LINES", 8)),
            chunk_trigger_chars=max(1, _env_int("LLM_CHUNK_TRIGGER_CHARS", 900)),
            chunk_max_lines=max(1, _env_int("LLM_CHUNK_MAX_LINES", 7)),
            chunk_max_chars=max(1, _env_int("LLM_CHUNK_MAX_CHARS", 1400)),
        )


def crop_line(image: Any, bbox: Any, pad: int = 6, max_side: int = 960) -> Any | None:
    """Safely crop a line bounding box [x1, y1, x2, y2] from BGR numpy image."""
    if image is None or not bbox or len(bbox) != 4:
        return None
    import cv2  # lazy import so text-only tests never load OpenCV

    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return None
    x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(w, x2 + pad)
    cy2 = min(h, y2 + pad)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    crop = image[cy1:cy2, cx1:cx2]
    ch, cw = crop.shape[:2]
    longest = max(ch, cw)
    if max_side > 0 and longest > max_side:
        scale = max_side / float(longest)
        new_w = max(1, int(round(cw * scale)))
        new_h = max(1, int(round(ch * scale)))
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return crop


def prepare_vlm_crop_jpeg(
    crop_bytes: bytes, *, max_side: int = 1024, quality: int = 90
) -> bytes:
    """Normalize crop image bytes to baseline JPEG (<= max_side px, quality 90, no metadata)."""
    if not crop_bytes:
        return crop_bytes
    try:
        import io
        from PIL import Image, ImageOps
    except ImportError:
        return crop_bytes

    try:
        with Image.open(io.BytesIO(crop_bytes)) as img:
            img.load()
            has_meta = bool(
                img.info.get("exif")
                or img.info.get("xmp")
                or img.info.get("icc_profile")
            )
            longest = max(img.size)
            if (
                img.format == "JPEG"
                and (max_side <= 0 or longest <= max_side)
                and not has_meta
                and img.mode in ("RGB", "L")
            ):
                return crop_bytes
            img_clean = ImageOps.exif_transpose(img)
            if img_clean.mode != "RGB":
                img_clean = img_clean.convert("RGB")
            w, h = img_clean.size
            if max_side > 0 and max(w, h) > max_side:
                scale = max_side / float(max(w, h))
                new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
                resampling = getattr(Image, "Resampling", Image).LANCZOS
                img_clean = img_clean.resize(new_size, resampling)
            stripped = Image.new("RGB", img_clean.size)
            stripped.paste(img_clean)
            buf = io.BytesIO()
            stripped.save(buf, format="JPEG", quality=quality)
            return buf.getvalue()
    except Exception:
        return crop_bytes


def _provider_label(client: LLMClient | None) -> str:
    if client is None:
        return "none"
    pname = getattr(client, "provider_name", None)
    if pname:
        return str(pname)
    return "cloud" if bool(getattr(client, "is_cloud", False)) else "local"


_SKIP_USAGE_STATUSES = {
    "CLOUD_NOT_CONSENTED",
    "LOCAL_ENDPOINT_UNAVAILABLE",
    "RATE_LIMITED",
    "BREAKER_OPEN",
    "DAILY_BUDGET_EXHAUSTED",
    "AUTH_ERROR",
    "MODEL_UNAVAILABLE",
}


def _meta(
    *,
    status: str,
    skipped_reason: str | None = None,
    candidate_lines: int = 0,
    suggestions_created: int = 0,
    rejections: dict[str, int] | None = None,
    provider: str = "none",
    llm_calls: int = 0,
    vlm_calls: int = 0,
    latency_ms: int = 0,
    llm_model: str | None = None,
    vlm_model: str | None = None,
    cloud_tier: str | None = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
    tokens_thought: int = 0,
    rate_limited: bool = False,
    reocr_latency_ms: int | None = None,
    reocr_lines: int | None = None,
    reocr_diffs: int | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "status": status,
        "skippedReason": skipped_reason,
        "candidateLines": candidate_lines,
        "suggestionsCreated": suggestions_created,
        "rejections": rejections or {},
        "provider": provider,
        "llmModel": llm_model,
        "vlmModel": vlm_model,
        "cloudTier": cloud_tier,
        "llmCalls": llm_calls,
        "vlmCalls": vlm_calls,
        "tokens": {
            "in": tokens_in,
            "out": tokens_out,
            "thought": tokens_thought,
        },
        "rateLimited": rate_limited,
        "latencyMs": latency_ms,
    }
    if reocr_latency_ms is not None:
        data["reocrLatencyMs"] = reocr_latency_ms
    if reocr_lines is not None:
        data["reocrLines"] = reocr_lines
    if reocr_diffs is not None:
        data["reocrDiffs"] = reocr_diffs
    return data


def _inc_rejection(rejections: dict[str, int], code: str | ReasonCode) -> None:
    key = code.value if isinstance(code, ReasonCode) else str(code)
    rejections[key] = rejections.get(key, 0) + 1


def _attach_reasons(ops: list[dict[str, Any]], llm_changes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reason_by_pair: dict[tuple[str, str], str] = {}
    for ch in llm_changes:
        r = sanitize_reason(ch.get("reason"))
        if r:
            reason_by_pair[(str(ch.get("from", "")), str(ch.get("to", "")))] = r
    enriched: list[dict[str, Any]] = []
    for op in ops:
        item = dict(op)
        matched = reason_by_pair.get((str(op.get("from", "")), str(op.get("to", ""))))
        if matched:
            item["reason"] = matched
        enriched.append(item)
    return enriched


def _partition_candidate_chunks(
    candidate_indices: list[int],
    lengths_by_idx: dict[int, int],
    *,
    trigger_lines: int = 8,
    trigger_chars: int = 900,
    target_lines_per_chunk: int = 7,
    target_chars_per_chunk: int = 1400,
) -> list[list[int]]:
    """Split candidate indices into balanced 6-8 line chunks when a page is long or text-heavy."""
    n = len(candidate_indices)
    if n <= 0:
        return []
    if n == 1:
        return [candidate_indices]
    total_chars = sum(lengths_by_idx.get(idx, 0) for idx in candidate_indices)
    if n <= trigger_lines and total_chars <= trigger_chars:
        return [candidate_indices]

    chunks_by_lines = math.ceil(n / max(1, target_lines_per_chunk))
    chunks_by_chars = (
        math.ceil(total_chars / max(1, target_chars_per_chunk))
        if total_chars > trigger_chars
        else 1
    )
    num_chunks = min(n, max(2, chunks_by_lines, chunks_by_chars))
    base, rem = divmod(n, num_chunks)
    chunks: list[list[int]] = []
    cursor = 0
    for k in range(num_chunks):
        size = base + (1 if k < rem else 0)
        chunks.append(candidate_indices[cursor : cursor + size])
        cursor += size
    return chunks


def _normalize_auto_apply_batch(
    parsed_resp: Any,
    expected_ids: list[str],
    locked_state: dict[str, tuple[dict[str, Any], str, str, list[LockedToken]]],
) -> Any:
    if not (isinstance(parsed_resp, dict) and isinstance(parsed_resp.get("lines"), list)):
        return parsed_resp
    expected_set = set(expected_ids)
    norm_lines: list[dict[str, Any]] = []
    seen_lids: set[str] = set()
    for raw_it in parsed_resp["lines"]:
        if not isinstance(raw_it, dict):
            continue
        lid = raw_it.get("lineId")
        if not isinstance(lid, str) or lid not in expected_set or lid in seen_lids:
            continue
        seen_lids.add(lid)
        corr_val = raw_it.get("corrected")
        if isinstance(corr_val, str) and ("\n" in corr_val or "\r" in corr_val):
            raw_it = {**raw_it, "corrected": " ".join(corr_val.splitlines()).strip()}
        norm_lines.append(raw_it)
    if seen_lids and len(seen_lids) < len(expected_ids):
        for lid in expected_ids:
            if lid not in seen_lids and lid in locked_state:
                norm_lines.append(
                    {
                        "lineId": lid,
                        "corrected": locked_state[lid][2],
                        "confidence": 1.0,
                        "changes": [],
                    }
                )
    return {**parsed_resp, "lines": norm_lines}


def _validate_vlm_batch(
    response: Any,
    expected_line_ids: list[str],
) -> tuple[bool, ReasonCode | None, dict[str, dict[str, Any]]]:
    """Validate batched VLM JSON schema and exact lineId set contract (C7)."""
    if not isinstance(response, dict) or "lines" not in response:
        return False, ReasonCode.REJECT_SCHEMA_INVALID, {}
    if not set(response.keys()).issubset({"lines"}):
        return False, ReasonCode.REJECT_SCHEMA_INVALID, {}

    raw_lines = response.get("lines")
    if not isinstance(raw_lines, list):
        return False, ReasonCode.REJECT_SCHEMA_INVALID, {}

    parsed: dict[str, dict[str, Any]] = {}
    req_keys = {"lineId", "transcription", "confidence", "uncertain", "uncertainSpans", "changes"}
    for item in raw_lines:
        if not isinstance(item, dict):
            return False, ReasonCode.REJECT_SCHEMA_INVALID, {}
        if not req_keys.issubset(item.keys()) or not set(item.keys()).issubset(req_keys):
            return False, ReasonCode.REJECT_SCHEMA_INVALID, {}
        line_id = item.get("lineId")
        tr = item.get("transcription")
        conf = item.get("confidence")
        unc = item.get("uncertain")
        spans = item.get("uncertainSpans")
        changes = item.get("changes")
        if not isinstance(line_id, str) or not line_id:
            return False, ReasonCode.REJECT_SCHEMA_INVALID, {}
        if not isinstance(tr, str) or not tr.strip():
            return False, ReasonCode.REJECT_SCHEMA_INVALID, {}
        if "\n" in tr or "\r" in tr:
            return False, ReasonCode.REJECT_LINE_COUNT_MISMATCH, {}
        if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
            return False, ReasonCode.REJECT_SCHEMA_INVALID, {}
        if not isinstance(unc, bool):
            return False, ReasonCode.REJECT_SCHEMA_INVALID, {}
        if not isinstance(spans, list) or not all(isinstance(s, str) for s in spans):
            return False, ReasonCode.REJECT_SCHEMA_INVALID, {}
        if not isinstance(changes, list):
            return False, ReasonCode.REJECT_SCHEMA_INVALID, {}
        for ch in changes:
            if not isinstance(ch, dict):
                return False, ReasonCode.REJECT_SCHEMA_INVALID, {}
            ch_keys = {"from", "to", "reason"}
            if not ch_keys.issubset(ch.keys()) or not set(ch.keys()).issubset(ch_keys):
                return False, ReasonCode.REJECT_SCHEMA_INVALID, {}
            if not all(isinstance(ch.get(k), str) for k in ch_keys):
                return False, ReasonCode.REJECT_SCHEMA_INVALID, {}
        if line_id in parsed:
            return False, ReasonCode.REJECT_LINE_COUNT_MISMATCH, {}
        parsed[line_id] = item

    if len(raw_lines) != len(expected_line_ids) or set(parsed.keys()) != set(expected_line_ids):
        return False, ReasonCode.REJECT_LINE_COUNT_MISMATCH, {}
    return True, None, parsed


def _validate_vlm_transcription(raw_text: str, vlm_text: str) -> ReasonCode | None:
    orig_len = max(1, len(raw_text))
    if orig_len < 20:
        if abs(len(vlm_text) - len(raw_text)) > 20:
            return ReasonCode.REJECT_LENGTH_RATIO_OUT_OF_BOUNDS
    else:
        ratio = len(vlm_text) / orig_len
        if ratio < 0.60 or ratio > 1.60:
            return ReasonCode.REJECT_LENGTH_RATIO_OUT_OF_BOUNDS

    # Non-damaged protected tokens in raw_text must remain intact in vlm_text
    orig_clean_prot = [(s[2], s[3]) for s in detect_protected_spans(raw_text) if not s[4]]
    vlm_prot_set = {(s[2], s[3]) for s in detect_protected_spans(vlm_text)}
    for item in orig_clean_prot:
        if item not in vlm_prot_set:
            return ReasonCode.REJECT_PROTECTED_TOKEN_MUTATED

    # Disallow unaligned extra word insertions/deletions in VLM output
    orig_toks = raw_text.split()
    vlm_toks = vlm_text.split()
    matcher = difflib.SequenceMatcher(a=orig_toks, b=vlm_toks, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("insert", "delete"):
            return ReasonCode.REJECT_WORD_INSERTION_OR_DELETION
        if tag == "replace" and (i2 - i1) != (j2 - j1) and (i2 - i1, j2 - j1) not in ((1, 2), (2, 1)):
            return ReasonCode.REJECT_WORD_INSERTION_OR_DELETION
    return None


def process_page(
    layout: dict[str, Any],
    options: dict[str, Any] | None = None,
    crop_provider: Callable[[dict[str, Any]], bytes | None] | None = None,
    reocr_provider: Callable[[list[dict[str, Any]]], list[str]] | None = None,
    *,
    config: PostprocessConfig | None = None,
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    """Run optional LLM_POSTPROCESS stage on a single page layout (never mutating block.text/confidence).

    When the stage is disabled (globally or per-job), returns `layout` completely unmodified (B11).
    """
    cfg = config or PostprocessConfig.from_env()
    opts = options or {}
    llm_flag = opts.get("llmPostprocess") if "llmPostprocess" in opts else opts.get("useLlm", False)
    if not cfg.enabled or not bool(llm_flag):
        return layout

    auto_apply = bool(opts.get("autoApply", False))
    t0 = time.monotonic()
    deadline_ts = t0 + cfg.page_deadline_seconds
    local_only = bool(opts.get("localOnly", False))
    redact_pii = bool(opts.get("redactPii", False))

    rejections: dict[str, int] = {}
    llm_calls = 0
    vlm_calls = 0
    suggestions_created = 0
    cand_count = 0
    provider_label = cfg.provider if cfg.provider == "gemini" else "none"
    llm_model: str | None = cfg.text_model if cfg.provider == "gemini" else None
    vlm_model: str | None = cfg.vision_model if cfg.provider == "gemini" else None
    cloud_tier: str | None = cfg.cloud_tier if cfg.provider == "gemini" else None
    tokens_in = 0
    tokens_out = 0
    tokens_thought = 0
    rate_limited = False
    reocr_latency_ms: int | None = None
    reocr_lines: int | None = None
    reocr_diffs: int | None = None

    def _accumulate_usage(usage: Any) -> None:
        nonlocal tokens_in, tokens_out, tokens_thought, rate_limited
        if usage is None:
            return
        tokens_in += int(getattr(usage, "prompt_tokens", 0) or 0)
        tokens_out += int(getattr(usage, "completion_tokens", 0) or 0)
        tokens_thought += int(getattr(usage, "thought_tokens", 0) or 0)
        if getattr(usage, "rate_limited", False) or getattr(usage, "status", None) in (
            "RATE_LIMITED",
            "BREAKER_OPEN",
        ):
            rate_limited = True

    def _build_meta(
        status: str,
        skipped_reason: str | None = None,
        *,
        suggestions_override: int | None = None,
    ) -> dict[str, Any]:
        return _meta(
            status=status,
            skipped_reason=skipped_reason,
            candidate_lines=cand_count,
            suggestions_created=(
                suggestions_created if suggestions_override is None else suggestions_override
            ),
            rejections=rejections,
            provider=provider_label,
            llm_calls=llm_calls,
            vlm_calls=vlm_calls,
            latency_ms=int((time.monotonic() - t0) * 1000),
            llm_model=llm_model,
            vlm_model=vlm_model,
            cloud_tier=cloud_tier,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tokens_thought=tokens_thought,
            rate_limited=rate_limited,
            reocr_latency_ms=reocr_latency_ms,
            reocr_lines=reocr_lines,
            reocr_diffs=reocr_diffs,
        )

    try:
        out_layout = copy.deepcopy(layout)
        blocks = out_layout.get("blocks")
        if not isinstance(blocks, list):
            out_layout["postprocessMeta"] = _build_meta("COMPLETED")
            return out_layout

        client = llm_client
        skip_reason: str | None = None
        if client is None:
            client, skip_reason = build_llm_client(cfg, opts)
        if client is None:
            reason_code = skip_reason or "PROVIDER_NOT_IMPLEMENTED"
            if reason_code in ("RATE_LIMITED", "BREAKER_OPEN"):
                rate_limited = True
            out_layout["postprocessMeta"] = _build_meta("SKIPPED", reason_code)
            return out_layout

        provider_label = _provider_label(client)
        llm_model = getattr(client, "text_model_name", getattr(client, "model_name", llm_model))
        vlm_model = getattr(client, "vision_model_name", vlm_model)
        if getattr(client, "cloud_tier", None) is not None:
            cloud_tier = getattr(client, "cloud_tier")
        elif not bool(getattr(client, "is_cloud", False)):
            cloud_tier = None

        # B9: localOnly + cloud client -> skip stage without touching external endpoints
        if local_only and bool(getattr(client, "is_cloud", False)):
            out_layout["postprocessMeta"] = _build_meta("SKIPPED", "LOCAL_ENDPOINT_UNAVAILABLE")
            return out_layout

        # Collect inspectable text blocks & measure page invalid token ratio (B7)
        inspectable: list[tuple[int, dict[str, Any], list[Any]]] = []
        total_words = 0
        total_invalid = 0
        for idx, blk in enumerate(blocks):
            if blk.get("type") == "table":
                continue
            txt = (blk.get("text") or "").strip()
            if not txt:
                continue
            total_words += len(txt.split())
            inv = invalid_syllables(txt, allow_loanwords=cfg.allow_common_loanwords)
            total_invalid += len(inv)
            inspectable.append((idx, blk, inv))

        if total_words > 0 and (total_invalid / total_words) > cfg.max_invalid_token_ratio:
            out_layout["postprocessMeta"] = _build_meta("SKIPPED", "PAGE_TOO_NOISY_OR_NOT_VIETNAMESE")
            return out_layout

        # Evaluate candidate rules (a) OR (b) OR (c) (B6, B7)
        raw_candidates: list[dict[str, Any]] = []
        selected_indices: set[int] = set()
        for idx, blk, inv in inspectable:
            txt = (blk.get("text") or "").strip()
            if has_placeholder_collision(txt):
                _inc_rejection(rejections, ReasonCode.SKIP_PLACEHOLDER_COLLISION)
                blk["postprocess"] = {
                    "candidate": False,
                    "reasons": [ReasonCode.SKIP_PLACEHOLDER_COLLISION.value],
                    "invalidSyllables": [],
                }
                continue

            reasons: list[str] = []
            conf = float(blk.get("confidence", 1.0))
            if conf < cfg.line_conf_threshold:
                reasons.append("LOW_CONFIDENCE")
            if inv:
                reasons.append("INVALID_SYLLABLE")
            prot = detect_protected_spans(txt)
            damaged_tokens = [s[2] for s in prot if s[4]]
            if damaged_tokens:
                reasons.append("DAMAGED_PROTECTED_TOKEN")
            protected_texts = {s[2] for s in prot}
            has_ocr_digit_noise = any(
                any(ch.isdigit() for ch in w)
                and any(ch.isalpha() for ch in w)
                and w.strip(".,;:!?\"'()[]{}") not in protected_texts
                for w in txt.split()
            )
            if has_ocr_digit_noise and "INVALID_SYLLABLE" not in reasons:
                reasons.append("INVALID_SYLLABLE")
            if auto_apply and not reasons:
                reasons.append("FULL_PAGE_POSTPROCESS")

            if reasons:
                raw_candidates.append(
                    {
                        "idx": idx,
                        "block": blk,
                        "text": txt,
                        "conf": conf,
                        "inv_tokens": [sp.token for sp in inv[:8]],
                        "inv_count": len(inv),
                        "reasons": reasons,
                        "damaged_tokens": damaged_tokens,
                        "priority": (
                            1 if damaged_tokens else 0,
                            len(inv),
                            -conf,
                            -idx,
                        ),
                    }
                )

        if not raw_candidates:
            out_layout["postprocessMeta"] = _build_meta("COMPLETED")
            return out_layout

        # Enforce candidate cap per page by priority (B7)
        raw_candidates.sort(key=lambda c: c["priority"], reverse=True)
        selected = raw_candidates[: cfg.max_candidates_per_page]
        cand_count = len(selected)
        cand_by_idx: dict[int, dict[str, Any]] = {}
        for c in selected:
            selected_indices.add(c["idx"])
            cand_by_idx[c["idx"]] = c
            c["block"]["postprocess"] = {
                "candidate": True,
                "reasons": list(c["reasons"]),
                "invalidSyllables": list(c["inv_tokens"]),
            }

        # Optional Re-OCR ONLY within selected candidates as evidence (B7)
        reocr_texts: dict[str, str] = {}
        if cfg.reocr_enabled and reocr_provider is not None and selected:
            rt0 = time.monotonic()
            reocr_targets = selected[: cfg.max_reocr_lines_per_page]
            reocr_blocks = [c["block"] for c in reocr_targets]
            reocr_out = reocr_provider(reocr_blocks)
            reocr_latency_ms = int((time.monotonic() - rt0) * 1000)
            reocr_lines = len(reocr_targets)
            diffs = 0
            for c, r_txt in zip(reocr_targets, reocr_out):
                r_clean = (r_txt or "").strip()
                if r_clean:
                    reocr_texts[str(c["block"].get("id", c["idx"]))] = r_clean
                    baseline = (c["block"].get("ocrText") or c["text"]).strip()
                    if r_clean != baseline:
                        diffs += 1
            reocr_diffs = diffs

        # B3: Lock ALL lines sent to Text-LLM (selected candidates + ±1 context lines) with page-unique ⟦Tn⟧
        send_indices: set[int] = set()
        eligible_indices = [item[0] for item in inspectable]
        pos_of_idx = {idx: pos for pos, idx in enumerate(eligible_indices)}
        for c_idx in selected_indices:
            send_indices.add(c_idx)
            pos = pos_of_idx[c_idx]
            if pos - 1 >= 0:
                send_indices.add(eligible_indices[pos - 1])
            if pos + 1 < len(eligible_indices):
                send_indices.add(eligible_indices[pos + 1])

        page_counter = PageTokenCounter()
        locked_entries: dict[int, dict[str, Any]] = {}
        locked_state: dict[str, tuple[dict[str, Any], str, str, list[LockedToken]]] = {}
        expected_candidate_indices: list[int] = []
        cand_lengths: dict[int, int] = {}

        for idx in sorted(send_indices):
            blk = blocks[idx]
            raw_txt = (blk.get("text") or "").strip()
            if has_placeholder_collision(raw_txt):
                continue
            line_id = str(blk.get("id") or f"line-{idx}")
            is_cand = idx in selected_indices
            locked_txt, locked_toks = TokenLocker.lock(raw_txt, page_counter)
            reocr_locked: str | None = None
            if is_cand:
                if line_id in reocr_texts and not has_placeholder_collision(reocr_texts[line_id]):
                    reocr_locked, _ = TokenLocker.lock(reocr_texts[line_id], page_counter)
                locked_state[line_id] = (blk, raw_txt, locked_txt, locked_toks)
                expected_candidate_indices.append(idx)
                cand_lengths[idx] = len(locked_txt)
            locked_entries[idx] = {
                "lineId": line_id,
                "order": blk.get("order", idx + 1),
                "text": locked_txt,
                "invalidSyllables": cand_by_idx[idx]["inv_tokens"] if is_cand else [],
                "reocrText": reocr_locked,
            }

        if not expected_candidate_indices:
            out_layout["postprocessMeta"] = _build_meta("COMPLETED")
            return out_layout

        # Check deadline before calling Text-LLM
        if time.monotonic() >= deadline_ts:
            out_layout["postprocessMeta"] = _build_meta("TIMEOUT", "PAGE_DEADLINE_EXCEEDED")
            return out_layout

        chunks = _partition_candidate_chunks(
            expected_candidate_indices,
            cand_lengths,
            trigger_lines=cfg.chunk_trigger_lines,
            trigger_chars=cfg.chunk_trigger_chars,
            target_lines_per_chunk=cfg.chunk_max_lines,
            target_chars_per_chunk=cfg.chunk_max_chars,
        )
        if len(chunks) > 1:
            LOG.info(
                "  -> Chia %d blocks cần sửa thành %d cụm nhỏ (%s dòng/cụm) để tránh timeout.",
                len(expected_candidate_indices),
                len(chunks),
                "/".join(str(len(ch)) for ch in chunks),
            )

        llm_accepted: dict[str, dict[str, Any]] = {}
        successful_chunks = 0
        last_ok_model: str | None = None
        last_seen_model: str | None = None
        last_skip_reason: str | None = None
        saw_timeout = False
        saw_failed_safe = False
        last_fail_reason: str | None = None

        for chunk_idx, chunk_cand_indices in enumerate(chunks, start=1):
            if time.monotonic() >= deadline_ts:
                saw_timeout = True
                break

            chunk_cand_set = set(chunk_cand_indices)
            chunk_send_indices: set[int] = set(chunk_cand_indices)
            for c_idx in chunk_cand_indices:
                pos = pos_of_idx[c_idx]
                if pos - 1 >= 0:
                    chunk_send_indices.add(eligible_indices[pos - 1])
                if pos + 1 < len(eligible_indices):
                    chunk_send_indices.add(eligible_indices[pos + 1])

            chunk_payload_lines: list[dict[str, Any]] = []
            chunk_candidate_ids: list[str] = []
            for s_idx in sorted(chunk_send_indices):
                info = locked_entries.get(s_idx)
                if info is None:
                    continue
                is_chunk_cand = s_idx in chunk_cand_set
                entry: dict[str, Any] = {
                    "lineId": info["lineId"],
                    "order": info["order"],
                    "isCandidate": is_chunk_cand,
                    "text": info["text"],
                }
                if is_chunk_cand:
                    entry["invalidSyllables"] = info["invalidSyllables"]
                    if info["reocrText"] is not None:
                        entry["reocrText"] = info["reocrText"]
                    chunk_candidate_ids.append(info["lineId"])
                chunk_payload_lines.append(entry)

            if not chunk_candidate_ids:
                continue

            page_payload = {
                "pageNumber": out_layout.get("pageNumber", 1),
                "lines": chunk_payload_lines,
            }
            raw_llm_resp, llm_usage = client.correct_lines_batch(
                page_payload, local_only=local_only, deadline_ts=deadline_ts
            )
            if getattr(llm_usage, "model", None):
                last_seen_model = str(llm_usage.model)
            _accumulate_usage(llm_usage)
            if llm_usage.status not in (
                "CLOUD_NOT_CONSENTED",
                "LOCAL_ENDPOINT_UNAVAILABLE",
                "BREAKER_OPEN",
                "DAILY_BUDGET_EXHAUSTED",
            ):
                llm_calls += 1

            if llm_usage.status in _SKIP_USAGE_STATUSES:
                last_skip_reason = llm_usage.status
                break

            if llm_usage.status == "timeout" or time.monotonic() >= deadline_ts:
                saw_timeout = True
                if time.monotonic() >= deadline_ts or len(chunks) == 1:
                    break
                continue

            if llm_usage.status == "FAILED_SAFE":
                saw_failed_safe = True
                last_fail_reason = getattr(llm_usage, "error_reason", None) or "FAILED_SAFE"
                if (deadline_ts - time.monotonic()) <= 1.0:
                    break
                continue

            if getattr(llm_usage, "model", None):
                last_ok_model = str(llm_usage.model)

            parsed_resp: Any = raw_llm_resp
            if isinstance(raw_llm_resp, str):
                try:
                    parsed_resp = json.loads(raw_llm_resp)
                except json.JSONDecodeError:
                    parsed_resp = None

            if auto_apply:
                parsed_resp = _normalize_auto_apply_batch(
                    parsed_resp, chunk_candidate_ids, locked_state
                )

            ok_page, page_err, parsed_by_id = TokenGuard.validate_page_batch(
                parsed_resp, chunk_candidate_ids
            )
            successful_chunks += 1
            if not ok_page:
                if page_err is not None:
                    _inc_rejection(rejections, page_err)
                continue

            for line_id in chunk_candidate_ids:
                blk, raw_txt, locked_txt, locked_toks = locked_state[line_id]
                item = parsed_by_id[line_id]
                guard_res = TokenGuard.check_line(
                    raw_txt,
                    locked_txt,
                    locked_toks,
                    item["corrected"],
                    float(item["confidence"]),
                    allow_paragraph_refinement=auto_apply,
                )
                if guard_res.unchanged:
                    continue
                if not guard_res.accepted or guard_res.restored_text is None:
                    for rc in guard_res.reason_codes:
                        _inc_rejection(rejections, rc)
                    continue
                ops = _attach_reasons(list(guard_res.ops), item.get("changes", []))
                llm_accepted[line_id] = {
                    "text": guard_res.restored_text,
                    "confidence": float(item["confidence"]),
                    "ops": ops,
                    "risk": guard_res.risk,
                }

        if last_ok_model:
            llm_model = last_ok_model
        elif last_seen_model:
            llm_model = last_seen_model

        had_chunk_failure = bool(last_skip_reason or saw_timeout or saw_failed_safe)
        if successful_chunks == 0 or (had_chunk_failure and not llm_accepted):
            if last_skip_reason is not None:
                out_layout["postprocessMeta"] = _build_meta("SKIPPED", last_skip_reason)
                return out_layout
            if saw_timeout:
                out_layout["postprocessMeta"] = _build_meta("TIMEOUT", "PAGE_DEADLINE_EXCEEDED")
                return out_layout
            if saw_failed_safe:
                out_layout["postprocessMeta"] = _build_meta(
                    "FAILED_SAFE",
                    last_fail_reason or "FAILED_SAFE",
                    suggestions_override=0,
                )
                return out_layout

        partial_chunk_skip_reason: str | None = None
        if had_chunk_failure and llm_accepted:
            fail_detail = (
                last_skip_reason
                or ("PAGE_DEADLINE_EXCEEDED" if saw_timeout else (last_fail_reason or "FAILED_SAFE"))
            )
            partial_chunk_skip_reason = f"PARTIAL_CHUNK_FAILURE:{fail_detail}"

        # Path L3: Batched VLM verification (<=1 request, <=5 crops) ONLY for damaged protected tokens (C7)
        damaged_candidates = [c for c in selected if c["damaged_tokens"]]
        vlm_results: dict[str, dict[str, Any]] = {}
        vlm_skipped_due_to_pii = bool(redact_pii and bool(getattr(client, "is_cloud", False)))
        vlm_skip_reason: str | None = None
        timed_out = False
        max_crops = min(5, max(0, cfg.max_vlm_crops))

        if damaged_candidates and not vlm_skipped_due_to_pii and max_crops > 0:
            needs_crop = bool(getattr(client, "needs_crop_image", provider_label != "mock"))
            vlm_items: list[dict[str, Any]] = []
            for c in damaged_candidates[:max_crops]:
                blk = c["block"]
                line_id = str(blk.get("id") or f"line-{c['idx']}")
                crop_bytes = b""
                if needs_crop:
                    if crop_provider is None:
                        continue
                    provided = crop_provider(blk)
                    if provided is None:
                        continue
                    crop_bytes = prepare_vlm_crop_jpeg(provided, max_side=1024, quality=90)
                vlm_items.append(
                    {
                        "lineId": line_id,
                        "cropBytes": crop_bytes,
                        "currentText": c["text"],
                        "focusTokens": list(c["damaged_tokens"]),
                    }
                )

            if vlm_items:
                if time.monotonic() >= deadline_ts:
                    timed_out = True
                else:
                    vlm_raw, vlm_usage = client.verify_crops_vlm(
                        vlm_items,
                        local_only=local_only,
                        deadline_ts=deadline_ts,
                    )
                    if getattr(vlm_usage, "model", None):
                        vlm_model = str(vlm_usage.model)
                    _accumulate_usage(vlm_usage)
                    if vlm_usage.status not in (
                        "CLOUD_NOT_CONSENTED",
                        "LOCAL_ENDPOINT_UNAVAILABLE",
                        "BREAKER_OPEN",
                        "DAILY_BUDGET_EXHAUSTED",
                    ):
                        vlm_calls += 1

                    if vlm_usage.status == "timeout" or time.monotonic() >= deadline_ts:
                        timed_out = True
                    elif vlm_usage.status == "FAILED_SAFE":
                        out_layout["postprocessMeta"] = _build_meta(
                            "FAILED_SAFE",
                            getattr(vlm_usage, "error_reason", None) or "FAILED_SAFE",
                            suggestions_override=0,
                        )
                        return out_layout
                    elif vlm_usage.status in ("AUTH_ERROR", "MODEL_UNAVAILABLE"):
                        out_layout["postprocessMeta"] = _build_meta(
                            "SKIPPED",
                            vlm_usage.status,
                            suggestions_override=0,
                        )
                        return out_layout
                    elif vlm_usage.status in _SKIP_USAGE_STATUSES:
                        vlm_skip_reason = vlm_usage.status
                    else:
                        vlm_parsed: Any = vlm_raw
                        if isinstance(vlm_raw, str):
                            try:
                                vlm_parsed = json.loads(vlm_raw)
                            except json.JSONDecodeError:
                                vlm_parsed = None
                        ok_vlm, vlm_err, vlm_by_id = _validate_vlm_batch(
                            vlm_parsed,
                            [it["lineId"] for it in vlm_items],
                        )
                        if not ok_vlm:
                            if vlm_err is not None:
                                _inc_rejection(rejections, vlm_err)
                        else:
                            vlm_results = vlm_by_id

        if timed_out:
            out_layout["postprocessMeta"] = _build_meta("TIMEOUT", "PAGE_DEADLINE_EXCEEDED")
            return out_layout

        # Assemble per-block suggestions and visual warnings (Policy D5.5 + B1 + B2 + B12 + B13)
        for c in selected:
            blk = c["block"]
            line_id = str(blk.get("id") or f"line-{c['idx']}")
            raw_txt = c["text"]
            llm_sug = llm_accepted.get(line_id)
            vlm_res = vlm_results.get(line_id)
            suggestions: list[dict[str, Any]] = []

            if c["damaged_tokens"]:
                if (
                    vlm_res is not None
                    and not vlm_res["uncertain"]
                    and vlm_res["transcription"].strip() != raw_txt
                ):
                    vlm_txt = vlm_res["transcription"].strip()
                    vlm_err = _validate_vlm_transcription(raw_txt, vlm_txt)
                    if vlm_err is not None:
                        _inc_rejection(rejections, vlm_err)
                        blk["isWarning"] = True
                        blk["warningReason"] = "Token số/mã hiệu nghi khuyết nét, cần kiểm tra mắt"
                    else:
                        vlm_ops = _attach_reasons(
                            compute_ops(raw_txt, vlm_txt), vlm_res.get("changes", [])
                        )
                        suggestions.append(
                            {
                                "id": f"sug-{line_id}-1",
                                "source": "vlm",
                                "provider": provider_label,
                                "text": vlm_txt,
                                "confidence": round(float(vlm_res["confidence"]), 3),
                                "risk": "high",
                                "requiresVisualCheck": True,
                                "status": "pending",
                                "reasonCodes": ["VLM_DAMAGED_TOKEN_VERIFIED"],
                                "ops": vlm_ops,
                            }
                        )
                        blk["isWarning"] = True
                        blk["warningReason"] = "Token số/mã hiệu nghi khuyết nét, cần kiểm tra mắt"
                else:
                    # C7: VLM uncertain / unchanged / skipped by redactPii -> set warning + keep safe word suggestion if any
                    blk["isWarning"] = True
                    blk["warningReason"] = "Token số/mã hiệu nghi khuyết nét, cần kiểm tra mắt"
                    if llm_sug is not None:
                        sug_status_dam = "applied" if auto_apply else "pending"
                        if auto_apply:
                            blk.setdefault("ocrText", raw_txt)
                            blk["text"] = llm_sug["text"]
                        suggestions.append(
                            {
                                "id": f"sug-{line_id}-1",
                                "source": "llm",
                                "provider": provider_label,
                                "text": llm_sug["text"],
                                "confidence": round(llm_sug["confidence"], 3),
                                "risk": llm_sug["risk"],
                                "requiresVisualCheck": True,
                                "status": sug_status_dam,
                                "reasonCodes": ["DAMAGED_PROTECTED_TOKEN_UNRESOLVED"],
                                "ops": llm_sug["ops"],
                            }
                        )
            elif llm_sug is not None:
                # Normal Text-LLM candidate (B2: NO isWarning when requiresVisualCheck == False)
                r_codes = ["INVALID_SYLLABLE_FIXED"] if c["inv_tokens"] else ["OCR_TEXT_REFINED"]
                sug_status = "applied" if auto_apply else "pending"
                if auto_apply:
                    blk.setdefault("ocrText", raw_txt)
                    blk["text"] = llm_sug["text"]
                suggestions.append(
                    {
                        "id": f"sug-{line_id}-1",
                        "source": "llm",
                        "provider": provider_label,
                        "text": llm_sug["text"],
                        "confidence": round(llm_sug["confidence"], 3),
                        "risk": llm_sug["risk"],
                        "requiresVisualCheck": False,
                        "status": sug_status,
                        "reasonCodes": r_codes,
                        "ops": llm_sug["ops"],
                    }
                )

            if suggestions:
                blk["suggestions"] = suggestions
                suggestions_created += len(suggestions)

        final_skip = (
            "VLM_SKIPPED_PII_REDACTION"
            if (damaged_candidates and vlm_skipped_due_to_pii)
            else (vlm_skip_reason or partial_chunk_skip_reason)
        )
        out_layout["postprocessMeta"] = _build_meta("COMPLETED", final_skip)
        return out_layout

    except Exception as exc:
        LOG.warning("LLM postprocess FAILED_SAFE: %s", type(exc).__name__)
        safe_layout = copy.deepcopy(layout)
        safe_layout["postprocessMeta"] = _build_meta(
            "FAILED_SAFE",
            f"EXCEPTION:{type(exc).__name__}",
            suggestions_override=0,
        )
        return safe_layout

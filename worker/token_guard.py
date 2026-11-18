"""Three-layer token protection (L1 locker, L2 deterministic guard, L3 damaged token flags)."""

from __future__ import annotations

import difflib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence


class ReasonCode(str, Enum):
    REJECT_SCHEMA_INVALID = "REJECT_SCHEMA_INVALID"
    REJECT_LINE_COUNT_MISMATCH = "REJECT_LINE_COUNT_MISMATCH"
    REJECT_PLACEHOLDER_ALTERED = "REJECT_PLACEHOLDER_ALTERED"
    REJECT_DIGIT_MULTISET_CHANGED = "REJECT_DIGIT_MULTISET_CHANGED"
    REJECT_PROTECTED_TOKEN_MUTATED = "REJECT_PROTECTED_TOKEN_MUTATED"
    REJECT_LENGTH_RATIO_OUT_OF_BOUNDS = "REJECT_LENGTH_RATIO_OUT_OF_BOUNDS"
    REJECT_WORD_INSERTION_OR_DELETION = "REJECT_WORD_INSERTION_OR_DELETION"
    REJECT_EXCESSIVE_CHAR_EDIT = "REJECT_EXCESSIVE_CHAR_EDIT"
    REJECT_LOW_LLM_CONFIDENCE = "REJECT_LOW_LLM_CONFIDENCE"
    SKIP_PLACEHOLDER_COLLISION = "SKIP_PLACEHOLDER_COLLISION"


@dataclass
class PageTokenCounter:
    """Monotonic counter ensuring ⟦Tn⟧ placeholders are unique across an entire page."""

    next_id: int = 1

    def allocate(self) -> str:
        placeholder = f"⟦T{self.next_id}⟧"
        self.next_id += 1
        return placeholder


@dataclass(frozen=True)
class LockedToken:
    placeholder: str
    text: str
    type: str
    damaged: bool
    start: int = 0
    end: int = 0


_PLACEHOLDER_RE = re.compile(r"⟦T\d+⟧")
_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"']+")
_EMAIL_RE = re.compile(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b")
_DOC_CODE_RE = re.compile(
    r"\b\d{1,5}(?:/\d{2,4})?/(?:[A-Za-zÀ-ỹĐđ0-9-]+)?(?=\s|$|[,.;:!?)\]}\"'])"
)
_CLEAN_DOC_CODE_SUFFIX_RE = re.compile(r"^[A-ZĐ0-9]{2,}(?:-[A-ZĐ0-9]+)*$")
_DATE_NUMERIC_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
_CURRENCY_RE = re.compile(
    r"(?i)(?:\$\s*\d[\d.,]*|\b\d[\d.,]*(?:\s*(?:đồng|vnđ|vnd|usd|\$|đ|tỷ|triệu|nghìn|ngàn))(?![A-Za-zÀ-ỹĐđ0-9_]))"
)
_ID_NUMBER_RE = re.compile(r"(?:(?<!\w)\+84\d{8,10}\b|\b0\d{8,10}\b|\b\d{9,14}(?:-\d{3})?\b)")
_NUMERIC_TOKEN_RE = re.compile(r"\S*\d\S*")
_PURE_NUMBER_RE = re.compile(r"^[+\-]?\d+(?:[.,:-]\d+)*%?$")
_UNIT_ALNUM_RE = re.compile(
    r"^(?:\d+(?:[.,]\d+)?(?:%|km|cm|mm|dm|kg|mg|ml|gb|mb|kb|hz|mhz|ghz|px|ms|pt|dpi|vnd|vnđ|usd|°c)|[A-ZĐ]{2,}-\d+(?:-[A-ZĐ0-9]+)*)$"
)
_PRECEDING_NUM_LABELS: frozenset[str] = frozenset(
    {
        "số", "s", "mục", "điều", "khoản", "chương", "phần", "bài", "câu", "trang",
        "tờ", "bảng", "hình", "biểu", "cột", "dòng", "hàng", "lần", "bước", "đợt",
        "kỳ", "quý", "thứ", "loại", "hạng", "bậc", "cấp", "tập", "quyển", "cuốn",
        "đội", "đôi", "tổ", "phòng", "ban", "quận", "phường", "xã", "khu", "nhóm",
        "ca", "tuyến", "mã", "no",
    }
)
_FOLLOWING_NUM_UNITS: frozenset[str] = frozenset(
    {
        "%", "tấc", "mét", "m", "km", "cm", "mm", "dm", "kg", "g", "mg", "lít",
        "tấn", "tạ", "yến", "lạng", "tuổi", "năm", "tháng", "ngày", "tuần", "giờ",
        "phút", "giây", "đồng", "vnđ", "vnd", "usd", "triệu", "tỷ", "nghìn", "ngàn",
        "trăm", "khổ", "sắc", "mảnh", "trang", "độ",
    }
)
_TONE_MARKS: frozenset[str] = frozenset({"\u0300", "\u0301", "\u0303", "\u0309", "\u0323"})
_SHAPE_MARKS: frozenset[str] = frozenset({"\u0302", "\u0306", "\u031b"})


def has_placeholder_collision(text: str) -> bool:
    """Return True if raw OCR text already contains sentinel brackets (B3)."""
    return ("⟦" in text) or ("⟧" in text)


def _strip_accents(text: str, convert_d: bool = True) -> str:
    if convert_d:
        text = text.replace("đ", "d").replace("Đ", "D")
    nfd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def _trim_outer_punct(text: str, start: int, end: int, keep_trailing_slash_hyphen: bool = False) -> tuple[int, int]:
    left_chars = "\"'([{«“‘"
    right_chars = ",.;:!?\"')]}»”’"
    if not keep_trailing_slash_hyphen:
        right_chars += "-"
    while start < end and text[start] in left_chars:
        start += 1
    while end > start and text[end - 1] in right_chars:
        end -= 1
    return start, end


def _is_valid_date_numeric(token: str) -> bool:
    parts = re.split(r"[/-]", token)
    if len(parts) != 3:
        return False
    d_s, m_s, y_s = parts
    if not (d_s.isdigit() and m_s.isdigit() and y_s.isdigit()):
        return False
    day, month = int(d_s), int(m_s)
    if day < 1 or day > 31 or month < 1 or month > 12:
        return False
    if len(y_s) == 3:
        return False
    return True


def _is_doc_code_damaged(token: str) -> bool:
    if token.endswith("/") or token.endswith("-"):
        return True
    first_slash = token.find("/")
    if first_slash < 0:
        return True
    suffix = token[first_slash + 1 :]
    if re.match(r"^\d{2,4}/", suffix):
        suffix = suffix.split("/", 1)[1]
    if _CLEAN_DOC_CODE_SUFFIX_RE.match(suffix):
        return False
    return True


def _detect_date_phrase_spans(text: str) -> list[tuple[int, int, str, bool]]:
    """Detect sliding-window 'ngày <X> tháng <Y> năm <Z>' value spans (including OCR-noised labels)."""
    raw_tokens = [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\S+", text)]
    if len(raw_tokens) < 4:
        return []

    def norm_label(tok: str) -> str:
        cleaned = tok.strip(".,;:!?\"'()[]{}")
        return _strip_accents(cleaned).lower()

    found: list[tuple[int, int, str, bool]] = []
    n = len(raw_tokens)
    i = 0
    while i < n:
        lbl_ngay = norm_label(raw_tokens[i][2])
        if len(lbl_ngay) < 3 or _levenshtein(lbl_ngay, "ngay") > 1:
            i += 1
            continue
        matched = False
        for j in range(i + 1, min(n, i + 4)):
            lbl_thang = norm_label(raw_tokens[j][2])
            if len(lbl_thang) < 4 or _levenshtein(lbl_thang, "thang") > 1:
                continue
            for k in range(j + 1, min(n, j + 4)):
                lbl_nam = norm_label(raw_tokens[k][2])
                if len(lbl_nam) < 2 or _levenshtein(lbl_nam, "nam") > 1:
                    continue
                if k + 1 >= n:
                    continue
                if k - j > 2 or j - i > 2:
                    continue

                y_raw_start, y_raw_end, _ = raw_tokens[k + 1]
                ys, ye = _trim_outer_punct(text, y_raw_start, y_raw_end)
                year_tok = text[ys:ye]
                day_tok = text[raw_tokens[i + 1][0] : raw_tokens[j - 1][1]].strip() if j - i > 1 else ""
                month_tok = text[raw_tokens[j + 1][0] : raw_tokens[k - 1][1]].strip() if k - j > 1 else ""

                has_date_evidence = (
                    year_tok.isdigit()
                    or day_tok.isdigit()
                    or month_tok.isdigit()
                    or day_tok in ("th", "..", "...", "?", "__", "___")
                    or lbl_ngay in ("ngay", "ngly")
                )
                if not has_date_evidence:
                    continue

                if j - i > 1:
                    raw_day = text[raw_tokens[i + 1][0] : raw_tokens[j - 1][1]]
                    ds, de = _trim_outer_punct(text, raw_tokens[i + 1][0], raw_tokens[j - 1][1])
                    if ds >= de and raw_day.strip():
                        ds, de = raw_tokens[i + 1][0], raw_tokens[j - 1][1]
                    d_val = text[ds:de]
                    if d_val and not (d_val.isdigit() and int(d_val) > 31):
                        d_dam = not (d_val.isdigit() and 1 <= int(d_val) <= 31)
                        found.append((ds, de, "DATE_PHRASE", d_dam))
                if k - j > 1:
                    raw_month = text[raw_tokens[j + 1][0] : raw_tokens[k - 1][1]]
                    ms, me = _trim_outer_punct(text, raw_tokens[j + 1][0], raw_tokens[k - 1][1])
                    if ms >= me and raw_month.strip():
                        ms, me = raw_tokens[j + 1][0], raw_tokens[k - 1][1]
                    m_val = text[ms:me]
                    if m_val and not (m_val.isdigit() and int(m_val) > 12):
                        m_dam = not (m_val.isdigit() and 1 <= int(m_val) <= 12)
                        found.append((ms, me, "DATE_PHRASE", m_dam))
                if year_tok and not (year_tok.isdigit() and 5 <= len(year_tok) <= 8):
                    y_dam = not (year_tok.isdigit() and len(year_tok) in (2, 4))
                    found.append((ys, ye, "DATE_PHRASE", y_dam))

                i = k + 2
                matched = True
                break
            if matched:
                break
        if not matched:
            i += 1
    return found


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def detect_protected_spans(text: str) -> list[tuple[int, int, str, str, bool]]:
    """Detect all protected spans ordered by position; returns (start, end, text, type, damaged)."""
    accepted: list[tuple[int, int, str, str, bool]] = []

    def try_add(s: int, e: int, tok_type: str, damaged: bool) -> None:
        if s >= e:
            return
        for ex_s, ex_e, _, _, _ in accepted:
            if _overlaps((s, e), (ex_s, ex_e)):
                return
        accepted.append((s, e, text[s:e], tok_type, damaged))

    # 1. URL & Email
    for m in _URL_RE.finditer(text):
        s, e = _trim_outer_punct(text, m.start(), m.end())
        try_add(s, e, "URL_EMAIL", False)
    for m in _EMAIL_RE.finditer(text):
        try_add(m.start(), m.end(), "URL_EMAIL", False)

    # 2. DOC_CODE (keep trailing '-'/'/' so '20/CV-' and '1027/' stay locked; skip pure numeric dates)
    for m in _DOC_CODE_RE.finditer(text):
        tok = m.group(0)
        has_alpha = any(ch.isalpha() for ch in tok)
        if not has_alpha and not (tok.endswith("/") or tok.endswith("-")):
            continue
        try_add(m.start(), m.end(), "DOC_CODE", _is_doc_code_damaged(tok))

    # 3. DATE_NUMERIC
    for m in _DATE_NUMERIC_RE.finditer(text):
        tok = m.group(0)
        try_add(m.start(), m.end(), "DATE_NUMERIC", not _is_valid_date_numeric(tok))

    # 4. DATE_PHRASE
    for s, e, tok_type, damaged in _detect_date_phrase_spans(text):
        try_add(s, e, tok_type, damaged)

    # 5. CURRENCY
    for m in _CURRENCY_RE.finditer(text):
        try_add(m.start(), m.end(), "CURRENCY", False)

    # 6. ID_NUMBER
    for m in _ID_NUMBER_RE.finditer(text):
        try_add(m.start(), m.end(), "ID_NUMBER", False)

    # 7. NUMERIC_TOKEN (smart locking: lock genuine structured/labeled/unit numbers,
    #    never lock OCR-garbled mixed alphanumeric tokens like '14p', '10p', 'g9', '1bể', 'Việt-81'
    #    or stray 1-2 digit numbers sitting inside ordinary Vietnamese prose)
    ws_tokens = [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\S+", text)]

    def _clean_neighbor_word(raw_w: str) -> str:
        return unicodedata.normalize("NFC", raw_w).strip(".,;:!?\"'()[]{}«»“”‘’—–-").lower()

    for m in _NUMERIC_TOKEN_RE.finditer(text):
        s, e = _trim_outer_punct(text, m.start(), m.end())
        if s >= e:
            continue
        tok = text[s:e]
        if not any(ch.isdigit() for ch in tok):
            continue

        # Never lock OCR-corrupted mixed digit+letter tokens unless they match explicit units or ASCII codes
        has_alpha = any(ch.isalpha() for ch in tok)
        if has_alpha:
            if _UNIT_ALNUM_RE.match(tok):
                try_add(s, e, "NUMERIC_TOKEN", False)
            continue

        # Pure/structured numeric token (only digits and optional . , : - %)
        if not _PURE_NUMBER_RE.match(tok):
            continue

        tok_idx = None
        for idx_w, (ws_s, ws_e, _) in enumerate(ws_tokens):
            if ws_s <= s and e <= ws_e:
                tok_idx = idx_w
                break

        prev_raw = ws_tokens[tok_idx - 1][2] if (tok_idx is not None and tok_idx > 0) else ""
        prev_clean = _clean_neighbor_word(prev_raw) if prev_raw else ""
        next_clean = _clean_neighbor_word(ws_tokens[tok_idx + 1][2]) if (tok_idx is not None and tok_idx + 1 < len(ws_tokens)) else ""

        # Do NOT lock a 5-8 digit OCR-garbled year immediately following 'năm'/'nam'/'nắm'/'1?năm' (e.g. 'năm 149733')
        if tok.isdigit() and 5 <= len(tok) <= 8 and prev_raw:
            alpha_tail_m = re.search(r"[A-Za-zÀ-ỹ]+$", prev_raw.strip(".,;:!?\"'()[]{}"))
            if alpha_tail_m:
                tail_ascii = _strip_accents(alpha_tail_m.group(0)).lower()
                if len(tail_ascii) >= 2 and _levenshtein(tail_ascii, "nam") <= 1:
                    continue

        # Do NOT lock impossible numeric day (>31) after 'ngày' or impossible month (>12) after 'tháng'
        if tok.isdigit() and prev_clean in ("ngày", "ngay", "ngly") and int(tok) > 31:
            continue
        if tok.isdigit() and prev_clean in ("tháng", "thang", "thíng") and int(tok) > 12:
            continue

        n_digits = sum(1 for ch in tok if ch.isdigit())
        is_structured = any(sep in tok for sep in (".", ",", ":", "-", "%"))
        if n_digits >= 4 or is_structured:
            try_add(s, e, "NUMERIC_TOKEN", False)
            continue

        # 1-3 digit standalone number: lock ONLY with structural/label/unit context
        raw_match = m.group(0)
        has_list_punct = raw_match.endswith((".", ")")) or (raw_match.startswith(("(", "[")) and raw_match.endswith((")", "]")))
        if has_list_punct or len(ws_tokens) == 1:
            try_add(s, e, "NUMERIC_TOKEN", False)
            continue

        # Allow page-number style separators like "— 19 —"
        is_framed = (
            tok_idx is not None
            and tok_idx > 0
            and tok_idx + 1 < len(ws_tokens)
            and ws_tokens[tok_idx - 1][2] in ("-", "–", "—")
            and ws_tokens[tok_idx + 1][2] in ("-", "–", "—")
        )
        if is_framed or prev_clean in _PRECEDING_NUM_LABELS or next_clean in _FOLLOWING_NUM_UNITS:
            try_add(s, e, "NUMERIC_TOKEN", False)

    accepted.sort(key=lambda item: item[0])
    return accepted


class TokenLocker:
    """L1 locker replacing protected tokens with page-unique immutable placeholders ⟦Tn⟧."""

    @staticmethod
    def lock(
        text: str, page_counter: PageTokenCounter | None = None
    ) -> tuple[str, list[LockedToken]]:
        counter = page_counter if page_counter is not None else PageTokenCounter()
        spans = detect_protected_spans(text)
        if not spans:
            return text, []

        out_parts: list[str] = []
        locked: list[LockedToken] = []
        cursor = 0
        for s, e, raw_tok, tok_type, damaged in spans:
            if s > cursor:
                out_parts.append(text[cursor:s])
            placeholder = counter.allocate()
            out_parts.append(placeholder)
            locked.append(
                LockedToken(
                    placeholder=placeholder,
                    text=raw_tok,
                    type=tok_type,
                    damaged=damaged,
                    start=s,
                    end=e,
                )
            )
            cursor = e
        if cursor < len(text):
            out_parts.append(text[cursor:])
        return "".join(out_parts), locked


def _decompose_char(ch: str) -> tuple[str, str, bool]:
    """Return (ascii_root_lower, viet_shape, has_foreign_mark)."""
    lower = ch.lower()
    if lower == "đ":
        return "d", "stroke", False
    nfd = unicodedata.normalize("NFD", lower)
    base = ""
    shape = ""
    has_foreign = False
    for part in nfd:
        if part in _TONE_MARKS:
            continue
        if part in _SHAPE_MARKS:
            shape = part
        elif unicodedata.category(part) == "Mn":
            has_foreign = True
        elif not base:
            base = part
    return base, shape, has_foreign


def _extract_word_tones(word: str) -> tuple[str, ...]:
    nfd = unicodedata.normalize("NFD", word.lower())
    return tuple(ch for ch in nfd if ch in _TONE_MARKS)


def _char_seq_units(word: str) -> list[tuple[str, str, bool]]:
    nfc = unicodedata.normalize("NFC", word)
    return [_decompose_char(ch) for ch in nfc]


def syllable_edit_cost(old_word: str, new_word: str) -> float:
    """Weighted Vietnamese syllable edit distance (D4.2):
    - Tone change across word: +0.5
    - Same base vowel/d with mũ/râu/gạch change (e<->ê, o<->ô<->ơ, u<->ư, a<->ă<->â, d<->đ): +0.5
    - Base character insert/delete: 1.0 (1.5 if introducing/removing vowel shape)
    - Different base character substitution: 1.0 (1.5 if shape also differs on different vowel)
    """
    if old_word == new_word:
        return 0.0

    u1 = _char_seq_units(old_word)
    u2 = _char_seq_units(new_word)
    m, n = len(u1), len(u2)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + (1.5 if u1[i - 1][1] else 1.0)
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + (1.5 if u2[j - 1][1] else 1.0)

    for i in range(1, m + 1):
        b1, s1, f1 = u1[i - 1]
        for j in range(1, n + 1):
            b2, s2, f2 = u2[j - 1]
            if b1 == b2:
                sub_cost = 0.0 if (s1 == s2 and f1 == f2) else 0.5
            else:
                sub_cost = 1.0 if s2 == "" else 1.25
            dp[i][j] = min(
                dp[i - 1][j] + (1.5 if s1 else 1.0),
                dp[i][j - 1] + (1.5 if s2 else 1.0),
                dp[i - 1][j - 1] + sub_cost,
            )

    tone_cost = 0.5 if _extract_word_tones(old_word) != _extract_word_tones(new_word) else 0.0
    return dp[m][n] + tone_cost


def compute_ops(old_text: str, new_text: str) -> list[dict[str, Any]]:
    """Deterministically compute word-level diff ops with tokenIndex in old_text (B4)."""
    old_tokens = old_text.split()
    new_tokens = new_text.split()
    matcher = difflib.SequenceMatcher(a=old_tokens, b=new_tokens, autojunk=False)
    ops: list[dict[str, Any]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace":
            if (i2 - i1) == (j2 - j1):
                for k in range(i2 - i1):
                    ops.append(
                        {
                            "op": "replace",
                            "tokenIndex": i1 + k,
                            "from": old_tokens[i1 + k],
                            "to": new_tokens[j1 + k],
                        }
                    )
            else:
                ops.append(
                    {
                        "op": "replace",
                        "tokenIndex": i1,
                        "from": " ".join(old_tokens[i1:i2]),
                        "to": " ".join(new_tokens[j1:j2]),
                    }
                )
        elif tag == "insert":
            ops.append(
                {
                    "op": "insert",
                    "tokenIndex": i1,
                    "from": "",
                    "to": " ".join(new_tokens[j1:j2]),
                }
            )
        elif tag == "delete":
            ops.append(
                {
                    "op": "delete",
                    "tokenIndex": i1,
                    "from": " ".join(old_tokens[i1:i2]),
                    "to": "",
                }
            )
    return ops


def sanitize_reason(reason: Any, max_len: int = 120) -> str | None:
    """Strip control characters and truncate LLM reason to <= 120 chars (B4)."""
    if not isinstance(reason, str):
        return None
    cleaned = "".join(ch for ch in reason if unicodedata.category(ch) != "Cc")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None
    return cleaned[:max_len]


def classify_risk(ops: Sequence[dict[str, Any]]) -> str:
    """Classify text-LLM edit risk according to B12: 'low' if diacritics/đ↔d only, else 'medium'."""
    for op in ops:
        if op.get("op") != "replace":
            return "medium"
        src = _strip_accents(str(op.get("from", "")), convert_d=True)
        dst = _strip_accents(str(op.get("to", "")), convert_d=True)
        if src != dst:
            return "medium"
    return "low"


@dataclass(frozen=True)
class LineGuardResult:
    accepted: bool
    unchanged: bool = False
    restored_text: str | None = None
    reason_codes: tuple[ReasonCode, ...] = field(default_factory=tuple)
    ops: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    risk: str = "low"


@dataclass(frozen=True)
class BatchGuardResult:
    accepted_page: bool
    page_reason_code: ReasonCode | None = None
    line_results: dict[str, LineGuardResult] = field(default_factory=dict)


class TokenGuard:
    """Deterministic L2 validator at page and line granularity."""

    MAX_SYLLABLE_EDIT_COST = 2.0
    MIN_LLM_CONFIDENCE = 0.70

    @staticmethod
    def validate_page_batch(
        response: Any,
        expected_line_ids: Sequence[str],
    ) -> tuple[bool, ReasonCode | None, dict[str, dict[str, Any]]]:
        """Validate Text-LLM page batch JSON schema and exact lineId contract (D4.2 + B1)."""
        if not isinstance(response, dict) or "lines" not in response:
            return False, ReasonCode.REJECT_SCHEMA_INVALID, {}
        allowed_root_keys = {"lines"}
        if not set(response.keys()).issubset(allowed_root_keys):
            return False, ReasonCode.REJECT_SCHEMA_INVALID, {}

        raw_lines = response.get("lines")
        if not isinstance(raw_lines, list):
            return False, ReasonCode.REJECT_SCHEMA_INVALID, {}

        parsed: dict[str, dict[str, Any]] = {}
        for item in raw_lines:
            if not isinstance(item, dict):
                return False, ReasonCode.REJECT_SCHEMA_INVALID, {}
            req = {"lineId", "corrected", "confidence", "changes"}
            if not req.issubset(item.keys()) or not set(item.keys()).issubset(req):
                return False, ReasonCode.REJECT_SCHEMA_INVALID, {}
            line_id = item.get("lineId")
            corrected = item.get("corrected")
            conf = item.get("confidence")
            changes = item.get("changes")
            if not isinstance(line_id, str) or not line_id:
                return False, ReasonCode.REJECT_SCHEMA_INVALID, {}
            if not isinstance(corrected, str):
                return False, ReasonCode.REJECT_SCHEMA_INVALID, {}
            if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
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
            if "\n" in corrected or "\r" in corrected:
                return False, ReasonCode.REJECT_LINE_COUNT_MISMATCH, {}
            parsed[line_id] = item

        if len(raw_lines) != len(expected_line_ids) or set(parsed.keys()) != set(expected_line_ids):
            return False, ReasonCode.REJECT_LINE_COUNT_MISMATCH, {}
        return True, None, parsed

    @classmethod
    def check_line(
        cls,
        raw_original: str,
        locked_original: str,
        locked_tokens: Sequence[LockedToken],
        llm_corrected: str,
        llm_confidence: float | None = 1.0,
        *,
        allow_paragraph_refinement: bool = False,
    ) -> LineGuardResult:
        """Validate a single line proposal through L2 guards and restore placeholders."""
        if "\n" in llm_corrected or "\r" in llm_corrected:
            return LineGuardResult(accepted=False, reason_codes=(ReasonCode.REJECT_LINE_COUNT_MISMATCH,))

        # 1. Strict placeholder preservation (B3: no bracket loosening)
        expected_placeholders = [tok.placeholder for tok in locked_tokens]
        found_placeholders = _PLACEHOLDER_RE.findall(llm_corrected)
        if (
            found_placeholders != expected_placeholders
            or llm_corrected.count("⟦") != len(expected_placeholders)
            or llm_corrected.count("⟧") != len(expected_placeholders)
        ):
            return LineGuardResult(accepted=False, reason_codes=(ReasonCode.REJECT_PLACEHOLDER_ALTERED,))

        # Restore placeholders in exact order
        restored = llm_corrected
        for tok in locked_tokens:
            restored = restored.replace(tok.placeholder, tok.text, 1)

        # Unchanged line is NOT an error and emits no suggestion (B1)
        if restored == raw_original and llm_corrected == locked_original:
            return LineGuardResult(accepted=False, unchanged=True, restored_text=restored)

        reasons: list[ReasonCode] = []

        # Low LLM confidence applies ONLY to changed lines (B1)
        if llm_confidence is not None and float(llm_confidence) < cls.MIN_LLM_CONFIDENCE:
            reasons.append(ReasonCode.REJECT_LOW_LLM_CONFIDENCE)

        # 2. Digit multiset invariance (strictly forbid inventing/mutating locked or new digit tokens;
        #    allow removing OCR-garbled digits belonging to unlocked tokens like '14p', 'g9', '1bể', '11')
        orig_digits = Counter(ch for ch in raw_original if ch.isdigit())
        restored_digits = Counter(ch for ch in restored if ch.isdigit())
        locked_digits = Counter(ch for tok in locked_tokens for ch in tok.text if ch.isdigit())
        locked_texts = {tok.text for tok in locked_tokens}

        locked_counts = Counter(tok.text for tok in locked_tokens)

        def _all_digit_tokens(s: str) -> Counter[str]:
            out: Counter[str] = Counter()
            for w in s.split():
                clean_w = w.strip(".,;:!?\"'()[]{}")
                if clean_w and any(ch.isdigit() for ch in clean_w):
                    out[clean_w] += 1
            return out

        orig_unlocked_toks = _all_digit_tokens(raw_original) - locked_counts
        restored_unlocked_toks = _all_digit_tokens(restored) - locked_counts

        def _is_subseq(sub: str, full: str) -> bool:
            it_f = iter(full)
            return all(ch in it_f for ch in sub)

        # Allow recovering intact maximal numeric runs that were glued into mixed OCR noise
        # (e.g. '6Kb8-168-đều' -> '6 ...', '1?nắm' -> '1 năm', '28?1' -> '28-1'), repairing an
        # unlocked 5-8 digit OCR-corrupted year ('149733' -> '1973'), or repairing an unlocked
        # out-of-range calendar day ('Ngày 37' -> 'Ngày 27') without inventing new digits.
        orig_mixed_runs: Counter[str] = Counter()
        for raw_u_tok, cnt in orig_unlocked_toks.items():
            if not raw_u_tok.isdigit():
                for run in re.findall(r"\d+", raw_u_tok):
                    orig_mixed_runs[run] += cnt
            elif 5 <= len(raw_u_tok) <= 8:
                for cand_t in restored_unlocked_toks:
                    if (
                        cand_t.isdigit()
                        and len(cand_t) == 4
                        and cand_t.startswith(("18", "19", "20"))
                        and _is_subseq(cand_t, raw_u_tok)
                    ):
                        orig_mixed_runs[cand_t] += cnt
            elif len(raw_u_tok) == 2 and int(raw_u_tok) > 31:
                for cand_t in restored_unlocked_toks:
                    if (
                        cand_t.isdigit()
                        and len(cand_t) == 2
                        and 1 <= int(cand_t) <= 31
                        and (cand_t[0] == raw_u_tok[0] or cand_t[1] == raw_u_tok[1])
                    ):
                        orig_mixed_runs[cand_t] += cnt
                        for ch_d in cand_t:
                            orig_digits[ch_d] += cnt

        avail_runs = Counter(orig_mixed_runs)
        unlocked_digit_invented = False
        for t, r_cnt in restored_unlocked_toks.items():
            extra_cnt = r_cnt - orig_unlocked_toks[t]
            if extra_cnt <= 0:
                continue
            if t.isdigit() and avail_runs[t] >= extra_cnt:
                avail_runs[t] -= extra_cnt
                continue
            if re.fullmatch(r"\d+(?:[/-]\d+)+", t):
                sub_runs = re.findall(r"\d+", t)
                if sub_runs and all(avail_runs[sr] >= extra_cnt for sr in sub_runs):
                    for sr in sub_runs:
                        avail_runs[sr] -= extra_cnt
                    continue
            unlocked_digit_invented = True
            break

        digit_added = any(restored_digits[d] > orig_digits[d] for d in restored_digits)
        locked_digit_lost = any(restored_digits[d] < locked_digits[d] for d in locked_digits)
        if digit_added or locked_digit_lost or unlocked_digit_invented:
            reasons.append(ReasonCode.REJECT_DIGIT_MULTISET_CHANGED)

        # 3. Protected tokens invariance:
        #    - Every originally locked token must remain intact in exact order (no deletion/mutation/adjacent char gluing).
        #    - Allow pre-existing unlocked numbers (e.g. '4' in '4 tác', '1' in '1?nắm', '28-1' from '28?1', or '1973' from '149733')
        #      to newly match NUMERIC_TOKEN or DATE_PHRASE when neighboring Vietnamese typos are repaired,
        #      and treat NUMERIC_TOKEN <-> DATE_PHRASE on the same number text as equivalent.
        def _canon_prot_type(ptype: str) -> str:
            return "NUM_OR_DATE" if ptype in ("NUMERIC_TOKEN", "DATE_PHRASE") else ptype

        expected_prot = [(tok.text, _canon_prot_type(tok.type)) for tok in locked_tokens]
        actual_prot = [(s[2], _canon_prot_type(s[3])) for s in detect_protected_spans(restored)]
        it_actual = iter(actual_prot)
        prot_missing_or_reordered = not all(ep in it_actual for ep in expected_prot)
        prot_unexpected_addition = False
        if not prot_missing_or_reordered:
            extra_prot = Counter(actual_prot) - Counter(expected_prot)
            for (ext_txt, ext_type), cnt in extra_prot.items():
                if ext_type != "NUM_OR_DATE":
                    prot_unexpected_addition = True
                    break
                allowed_existing = orig_unlocked_toks[ext_txt] + (orig_mixed_runs[ext_txt] if ext_txt.isdigit() else 0)
                if allowed_existing < cnt and re.fullmatch(r"\d+(?:[/-]\d+)+", ext_txt):
                    sub_runs = re.findall(r"\d+", ext_txt)
                    if sub_runs and all((orig_mixed_runs[sr] + orig_unlocked_toks[sr]) >= cnt for sr in sub_runs):
                        allowed_existing = cnt
                if allowed_existing < cnt:
                    prot_unexpected_addition = True
                    break
        if prot_missing_or_reordered or prot_unexpected_addition:
            reasons.append(ReasonCode.REJECT_PROTECTED_TOKEN_MUTATED)

        # Measure OCR corruption severity via Vietnamese syllable validator so heavy OCR recovery is not unfairly blocked
        try:
            from vietnamese_syllable_validator import invalid_syllables
        except ImportError:
            from worker.vietnamese_syllable_validator import invalid_syllables

        orig_words = raw_original.split()
        n_orig_words = max(1, len(orig_words))
        orig_inv_count = len(invalid_syllables(raw_original, allow_loanwords=True))
        orig_unlocked_digit_noise = sum(
            1
            for w in orig_words
            if any(ch.isdigit() for ch in w) and w.strip(".,;:!?\"'()[]{}") not in locked_texts
        )
        orig_corrupt = orig_inv_count + orig_unlocked_digit_noise
        corrupt_ratio = orig_corrupt / float(n_orig_words)

        restored_words = restored.split()
        restored_inv_count = len(invalid_syllables(restored, allow_loanwords=True))
        restored_unlocked_digit_noise = sum(
            1
            for w in restored_words
            if any(ch.isdigit() for ch in w) and w.strip(".,;:!?\"'()[]{}") not in locked_texts
        )
        restored_corrupt = restored_inv_count + restored_unlocked_digit_noise
        is_heavy_recovery = (orig_corrupt >= 1 and corrupt_ratio >= 0.15 and restored_corrupt < orig_corrupt)

        # 4. Length bounds (B10 for < 20 chars, [0.80, 1.25] for >= 20 chars; adaptive for heavy OCR recovery)
        orig_len = len(raw_original)
        corr_len = len(restored)
        if orig_len < 20:
            max_delta = 8 if is_heavy_recovery else 4
            if abs(corr_len - orig_len) > max_delta:
                reasons.append(ReasonCode.REJECT_LENGTH_RATIO_OUT_OF_BOUNDS)
        else:
            ratio = corr_len / max(1, orig_len)
            lo_ratio, hi_ratio = (0.70, 1.40) if is_heavy_recovery else (0.80, 1.25)
            if ratio < lo_ratio or ratio > hi_ratio:
                reasons.append(ReasonCode.REJECT_LENGTH_RATIO_OUT_OF_BOUNDS)

        # 5 & 6. Word insertion/deletion and weighted syllable edit distance
        if allow_paragraph_refinement:
            # For merged paragraph blocks in auto_apply mode: allow hyphen/compound merges
            # while enforcing base-character similarity (relaxed proportionally on heavily corrupted OCR).
            norm_orig = re.sub(r"[\s\-.,;:!?()\"']+", "", _strip_accents(raw_original).lower())
            norm_corr = re.sub(r"[\s\-.,;:!?()\"']+", "", _strip_accents(restored).lower())
            if norm_orig and norm_corr:
                sim = difflib.SequenceMatcher(a=norm_orig, b=norm_corr, autojunk=False).ratio()
                min_sim = max(0.45, 0.80 - 0.60 * corrupt_ratio) if is_heavy_recovery else 0.80
                if sim < min_sim:
                    reasons.append(ReasonCode.REJECT_EXCESSIVE_CHAR_EDIT)
        else:
            orig_tokens = orig_words
            corr_tokens = restored_words
            matcher = difflib.SequenceMatcher(a=orig_tokens, b=corr_tokens, autojunk=False)
            word_err = False
            char_err = False

            def _is_ocr_noise_tok(w: str) -> bool:
                if any(ch.isdigit() for ch in w) and any(ch.isalpha() for ch in w):
                    return True
                if w.isdigit() and len(w) <= 2:
                    return True
                return bool(invalid_syllables(w, allow_loanwords=True))

            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == "equal":
                    continue
                if tag in ("insert", "delete"):
                    word_err = True
                    span_txt = (
                        " ".join(corr_tokens[j1:j2]) if tag == "insert" else " ".join(orig_tokens[i1:i2])
                    )
                    if syllable_edit_cost(span_txt, "") > cls.MAX_SYLLABLE_EDIT_COST:
                        char_err = True
                    continue
                old_slice = orig_tokens[i1:i2]
                new_slice = corr_tokens[j1:j2]
                if len(old_slice) == len(new_slice):
                    for ow, nw in zip(old_slice, new_slice):
                        lim = 5.0 if _is_ocr_noise_tok(ow) else cls.MAX_SYLLABLE_EDIT_COST
                        if syllable_edit_cost(ow, nw) > lim:
                            char_err = True
                elif (len(old_slice), len(new_slice)) in ((1, 2), (2, 1)):
                    old_joined = "".join(old_slice)
                    new_joined = "".join(new_slice)
                    if len(old_slice) == 1 and _is_ocr_noise_tok(old_slice[0]) and syllable_edit_cost(old_joined, new_joined) <= 5.5:
                        continue
                    if syllable_edit_cost(old_joined, new_joined) > cls.MAX_SYLLABLE_EDIT_COST:
                        word_err = True
                        char_err = True
                else:
                    word_err = True
                    if syllable_edit_cost(" ".join(old_slice), " ".join(new_slice)) > cls.MAX_SYLLABLE_EDIT_COST:
                        char_err = True

            if word_err:
                reasons.append(ReasonCode.REJECT_WORD_INSERTION_OR_DELETION)
            if char_err:
                reasons.append(ReasonCode.REJECT_EXCESSIVE_CHAR_EDIT)

        if reasons:
            # Preserve insertion order without duplicates
            deduped = tuple(dict.fromkeys(reasons))
            return LineGuardResult(accepted=False, restored_text=restored, reason_codes=deduped)

        ops = tuple(compute_ops(raw_original, restored))
        risk = classify_risk(ops)
        return LineGuardResult(
            accepted=True,
            unchanged=False,
            restored_text=restored,
            ops=ops,
            risk=risk,
        )

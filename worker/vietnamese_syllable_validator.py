"""Deterministic Vietnamese syllable phonotactic and orthographic validator."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

InvalidReason = Literal[
    "MULTI_TONE",
    "NON_VIET_CHAR",
    "INVALID_RHYME",
    "STOP_FINAL_TONE",
    "ORTHO_CONSONANT",
    "BOUND_NUCLEUS",
]


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    token: str
    reason: InvalidReason


_TONE_MARKS: dict[str, str] = {
    "\u0300": "huyen",
    "\u0301": "sac",
    "\u0303": "nga",
    "\u0309": "hoi",
    "\u0323": "nang",
}

_VIET_BASE_ALPHABET: frozenset[str] = frozenset("aăâbcdđeêghiklmnoôơpqrstuưvxy")

_VALID_INITIALS: tuple[str, ...] = (
    "ngh",
    "ng",
    "nh",
    "ch",
    "gh",
    "gi",
    "kh",
    "ph",
    "qu",
    "th",
    "tr",
    "b",
    "c",
    "d",
    "đ",
    "g",
    "h",
    "k",
    "l",
    "m",
    "n",
    "p",
    "r",
    "s",
    "t",
    "v",
    "x",
    "",
)

_BOUND_NUCLEI: frozenset[str] = frozenset(
    {"ă", "â", "iê", "uô", "ươ", "yê", "oă", "uă", "uâ", "uyê"}
)

_VALID_RHYMES: frozenset[str] = frozenset(
    {
        # Nguyên âm đơn (12)
        "a", "ă", "â", "e", "ê", "i", "o", "ô", "ơ", "u", "ư", "y",
        # Vần mở / bán âm cuối (D3.1 + uơ cho "thuở", "huơ")
        "ai", "ao", "au", "ay", "âu", "ây", "eo", "êu", "ia", "iê", "iu", "iêu",
        "oa", "oă", "oe", "oi", "ôi", "ơi", "oo", "ua", "uă", "uâ", "ưa", "uê",
        "ui", "ưi", "uô", "uơ", "ươ", "ưu", "uy", "oai", "oao", "oay", "oeo",
        "uây", "uôi", "ươi", "ươu", "uya", "uyê", "uyu", "ya", "yê", "yêu",
        # Vần kết thúc bằng -m (18)
        "am", "ăm", "âm", "em", " êm".strip(), "im", "om", "ôm", "ơm", "um",
        "ưm", "iêm", "oam", "oăm", "oem", "uôm", "ươm", "yêm",
        # Vần kết thúc bằng -n (21)
        "an", "ăn", "ân", "en", "ên", "in", "on", "ôn", "ơn", "un", "ưn",
        "iên", "oan", "oăn", "oen", "uân", "uôn", "ươn", "uyên", "uyn", "yên",
        # Vần kết thúc bằng -ng (16)
        "ang", "ăng", "âng", "eng", "ong", "ông", "ưng", "ung", "iêng", "oang",
        "oăng", "oong", "uâng", "uông", "ương", "yêng",
        # Vần kết thúc bằng -nh (6)
        "anh", "ênh", "inh", "oanh", "uênh", "uynh",
        # Vần kết thúc bằng -p (19)
        "ap", "ăp", "âp", "ep", "êp", "ip", "op", "ôp", "ơp", "up", "ưp",
        "iêp", "oap", "oăp", "oep", "uôp", "ươp", "uyp", "yêp",
        # Vần kết thúc bằng -t (21)
        "at", "ăt", "ât", "et", "êt", "it", "ot", "ôt", "ơt", "ut", "ưt",
        "iêt", "oat", "oăt", "oet", "uât", "uôt", "ươt", "uyêt", "uyt", "yêt",
        # Vần kết thúc bằng -c (15)
        "ac", "ăc", "âc", "ec", "oc", "ôc", "ơc", "uc", "ưc", "iêc", "oac",
        "oăc", "ooc", "uôc", "ươc",
        # Vần kết thúc bằng -ch (6)
        "ach", "êch", "ich", "oach", "uêch", "uych",
    }
)

_STOP_FINALS: tuple[str, ...] = ("ch", "p", "t", "c")

_FRONT_VOWELS: frozenset[str] = frozenset({"i", "e", "ê", "y"})

# Từ viết tắt đơn vị & hành chính chữ thường (D3.2 mục 4 + Quyết định B5: thêm tt, stt)
UNIT_OR_ABBR: frozenset[str] = frozenset(
    {
        "km", "cm", "mm", "dm", "kg", "mg", "ml", "gb", "mb", "kb",
        "hz", "mhz", "ghz", "px", "ms", "tp", "tw", "ubnd", "hđnd",
        "tnhh", "qltt", "ths", "ts", "pgs", "gs", "cn", "ks", "đc", "đt",
        "tt", "stt",
    }
)

# Nguồn gốc: Tự soạn ~138 thuật ngữ CNTT/hành chính tiếng Anh phổ biến từ kiến thức chung
# (theo D3.2 & Quyết định B5; KHÔNG bổ sung dựa trên danh sách FP của datasets/raw_docs).
COMMON_LOANWORDS: frozenset[str] = frozenset(
    {
        "api", "app", "backend", "frontend", "fullstack", "client", "server",
        "service", "services", "microservice", "database", "sql", "nosql",
        "schema", "query", "table", "json", "xml", "yaml", "html", "css",
        "http", "https", "rest", "restful", "graphql", "grpc", "endpoint",
        "router", "route", "controller", "middleware", "payload", "header",
        "headers", "body", "response", "request", "status", "token", "jwt",
        "oauth", "session", "cookie", "cache",
        "java", "python", "javascript", "typescript", "node", "nodejs", "react",
        "nextjs", "spring", "boot", "gradle", "maven", "docker", "compose",
        "container", "image", "kubernetes", "linux", "ubuntu", "windows",
        "git", "github", "commit", "branch", "merge", "pull", "push", "build",
        "deploy", "pipeline", "worker", "queue", "thread", "async", "sync",
        "port", "host", "localhost", "domain", "ssl", "tls", "ssh", "cli",
        "sdk", "ide",
        "code", "file", "files", "folder", "path", "config", "debug", "error",
        "test", "tests", "unit", "mock", "stub", "script", "module", "package",
        "import", "class", "object", "function", "method", "model", "models",
        "dataset", "train", "training", "batch", "epoch", "loss", "score",
        "precision", "recall", "metric", "ocr", "llm", "vlm", "prompt", "pdf",
        "docx", "png", "jpg", "jpeg", "email", "user", "admin", "login", "logout",
        "wifi", "internet", "web", "website", "url", "uri", "data",
    }
)

_OUTER_LEFT_PUNCT = "\"'([{«“‘"
_OUTER_RIGHT_PUNCT = ".,;:!?\"')]}»”’…"
_TOKEN_ID_CHARS: frozenset[str] = frozenset({"@", "/", "_", "\\", ":"})
_LIST_ITEM_RE = re.compile(r"^[a-zA-ZđĐ][.)]$")
_ABBR_DOT_RE = re.compile(r"^(?:[a-zA-ZđĐ]\.){2,}$")


def _has_non_latin_alpha(token: str) -> bool:
    for ch in token:
        if ch.isalpha() and "LATIN" not in unicodedata.name(ch, ""):
            return True
    return False


def _is_camel_or_pascal(token: str) -> bool:
    return len(token) > 1 and any(ch.isupper() for ch in token[1:])


def _should_exempt_whole_token(raw_tok: str) -> bool:
    """Exempt tokens containing digits, URL/path symbols, list prefixes, or non-Latin scripts."""
    if not raw_tok:
        return True
    if any(ch.isdigit() for ch in raw_tok):
        return True
    if _has_non_latin_alpha(raw_tok):
        return True
    if _LIST_ITEM_RE.match(raw_tok):
        return True

    trimmed = raw_tok.lstrip(_OUTER_LEFT_PUNCT).rstrip(".,;!?\"')]}»”’…")
    if _ABBR_DOT_RE.match(raw_tok) or _ABBR_DOT_RE.match(trimmed):
        return True

    # Inspect ':' only after stripping trailing ':' before whitespace so 's:' (GC3) still checks 's',
    # while 'https://...' and 'foo:bar' keep ':' internally and are exempted.
    trimmed_no_trailing_colon = trimmed.rstrip(":")
    if any(ch in _TOKEN_ID_CHARS for ch in trimmed_no_trailing_colon):
        return True
    return False


def _is_exempt_subtoken(sub: str, allow_loanwords: bool) -> bool:
    if not sub:
        return True
    if any(ch.isdigit() for ch in sub) or _has_non_latin_alpha(sub):
        return True
    if not any(ch.isalpha() for ch in sub):
        return True
    if sub.isupper() or _is_camel_or_pascal(sub):
        return True
    lower = unicodedata.normalize("NFC", sub.lower())
    if lower in UNIT_OR_ABBR:
        return True
    if allow_loanwords and lower in COMMON_LOANWORDS:
        return True
    return False


def _check_orthography(initial: str, rhyme: str) -> bool:
    if not rhyme:
        return True
    first = rhyme[0]
    if initial in ("k", "gh", "ngh"):
        return first in _FRONT_VOWELS
    if initial == "c":
        return first not in _FRONT_VOWELS
    if initial == "g":
        return first not in ("e", "ê", "y")
    if initial == "ng":
        return first not in _FRONT_VOWELS
    return True


def _validate_split(initial: str, rhyme: str, tone: str) -> InvalidReason | None:
    if rhyme in _BOUND_NUCLEI:
        return "BOUND_NUCLEUS"
    if rhyme not in _VALID_RHYMES:
        return "INVALID_RHYME"
    if not _check_orthography(initial, rhyme):
        return "ORTHO_CONSONANT"
    if rhyme.endswith(_STOP_FINALS) and tone not in ("sac", "nang"):
        return "STOP_FINAL_TONE"
    return None


def _candidate_splits(base: str) -> list[tuple[str, str]]:
    splits: list[tuple[str, str]] = []
    if base.startswith("gi"):
        rest = base[2:]
        if rest:
            splits.append(("gi", rest))
        splits.append(("g", "i" + rest))
        return splits
    if base.startswith("qu"):
        rest = base[2:]
        if rest and not rest.startswith("u"):
            splits.append(("qu", rest))
        if rest.startswith("y"):
            splits.append(("qu", "u" + rest))
        return splits

    for init in _VALID_INITIALS:
        if init == "":
            splits.append(("", base))
        elif base.startswith(init):
            splits.append((init, base[len(init) :]))
    return splits


_REASON_PRIORITY: dict[InvalidReason, int] = {
    "STOP_FINAL_TONE": 4,
    "ORTHO_CONSONANT": 3,
    "BOUND_NUCLEUS": 2,
    "INVALID_RHYME": 1,
}


def check_syllable(word: str) -> InvalidReason | None:
    """Validate a single non-exempted word; returns InvalidReason or None if valid."""
    nfd = unicodedata.normalize("NFD", word.lower())
    tones = [ch for ch in nfd if ch in _TONE_MARKS]
    if len(tones) > 1:
        return "MULTI_TONE"
    tone = _TONE_MARKS[tones[0]] if tones else "ngang"

    base_chars = [ch for ch in nfd if ch not in _TONE_MARKS]
    base = unicodedata.normalize("NFC", "".join(base_chars))
    if not base or any(ch not in _VIET_BASE_ALPHABET for ch in base):
        return "NON_VIET_CHAR"

    if base.startswith("q") and not base.startswith("qu"):
        return "ORTHO_CONSONANT"

    splits = _candidate_splits(base)
    if not splits:
        return "INVALID_RHYME"

    best_err: InvalidReason | None = None
    best_rank = -1
    for initial, rhyme in splits:
        err = _validate_split(initial, rhyme, tone)
        if err is None:
            return None
        rank = _REASON_PRIORITY.get(err, 0)
        if rank > best_rank:
            best_rank = rank
            best_err = err
    return best_err


def _iter_subtokens(raw_tok: str, tok_start: int) -> list[tuple[int, int, str]]:
    """Split a non-exempted whitespace token into hyphen/dot-separated word spans."""
    left_trim = len(raw_tok) - len(raw_tok.lstrip(_OUTER_LEFT_PUNCT))
    right_trim = len(raw_tok) - len(raw_tok.rstrip(_OUTER_RIGHT_PUNCT))
    end_idx = len(raw_tok) - right_trim
    if left_trim >= end_idx:
        return []
    core = raw_tok[left_trim:end_idx]
    core_start = tok_start + left_trim
    out: list[tuple[int, int, str]] = []
    for sub in re.finditer(r"[^\-–—.]+", core):
        sub_txt = sub.group(0).strip(_OUTER_LEFT_PUNCT + _OUTER_RIGHT_PUNCT)
        if not sub_txt:
            continue
        sub_offset = sub.group(0).find(sub_txt)
        s = core_start + sub.start() + sub_offset
        out.append((s, s + len(sub_txt), sub_txt))
    return out


def invalid_syllables(text: str, allow_loanwords: bool = True) -> list[Span]:
    """Return invalid Vietnamese syllable spans found in line `text`."""
    if not text or not text.strip():
        return []
    spans: list[Span] = []
    for match in re.finditer(r"\S+", text):
        raw_tok = match.group(0)
        if _should_exempt_whole_token(raw_tok):
            continue
        for s, e, sub_txt in _iter_subtokens(raw_tok, match.start()):
            if _is_exempt_subtoken(sub_txt, allow_loanwords=allow_loanwords):
                continue
            reason = check_syllable(sub_txt)
            if reason is not None:
                spans.append(Span(start=s, end=e, token=sub_txt, reason=reason))
    return spans

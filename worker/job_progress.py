"""Pure stdlib helpers for worker job options parsing and monotonic progress calculation."""

from __future__ import annotations

from typing import Any

# Subset of Backend JobDtos.JobHeartbeatRequest stage @Pattern values used in OCR jobs.
OCR_STAGE_ORDER: tuple[str, ...] = (
    "PREPROCESSING",
    "RESTORING",
    "OCR",
    "LAYOUT",
    "LLM_POSTPROCESS",
)

STAGE_FRACTIONS: dict[str, float] = {
    "PREPROCESSING": 0.0,
    "RESTORING": 0.25,
    "OCR": 0.50,
    "LAYOUT": 0.75,
    "LLM_POSTPROCESS": 0.90,
}

VALID_HEARTBEAT_STAGES: frozenset[str] = frozenset(
    OCR_STAGE_ORDER + ("EXPORTING",)
)


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no", ""):
            return False
    return default


def parse_job_options(job_or_options: dict[str, Any] | None) -> dict[str, bool]:
    """Extract only supported Commit-7 boolean flags from a claimed job or options dict.

    Intentionally ignores unsupported/legacy option keys (e.g. deskew, restoreMode).
    """
    if not isinstance(job_or_options, dict):
        opts: Any = {}
    elif "options" in job_or_options and (
        isinstance(job_or_options.get("options"), dict) or job_or_options.get("options") is None
    ):
        opts = job_or_options.get("options") or {}
    else:
        opts = job_or_options

    if not isinstance(opts, dict):
        opts = {}

    llm_raw = opts.get("llmPostprocess") if "llmPostprocess" in opts else opts.get("useLlm")
    return {
        "llmPostprocess": _as_bool(llm_raw, False),
        "localOnly": _as_bool(opts.get("localOnly"), False),
        "redactPii": _as_bool(opts.get("redactPii"), False),
    }


def parse_scan_options(
    job_or_options: dict[str, Any] | None,
    *,
    default_llm: bool | None = None,
    default_unet: bool | None = None,
) -> dict[str, bool]:
    """Parse per-job scan & postprocess flags (`useUnet`, `useLlm`/`llmPostprocess`, `localOnly`, `redactPii`)."""
    import os

    if default_llm is None:
        raw_env = (
            os.getenv("ENABLE_LLM_POSTPROCESS")
            or os.getenv("LLM_POSTPROCESS_DEFAULT")
            or os.getenv("LLM_POSTPROCESS_ENABLED")
            or "true"
        )
        default_llm = _as_bool(raw_env, True)

    if default_unet is None:
        raw_unet_env = os.getenv("ENABLE_IMAGE_RESTORATION", "true")
        default_unet = _as_bool(raw_unet_env, True)

    if not isinstance(job_or_options, dict):
        opts: Any = {}
    elif "options" in job_or_options and (
        isinstance(job_or_options.get("options"), dict) or job_or_options.get("options") is None
    ):
        opts = job_or_options.get("options") or {}
    else:
        opts = job_or_options

    if not isinstance(opts, dict):
        opts = {}

    if "llmPostprocess" in opts:
        use_llm = _as_bool(opts.get("llmPostprocess"), default_llm)
    elif "useLlm" in opts:
        use_llm = _as_bool(opts.get("useLlm"), default_llm)
    else:
        use_llm = bool(default_llm)

    if "useUnet" in opts:
        use_unet = _as_bool(opts.get("useUnet"), default_unet)
    elif str(opts.get("restoreMode") or "").strip().lower() in ("unet", "unet_v5_1", "unet_hybrid"):
        use_unet = True
    else:
        use_unet = bool(default_unet)

    return {
        "llmPostprocess": use_llm,
        "useLlm": use_llm,
        "useUnet": use_unet,
        "localOnly": _as_bool(opts.get("localOnly"), False),
        "redactPii": _as_bool(opts.get("redactPii"), False),
    }


def compute_progress(page_idx: int, n_pages: int, stage: str) -> int:
    """Compute monotonically non-decreasing OCR progress in [5, 95].

    Args:
        page_idx: Zero-based page index (clamped into ``[0, max(1, n_pages) - 1]``).
        n_pages: Total number of pages in the job (values <= 0 are treated as 1).
        stage: Current pipeline stage name.
    """
    total = max(1, int(n_pages))
    idx = max(0, min(int(page_idx), total - 1))
    frac = STAGE_FRACTIONS.get(str(stage or "").upper(), 0.0)
    raw = 5 + int(round(90.0 * ((idx + frac) / float(total))))
    return max(5, min(95, raw))

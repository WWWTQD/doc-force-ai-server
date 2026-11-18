"""LLM/VLM client protocol, deterministic offline MockLLMClient, and client factory."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

LOG = logging.getLogger("docforge.worker.llm_client")

_PLACEHOLDER_RE = re.compile(r"⟦T\d+⟧")
_DEFAULT_FIXTURE_PATH = Path(__file__).resolve().parent / "mock_fixtures" / "golden_cases.json"
PROMPT_VERSION = "v1"


@dataclass(frozen=True)
class LLMUsageLog:
    provider_type: str  # "mock" | "cloud" | "local"
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thought_tokens: int = 0
    latency_ms: int = 0
    cache_hit: bool = False
    status: str = "ok"  # "ok" | "timeout" | "error" | "skipped" | "rate_limited" | "FAILED_SAFE" | ...
    rate_limited: bool = False
    error_reason: str | None = None


@runtime_checkable
class LLMClient(Protocol):
    is_cloud: bool
    provider_name: str
    model_name: str

    def correct_lines_batch(
        self,
        page_payload: dict[str, Any],
        *,
        local_only: bool = False,
        deadline_ts: float,
    ) -> tuple[Any, LLMUsageLog]:
        ...

    def verify_crops_vlm(
        self,
        items: list[dict[str, Any]],
        *,
        local_only: bool = False,
        deadline_ts: float,
    ) -> tuple[Any, LLMUsageLog]:
        ...


def load_golden_fixtures(path: Path | None = None) -> dict[str, Any]:
    target = path or _DEFAULT_FIXTURE_PATH
    with target.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _canonicalize_placeholders(locked_text: str) -> tuple[str, list[str]]:
    """Map page-unique placeholders (e.g. ⟦T4⟧, ⟦T5⟧) to canonical ⟦T1⟧, ⟦T2⟧... for fixture lookup."""
    found = _PLACEHOLDER_RE.findall(locked_text)
    if not found:
        return locked_text, []
    canonical = locked_text
    # Two-pass substitution avoids collision when e.g. T2 becomes T1 while T1 exists later.
    tokens: list[str] = []
    for idx, ph in enumerate(found, start=1):
        tokens.append(ph)
        canonical = canonical.replace(ph, f"\x00CANON_{idx}\x00", 1)
    for idx in range(1, len(tokens) + 1):
        canonical = canonical.replace(f"\x00CANON_{idx}\x00", f"⟦T{idx}⟧", 1)
    return canonical, tokens


def _restore_page_placeholders(canonical_corrected: str, page_placeholders: list[str]) -> str:
    if not page_placeholders:
        return canonical_corrected
    out = canonical_corrected
    for idx in range(1, len(page_placeholders) + 1):
        out = out.replace(f"⟦T{idx}⟧", f"\x00PAGE_{idx}\x00", 1)
    for idx, actual in enumerate(page_placeholders, start=1):
        out = out.replace(f"\x00PAGE_{idx}\x00", actual, 1)
    return out


class MockLLMClient:
    """Zero-network deterministic mock LLM/VLM client (Decision B13 & R4)."""

    is_cloud: bool = False
    provider_name: str = "mock"
    needs_crop_image: bool = False

    def __init__(
        self,
        mode: str = "normal",
        fixture_path: Path | None = None,
        adversarial_step: int | None = None,
    ) -> None:
        normalized_mode = (mode or "normal").strip().lower()
        if normalized_mode not in ("normal", "adversarial"):
            normalized_mode = "normal"
        self.mode = normalized_mode
        self.model_name = f"mock-{self.mode}"
        self._fixtures = load_golden_fixtures(fixture_path)
        self._call_counter = 0
        self._fixed_adversarial_step = adversarial_step

    def correct_lines_batch(
        self,
        page_payload: dict[str, Any],
        *,
        local_only: bool = False,
        deadline_ts: float,
    ) -> tuple[Any, LLMUsageLog]:
        del local_only, deadline_ts
        lines = [ln for ln in page_payload.get("lines", []) if ln.get("isCandidate")]
        if self.mode == "adversarial":
            return self._adversarial_batch(lines)

        text_table = self._fixtures.get("text_llm", {})
        resp_lines: list[dict[str, Any]] = []
        for ln in lines:
            raw_locked = ln.get("text", "")
            canon_key, orig_phs = _canonicalize_placeholders(raw_locked)
            entry = text_table.get(canon_key)
            if entry is None:
                resp_lines.append(
                    {
                        "lineId": ln["lineId"],
                        "corrected": raw_locked,
                        "confidence": 0.99,
                        "changes": [],
                    }
                )
            else:
                corrected = _restore_page_placeholders(entry["corrected"], orig_phs)
                resp_lines.append(
                    {
                        "lineId": ln["lineId"],
                        "corrected": corrected,
                        "confidence": float(entry.get("confidence", 0.95)),
                        "changes": list(entry.get("changes", [])),
                    }
                )
        return (
            {"lines": resp_lines},
            LLMUsageLog(provider_type="mock", model=self.model_name, status="ok"),
        )

    def _adversarial_batch(self, candidate_lines: list[dict[str, Any]]) -> tuple[Any, LLMUsageLog]:
        step = (
            self._fixed_adversarial_step
            if self._fixed_adversarial_step is not None
            else (self._call_counter % 4)
        )
        self._call_counter += 1
        usage = LLMUsageLog(provider_type="mock", model=self.model_name, status="ok")

        # 4: malformed JSON string
        if step == 3:
            return "{invalid_json_from_adversarial_mock", usage

        # 2: line count mismatch (GC11)
        if step == 1:
            truncated = [
                {
                    "lineId": candidate_lines[0]["lineId"],
                    "corrected": candidate_lines[0].get("text", ""),
                    "confidence": 0.95,
                    "changes": [],
                }
            ] if len(candidate_lines) > 1 else []
            return {"lines": truncated}, usage

        out_lines: list[dict[str, Any]] = []
        for idx, ln in enumerate(candidate_lines):
            txt = ln.get("text", "")
            if idx == 0:
                if step == 0:
                    # 1: mutate digit / remove placeholder (GC10)
                    phs = _PLACEHOLDER_RE.findall(txt)
                    mutated = txt.replace(phs[0], "08/01/2021", 1) if phs else f"{txt} 999"
                    out_lines.append(
                        {
                            "lineId": ln["lineId"],
                            "corrected": mutated,
                            "confidence": 0.95,
                            "changes": [{"from": txt, "to": mutated, "reason": "mutate digits"}],
                        }
                    )
                    continue
                if step == 2:
                    # 3: obey injected command (GC12) while keeping placeholders so length/edit guards fire
                    phs = _PLACEHOLDER_RE.findall(txt)
                    injected = f"{' '.join(phs)} HACKED".strip()
                    out_lines.append(
                        {
                            "lineId": ln["lineId"],
                            "corrected": injected,
                            "confidence": 0.99,
                            "changes": [{"from": txt, "to": injected, "reason": "injected"}],
                        }
                    )
                    continue
            out_lines.append(
                {
                    "lineId": ln["lineId"],
                    "corrected": txt,
                    "confidence": 0.99,
                    "changes": [],
                }
            )
        return {"lines": out_lines}, usage

    def verify_crops_vlm(
        self,
        items: list[dict[str, Any]],
        *,
        local_only: bool = False,
        deadline_ts: float,
    ) -> tuple[Any, LLMUsageLog]:
        del local_only, deadline_ts
        usage = LLMUsageLog(provider_type="mock", model=self.model_name, status="ok")
        out_lines: list[dict[str, Any]] = []
        if self.mode == "adversarial":
            for it in items:
                out_lines.append(
                    {
                        "lineId": it["lineId"],
                        "transcription": "HACKED 99999 88888",
                        "confidence": 0.95,
                        "uncertain": False,
                        "uncertainSpans": [],
                        "changes": [],
                    }
                )
            return {"lines": out_lines}, usage

        vlm_table = self._fixtures.get("vlm", {})
        for it in items:
            line_id = it["lineId"]
            current_text = str(it.get("currentText", ""))
            entry = vlm_table.get(current_text)
            if entry is None:
                out_lines.append(
                    {
                        "lineId": line_id,
                        "transcription": current_text,
                        "confidence": 0.5,
                        "uncertain": True,
                        "uncertainSpans": [],
                        "changes": [],
                    }
                )
            else:
                out_lines.append(
                    {
                        "lineId": line_id,
                        "transcription": entry["transcription"],
                        "confidence": float(entry.get("confidence", 0.90)),
                        "uncertain": bool(entry.get("uncertain", False)),
                        "uncertainSpans": list(entry.get("uncertainSpans", [])),
                        "changes": list(entry.get("changes", [])),
                    }
                )
        return {"lines": out_lines}, usage


def build_llm_client(config: Any, options: dict[str, Any] | None = None) -> tuple[LLMClient | None, str | None]:
    """Instantiate LLMClient based on config.provider; enforces consent, localOnly, and fatal disable state."""
    opts = options or {}
    provider = str(getattr(config, "provider", getattr(config, "LLM_PROVIDER", "mock"))).strip().lower()
    if provider == "mock":
        mock_mode = str(getattr(config, "mock_mode", getattr(config, "LLM_MOCK_MODE", "normal")))
        return MockLLMClient(mode=mock_mode), None

    if provider == "gemini":
        local_only = bool(opts.get("localOnly", False))
        if local_only:
            return None, "LOCAL_ENDPOINT_UNAVAILABLE"

        cloud_consent = bool(
            getattr(config, "cloud_consent", getattr(config, "LLM_CLOUD_CONSENT", False))
        )
        if not cloud_consent:
            return None, "CLOUD_NOT_CONSENTED"

        try:
            from gemini_provider import GeminiProvider, get_shared_resilience_controller
        except ImportError:
            from worker.gemini_provider import GeminiProvider, get_shared_resilience_controller

        rpm_limit = int(getattr(config, "rpm_limit", getattr(config, "LLM_RPM_LIMIT", 5)))
        daily_budget = int(
            getattr(config, "daily_request_budget", getattr(config, "LLM_DAILY_REQUEST_BUDGET", 150))
        )
        ctrl = get_shared_resilience_controller(rpm_limit=rpm_limit, daily_budget=daily_budget)
        if ctrl.disabled_reason is not None:
            return None, ctrl.disabled_reason
        if ctrl.is_breaker_open():
            return None, "BREAKER_OPEN"

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            if ctrl.disable_provider("AUTH_ERROR"):
                LOG.warning("Gemini provider disabled: GEMINI_API_KEY is not set (AUTH_ERROR).")
            return None, "AUTH_ERROR"

        return GeminiProvider.from_config(config, api_key=api_key, controller=ctrl), None

    return None, "PROVIDER_NOT_IMPLEMENTED"

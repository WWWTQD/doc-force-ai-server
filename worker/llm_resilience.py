"""Pure, thread-safe resilience primitives for cloud LLM/VLM calls (C3, C4, C6, C8)."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class AcquireResult:
    """Outcome of attempting to acquire rate-limit, breaker, and daily budget."""

    allowed: bool
    reason: str | None = None  # "BREAKER_OPEN" | "DAILY_BUDGET_EXHAUSTED" | "RATE_LIMITED"
    waited_seconds: float = 0.0
    rate_limited: bool = False


def make_cache_key(
    *,
    model: str,
    prompt_version: str,
    payload: Any,
    image_bytes_list: Sequence[bytes] | None = None,
) -> str:
    """Compute deterministic SHA-256 cache key over model, prompt version, payload, and image hashes."""
    img_hashes = [
        hashlib.sha256(img or b"").hexdigest() for img in (image_bytes_list or ())
    ]
    canonical = json.dumps(
        {
            "model": model,
            "promptVersion": prompt_version,
            "payload": payload,
            "imageHashes": img_hashes,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class InMemoryLRUCache:
    """Thread-safe in-memory LRU cache (default 256 entries, never writes to disk)."""

    def __init__(self, max_entries: int = 256) -> None:
        self.max_entries = max(1, int(max_entries))
        self._store: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key not in self._store:
                return None
            self._store.move_to_end(key)
            return copy.deepcopy(self._store[key])

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = copy.deepcopy(value)
            self._store.move_to_end(key)
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


class LLMResilienceController:
    """Coordinates sliding-window RPM limiting, escalating circuit breaker, and daily budget."""

    COOLDOWN_SEQUENCE_SECONDS = (60.0, 300.0)

    def __init__(
        self,
        *,
        rpm_limit: int = 5,
        daily_budget: int = 150,
        window_seconds: float = 60.0,
        deadline_reserve_seconds: float = 2.0,
        clock: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self.rpm_limit = int(rpm_limit)
        self.daily_budget = max(0, int(daily_budget))
        self.window_seconds = float(window_seconds)
        self.deadline_reserve_seconds = float(deadline_reserve_seconds)
        self._clock = clock or time.monotonic
        self._sleep = sleep_fn or time.sleep

        self._lock = threading.Lock()
        self._timestamps: deque[float] = deque()
        self._daily_used: int = 0
        self._breaker_strikes: int = 0
        self._cooldown_until: float = 0.0
        self._permanent_breaker_open: bool = False
        self._disabled_reason: str | None = None
        self._thinking_disabled: bool = False
        self._warned_tags: set[str] = set()
        self._model_overrides: dict[str, str] = {}
        self._unavailable_models: set[str] = set()

    def get_model_override(self, kind: str) -> str | None:
        with self._lock:
            return self._model_overrides.get(kind)

    def set_model_override(self, kind: str, model_name: str) -> None:
        with self._lock:
            self._model_overrides[kind] = model_name

    def mark_model_unavailable(self, model_name: str) -> None:
        if not model_name:
            return
        with self._lock:
            self._unavailable_models.add(model_name)

    def is_model_unavailable(self, model_name: str) -> bool:
        if not model_name:
            return False
        with self._lock:
            return model_name in self._unavailable_models

    @property
    def disabled_reason(self) -> str | None:
        with self._lock:
            return self._disabled_reason

    @property
    def thinking_disabled(self) -> bool:
        with self._lock:
            return self._thinking_disabled

    @property
    def daily_used(self) -> int:
        with self._lock:
            return self._daily_used

    def disable_provider(self, reason: str) -> bool:
        """Disable cloud provider for the rest of the process; returns True only on first disable."""
        with self._lock:
            if self._disabled_reason == reason:
                return False
            self._disabled_reason = reason
            return True

    def disable_thinking(self) -> None:
        with self._lock:
            self._thinking_disabled = True

    def mark_warned_once(self, tag: str) -> bool:
        """Return True only the first time `tag` is checked (prevents per-page log spam)."""
        with self._lock:
            if tag in self._warned_tags:
                return False
            self._warned_tags.add(tag)
            return True

    def is_breaker_open(self) -> bool:
        now = self._clock()
        with self._lock:
            return self._permanent_breaker_open or now < self._cooldown_until

    def record_rate_limit_error(self) -> str:
        """Trip escalating circuit breaker on 429 / RESOURCE_EXHAUSTED: 60s -> 300s -> process."""
        now = self._clock()
        with self._lock:
            self._breaker_strikes += 1
            if self._breaker_strikes == 1:
                self._cooldown_until = now + self.COOLDOWN_SEQUENCE_SECONDS[0]
                return "COOLDOWN_60S"
            if self._breaker_strikes == 2:
                self._cooldown_until = now + self.COOLDOWN_SEQUENCE_SECONDS[1]
                return "COOLDOWN_300S"
            self._permanent_breaker_open = True
            return "OPEN_PROCESS"

    def acquire(self, *, deadline_ts: float) -> AcquireResult:
        """Acquire a request slot respecting breaker cooldown, daily budget, RPM window, and page deadline."""
        total_waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                if self._disabled_reason is not None:
                    return AcquireResult(allowed=False, reason=self._disabled_reason)

                if self._permanent_breaker_open or now < self._cooldown_until:
                    return AcquireResult(
                        allowed=False,
                        reason="BREAKER_OPEN",
                        waited_seconds=total_waited,
                        rate_limited=True,
                    )

                if self._daily_used >= self.daily_budget:
                    return AcquireResult(
                        allowed=False,
                        reason="DAILY_BUDGET_EXHAUSTED",
                        waited_seconds=total_waited,
                        rate_limited=True,
                    )

                if self.rpm_limit <= 0:
                    return AcquireResult(
                        allowed=False,
                        reason="RATE_LIMITED",
                        waited_seconds=total_waited,
                        rate_limited=True,
                    )

                cutoff = now - self.window_seconds
                while self._timestamps and self._timestamps[0] <= cutoff:
                    self._timestamps.popleft()

                if len(self._timestamps) < self.rpm_limit:
                    self._timestamps.append(now)
                    self._daily_used += 1
                    return AcquireResult(
                        allowed=True,
                        reason=None,
                        waited_seconds=total_waited,
                        rate_limited=False,
                    )

                next_slot_ts = self._timestamps[0] + self.window_seconds
                wait_needed = max(0.0, next_slot_ts - now)
                remaining = deadline_ts - now
                if wait_needed > (remaining - self.deadline_reserve_seconds):
                    return AcquireResult(
                        allowed=False,
                        reason="RATE_LIMITED",
                        waited_seconds=total_waited,
                        rate_limited=True,
                    )

            # Sleep outside the lock so other pages don't serialize on sleep and notice cooldown immediately.
            sleep_for = max(0.001, wait_needed)
            self._sleep(sleep_for)
            total_waited += sleep_for


def check_cloud_consent(
    *,
    is_cloud: bool,
    local_only: bool,
    cloud_consent: bool,
) -> str | None:
    """Enforce privacy/consent gates (B9 + C6) before any cloud request or client acquisition."""
    if not is_cloud:
        return None
    if local_only:
        return "LOCAL_ENDPOINT_UNAVAILABLE"
    if not cloud_consent:
        return "CLOUD_NOT_CONSENTED"
    return None

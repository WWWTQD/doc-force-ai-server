"""Unit tests for worker.llm_resilience (pure sliding window, breaker, budget, consent, LRU cache)."""

from __future__ import annotations

import threading

from worker.llm_resilience import (
    InMemoryLRUCache,
    LLMResilienceController,
    check_cloud_consent,
    make_cache_key,
)


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []
        self._lock = threading.Lock()

    def time(self) -> float:
        with self._lock:
            return self.now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.now += seconds

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self.sleeps.append(seconds)
            self.now += seconds


def test_sliding_window_wait_vs_deadline_skip() -> None:
    clk = FakeClock(start=100.0)
    ctrl = LLMResilienceController(
        rpm_limit=2,
        daily_budget=150,
        clock=clk.time,
        sleep_fn=clk.sleep,
    )

    # First 2 requests succeed immediately
    r1 = ctrl.acquire(deadline_ts=clk.time() + 15.0)
    clk.advance(5.0)
    r2 = ctrl.acquire(deadline_ts=clk.time() + 15.0)
    assert r1.allowed and r2.allowed
    assert clk.sleeps == []

    # At t=105, window holds [100, 105]. Next free slot is at t=160 (wait = 55s).
    # With 15s page deadline (remaining=15, reserve=2 -> max wait 13s), MUST skip immediately without sleeping!
    r3_skip = ctrl.acquire(deadline_ts=clk.time() + 15.0)
    assert not r3_skip.allowed
    assert r3_skip.reason == "RATE_LIMITED"
    assert r3_skip.rate_limited is True
    assert clk.sleeps == []

    # If remaining budget permits waiting (e.g. t=150 -> wait=10s <= 15-2=13s), it sleeps and succeeds!
    clk.advance(45.0)  # t = 150.0 -> wait = 10.0s
    r4_wait = ctrl.acquire(deadline_ts=clk.time() + 15.0)
    assert r4_wait.allowed is True
    assert r4_wait.waited_seconds == 10.0
    assert clk.sleeps == [10.0]
    assert ctrl.daily_used == 3


def test_escalating_circuit_breaker_60_300_process() -> None:
    clk = FakeClock(start=500.0)
    ctrl = LLMResilienceController(
        rpm_limit=100,
        daily_budget=150,
        clock=clk.time,
        sleep_fn=clk.sleep,
    )

    # Strike 1 -> 60s cooldown, immediate BREAKER_OPEN skip
    assert ctrl.record_rate_limit_error() == "COOLDOWN_60S"
    assert ctrl.is_breaker_open() is True
    res_during_1 = ctrl.acquire(deadline_ts=clk.time() + 120.0)
    assert not res_during_1.allowed and res_during_1.reason == "BREAKER_OPEN"
    assert clk.sleeps == []  # Never sleeps during cooldown

    clk.advance(59.0)
    assert ctrl.is_breaker_open() is True
    clk.advance(1.5)  # +60.5s total -> recovered
    assert ctrl.is_breaker_open() is False
    assert ctrl.acquire(deadline_ts=clk.time() + 15.0).allowed is True

    # Strike 2 -> 300s cooldown
    assert ctrl.record_rate_limit_error() == "COOLDOWN_300S"
    clk.advance(299.0)
    assert not ctrl.acquire(deadline_ts=clk.time() + 500.0).allowed
    clk.advance(2.0)  # +301s -> recovered
    assert ctrl.acquire(deadline_ts=clk.time() + 15.0).allowed is True

    # Strike 3 -> permanent for rest of process
    assert ctrl.record_rate_limit_error() == "OPEN_PROCESS"
    clk.advance(86400.0)
    res_perm = ctrl.acquire(deadline_ts=clk.time() + 100.0)
    assert not res_perm.allowed and res_perm.reason == "BREAKER_OPEN"


def test_daily_budget_and_consent_gate() -> None:
    clk = FakeClock(start=0.0)
    ctrl = LLMResilienceController(rpm_limit=50, daily_budget=3, clock=clk.time, sleep_fn=clk.sleep)

    assert ctrl.acquire(deadline_ts=15.0).allowed
    assert ctrl.acquire(deadline_ts=15.0).allowed
    assert ctrl.acquire(deadline_ts=15.0).allowed
    r4 = ctrl.acquire(deadline_ts=15.0)
    assert not r4.allowed
    assert r4.reason == "DAILY_BUDGET_EXHAUSTED"
    assert ctrl.daily_used == 3

    # Consent defaults False -> blocks cloud requests immediately
    assert check_cloud_consent(is_cloud=True, local_only=False, cloud_consent=False) == "CLOUD_NOT_CONSENTED"
    assert check_cloud_consent(is_cloud=True, local_only=True, cloud_consent=True) == "LOCAL_ENDPOINT_UNAVAILABLE"
    assert check_cloud_consent(is_cloud=True, local_only=False, cloud_consent=True) is None
    assert check_cloud_consent(is_cloud=False, local_only=True, cloud_consent=False) is None


def test_thread_safe_rpm_and_in_memory_lru_cache() -> None:
    clk = FakeClock(start=10.0)
    ctrl = LLMResilienceController(rpm_limit=4, daily_budget=10, clock=clk.time, sleep_fn=clk.sleep)
    allowed_count = 0
    lock = threading.Lock()

    def worker() -> None:
        nonlocal allowed_count
        res = ctrl.acquire(deadline_ts=20.0)
        if res.allowed:
            with lock:
                allowed_count += 1

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert allowed_count == 4
    assert ctrl.daily_used == 4

    # LRU cache 256 limit, SHA-256 key sensitivity, deep-copy safety
    cache = InMemoryLRUCache(max_entries=256)
    k1 = make_cache_key(model="gemini-3.5-flash-lite", prompt_version="v1", payload={"a": 1}, image_bytes_list=[b"img1"])
    k2 = make_cache_key(model="gemini-3.5-flash-lite", prompt_version="v2", payload={"a": 1}, image_bytes_list=[b"img1"])
    k3 = make_cache_key(model="gemini-3.5-flash-lite", prompt_version="v1", payload={"a": 1}, image_bytes_list=[b"img2"])
    assert len({k1, k2, k3}) == 3

    for i in range(260):
        cache.set(f"k-{i}", {"val": i})
    assert len(cache) == 256
    assert cache.get("k-0") is None  # oldest evicted
    got = cache.get("k-259")
    assert got == {"val": 259}
    got["val"] = -1  # mutating copy must not pollute cached value
    assert cache.get("k-259") == {"val": 259}

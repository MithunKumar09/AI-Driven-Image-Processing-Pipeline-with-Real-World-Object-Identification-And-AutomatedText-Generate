"""Per-session rate limiting for the public demo.

**This is a per-session limiter, not a server-side one.**  State lives in
``st.session_state``, so a new browser session resets it and it provides no
protection against a determined caller.  Its purpose is to stop a public demo
from being casually used as free compute, and the UI says so plainly.

Keyed to actual pipeline executions, never to script reruns - Streamlit reruns
the script on every widget interaction, so counting reruns would exhaust the
budget without any inference happening.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

_STATE_KEY = "_pipeline_run_timestamps"
_WINDOW_SECONDS = 3600


@dataclass(frozen=True)
class RateLimitStatus:
    allowed: bool
    remaining: int
    retry_after_seconds: int


def _prune(timestamps: list[float], now: float) -> list[float]:
    return [t for t in timestamps if now - t < _WINDOW_SECONDS]


def check(session_state, *, max_runs_per_hour: int, now: float | None = None) -> RateLimitStatus:
    """Report whether another run is allowed. Does not consume budget."""
    now = time.time() if now is None else now
    timestamps = _prune(list(session_state.get(_STATE_KEY, [])), now)
    session_state[_STATE_KEY] = timestamps

    remaining = max_runs_per_hour - len(timestamps)
    if remaining > 0:
        return RateLimitStatus(allowed=True, remaining=remaining, retry_after_seconds=0)

    oldest = min(timestamps)
    retry_after = int(_WINDOW_SECONDS - (now - oldest)) + 1
    return RateLimitStatus(allowed=False, remaining=0, retry_after_seconds=max(1, retry_after))


def record_run(session_state, *, now: float | None = None) -> None:
    """Consume one unit of budget. Call only when inference actually runs."""
    now = time.time() if now is None else now
    timestamps = _prune(list(session_state.get(_STATE_KEY, [])), now)
    timestamps.append(now)
    session_state[_STATE_KEY] = timestamps


def describe_limit(max_runs_per_hour: int) -> str:
    return (
        f"Demo limit: {max_runs_per_hour} analyses per hour per browser session. "
        "This is a per-session guard, not a server-side rate limit."
    )

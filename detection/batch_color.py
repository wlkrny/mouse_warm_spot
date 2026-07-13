"""Pure helpers shared by the GUI's bounded color-identification batch worker."""

from __future__ import annotations

import math
import os

DEFAULT_BATCH_CONCURRENCY = 6
MAX_BATCH_CONCURRENCY = 6


def batch_color_concurrency(value: str | None = None) -> int:
    """Read a bounded positive concurrency setting; invalid values are safe."""
    raw = os.environ.get("MOUSE_COLOR_BATCH_CONCURRENCY", "") if value is None else value
    try:
        parsed = int(str(raw).strip())
    except (TypeError, ValueError):
        return DEFAULT_BATCH_CONCURRENCY
    return parsed if 1 <= parsed <= MAX_BATCH_CONCURRENCY else DEFAULT_BATCH_CONCURRENCY


def is_single_mouse_segment(segment: dict) -> bool:
    """Return whether a segment is eligible for one-mouse color recognition.

    A manual confirmed count is more trustworthy than the automatic estimate.
    When no valid confirmation exists, fall back to ``estimated_mouse_count``.
    """
    confirmed = segment.get("confirmed_mouse_count")
    count = confirmed if isinstance(confirmed, int) and not isinstance(confirmed, bool) else \
        segment.get("estimated_mouse_count")
    return count == 1


def batch_items(segments: list[dict]) -> list[tuple[int, dict]]:
    """Return valid one-mouse timeline segments without mutating them."""
    return [(index, segment) for index, segment in enumerate(segments)
            if is_single_mouse_segment(segment)
            and segment.get("start_frame") is not None
            and segment.get("end_frame") is not None
            and segment["end_frame"] >= segment["start_frame"]]


def batch_summary(results: list[dict], failures: int, cancelled: bool) -> dict:
    """Summarize completed outcomes; cancellation is reported, not fabricated."""
    thermometer = sum(bool(result.get("thermometer_present")) for result in results)
    review = sum(bool(result.get("identity_needs_review")) for result in results)
    total_cost_usd = math.fsum(
        _safe_cost_usd(result.get("ai_api_cost_usd", 0.0)) for result in results
    )
    return {
        "total_completed": len(results),
        "success": sum(not result.get("identity_needs_review", False)
                       and not result.get("thermometer_present", False) for result in results),
        "thermometer": thermometer,
        "needs_review": review,
        "failed": failures,
        "cancelled": cancelled,
        "total_cost_usd": total_cost_usd,
    }


def _safe_cost_usd(value) -> float:
    """Return a finite numeric USD cost; malformed worker output costs zero."""
    try:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return 0.0
        value = float(value)
        return value if value >= 0.0 and math.isfinite(value) else 0.0
    except (TypeError, ValueError):
        return 0.0

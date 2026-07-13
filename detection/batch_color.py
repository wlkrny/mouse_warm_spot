"""Pure helpers shared by the GUI's bounded color-identification batch worker."""

from __future__ import annotations

import os

DEFAULT_BATCH_CONCURRENCY = 2
MAX_BATCH_CONCURRENCY = 4


def batch_color_concurrency(value: str | None = None) -> int:
    """Read a bounded positive concurrency setting; invalid values are safe."""
    raw = os.environ.get("MOUSE_COLOR_BATCH_CONCURRENCY", "") if value is None else value
    try:
        parsed = int(str(raw).strip())
    except (TypeError, ValueError):
        return DEFAULT_BATCH_CONCURRENCY
    return parsed if 1 <= parsed <= MAX_BATCH_CONCURRENCY else DEFAULT_BATCH_CONCURRENCY


def batch_items(segments: list[dict]) -> list[tuple[int, dict]]:
    """Return eligible non-zero timeline segments without mutating them."""
    return [(index, segment) for index, segment in enumerate(segments)
            if segment.get("estimated_mouse_count") != 0
            and segment.get("start_frame") is not None
            and segment.get("end_frame") is not None
            and segment["end_frame"] >= segment["start_frame"]]


def batch_summary(results: list[dict], failures: int, cancelled: bool) -> dict:
    """Summarize completed outcomes; cancellation is reported, not fabricated."""
    thermometer = sum(bool(result.get("thermometer_present")) for result in results)
    review = sum(bool(result.get("identity_needs_review")) for result in results)
    return {
        "total_completed": len(results),
        "success": sum(not result.get("identity_needs_review", False)
                       and not result.get("thermometer_present", False) for result in results),
        "thermometer": thermometer,
        "needs_review": review,
        "failed": failures,
        "cancelled": cancelled,
    }

"""Synthetic regressions for Core-connected warm-spot counting."""

import cv2
import numpy as np

from detection.counter import MouseCounter


def _roi():
    return (
        {"cx": 100, "cy": 100, "a": 20, "b": 20, "angle": 0},
        {"cx": 100, "cy": 100, "a": 70, "b": 70, "angle": 0},
    )


def _frame(rectangles):
    frame = np.full((200, 200, 3), 220, dtype=np.uint8)
    for x1, y1, x2, y2 in rectangles:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (10, 10, 10), -1)
    return frame


def test_outer_independent_blob_does_not_increase_core_occupant_count():
    core, count = _roi()
    # First body crosses the Core; second body stays independent in ROI Count.
    result = MouseCounter(single_mouse_area_ref=400).estimate_count(
        _frame([(88, 88, 108, 112), (135, 88, 154, 112)]), core, count,
        np.full((200, 200, 3), 220, dtype=np.uint8),
    )
    assert result["estimated_mouse_count"] == 1
    assert result["core_connected_blob_count"] == 1
    assert result["ignored_outer_blob_count"] == 1
    assert result["core_connected_area"] == result["total_mouse_area"]


def test_two_separate_core_connected_bodies_can_count_as_two():
    core, count = _roi()
    result = MouseCounter(single_mouse_area_ref=300).estimate_count(
        _frame([(76, 88, 94, 112), (106, 88, 124, 112)]), core, count,
        np.full((200, 200, 3), 220, dtype=np.uint8),
    )
    assert result["core_connected_blob_count"] == 2
    assert result["ignored_outer_blob_count"] == 0
    assert result["estimated_mouse_count"] == 2

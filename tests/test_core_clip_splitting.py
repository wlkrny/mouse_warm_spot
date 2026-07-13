"""Regression tests for Core-ROI occupancy episode boundaries."""

import numpy as np

from detection.engine import DetectionEngine


class FakeCapture:
    """Small OpenCV VideoCapture substitute with deterministic frame positions."""

    def __init__(self, frame_count, fps=10.0):
        self.frame_count = frame_count
        self.fps = fps
        self.position = 0

    def get(self, prop):
        return {5: self.fps, 7: self.frame_count}.get(prop, 0)

    def set(self, prop, value):
        if prop == 1:
            self.position = int(value)

    def read(self):
        if self.position >= self.frame_count:
            return False, None
        frame = np.full((1, 1, 3), self.position, dtype=np.uint8)
        self.position += 1
        return True, frame


class SequenceMetrics:
    def __init__(self, core_occupied, count_occupied=None):
        self.core_occupied = core_occupied
        self.count_occupied = count_occupied or core_occupied

    def compute(self, frame, roi, background):
        index = int(frame[0, 0, 0])
        occupied = (self.core_occupied if roi["name"] == "core"
                    else self.count_occupied)[index]
        return {
            "occlusion_area_ratio": 0.3 if occupied else 0.0,
            "dark_pixel_ratio": 0.0,
            "background_diff_score": 0.0,
        }


class ZeroCounter:
    def estimate_count(self, *args):
        return {"estimated_mouse_count": 0, "count_by_blob": 0,
                "count_by_area": 0, "total_mouse_area": 0,
                "largest_blob_area": 0, "core_touching_blob_count": 0,
                "count_confidence": 1, "blob_count": 0}

    def get_single_mouse_area(self):
        return None


def _engine(core, count=None):
    return DetectionEngine(SequenceMetrics(core, count))


def _params(**extra):
    result = dict(DetectionEngine.DEFAULT_PARAMS)
    result.update({
        "min_on_frames": 1,
        "core_gap_tolerance_seconds": 0.3,  # three frames at 10 fps
        "coarse_stride_frames": 1,
        "backtrack_seconds": 0.1,
        "forward_confirm_seconds": 0.1,
        "review_padding_seconds": 0,
    })
    result.update(extra)
    return result


def _roi():
    return {
        "roi_core": {"name": "core", "cx": 0, "cy": 0, "a": 1, "b": 1},
        "roi_count": {"name": "count", "cx": 0, "cy": 0, "a": 1, "b": 1},
    }


def _fine_events(core, count=None, **params):
    cap = FakeCapture(len(core))
    return _engine(core, count)._fine_scan(
        cap, 0, len(core) - 1, 10.0, _roi(), None, _params(**params)
    )


def test_continuous_core_occupancy_is_one_clip():
    events = _fine_events([True] * 10)
    assert [(e["start_frame"], e["end_frame"]) for e in events] == [(0, 9)]


def test_core_gap_at_tolerance_splits_clips():
    events = _fine_events([True] * 5 + [False] * 3 + [True] * 5)
    assert [(e["start_frame"], e["end_frame"]) for e in events] == [(0, 4), (8, 12)]
    assert events[0]["end_reason"] == "core_gap"
    assert events[0]["core_gap_frames"] == 3


def test_core_gap_shorter_than_tolerance_does_not_split():
    events = _fine_events([True] * 5 + [False] * 2 + [True] * 5)
    assert [(e["start_frame"], e["end_frame"]) for e in events] == [(0, 11)]


def test_count_roi_activity_cannot_keep_core_clip_open():
    core = [True] * 5 + [False] * 3 + [True] * 5
    events = _fine_events(core, [True] * len(core))
    assert [(e["start_frame"], e["end_frame"]) for e in events] == [(0, 4), (8, 12)]


def test_detect_and_counting_emits_visible_zero_clip_for_qualifying_core_gap():
    core = [True] * 5 + [False] * 3 + [True] * 5
    # Count ROI remains active through the Core gap, so coarse scan covers it.
    engine = _engine(core, [True] * len(core))
    episodes, segments = engine.detect_with_counting(
        FakeCapture(len(core)), _roi(), None, _params(), counter=ZeroCounter()
    )
    assert [(e["start_frame"], e["end_frame"]) for e in episodes] == [(0, 4), (8, 12)]
    assert [(s["start_frame"], s["end_frame"], s["estimated_mouse_count"])
            for s in segments] == [(0, 4, 0), (5, 7, 0), (8, 12, 0)]
    gap = segments[1]
    assert gap["start_reason"] == "core_empty_gap"
    assert gap["end_reason"] == "core_gap_end"
    assert gap["is_short_event"] is False


def test_short_core_gap_does_not_emit_zero_clip():
    core = [True] * 5 + [False] * 2 + [True] * 5
    episodes, segments = _engine(core, [True] * len(core)).detect_with_counting(
        FakeCapture(len(core)), _roi(), None, _params(), counter=ZeroCounter()
    )
    assert len(episodes) == 1
    assert not any(s.get("start_reason") == "core_empty_gap" for s in segments)


def test_zero_gap_segment_survives_count_segment_filters_and_merges():
    engine = _engine([True])
    events = [{"start_frame": 0, "end_frame": 4},
              {"start_frame": 8, "end_frame": 12}]
    # The canonical zero segment is built after occupied-only filtering and is
    # never passed to same-count merging, so it remains a standalone boundary.
    gaps = engine._build_core_empty_gap_segments(events, 10.0, _params())
    assert [(s["start_frame"], s["end_frame"], s["estimated_mouse_count"])
            for s in gaps] == [(5, 7, 0)]
    assert engine._merge_same_count_segments(gaps, 99, 10.0) == gaps

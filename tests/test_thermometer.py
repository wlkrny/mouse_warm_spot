"""No-network thermometer interference regression tests."""
import os
import unittest
from unittest.mock import MagicMock
import numpy as np


class TestThermometerContract(unittest.TestCase):
    def test_missing_field_is_legacy_safe_false(self):
        from detection.models.vision_provider import _parse_segment_json
        result = _parse_segment_json('{"mouse_count":1,"colors":["green"],"confidence":0.9}')
        self.assertFalse(result["thermometer_present"])

    def test_boolean_is_strict(self):
        from detection.models.vision_provider import _parse_segment_json
        self.assertIsNone(_parse_segment_json(
            '{"mouse_count":1,"colors":["green"],"confidence":0.9,"thermometer_present":"true"}'
        ))

    def test_true_zeros_identity_and_apply_preserves_confirmed(self):
        from detection.identity_assist import apply_identity_to_segment
        result = {
            "target_count": 1, "auto_mouse_colors": ["green"], "auto_mouse_ids": ["auto_green"],
            "identity_confidence": 0.0, "identity_needs_review": True, "identity_conflict": False,
            "identity_method": "ear_tag_color_vlm|thermometer_detected",
            "identity_note": "检测到测温器/探头，颜色识别置信度已置零，请人工复核",
            "thermometer_present": True,
        }
        segment = {"count_status": "pending"}
        apply_identity_to_segment(segment, result)
        self.assertEqual(segment["identity_confidence"], 0.0)
        self.assertTrue(segment["identity_needs_review"])
        self.assertFalse(segment["identity_conflict"])
        self.assertIn("thermometer_detected", segment["identity_method"])
        confirmed = {"count_status": "confirmed", "identity_confidence": 0.8}
        apply_identity_to_segment(confirmed, result)
        self.assertEqual(confirmed["identity_confidence"], 0.8)


class TestBatchScope(unittest.TestCase):
    def test_only_nonempty_segments_are_selectable_for_batch(self):
        # This mirrors MainWindow's deliberately simple, testable batch filter.
        segments = [{"start_frame": 0, "end_frame": 2}, {"start_frame": 4, "end_frame": 3},
                    {"start_frame": None, "end_frame": 5}]
        items = [(i, s) for i, s in enumerate(segments)
                 if s.get("start_frame") is not None and s.get("end_frame") is not None
                 and s["end_frame"] >= s["start_frame"]]
        self.assertEqual([i for i, _ in items], [0])


if __name__ == "__main__":
    unittest.main()

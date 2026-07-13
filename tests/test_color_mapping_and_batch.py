"""No-network tests for durable color IDs and bounded batch helpers."""

import json

from detection.batch_color import (DEFAULT_BATCH_CONCURRENCY, batch_color_concurrency,
                                   batch_items, batch_summary)
from detection.color_mouse_mapping import ColorMouseMappingStore
from detection.identity_assist import apply_identity_to_segment


def _result(colors, **extra):
    result = {
        "target_count": len(colors), "auto_mouse_colors": colors,
        "auto_mouse_ids": [f"auto_{color}" for color in colors],
        "identity_confidence": .9, "identity_needs_review": False,
        "identity_conflict": False, "identity_method": "rule", "identity_note": "",
    }
    result.update(extra)
    return result


def test_mapping_assigns_lowest_reuses_and_round_trips(tmp_path):
    path = tmp_path / "mapping.json"
    store = ColorMouseMappingStore(path)
    assert store.assign_colors(["green"]) == [1]
    assert store.assign_colors(["red"]) == [2]
    assert store.assign_colors(["green"]) == [1]
    assert ColorMouseMappingStore(path).get_mapping() == {"green": 1, "red": 2}


def test_mapping_rejects_unknown_duplicate_and_conflict(tmp_path):
    store = ColorMouseMappingStore(tmp_path / "mapping.json")
    assert store.assign_colors(["unknown"]) is None
    assert store.assign_colors(["green", "green"]) is None
    (tmp_path / "bad.json").write_text(json.dumps({"mappings": {"green": 1, "red": 1}}))
    assert ColorMouseMappingStore(tmp_path / "bad.json").get_mapping() == {"green": 1}


def test_apply_mapping_protects_confirmed_zero_and_ambiguous(tmp_path):
    store = ColorMouseMappingStore(tmp_path / "mapping.json")
    segment = {"count_status": "pending", "estimated_mouse_count": 1}
    apply_identity_to_segment(segment, _result(["green"]), store)
    assert segment["mouse_ids"] == [1]
    confirmed = {"count_status": "confirmed", "estimated_mouse_count": 1, "mouse_ids": [4]}
    apply_identity_to_segment(confirmed, _result(["red"]), store)
    assert confirmed["mouse_ids"] == [4]
    zero = {"count_status": "pending", "estimated_mouse_count": 0}
    apply_identity_to_segment(zero, _result(["red"]), store)
    assert "auto_mouse_colors" not in zero
    ambiguous = {"count_status": "pending", "estimated_mouse_count": 2}
    apply_identity_to_segment(ambiguous, _result(["red", "red"]), store)
    assert ambiguous["identity_needs_review"]
    assert ambiguous.get("mouse_ids", []) == []


def test_batch_config_scope_and_summary(monkeypatch):
    assert batch_color_concurrency("1") == 1
    assert batch_color_concurrency("4") == 4
    assert batch_color_concurrency("0") == DEFAULT_BATCH_CONCURRENCY
    assert batch_color_concurrency("5") == DEFAULT_BATCH_CONCURRENCY
    assert batch_color_concurrency("invalid") == DEFAULT_BATCH_CONCURRENCY
    segments = [{"start_frame": 0, "end_frame": 1, "estimated_mouse_count": 1},
                {"start_frame": 2, "end_frame": 3, "estimated_mouse_count": 0},
                {"start_frame": 4, "end_frame": 3, "estimated_mouse_count": 1}]
    assert [index for index, _ in batch_items(segments)] == [0]
    summary = batch_summary([_result(["green"]), _result(["red"], thermometer_present=True,
                                                     identity_needs_review=True)], 1, True)
    assert summary == {"total_completed": 2, "success": 1, "thermometer": 1,
                       "needs_review": 1, "failed": 1, "cancelled": True}

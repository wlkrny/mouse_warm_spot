"""No-network tests for durable color IDs and bounded batch helpers."""

import json

import pytest

from detection.batch_color import (DEFAULT_BATCH_CONCURRENCY, batch_color_concurrency,
                                   batch_items, batch_summary, is_single_mouse_segment)
from detection.color_mouse_mapping import ColorMouseMappingStore
from detection.identity_assist import IdentityAssist, apply_identity_to_segment


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
    assert batch_color_concurrency("6") == 6
    assert batch_color_concurrency("0") == DEFAULT_BATCH_CONCURRENCY
    assert batch_color_concurrency("7") == DEFAULT_BATCH_CONCURRENCY
    assert batch_color_concurrency("invalid") == DEFAULT_BATCH_CONCURRENCY
    segments = [
        {"start_frame": 0, "end_frame": 1, "estimated_mouse_count": 1},
        {"start_frame": 2, "end_frame": 3, "estimated_mouse_count": 0},
        {"start_frame": 4, "end_frame": 5, "estimated_mouse_count": 2},
        {"start_frame": 6, "end_frame": 7, "estimated_mouse_count": 2,
         "confirmed_mouse_count": 1},
        {"start_frame": 8, "end_frame": 7, "estimated_mouse_count": 1},
    ]
    assert [index for index, _ in batch_items(segments)] == [0, 3]
    assert is_single_mouse_segment(segments[0])
    assert not is_single_mouse_segment(segments[2])
    assert is_single_mouse_segment(segments[3])
    summary = batch_summary([
        _result(["green"], ai_api_cost_usd=.0012),
        _result(["red"], thermometer_present=True, identity_needs_review=True,
                ai_api_cost_usd=.0003),
    ], 1, True)
    assert {key: value for key, value in summary.items() if key != "total_cost_usd"} == {
        "total_completed": 2, "success": 1, "thermometer": 1,
        "needs_review": 1, "failed": 1, "cancelled": True,
    }
    assert summary["total_cost_usd"] == pytest.approx(.0015)


def test_extract_ai_api_cost_usd_is_safe_for_malformed_responses():
    assert IdentityAssist._extract_ai_api_cost_usd({"usage": {"cost": .0012}}) == .0012
    assert IdentityAssist._extract_ai_api_cost_usd(None) == 0.0
    assert IdentityAssist._extract_ai_api_cost_usd({"usage": {"cost": "0.0012"}}) == 0.0


def test_batch_summary_ignores_invalid_costs():
    summary = batch_summary([
        _result(["green"], ai_api_cost_usd=".002"),
        _result(["red"], ai_api_cost_usd=float("nan")),
        _result(["blue"], ai_api_cost_usd=-.001),
    ], 0, False)
    assert summary["total_cost_usd"] == 0.0

"""Persistent, conservative ear-tag color to mouse-number mapping."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

KNOWN_COLORS = frozenset({"red", "yellow", "blue", "green", "white"})
MOUSE_ID_RANGE = range(1, 5)


class ColorMouseMappingStore:
    """Store one-to-one known-color -> mouse-ID assignments in user config.

    Invalid or unavailable storage never prevents color recognition; callers get
    an empty mapping and can still ask for a conservative assignment.
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else (
            Path.home() / ".mouse_warm_spot" / "color_mouse_mapping.json"
        )
        self._mapping: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            raw = data.get("mappings", data) if isinstance(data, dict) else {}
            if not isinstance(raw, dict):
                raise ValueError("mappings is not an object")
            mapping: dict[str, int] = {}
            used: set[int] = set()
            for color, mouse_id in raw.items():
                if color in KNOWN_COLORS and isinstance(mouse_id, int) and mouse_id in MOUSE_ID_RANGE and mouse_id not in used:
                    mapping[color] = mouse_id
                    used.add(mouse_id)
            self._mapping = mapping
        except FileNotFoundError:
            self._mapping = {}
        except (OSError, ValueError, json.JSONDecodeError, TypeError):
            # Do not expose file contents (which may be malformed user data).
            logger.warning("Color mouse mapping could not be loaded; starting with an empty mapping")
            self._mapping = {}

    def get_mapping(self) -> dict[str, int]:
        return dict(self._mapping)

    def summary(self) -> str:
        return ", ".join(f"{color}→{mouse_id}" for color, mouse_id in sorted(self._mapping.items())) or "（暂无映射）"

    def reset(self) -> bool:
        self._mapping = {}
        return self._save()

    def assign_colors(self, colors: list[str]) -> list[int] | None:
        """Assign/reuse IDs atomically, returning None for ambiguous input.

        A batch with unknown/repeated colors, more than four colors, conflicting
        persisted IDs, or no available IDs is deliberately left for review.
        """
        normalized = list(colors)
        if (not normalized or len(normalized) > len(MOUSE_ID_RANGE)
                or any(color not in KNOWN_COLORS for color in normalized)
                or len(set(normalized)) != len(normalized)):
            return None
        if len(set(self._mapping.values())) != len(self._mapping):
            return None
        proposed = dict(self._mapping)
        used = set(proposed.values())
        for color in normalized:
            if color in proposed:
                continue
            available = [mouse_id for mouse_id in MOUSE_ID_RANGE if mouse_id not in used]
            if not available:
                return None
            proposed[color] = available[0]
            used.add(available[0])
        old = self._mapping
        self._mapping = proposed
        if not self._save():
            self._mapping = old
            return None
        return [proposed[color] for color in normalized]

    def _save(self) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"version": 1, "mappings": self._mapping}
            temp = self.path.with_suffix(self.path.suffix + ".tmp")
            temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
            temp.replace(self.path)
            return True
        except OSError:
            logger.warning("Color mouse mapping could not be saved; automatic assignment was not applied")
            return False


_default_store: ColorMouseMappingStore | None = None


def default_color_mouse_mapping_store() -> ColorMouseMappingStore:
    global _default_store
    if _default_store is None:
        _default_store = ColorMouseMappingStore()
    return _default_store

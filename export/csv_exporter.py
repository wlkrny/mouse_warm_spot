"""
CSV 导出器 — 以 CountSegment 为单位导出/导入 CSV

CSV 字段规范:
video_file|episode_id|segment_id|start_frame|end_frame|
start_time_sec|end_time_sec|duration_sec|fps|count|count_confidence|
count_method|manual_mouse_ids|manual_confirmed|auto_mouse_ids|
auto_mouse_colors|identity_confidence|identity_method|needs_review|
identity_needs_review|identity_conflict|is_short_event|
is_possible_false_positive|occupancy_confidence|
roi_core_cx|roi_core_cy|roi_core_rx|roi_core_ry|roi_core_angle|notes
"""

import csv
import os
from typing import Any


# CSV 列名 (规范)
CSV_FIELDS = [
    "video_file",
    "episode_id",
    "segment_id",
    "start_frame",
    "end_frame",
    "start_time_sec",
    "end_time_sec",
    "duration_sec",
    "fps",
    "count",
    "count_confidence",
    "count_method",
    "manual_mouse_ids",
    "manual_confirmed",
    "auto_mouse_ids",
    "auto_mouse_colors",
    "identity_confidence",
    "identity_method",
    "needs_review",
    "identity_needs_review",
    "identity_conflict",
    "is_short_event",
    "is_possible_false_positive",
    "occupancy_confidence",
    "roi_core_cx",
    "roi_core_cy",
    "roi_core_rx",
    "roi_core_ry",
    "roi_core_angle",
    "notes",
]


def _bool_str(value: Any) -> str:
    """将值转为小写布尔字符串 true/false"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return "true" if value.lower() in ("true", "yes", "1") else "false"
    if isinstance(value, (int, float)):
        return "true" if value else "false"
    return "false"


def _float3(value: Any, default: float = 0.0) -> str:
    """格式化为 3 位小数"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = default
    return f"{v:.3f}"


def _join_ids(ids: Any) -> str:
    """用分号连接 mouse_ids 列表, 空列表返回 ''"""
    if not ids:
        return ""
    if isinstance(ids, list):
        return ";".join(str(i) for i in ids)
    return str(ids)


class CsvExporter:
    """CSV 导出器 — 静态方法, 无 Qt 依赖"""

    @staticmethod
    def export_segments(
        segments: list[dict],
        video_file: str,
        fps: float,
        output_path: str,
        roi_data: dict | None = None,
    ) -> None:
        """
        将 CountSegment 列表导出为 CSV 文件

        :param segments:   EventListWidget.get_segments() 返回的片段列表
        :param video_file: 视频文件路径
        :param fps:        视频 FPS
        :param output_path: 输出 CSV 文件路径
        :param roi_data:   视频 ROI 数据 (可选, 用于 roi_core_* 字段)
        """
        # 提取视频文件名为简短标识
        video_name = os.path.basename(video_file) if video_file else ""

        # ROI 核心参数
        roi_core = roi_data.get("roi_core", {}) if roi_data else {}

        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, delimiter="|")
            writer.writeheader()

            for seg in segments:
                # 确定 count 字段: 优先确认数量, 否则用估计数量 (cap=2)
                count = seg.get("confirmed_mouse_count")
                if count is None:
                    count = seg.get("estimated_mouse_count", 0)
                # cap=2 硬约束: 向前兼容旧数据 count>2 → 降级为2
                count = min(count if count is not None else 0, 2)

                # 是否已确认
                count_status = seg.get("count_status", "pending")
                manual_confirmed = count_status == "confirmed"

                row = {
                    "video_file": video_name,
                    "episode_id": seg.get("episode_id", ""),
                    "segment_id": seg.get("segment_id", ""),
                    "start_frame": seg.get("start_frame", 0),
                    "end_frame": seg.get("end_frame", 0),
                    "start_time_sec": _float3(seg.get("start_time", 0.0)),
                    "end_time_sec": _float3(seg.get("end_time", 0.0)),
                    "duration_sec": _float3(seg.get("duration", 0.0)),
                    "fps": f"{fps:.2f}" if fps > 0 else "0.00",
                    "count": count if count is not None else 0,
                    "count_confidence": _float3(seg.get("count_confidence", 0.0)),
                    "count_method": seg.get("count_method", ""),
                    "manual_mouse_ids": _join_ids(seg.get("mouse_ids", [])),
                    "manual_confirmed": _bool_str(manual_confirmed),
                    "auto_mouse_ids": _join_ids(seg.get("auto_mouse_ids", [])),
                    "auto_mouse_colors": _join_ids(seg.get("auto_mouse_colors", [])),
                    "identity_confidence": seg.get("identity_confidence", 0),
                    "identity_method": seg.get("identity_method", ""),
                    "needs_review": _bool_str(seg.get("needs_review", False)),
                    "identity_needs_review": _bool_str(seg.get("identity_needs_review", False)),
                    "identity_conflict": _bool_str(seg.get("identity_conflict", False)),
                    "is_short_event": _bool_str(seg.get("is_short_event", False)),
                    "is_possible_false_positive": _bool_str(seg.get("is_possible_false_positive", False)),
                    "occupancy_confidence": _float3(seg.get("confidence", 0.0)),
                    "roi_core_cx": f"{roi_core.get('cx', 0.0):.1f}",
                    "roi_core_cy": f"{roi_core.get('cy', 0.0):.1f}",
                    "roi_core_rx": f"{roi_core.get('a', 0.0):.1f}",
                    "roi_core_ry": f"{roi_core.get('b', 0.0):.1f}",
                    "roi_core_angle": f"{roi_core.get('angle', 0.0):.1f}",
                    "notes": (seg.get("note") or seg.get("count_note") or ""),
                }
                writer.writerow(row)

    # ------------------------------------------------------------------
    # Phase 9: 向后兼容 CSV 加载
    # ------------------------------------------------------------------
    @staticmethod
    def load_segments(input_path: str) -> list[dict]:
        """
        从 CSV 文件加载 CountSegment 列表, 向后兼容旧格式.

        兼容规则:
        - count>2 降级为 2, notes 追加 "Legacy count capped"
        - 缺失字段使用默认值
        - 确保导出的 CSV 再加载不丢失新增字段
        """
        segments: list[dict] = []
        with open(input_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter="|")
            for row in reader:
                seg = CsvExporter._row_to_segment(row)
                segments.append(seg)
        return segments

    @staticmethod
    def _row_to_segment(row: dict[str, str]) -> dict:
        """Convert a single CSV row to a CountSegment dict with legacy fixes."""
        seg: dict[str, Any] = {
            "segment_id": row.get("segment_id", ""),
            "episode_id": row.get("episode_id", ""),
            "start_frame": int(row.get("start_frame", 0)),
            "end_frame": int(row.get("end_frame", 0)),
            "start_time": float(row.get("start_time_sec", 0.0)),
            "end_time": float(row.get("end_time_sec", 0.0)),
            "duration": float(row.get("duration_sec", 0.0)),
            "count_confidence": float(row.get("count_confidence", 0.0)),
            "count_method": row.get("count_method", ""),
            "confidence": float(row.get("occupancy_confidence", 0.0)),
            "note": row.get("notes", ""),
            "count_note": row.get("notes", ""),
        }

        # count — 向前兼容 cap=2: >2 降级
        raw_count = int(row.get("count", 0))
        if raw_count > 2:
            seg["estimated_mouse_count"] = 2
            seg["confirmed_mouse_count"] = 2
            old_note = seg.get("note", "")
            seg["note"] = f"{old_note}; Legacy count capped from {raw_count}" if old_note else f"Legacy count capped from {raw_count}"
        else:
            seg["estimated_mouse_count"] = raw_count

        # manual fields
        raw_confs = row.get("manual_confirmed", "false")
        manual_confirmed = raw_confs.lower() == "true" if raw_confs else False
        seg["manual_confirmed"] = manual_confirmed

        mouse_str = row.get("manual_mouse_ids", "")
        seg["mouse_ids"] = [int(x) for x in mouse_str.split(";") if x.strip().isdigit()] if mouse_str else []

        # status
        if manual_confirmed:
            seg["count_status"] = "confirmed"
        elif row.get("is_possible_false_positive", "false").lower() == "true":
            seg["count_status"] = "rejected"
        else:
            seg["count_status"] = "pending"

        # auto identity fields — missing → defaults
        seg["auto_mouse_colors"] = [x for x in row.get("auto_mouse_colors", "").split(";") if x.strip()] if row.get("auto_mouse_colors", "") else []
        auto_ids_str = row.get("auto_mouse_ids", "")
        seg["auto_mouse_ids"] = auto_ids_str.split(";") if auto_ids_str else []
        seg["identity_confidence"] = float(row.get("identity_confidence", 0.0))
        seg["identity_method"] = row.get("identity_method", "")
        seg["identity_needs_review"] = row.get("identity_needs_review", "false").lower() == "true"
        seg["identity_conflict"] = row.get("identity_conflict", "false").lower() == "true"

        # review fields
        seg["needs_review"] = row.get("needs_review", "false").lower() == "true"
        seg["is_short_event"] = row.get("is_short_event", "false").lower() == "true"
        seg["is_possible_false_positive"] = row.get("is_possible_false_positive", "false").lower() == "true"

        # other defaults
        seg.setdefault("mouse_count", raw_count)
        seg.setdefault("detected_by", "csv_import")
        seg.setdefault("modified_by_user", False)
        seg.setdefault("reviewer", "")

        return seg

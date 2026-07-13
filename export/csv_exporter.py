"""
CSV 导出器 — 以 CountSegment 为单位导出/导入 CSV

CSV 字段规范 (精简):
segment_id|start_time_sec|end_time_sec|mouse_count|mouse_ids
"""

import csv
import os
from typing import Any


# CSV 列名 (精简: 事件编号/起止时间/老鼠数量/老鼠编号)
CSV_FIELDS = [
    "segment_id",
    "start_time_sec",
    "end_time_sec",
    "mouse_count",
    "mouse_ids",
]


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
        将 CountSegment 列表导出为精简 CSV 文件.
        字段: 事件编号 | 开始秒 | 结束秒 | 老鼠数量 | 老鼠编号
        以事件列表中手动标记的 mouse_ids 为准, 未标记则数量为 0.
        """
        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, delimiter="|")
            writer.writeheader()

            for seg in segments:
                mouse_ids = seg.get("mouse_ids", [])
                count = len(mouse_ids)
                # 数量为 0 时 mouse_ids 填 "0"
                mouse_ids_str = "0" if count == 0 else _join_ids(mouse_ids)

                row = {
                    "segment_id": seg.get("segment_id", ""),
                    "start_time_sec": _float3(seg.get("start_time", 0.0)),
                    "end_time_sec": _float3(seg.get("end_time", 0.0)),
                    "mouse_count": count,
                    "mouse_ids": mouse_ids_str,
                }
                writer.writerow(row)

    # ------------------------------------------------------------------
    # Phase 9: 向后兼容旧格式 CSV 加载
    # ------------------------------------------------------------------
    @staticmethod
    def load_segments(input_path: str) -> list[dict]:
        """
        从 CSV 文件加载 CountSegment 列表, 兼容新旧格式.
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
        """将 CSV 行转换为 CountSegment dict, 兼容新旧格式."""
        seg: dict[str, Any] = {
            "segment_id": row.get("segment_id", ""),
            "start_frame": int(row.get("start_frame", 0)),
            "end_frame": int(row.get("end_frame", 0)),
            "start_time": float(row.get("start_time_sec", 0.0)),
            "end_time": float(row.get("end_time_sec", 0.0)),
            "duration": float(row.get("duration_sec", 0.0)) if row.get("duration_sec") else float(row.get("end_time_sec", 0.0)) - float(row.get("start_time_sec", 0.0)),
        }

        # 新格式: mouse_count + mouse_ids
        if "mouse_count" in row:
            count = int(row.get("mouse_count", 0))
            seg["estimated_mouse_count"] = count
            seg["confirmed_mouse_count"] = count
            mouse_str = row.get("mouse_ids", "")
            seg["mouse_ids"] = [int(x) for x in mouse_str.split(";") if x.strip().isdigit()] if mouse_str else []
            seg["count_status"] = "confirmed"
        else:
            # 旧格式兼容
            raw_count = int(row.get("count", 0))
            if raw_count > 2:
                raw_count = 2
            seg["estimated_mouse_count"] = raw_count
            seg["confirmed_mouse_count"] = raw_count
            mouse_str = row.get("manual_mouse_ids", "")
            seg["mouse_ids"] = [int(x) for x in mouse_str.split(";") if x.strip().isdigit()] if mouse_str else []
            seg["count_status"] = "confirmed" if row.get("manual_confirmed", "false").lower() == "true" else "pending"
            seg["confidence"] = float(row.get("occupancy_confidence", 0.0))
            seg["auto_mouse_colors"] = [x for x in row.get("auto_mouse_colors", "").split(";") if x.strip()] if row.get("auto_mouse_colors", "") else []
            seg["note"] = row.get("notes", "")

        seg.setdefault("detected_by", "csv_import")
        return seg

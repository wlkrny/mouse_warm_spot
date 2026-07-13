"""
全视频自动检测引擎
实现规范 8-14 节：隔帧粗筛 → 局部逐帧精查 → 状态机 → 后处理
改进.md：两层检测架构 OccupancyEpisode + CountSegment
"""

import time
import logging
import numpy as np
from collections.abc import Callable

from .metrics import DetectionMetrics
from .counter import MouseCounter

# ---- Phase 10: 检测日志 logger ----
logger = logging.getLogger("DetectionEngine")


class DetectionEngine:
    """全视频自动检测引擎"""

    # ---- 短事件策略修正 (Phase 3) ----
    MIN_HARD_EVENT_DURATION_SEC = 0.3   # <0.3s 硬过滤
    SHORT_EVENT_DURATION_SEC = 0.8      # >=0.3s 且 <0.8s: 保留但标记 is_short_event

    # ---- Phase 4: 集中阈值管理 (引用自 DetectionMetrics) ----
    OCCUPANCY_ENTER_THRESHOLD = DetectionMetrics.OCCUPANCY_ENTER_THRESHOLD
    OCCUPANCY_EXIT_THRESHOLD = DetectionMetrics.OCCUPANCY_EXIT_THRESHOLD
    CONFIRM_ON_FRAMES = DetectionMetrics.CONFIRM_ON_FRAMES
    CONFIRM_OFF_FRAMES = DetectionMetrics.CONFIRM_OFF_FRAMES
    OCCUPANCY_NEAR_MARGIN = DetectionMetrics.OCCUPANCY_NEAR_THRESHOLD_MARGIN

    # ---- Phase 5: CountSegment 切分优化阈值 ----
    COUNT2_SMALL_AREA_RATIO = 1.5          # count=2时面积比<此值降低置信度
    IDENTITY_CONFLICT_BLOB_COUNT = 2       # count=1但blob>=2触发冲突

    # 默认检测参数 (规范 18 节)
    DEFAULT_PARAMS = {
        "coarse_stride_frames": 10,
        "backtrack_seconds": 2.0,
        "forward_confirm_seconds": 1.0,
        "min_on_frames": DetectionMetrics.CONFIRM_ON_FRAMES,
        "min_off_frames": DetectionMetrics.CONFIRM_OFF_FRAMES,
        # Core must stay empty this long before an OccupancyEpisode ends.
        # This replaces cross-episode time-gap merging: only Core occupancy
        # continuity may determine a clip boundary.
        "core_gap_tolerance_seconds": 0.3,
        # Below this score, Core is empty even if a custom/stale occupied flag
        # says otherwise. It is deliberately below the release threshold.
        "core_empty_occupancy_threshold": 0.04,
        "min_event_duration_seconds": 0.8,
        "merge_gap_seconds": 0.8,
        "review_padding_seconds": 1.0,
        "occupy_area_threshold": DetectionMetrics.OCCUPANCY_ENTER_THRESHOLD,
        "release_area_threshold": DetectionMetrics.OCCUPANCY_EXIT_THRESHOLD,
        # 计数相关参数 (改进.md 9 节)
        "count_smooth_window_frames": 5,
        "min_count_change_frames": 5,
        "min_count_segment_duration_sec": 0.5,
        "merge_same_count_gap_sec": 0.3,
        "rapid_alternation_max_sec": 0.5,
    }

    # 状态机状态 (规范 13.1)
    IDLE = 0
    CANDIDATE_ON = 1
    OCCUPIED = 2
    CANDIDATE_OFF = 3

    def __init__(self, metrics_calc: DetectionMetrics):
        """
        :param metrics_calc: DetectionMetrics 实例，用于单帧指标计算
        """
        self._metrics = metrics_calc

    # ------------------------------------------------------------------
    # 主检测入口
    # ------------------------------------------------------------------
    def detect(
        self,
        cap,  # cv2.VideoCapture (已打开)
        roi_data: dict,
        background_bgr: np.ndarray,
        params: dict | None = None,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> list[dict]:
        """
        全视频检测主入口

        :param cap:              已打开的 cv2.VideoCapture
        :param roi_data:         ROI 数据 {"roi_core": {...}, "roi_count": {...}}
        :param background_bgr:   背景帧 BGR 图像
        :param params:           检测参数字典 (缺省使用 DEFAULT_PARAMS)
        :param progress_callback: 进度回调 (percent, message)
        :return: 事件列表 [{"start_frame", "end_frame", "start_time", "end_time",
                            "confidence", "avg_occ_ratio", "max_occ_ratio",
                            "mean_occupancy_score", "max_dark_pixel_ratio",
                            "max_background_diff_score"}, ...]
        """
        # 合并参数
        p = dict(self.DEFAULT_PARAMS)
        if params:
            p.update(params)
        self._normalize_core_empty_params(p)

        total_frames = int(cap.get(7))
        fps = cap.get(5)
        if fps <= 0:
            fps = 30.0
        cap.set(1, 0)  # 回到起始

        self._notify(progress_callback, 0, "开始粗筛扫描...")

        # ---- Phase 1: 粗筛 (规范 11 节) ----
        triggers = self._coarse_scan(cap, total_frames, fps, roi_data, background_bgr, p, progress_callback)

        self._notify(progress_callback, 30, f"粗筛完成, 触发 {len(triggers)} 次, 开始合并重叠触发区间...")

        # 合并重叠的触发窗口
        merged_windows = self._merge_trigger_windows(triggers, fps, p)

        self._notify(progress_callback, 35, f"合并后 {len(merged_windows)} 个检测窗口, 开始局部精查...")

        # ---- Phase 2: 局部逐帧精查 (规范 12 节) ----
        raw_events = []
        for i, (win_start, win_end) in enumerate(merged_windows):
            sub_pct = 35 + int((i / max(1, len(merged_windows))) * 55)
            self._notify(progress_callback, sub_pct,
                         f"精查窗口 {i+1}/{len(merged_windows)} (帧 {win_start}-{win_end})")

            events_in_window = self._fine_scan(
                cap, win_start, win_end, fps, roi_data, background_bgr, p
            )
            raw_events.extend(events_in_window)

        self._notify(progress_callback, 90, f"精查完成, 原始事件 {len(raw_events)} 个, 开始后处理...")

        # ---- Phase 3: 后处理 (规范 14 节) ----
        n_before_filter = len(raw_events)
        events = self._filter_short_events(raw_events, fps, p)
        n_filtered = n_before_filter - len(events)

        self._notify(progress_callback, 93, f"过滤 {n_filtered} 个过短事件, 剩余 {len(events)} 个")

        # Fine scans can overlap because their coarse-trigger windows overlap.
        # Deduplicate only overlapping copies of the same Core episode; never
        # merge distinct episodes across a Core-empty gap.
        events = self._merge_adjacent_events(events, fps, p)

        self._notify(progress_callback, 96, f"去重后 {len(events)} 个事件（Core 空档保留为分割）")

        events = self._add_review_padding(events, fps, p)

        self._notify(progress_callback, 100, f"检测完成! 最终 {len(events)} 个候选事件")

        # 将帧号转换为时间 (秒)
        for evt in events:
            evt["start_time"] = evt["start_frame"] / fps
            evt["end_time"] = evt["end_frame"] / fps
            evt["review_start_time"] = evt["review_start_frame"] / fps
            evt["review_end_time"] = evt["review_end_frame"] / fps
            evt["duration"] = evt["end_time"] - evt["start_time"]
            # Phase 4: 确保均值指标字段存在
            evt.setdefault("mean_occupancy_score", evt.get("avg_occ_ratio", 0.0))
            evt.setdefault("max_dark_pixel_ratio", evt.get("dark_pixel_max", 0.0))
            evt.setdefault("max_background_diff_score", evt.get("bg_diff_max", 0.0))

        return events

    @classmethod
    def _normalize_core_empty_params(cls, params: dict) -> None:
        """Clamp bad Core-gap options instead of letting user config abort detection."""
        try:
            params["core_gap_tolerance_seconds"] = max(0.0, float(params["core_gap_tolerance_seconds"]))
        except (TypeError, ValueError):
            params["core_gap_tolerance_seconds"] = cls.DEFAULT_PARAMS["core_gap_tolerance_seconds"]
        try:
            params["core_empty_occupancy_threshold"] = min(1.0, max(
                0.0, float(params["core_empty_occupancy_threshold"])
            ))
        except (TypeError, ValueError):
            params["core_empty_occupancy_threshold"] = cls.DEFAULT_PARAMS["core_empty_occupancy_threshold"]

    # ------------------------------------------------------------------
    # 两层检测: 占据大事件 + 计数子片段 (改进.md)
    # ------------------------------------------------------------------
    def detect_with_counting(
        self,
        cap,
        roi_data: dict,
        background_bgr: np.ndarray,
        params: dict | None = None,
        progress_callback: Callable[[int, str], None] | None = None,
        counter: MouseCounter | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """
        两层检测:
        1. 先调用 detect() 得到 OccupancyEpisode 列表
        2. 对每个 episode 内部调用 MouseCounter 逐帧估计数量
        3. 用计数状态机切分 CountSegment

        :return: (episodes, segments) 两个列表
        """
        # ---- Phase 1: 原有占据检测 ----
        self._notify(progress_callback, 0, "Phase 1/2: 检测占据大事件...")
        raw_events = self.detect(cap, roi_data, background_bgr, params, progress_callback)

        # ---- Phase 2: 计数检测 ----
        p = dict(self.DEFAULT_PARAMS)
        if params:
            p.update(params)
        self._normalize_core_empty_params(p)

        roi_core = roi_data["roi_core"]
        roi_count = roi_data.get("roi_count")
        fps = cap.get(5)
        if fps <= 0:
            fps = 30.0

        if counter is None:
            counter = MouseCounter()

        self._notify(progress_callback, 50, f"Phase 2/2: 在 {len(raw_events)} 个占据事件内逐帧计数...")

        episodes = []
        all_segments = []

        for ep_idx, evt in enumerate(raw_events):
            sub_pct = 50 + int((ep_idx / max(1, len(raw_events))) * 45)
            self._notify(progress_callback, sub_pct,
                         f"计数事件 {ep_idx+1}/{len(raw_events)} (帧 {evt['start_frame']}-{evt['end_frame']})")

            episode, segments = self._count_within_episode(
                cap, evt, roi_core, roi_count, background_bgr, fps, p, counter, ep_idx
            )
            episodes.append(episode)
            all_segments.extend(segments)

        # A qualifying Core-empty interval belongs to the timeline too.  Emit a
        # canonical CountSegment after occupied clips have been counted; it is
        # deliberately not fed through occupied-only smoothing/short filters.
        gap_segments = self._build_core_empty_gap_segments(raw_events, fps, p)
        all_segments.extend(gap_segments)
        all_segments.sort(key=lambda s: (s["start_frame"], s["end_frame"]))

        self._notify(progress_callback, 100,
                     f"两层检测完成! {len(episodes)} 个占据大事件, {len(all_segments)} 个计数子片段（含 {len(gap_segments)} 个0鼠空档）")

        # ---- Phase 10: 输出检测日志 ----
        self._log_detection_summary(
            video_file=roi_data.get("video_path", "unknown"),
            roi_data=roi_data,
            params=p,
            episodes=episodes,
            segments=all_segments,
            fps=fps,
        )

        return episodes, all_segments

    def _count_within_episode(
        self, cap, evt: dict, roi_core: dict, roi_count: dict | None,
        background_bgr, fps, params, counter, ep_idx: int
    ) -> tuple[dict, list[dict]]:
        """
        在一个占据事件内部逐帧估计小鼠数量, 并按数量变化切分为 CountSegment

        Phase 4: 记录 mean_occupancy_score, max_dark_pixel_ratio, max_background_diff_score
        Phase 5: Count=0 标记 is_possible_false_positive; count=2+小面积降低置信; count=1+多blob标记冲突
        """
        start_f = evt["start_frame"]
        end_f = evt["end_frame"]

        # Phase 4: 均值指标
        mean_occ = evt.get("mean_occupancy_score", evt.get("avg_occ_ratio", 0.0))
        max_dark = evt.get("max_dark_pixel_ratio", evt.get("dark_pixel_max", 0.0))
        max_bg_diff = evt.get("max_background_diff_score", evt.get("bg_diff_max", 0.0))

        # Phase 4: 长期贴近阈值判定
        occ_near_threshold = (
            self.OCCUPANCY_EXIT_THRESHOLD + self.OCCUPANCY_NEAR_MARGIN
            <= mean_occ
            <= self.OCCUPANCY_ENTER_THRESHOLD + self.OCCUPANCY_NEAR_MARGIN
        )

        # 构建 episode
        episode_id = f"{ep_idx + 1:03d}"
        duration = evt.get("duration", (end_f - start_f) / fps)
        episode = {
            "episode_id": episode_id,
            "start_frame": start_f,
            "end_frame": end_f,
            "start_time": evt.get("start_time", start_f / fps),
            "end_time": evt.get("end_time", end_f / fps),
            "duration_sec": duration,
            "detected_by": "auto",
            "confidence": evt.get("confidence", 0.0),
            "status": "pending",
            "child_segment_ids": [],
            "note": "",
            # Phase 4: 均值指标
            "mean_occupancy_score": round(mean_occ, 4),
            "max_dark_pixel_ratio": round(max_dark, 4),
            "max_background_diff_score": round(max_bg_diff, 4),
            "needs_review": occ_near_threshold or evt.get("is_short_event", False),
        }

        # ---- 逐帧估计数量 ----
        cap.set(1, start_f)
        raw_counts = []  # list of (frame, estimated_count, count_confidence, ...)

        for local_idx in range(start_f, end_f + 1):
            ret, frame = cap.read()
            if not ret:
                break

            count_result = counter.estimate_count(
                frame, roi_core, roi_count, background_bgr
            )
            raw_counts.append({
                "frame": local_idx,
                "estimated_mouse_count": count_result["estimated_mouse_count"],
                "count_by_blob": count_result["count_by_blob"],
                "count_by_area": count_result["count_by_area"],
                "total_mouse_area": count_result["total_mouse_area"],
                "largest_blob_area": count_result["largest_blob_area"],
                "core_touching_blob_count": count_result["core_touching_blob_count"],
                "count_confidence": count_result["count_confidence"],
                "blob_count": count_result["blob_count"],
            })

        if not raw_counts:
            episode["child_segment_ids"] = []
            return episode, []

        # ---- 滑动窗口中位数平滑 (改进.md 9.2) ----
        smooth_window = params.get("count_smooth_window_frames", 5)
        smoothed_counts = self._median_smooth_counts(raw_counts, smooth_window)

        # ---- 计数状态机: 保守转换阈值 (改进.md) ----
        segments_raw = self._segment_by_count_change_v2(
            smoothed_counts, raw_counts, fps
        )

        # ---- 后处理: 过滤过短片段 (改进.md 9 节) ----
        min_seg_dur = params.get("min_count_segment_duration_sec", 0.5)
        segments = self._filter_short_count_segments(segments_raw, fps, min_seg_dur)

        # ---- 合并相邻相同数量片段 (改进.md 9 节) ----
        merge_gap = params.get("merge_same_count_gap_sec", 0.3)
        segments = self._merge_same_count_segments(segments, merge_gap, fps)

        # ---- 快速交替切割优化: 检测短暂 count=2 尖刺并拆分 ----
        segments = self._handle_rapid_alternation(segments, fps, params)

        # ---- 构建 CountSegment 列表 ----
        count_segments = []
        child_ids = []
        for seg_idx, seg in enumerate(segments):
            seg_id = f"{episode_id}-{seg_idx + 1:02d}"  # 001-01, 001-02, ...无溢出
            # seg["start_frame"] / seg["end_frame"] 是 raw_counts 数组中的索引 (0-based)
            # 需要加上 episode 的起始帧 start_f 转换为绝对帧号
            seg_start_f = start_f + seg["start_frame"]
            seg_end_f = start_f + seg["end_frame"]
            seg_start_t = seg_start_f / fps
            seg_end_t = seg_end_f / fps
            seg_dur = seg_end_t - seg_start_t

            # 计算该片段的平均置信度 (使用绝对帧号匹配)
            seg_counts = [
                c for c in raw_counts
                if seg_start_f <= c["frame"] <= seg_end_f
            ]
            avg_confidence = (
                np.mean([c["count_confidence"] for c in seg_counts])
                if seg_counts else 0.5
            )

            # Phase 5: 计算片段内 blob_count 最大值和总前景面积
            seg_max_blob_count = max(
                (c.get("blob_count", 0) for c in seg_counts), default=0
            )
            seg_avg_total_area = np.mean(
                [c.get("total_mouse_area", 0.0) for c in seg_counts]
            ) if seg_counts else 0.0

            # Phase 5: 单鼠面积参考 (用于 count=2 小面积判定)
            area_ref = counter.get_single_mouse_area()

            seg_entry = {
                "segment_id": seg_id,
                "episode_id": episode_id,
                "start_frame": seg_start_f,
                "end_frame": seg_end_f,
                "start_time": seg_start_t,
                "end_time": seg_end_t,
                "duration": seg_dur,
                "estimated_mouse_count": seg["count"],
                "confirmed_mouse_count": None,
                "mouse_ids": [],
                "mouse_count": 0,
                "count_confidence": round(float(avg_confidence), 3),
                "count_status": "pending",
                "detected_by": "auto",
                "modified_by_user": False,
                "reviewer": "",
                "note": "",
                "needs_review": False,
                "count_note": "",
                # Phase 7+8: identity assist defaults
                "auto_mouse_colors": [],
                "auto_mouse_ids": [],
                "identity_confidence": 0.0,
                "identity_needs_review": False,
                "identity_conflict": False,
                "identity_method": "",
                # Phase 3: 继承父 episode 的 is_short_event 标记
                "is_short_event": evt.get("is_short_event", seg.get("is_short_event", False)),
                # Phase 5: 额外标记
                "is_possible_false_positive": False,
                "identity_conflict": False,
            }

            # ---- Phase 4: count=0 但在 OccupancyEpisode 内 → false positive ----
            if seg["count"] == 0:
                seg_entry["is_possible_false_positive"] = True
                seg_entry["needs_review"] = True
                seg_entry["count_note"] = "占据事件内计数为0, 可能误检"

            # ---- Phase 5: count=2 但前景面积很小 → 降低置信 ----
            if seg["count"] == 2 and area_ref is not None and area_ref > 0:
                area_ratio = seg_avg_total_area / area_ref
                if area_ratio < self.COUNT2_SMALL_AREA_RATIO:
                    seg_entry["count_confidence"] = round(
                        min(seg_entry["count_confidence"], 0.4), 3
                    )
                    seg_entry["needs_review"] = True
                    existing_note = seg_entry.get("count_note", "")
                    small_area_note = f"count=2但前景面积比仅{area_ratio:.2f}, 建议人工审核"
                    seg_entry["count_note"] = (
                        f"{existing_note}; {small_area_note}" if existing_note else small_area_note
                    )

            # ---- Phase 5: count=1 但 blob_count≥2 → 身份冲突 ----
            if seg["count"] == 1 and seg_max_blob_count >= self.IDENTITY_CONFLICT_BLOB_COUNT:
                seg_entry["identity_conflict"] = True
                seg_entry["needs_review"] = True
                existing_note = seg_entry.get("count_note", "")
                conflict_note = f"count=1但存在{seg_max_blob_count}个blob, 身份冲突"
                seg_entry["count_note"] = (
                    f"{existing_note}; {conflict_note}" if existing_note else conflict_note
                )

            # Phase 3: 短事件标记 (继承 episode 或计数片段自身)
            if seg_entry["is_short_event"]:
                seg_entry["needs_review"] = True
                existing_note = seg_entry.get("count_note", "")
                short_note = "短事件, 建议人工审核"
                seg_entry["count_note"] = (
                    f"{existing_note}; {short_note}" if existing_note else short_note
                )

            # Phase 4: 继承 episode 的 needs_review
            if episode.get("needs_review", False):
                seg_entry["needs_review"] = True
                existing_note = seg_entry.get("count_note", "")
                ep_note = "父episode标记需审核"
                seg_entry["count_note"] = (
                    f"{existing_note}; {ep_note}" if existing_note else ep_note
                )

            # 低置信度片段标记
            if avg_confidence < 0.5:
                seg_entry["needs_review"] = True
                existing_note = seg_entry.get("count_note", "")
                low_conf_note = f"低置信度计数(conf={avg_confidence:.2f})，建议人工审核"
                seg_entry["count_note"] = (
                    f"{existing_note}; {low_conf_note}" if existing_note else low_conf_note
                )

            count_segments.append(seg_entry)
            child_ids.append(seg_id)

        episode["child_segment_ids"] = child_ids
        return episode, count_segments

    @staticmethod
    def _build_core_empty_gap_segments(events: list[dict], fps: float, params: dict) -> list[dict]:
        """Build visible zero-count timeline segments between confirmed Core clips.

        The gap starts at the first confirmed Core-empty frame (previous end +
        1) and ends immediately before the next occupied clip.  It is emitted
        only between two occupied episodes and only when it meets the same
        ``core_gap_tolerance_seconds`` used to split them, avoiding background
        zero clips before the first or after the last occupancy.
        """
        if fps <= 0:
            return []
        ordered = sorted(events, key=lambda e: e["start_frame"])
        min_gap_frames = max(1, int(np.ceil(
            params["core_gap_tolerance_seconds"] * fps
        )))
        result = []
        for previous, following in zip(ordered, ordered[1:]):
            start_frame = previous["end_frame"] + 1
            end_frame = following["start_frame"] - 1
            frame_count = end_frame - start_frame + 1
            if frame_count < min_gap_frames:
                continue
            gap_index = len(result) + 1
            result.append({
                "segment_id": f"core-empty-{gap_index:03d}",
                "episode_id": "core-empty-gap",
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time": start_frame / fps,
                "end_time": end_frame / fps,
                "duration": (end_frame - start_frame) / fps,
                "estimated_mouse_count": 0,
                "confirmed_mouse_count": None,
                "mouse_ids": [],
                "mouse_count": 0,
                "count_confidence": 1.0,
                "count_status": "pending",
                "detected_by": "auto",
                "modified_by_user": False,
                "reviewer": "",
                "note": "ROI Core confirmed empty gap",
                "count_note": "ROI Core confirmed empty gap",
                "needs_review": False,
                "is_short_event": False,
                "is_possible_false_positive": False,
                "start_reason": "core_empty_gap",
                "end_reason": "core_gap_end",
                "auto_mouse_colors": [],
                "auto_mouse_ids": [],
                "identity_confidence": 0.0,
                "identity_needs_review": False,
                "identity_conflict": False,
                "identity_method": "not_applicable_core_empty_gap",
                "core_empty_reason": previous.get("core_empty_reason", "not_occupied"),
                "core_empty_occupancy": previous.get("core_empty_occupancy", {}),
            })
        return result

    @staticmethod
    def _median_smooth_counts(raw_counts: list[dict], window: int) -> list[int]:
        """滑动窗口中位数平滑"""
        counts = [c["estimated_mouse_count"] for c in raw_counts]
        n = len(counts)
        if n == 0:
            return []
        if window >= n or window < 2:
            return counts

        half = window // 2
        smoothed = []
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            window_vals = counts[lo:hi]
            smoothed.append(int(round(np.median(window_vals))))
        return smoothed

    @staticmethod
    def _segment_by_count_change(
        smoothed_counts: list[int], fps: float, min_change_frames: int
    ) -> list[dict]:
        """
        按数量变化切分为稳定片段 (兼容遗留调用)
        :return: [{"start_frame", "end_frame", "count"}, ...]
        """
        if not smoothed_counts:
            return []

        current_count = smoothed_counts[0]
        change_counter = 0
        segment_start = 0

        segments = []

        for i in range(1, len(smoothed_counts)):
            if smoothed_counts[i] != current_count:
                change_counter += 1
                if change_counter >= min_change_frames:
                    seg_end = i - change_counter
                    segments.append({
                        "start_frame": segment_start,
                        "end_frame": seg_end,
                        "count": current_count,
                    })
                    segment_start = i - change_counter + 1
                    current_count = smoothed_counts[i]
                    change_counter = 0
            else:
                change_counter = 0

        segments.append({
            "start_frame": segment_start,
            "end_frame": len(smoothed_counts) - 1,
            "count": current_count,
        })

        return segments

    @staticmethod
    def _segment_by_count_change_v2(
        smoothed_counts: list[int],
        raw_counts: list[dict],
        fps: float,
    ) -> list[dict]:
        """
        保守计数状态机 (cap=2): 不同数量转换需要不同连续帧数
        Transition table (rapid alternation optimized):
          0→1: 4帧,  1→0: 8帧
          1→2: 4帧,  2→1: 3帧
          0→2: 6帧,  2→0: 8帧
        :return: [{"start_frame", "end_frame", "count"}, ...]
        """
        if not smoothed_counts:
            return []

        # Transition table: (from, to) → required_frames
        TRANSITION_FRAMES: dict[tuple[int, int], int] = {
            (0, 1): 4, (1, 0): 8,
            (1, 2): 4, (2, 1): 3,
            (0, 2): 6, (2, 0): 8,
        }
        DEFAULT_REQUIRED = 5

        current_count = smoothed_counts[0]
        candidate_count = None
        change_counter = 0
        segment_start = 0
        segments = []

        for i in range(1, len(smoothed_counts)):
            val = smoothed_counts[i]
            if val == current_count:
                change_counter = 0
                candidate_count = None
                continue

            if candidate_count is None or val != candidate_count:
                candidate_count = val
                change_counter = 1
            else:
                change_counter += 1

            # 查询转换表确定需要的连续帧数
            required = TRANSITION_FRAMES.get(
                (current_count, candidate_count), DEFAULT_REQUIRED
            )

            if change_counter >= required:
                # 确认数量变化
                seg_end = i - change_counter
                segments.append({
                    "start_frame": segment_start,
                    "end_frame": seg_end,
                    "count": current_count,
                })
                segment_start = i - change_counter + 1
                current_count = candidate_count
                change_counter = 0
                candidate_count = None

        # 收尾最后一段
        segments.append({
            "start_frame": segment_start,
            "end_frame": len(smoothed_counts) - 1,
            "count": current_count,
        })

        return segments

    @staticmethod
    def _filter_short_count_segments(
        segments: list[dict], fps: float, min_dur_sec: float
    ) -> list[dict]:
        """短事件策略修正 (Phase 3): 计数片段同样处理
        - <0.3s: 过滤
        - 0.3-0.8s: 保留但标记 is_short_event
        - >=0.8s: 正常
        """
        min_hard_frames = int(DetectionEngine.MIN_HARD_EVENT_DURATION_SEC * fps)
        short_event_frames = int(DetectionEngine.SHORT_EVENT_DURATION_SEC * fps)
        result = []
        for s in segments:
            dur_frames = s["end_frame"] - s["start_frame"]
            if dur_frames < min_hard_frames:
                continue
            if dur_frames < short_event_frames:
                s["is_short_event"] = True
            result.append(s)
        return result

    @staticmethod
    def _merge_same_count_segments(
        segments: list[dict], merge_gap_sec: float, fps: float
    ) -> list[dict]:
        """合并相邻相同数量的片段 (改进.md 9 节)"""
        if not segments:
            return []

        merged = [segments[0]]
        for seg in segments[1:]:
            prev = merged[-1]
            if seg["count"] == prev["count"]:
                gap_frames = seg["start_frame"] - prev["end_frame"]
                gap_sec = gap_frames / fps if fps > 0 else 0
                if gap_sec <= merge_gap_sec:
                    # 合并
                    merged[-1] = {
                        "start_frame": prev["start_frame"],
                        "end_frame": seg["end_frame"],
                        "count": prev["count"],
                    }
                else:
                    merged.append(seg)
            else:
                merged.append(seg)

        return merged

    def _handle_rapid_alternation(self, segments, fps, params):
        """处理快速交替切割优化

        当两只鼠快速交替占据暖点时，计数会短暂出现 1→2→1 的模式。
        该方法扫描 segments 中 count=2 且持续时间低于快速交替阈值
        （默认 0.5s）的片段。对于匹配的片段，将 count=2 帧按中点
        分配给前后 count=1（或 count=0）片段，删除中间的 count=2
        片段，从而将 1 个 count=1 clip 拆分为 2 个独立 clip。

        :param segments: [{"start_frame", "end_frame", "count"}, ...]
                         注意: start_frame/end_frame 是相对于 episode 的帧号
        :param fps: 帧率
        :param params: 参数字典, 包含 rapid_alternation_max_sec
        :return: 处理后的 segments 列表
        """
        if not segments or len(segments) < 3:
            return segments

        rapid_threshold_sec = params.get("rapid_alternation_max_sec", 0.5)
        rapid_threshold_frames = rapid_threshold_sec * fps

        result = []
        i = 0
        n = len(segments)

        while i < n:
            seg = segments[i]

            # 检测快速交替模式: [count=1/0, count=2(短), count=1/0]
            if (seg["count"] == 2
                    and i > 0 and i < n - 1):
                prev_seg = segments[i - 1]
                next_seg = segments[i + 1]

                # 前后 count 相同且为 0 或 1
                if (prev_seg["count"] in (0, 1)
                        and next_seg["count"] == prev_seg["count"]):

                    dur_frames = seg["end_frame"] - seg["start_frame"] + 1
                    if dur_frames < rapid_threshold_frames:
                        # 快速交替: 将 count=2 帧按中点分配给前后片段
                        prev_copy = result.pop()  # 已添加的前一个片段

                        mid = (seg["start_frame"] + seg["end_frame"]) // 2

                        # 前一个 count=1 扩展到中点-1
                        prev_copy["end_frame"] = mid - 1
                        if prev_copy["end_frame"] >= prev_copy["start_frame"]:
                            result.append(prev_copy)

                        # 后一个 count=1 从中点开始
                        next_copy = dict(next_seg)
                        next_copy["start_frame"] = mid
                        if next_copy["end_frame"] >= next_copy["start_frame"]:
                            result.append(next_copy)

                        # 跳过 count=2 和已处理的 next
                        i += 2
                        continue

            result.append(seg)
            i += 1

        return result

    # ------------------------------------------------------------------
    # 粗筛 (规范 11 节)
    # ------------------------------------------------------------------
    def _coarse_scan(self, cap, total_frames, fps, roi_data, background_bgr, params, progress_callback):
        """
        每隔 coarse_stride_frames 帧对 ROI B (缓冲ROI) 采样, 检测遮挡比例 (规范 11.2-11.3 节)
        返回触发帧列表
        """
        stride = params["coarse_stride_frames"]
        threshold = params["occupy_area_threshold"]
        # ROI Count (外圈) 用于粗筛
        roi_count = roi_data.get("roi_count")
        if roi_count is None:
            roi_core = roi_data["roi_core"]
            roi_count = {
                "cx": roi_core["cx"], "cy": roi_core["cy"],
                "a": roi_core["a"] * 1.8, "b": roi_core["b"] * 1.8,
                "angle": roi_core.get("angle", 0.0),
            }

        triggers = set()
        sample_count = (total_frames // stride) + 1

        for idx in range(0, total_frames, stride):
            sample_idx = idx // stride
            if sample_idx % 100 == 0:
                pct = int((sample_idx / max(1, sample_count)) * 28)
                self._notify(progress_callback, pct, f"粗筛(ROI Count): {sample_idx}/{sample_count}")

            cap.set(1, idx)  # cv2.CAP_PROP_POS_FRAMES
            ret, frame = cap.read()
            if not ret:
                continue

            metrics = self._metrics.compute(frame, roi_count, background_bgr)
            occ_ratio = metrics.get("occlusion_area_ratio", 0.0)

            if occ_ratio >= threshold:
                triggers.add(idx)

        # 排序
        return sorted(triggers)

    # ------------------------------------------------------------------
    # 合并重叠触发窗口
    # ------------------------------------------------------------------
    def _merge_trigger_windows(self, triggers, fps, params):
        """
        将粗筛触发帧按照精细扫描窗口合并, 避免重复检测
        """
        if not triggers:
            return []

        backtrack_frames = int(params["backtrack_seconds"] * fps)
        forward_frames = int(params["forward_confirm_seconds"] * fps)

        # 每个触发帧的扫描窗口
        windows = []
        for t in triggers:
            ws = max(0, t - backtrack_frames)
            we = t + forward_frames
            windows.append((ws, we))

        # 排序后合并重叠区间
        windows.sort()
        merged = []
        for ws, we in windows:
            if merged and ws <= merged[-1][1]:
                # 重叠或相邻, 扩展
                merged[-1] = (merged[-1][0], max(merged[-1][1], we))
            else:
                merged.append((ws, we))

        return merged

    # ------------------------------------------------------------------
    # 局部逐帧精查 (规范 12 节)
    # ------------------------------------------------------------------
    def _fine_scan(self, cap, search_start, search_end, fps, roi_data, background_bgr, params):
        """
        在 [search_start, search_end] 范围内逐帧检测, 使用状态机判断事件边界
        返回该窗口内的原始事件列表
        """
        roi_core = roi_data["roi_core"]
        occupy_threshold = params["occupy_area_threshold"]
        release_threshold = params["release_area_threshold"]
        min_on = params["min_on_frames"]
        # The clip split tolerance is time-based so it remains stable across
        # videos with different FPS.  ceil means an empty run at the configured
        # duration is sufficient, while shorter runs are treated as jitter.
        core_gap_frames = max(1, int(np.ceil(
            params["core_gap_tolerance_seconds"] * fps
        )))

        state = self.IDLE
        on_counter = 0          # 连续满足进入条件的帧数
        off_counter = 0         # 连续满足离开条件的帧数
        empty_ratios = []       # qualifying Core-empty occupancy scores
        empty_reasons = []
        event_start = -1        # 当前事件开始帧
        current_occ_ratios = [] # 当前事件的遮挡比例记录

        events = []

        # Phase 4: 逐帧收集辅助指标 (dark_pixel_ratio, background_diff_score)
        current_dark_ratios = []
        current_bg_diff_scores = []

        # 逐帧遍历
        total = search_end - search_start + 1
        cap.set(1, search_start)
        last_frame_read = search_start - 1

        for local_idx in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            global_frame = search_start + local_idx
            last_frame_read = global_frame

            metrics = self._metrics.compute(frame, roi_core, background_bgr)
            occ_ratio = float(metrics.get("occlusion_area_ratio", 0.0))
            # Native DetectionMetrics uses score >= 0.20 for occupation. A
            # score below the explicit empty threshold wins over stale flags.
            is_occupied = bool(metrics.get("is_occupied", occ_ratio >= occupy_threshold))
            low_occupancy = occ_ratio < params["core_empty_occupancy_threshold"]
            core_empty = (not is_occupied) or low_occupancy
            core_empty_reason = "low_occupancy" if low_occupancy else "not_occupied"
            dark_ratio = metrics.get("dark_pixel_ratio", 0.0)
            bg_diff_score = metrics.get("background_diff_score", 0.0)

            # ---- 状态机 (规范 13 节) ----
            if state == self.IDLE:
                if occ_ratio >= occupy_threshold:
                    state = self.CANDIDATE_ON
                    on_counter = 1
                    off_counter = 0
                else:
                    on_counter = 0

            elif state == self.CANDIDATE_ON:
                if occ_ratio >= occupy_threshold:
                    on_counter += 1
                    if on_counter >= min_on:
                        # 确认进入: 事件开始帧 = 连续满足的第一帧
                        state = self.OCCUPIED
                        event_start = global_frame - on_counter + 1
                        current_occ_ratios = []
                        current_dark_ratios = []
                        current_bg_diff_scores = []
                        off_counter = 0
                else:
                    # 未满足, 回到 IDLE (噪声/尾巴)
                    state = self.IDLE
                    on_counter = 0

            elif state == self.OCCUPIED:
                current_occ_ratios.append(occ_ratio)
                current_dark_ratios.append(dark_ratio)
                current_bg_diff_scores.append(bg_diff_score)
                if core_empty:
                    state = self.CANDIDATE_OFF
                    off_counter = 1
                    empty_ratios = [occ_ratio]
                    empty_reasons = [core_empty_reason]
                else:
                    off_counter = 0

            elif state == self.CANDIDATE_OFF:
                current_occ_ratios.append(occ_ratio)
                current_dark_ratios.append(dark_ratio)
                current_bg_diff_scores.append(bg_diff_score)
                if not core_empty:
                    # Intermediate but non-empty scores cancel a gap, avoiding
                    # splits caused by normal near-threshold score noise.
                    state = self.OCCUPIED
                    off_counter = 0
                    empty_ratios = []
                    empty_reasons = []
                else:
                    off_counter += 1
                    empty_ratios.append(occ_ratio)
                    empty_reasons.append(core_empty_reason)
                    if off_counter >= core_gap_frames:
                        # Core has been empty for the configured tolerance.
                        # End on the last Core-occupied frame; ROI Count is not
                        # consulted here and therefore cannot bridge this gap.
                        event_end = global_frame - off_counter
                        state = self.IDLE

                        # 生成事件
                        if event_end >= event_start:
                            occ_slice = current_occ_ratios[:-off_counter] if len(current_occ_ratios) > off_counter else [occ_ratio]
                            dark_slice = current_dark_ratios[:-off_counter] if len(current_dark_ratios) > off_counter else [dark_ratio]
                            bg_diff_slice = current_bg_diff_scores[:-off_counter] if len(current_bg_diff_scores) > off_counter else [bg_diff_score]

                            avg_occ = np.mean(occ_slice) if occ_slice else occ_ratio
                            max_occ = max(occ_slice) if occ_slice else occ_ratio
                            max_dark = max(dark_slice) if dark_slice else dark_ratio
                            max_bg = max(bg_diff_slice) if bg_diff_slice else bg_diff_score
                            confidence = min(1.0, avg_occ * 3.0)  # 简单置信度映射

                            events.append({
                                "start_frame": event_start,
                                "end_frame": event_end,
                                "avg_occ_ratio": float(avg_occ),
                                "max_occ_ratio": float(max_occ),
                                "confidence": float(confidence),
                                # Phase 4: 附属指标
                                "mean_occupancy_score": float(avg_occ),
                                "max_dark_pixel_ratio": float(max_dark),
                                "max_background_diff_score": float(max_bg),
                                "dark_pixel_max": float(max_dark),
                                "bg_diff_max": float(max_bg),
                                "end_reason": "core_gap",
                                "core_gap_frames": off_counter,
                                "core_empty_reason": ("low_occupancy" if "low_occupancy" in empty_reasons else "not_occupied"),
                                "core_empty_occupancy": {
                                    "min": float(min(empty_ratios)) if empty_ratios else 0.0,
                                    "max": float(max(empty_ratios)) if empty_ratios else 0.0,
                                    "mean": float(np.mean(empty_ratios)) if empty_ratios else 0.0,
                                    "threshold": float(params["core_empty_occupancy_threshold"]),
                                },
                            })

                        on_counter = 0
                        off_counter = 0
                        current_occ_ratios = []
                        current_dark_ratios = []
                        current_bg_diff_scores = []
                        empty_ratios = []
                        empty_reasons = []
                        event_start = -1

        # 如果扫描结束时仍处于 OCCUPIED/CANDIDATE 状态, 收尾
        if state in (self.OCCUPIED, self.CANDIDATE_OFF) and event_start >= 0:
            event_end = last_frame_read
            if event_end >= event_start and current_occ_ratios:
                avg_occ = float(np.mean(current_occ_ratios))
                max_occ = float(max(current_occ_ratios))
                max_dark = float(max(current_dark_ratios)) if current_dark_ratios else 0.0
                max_bg = float(max(current_bg_diff_scores)) if current_bg_diff_scores else 0.0
                confidence = min(1.0, avg_occ * 3.0)
                events.append({
                    "start_frame": event_start,
                    "end_frame": event_end,
                    "avg_occ_ratio": avg_occ,
                    "max_occ_ratio": max_occ,
                    "confidence": confidence,
                    # Phase 4: 附属指标
                    "mean_occupancy_score": avg_occ,
                    "max_dark_pixel_ratio": max_dark,
                    "max_background_diff_score": max_bg,
                    "dark_pixel_max": max_dark,
                    "bg_diff_max": max_bg,
                    "end_reason": "scan_end",
                    "core_gap_frames": 0,
                })

        return events

    # ------------------------------------------------------------------
    # 后处理: 过滤过短事件 (规范 14.1)
    # ------------------------------------------------------------------
    def _filter_short_events(self, events, fps, params):
        """短事件策略修正 (Phase 3):
        - duration < MIN_HARD_EVENT_DURATION_SEC (0.3s): 过滤
        - 0.3s <= duration < SHORT_EVENT_DURATION_SEC (0.8s): 保留但标记 is_short_event+needs_review
        - duration >= 0.8s: 正常保留
        """
        min_hard_frames = int(self.MIN_HARD_EVENT_DURATION_SEC * fps)
        short_event_frames = int(self.SHORT_EVENT_DURATION_SEC * fps)
        result = []
        for e in events:
            dur_frames = e["end_frame"] - e["start_frame"]
            if dur_frames < min_hard_frames:
                continue
            if dur_frames < short_event_frames:
                e["is_short_event"] = True
                e["needs_review"] = True
            else:
                e["is_short_event"] = False
            result.append(e)
        return result

    # ------------------------------------------------------------------
    # 后处理: 合并相邻事件 (规范 14.2)
    # ------------------------------------------------------------------
    def _merge_adjacent_events(self, events, fps, params):
        """Deduplicate overlapping fine-scan copies without crossing Core gaps.

        ``merge_gap_seconds`` is intentionally not used for OccupancyEpisodes:
        merging separated events would make an empty ROI Core gap disappear.
        """
        if not events:
            return []

        events = sorted(events, key=lambda e: e["start_frame"])

        merged = [events[0]]
        for evt in events[1:]:
            prev = merged[-1]
            if evt["start_frame"] <= prev["end_frame"]:
                # 合并
                merged[-1] = {
                    **prev,
                    "end_frame": evt["end_frame"],
                    "end_time": evt.get("end_time", prev.get("end_time", 0)),
                    "avg_occ_ratio": (prev["avg_occ_ratio"] + evt["avg_occ_ratio"]) / 2.0,
                    "max_occ_ratio": max(prev["max_occ_ratio"], evt["max_occ_ratio"]),
                    "mean_occupancy_score": max(prev.get("mean_occupancy_score", 0), evt.get("mean_occupancy_score", 0)),
                    "max_dark_pixel_ratio": max(prev.get("max_dark_pixel_ratio", 0), evt.get("max_dark_pixel_ratio", 0)),
                    "max_background_diff_score": max(prev.get("max_background_diff_score", 0), evt.get("max_background_diff_score", 0)),
                    "dark_pixel_max": max(prev.get("dark_pixel_max", 0), evt.get("dark_pixel_max", 0)),
                    "bg_diff_max": max(prev.get("bg_diff_max", 0), evt.get("bg_diff_max", 0)),
                    "confidence": min(1.0, max(prev["confidence"], evt["confidence"]) + 0.05),
                }
            else:
                merged.append(evt)
        return merged

    # ------------------------------------------------------------------
    # 后处理: 增加审核缓冲 (规范 14.3)
    # ------------------------------------------------------------------
    def _add_review_padding(self, events, fps, params):
        """为事件增加前后审核缓冲"""
        pad_frames = int(params["review_padding_seconds"] * fps)
        for evt in events:
            evt["review_start_frame"] = max(0, evt["start_frame"] - pad_frames)
            evt["review_end_frame"] = evt["end_frame"] + pad_frames
        return events

    # ------------------------------------------------------------------
    # Phase 10: 检测日志摘要
    # ------------------------------------------------------------------
    @staticmethod
    def _log_detection_summary(
        video_file: str,
        roi_data: dict,
        params: dict,
        episodes: list[dict],
        segments: list[dict],
        fps: float,
    ) -> None:
        """每次检测运行后输出结构化日志"""
        roi_core = roi_data.get("roi_core", {})
        roi_count = roi_data.get("roi_count", {})
        log_lines = [
            "=" * 60,
            "[DetectionEngine] 检测摘要",
            f"  视频: {video_file}",
            f"  ROI Core: cx={roi_core.get('cx', '?'):.0f} cy={roi_core.get('cy', '?'):.0f} "
            f"a={roi_core.get('a', '?'):.0f} b={roi_core.get('b', '?'):.0f}",
            f"  ROI Count: cx={roi_count.get('cx', '?'):.0f} cy={roi_count.get('cy', '?'):.0f} "
            f"a={roi_count.get('a', '?'):.0f} b={roi_count.get('b', '?'):.0f}",
            f"  参数: stride={params.get('coarse_stride_frames','?')} "
            f"occ_thr={params.get('occupy_area_threshold','?')} "
            f"rel_thr={params.get('release_area_threshold','?')} "
            f"on={params.get('min_on_frames','?')} core_gap={params.get('core_gap_tolerance_seconds','?')}s "
            f"(Core-empty gap splits clip)",
            f"  FPS: {fps:.2f}",
        ]

        # 大事件统计
        log_lines.append(f"  OccupancyEpisodes: {len(episodes)}")
        core_gap_splits = sum(1 for e in episodes if e.get("end_reason") == "core_gap")
        log_lines.append(
            f"    Core-gap splits: {core_gap_splits} "
            f"(tolerance={params.get('core_gap_tolerance_seconds', '?')}s)"
        )
        short_ep = sum(1 for e in episodes if e.get("is_short_event", False))
        if short_ep:
            log_lines.append(f"    短事件(<0.8s): {short_ep}")
        needs_review_ep = sum(1 for e in episodes if e.get("needs_review", False))
        if needs_review_ep:
            log_lines.append(f"    需审核: {needs_review_ep}")

        # 子片段统计
        log_lines.append(f"  CountSegments: {len(segments)}")
        zero_gaps = sum(1 for s in segments if s.get("start_reason") == "core_empty_gap")
        log_lines.append(f"    Confirmed Core-empty 0-count gaps: {zero_gaps}")

        # count 分布
        count_dist: dict[int, int] = {}
        for s in segments:
            c = s.get("estimated_mouse_count", 0)
            count_dist[c] = count_dist.get(c, 0) + 1
        dist_parts = [f"count={c}:{n}" for c, n in sorted(count_dist.items())]
        log_lines.append(f"    数量分布: {', '.join(dist_parts)}")

        # 短事件
        short_segs = sum(1 for s in segments if s.get("is_short_event", False))
        if short_segs:
            log_lines.append(f"    短片段(<0.8s): {short_segs}")

        # needs_review 数量
        needs_review_segs = sum(1 for s in segments if s.get("needs_review", False))
        identity_conflict = sum(1 for s in segments if s.get("identity_conflict", False))
        false_positives = sum(1 for s in segments if s.get("is_possible_false_positive", False))
        log_lines.append(
            f"    需审核: {needs_review_segs} "
            f"(冲突={identity_conflict} 可能误检={false_positives})"
        )

        log_lines.append("=" * 60)

        for line in log_lines:
            logger.info(line)

    # ------------------------------------------------------------------
    # 进度通知
    # ------------------------------------------------------------------
    @staticmethod
    def _notify(callback, percent, message):
        if callback:
            callback(percent, message)

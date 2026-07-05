"""
全视频自动检测引擎
实现规范 8-14 节：隔帧粗筛 → 局部逐帧精查 → 状态机 → 后处理
改进.md：两层检测架构 OccupancyEpisode + CountSegment
"""

import time
import numpy as np
from collections.abc import Callable

from .metrics import DetectionMetrics
from .counter import MouseCounter


class DetectionEngine:
    """全视频自动检测引擎"""

    # 默认检测参数 (规范 18 节)
    DEFAULT_PARAMS = {
        "coarse_stride_frames": 10,
        "backtrack_seconds": 2.0,
        "forward_confirm_seconds": 1.0,
        "min_on_frames": 4,
        "min_off_frames": 10,
        "min_event_duration_seconds": 0.8,
        "merge_gap_seconds": 0.8,
        "review_padding_seconds": 1.0,
        "occupy_area_threshold": 0.20,
        "release_area_threshold": 0.08,
        # 计数相关参数 (改进.md 9 节)
        "roi_c_scale": 1.6,
        "count_smooth_window_frames": 5,
        "min_count_change_frames": 5,
        "min_count_segment_duration_sec": 0.5,
        "merge_same_count_gap_sec": 0.3,
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
        :param roi_data:         ROI 数据 {"roi_a": {...}, "buffer_roi_scale": ...}
        :param background_bgr:   背景帧 BGR 图像
        :param params:           检测参数字典 (缺省使用 DEFAULT_PARAMS)
        :param progress_callback: 进度回调 (percent, message)
        :return: 事件列表 [{"start_frame", "end_frame", "start_time", "end_time",
                            "confidence", "avg_occ_ratio", "max_occ_ratio"}, ...]
        """
        # 合并参数
        p = dict(self.DEFAULT_PARAMS)
        if params:
            p.update(params)

        total_frames = int(cap.get(3))  # cv2.CAP_PROP_FRAME_COUNT = 7? Actually cv2.CAP_PROP_FRAME_COUNT = 7
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

        events = self._merge_adjacent_events(events, fps, p)

        self._notify(progress_callback, 96, f"合并后 {len(events)} 个事件")

        events = self._add_review_padding(events, fps, p)

        self._notify(progress_callback, 100, f"检测完成! 最终 {len(events)} 个候选事件")

        # 将帧号转换为时间 (秒)
        for evt in events:
            evt["start_time"] = evt["start_frame"] / fps
            evt["end_time"] = evt["end_frame"] / fps
            evt["review_start_time"] = evt["review_start_frame"] / fps
            evt["review_end_time"] = evt["review_end_frame"] / fps
            evt["duration"] = evt["end_time"] - evt["start_time"]

        return events

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

        roi_a = roi_data["roi_a"]
        roi_c_scale = roi_data.get("roi_c_scale", p.get("roi_c_scale", 1.6))
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
                cap, evt, roi_a, roi_c_scale, background_bgr, fps, p, counter, ep_idx
            )
            episodes.append(episode)
            all_segments.extend(segments)

        self._notify(progress_callback, 100,
                     f"两层检测完成! {len(episodes)} 个占据大事件, {len(all_segments)} 个计数子片段")

        return episodes, all_segments

    def _count_within_episode(
        self, cap, evt: dict, roi_a: dict, roi_c_scale: float,
        background_bgr, fps, params, counter, ep_idx: int
    ) -> tuple[dict, list[dict]]:
        """
        在一个占据事件内部逐帧估计小鼠数量, 并按数量变化切分为 CountSegment
        """
        start_f = evt["start_frame"]
        end_f = evt["end_frame"]

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
        }

        # ---- 逐帧估计数量 ----
        cap.set(1, start_f)
        raw_counts = []  # list of (frame, estimated_count, count_confidence, ...)

        for local_idx in range(start_f, end_f + 1):
            ret, frame = cap.read()
            if not ret:
                break

            count_result = counter.estimate_count(
                frame, roi_a, roi_c_scale, background_bgr
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
        segments = self._merge_same_count_segments(segments, merge_gap)

        # ---- 构建 CountSegment 列表 ----
        count_segments = []
        child_ids = []
        for seg_idx, seg in enumerate(segments):
            seg_id = f"{episode_id}-{chr(65 + seg_idx)}"  # 001-A, 001-B, ...
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
            }

            # 低置信度片段标记
            if avg_confidence < 0.5:
                seg_entry["needs_review"] = True
                seg_entry["count_note"] = (
                    f"低置信度计数(conf={avg_confidence:.2f})，建议人工审核"
                )

            count_segments.append(seg_entry)
            child_ids.append(seg_id)

        episode["child_segment_ids"] = child_ids
        return episode, count_segments

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
            smoothed.append(int(np.median(window_vals)))
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
        保守计数状态机: 不同数量转换需要不同连续帧数
        1→2: 连续8帧
        1→3: 连续15帧 + 需要额外证据(count_by_blob>=2)
        2→3: 连续10帧
        其他: 连续5帧
        :return: [{"start_frame", "end_frame", "count"}, ...]
        """
        if not smoothed_counts:
            return []

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

            # 确定需要的连续帧数
            required = 5  # 默认
            if current_count == 1 and candidate_count == 2:
                required = 8
            elif current_count == 1 and candidate_count == 3:
                required = 15
            elif current_count == 2 and candidate_count == 3:
                required = 10

            if change_counter >= required:
                # 1→3 额外证据检查
                if current_count == 1 and candidate_count == 3:
                    if not DetectionEngine._check_extra_evidence_1to3(
                        raw_counts, i - change_counter + 1, i
                    ):
                        # 证据不足, 继续等待 (但不重置计数器)
                        continue

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
    def _check_extra_evidence_1to3(
        raw_counts: list[dict], start_idx: int, end_idx: int
    ) -> bool:
        """
        检查1→3转换窗口内是否有 count_by_blob >= 2 的帧
        """
        for i in range(start_idx, min(end_idx + 1, len(raw_counts))):
            if raw_counts[i].get("count_by_blob", 0) >= 2:
                return True
        return False

    @staticmethod
    def _filter_short_count_segments(
        segments: list[dict], fps: float, min_dur_sec: float
    ) -> list[dict]:
        """过滤过短计数片段 (改进.md 9 节)"""
        min_frames = int(min_dur_sec * fps)
        return [s for s in segments if (s["end_frame"] - s["start_frame"]) >= min_frames]

    @staticmethod
    def _merge_same_count_segments(
        segments: list[dict], merge_gap_sec: float
    ) -> list[dict]:
        """合并相邻相同数量的片段 (改进.md 9 节)"""
        if not segments:
            return []

        # 需要实际帧间隔来判断, 这里用 end_frame 和 start_frame 差值
        # gap 单位是帧, 需要从调用方传入 fps
        # 简化处理: 相邻相同 count 直接合并 (因为计数状态机已确保连续)
        merged = [segments[0]]
        for seg in segments[1:]:
            prev = merged[-1]
            if seg["count"] == prev["count"]:
                # 合并
                merged[-1] = {
                    "start_frame": prev["start_frame"],
                    "end_frame": seg["end_frame"],
                    "count": prev["count"],
                }
            else:
                merged.append(seg)

        return merged

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
        roi_a = roi_data["roi_a"]
        buffer_scale = roi_data.get("buffer_roi_scale", 1.8)

        # ROI B = 将 ROI A 外扩 buffer_roi_scale 倍用于粗筛 (规范 11.2 节)
        roi_b = dict(roi_a)
        roi_b["a"] = roi_a["a"] * buffer_scale
        roi_b["b"] = roi_a["b"] * buffer_scale

        triggers = set()
        sample_count = (total_frames // stride) + 1

        for idx in range(0, total_frames, stride):
            sample_idx = idx // stride
            if sample_idx % 100 == 0:
                pct = int((sample_idx / max(1, sample_count)) * 28)
                self._notify(progress_callback, pct, f"粗筛(ROI B): {sample_idx}/{sample_count}")

            cap.set(1, idx)  # cv2.CAP_PROP_POS_FRAMES
            ret, frame = cap.read()
            if not ret:
                continue

            metrics = self._metrics.compute(frame, roi_b, background_bgr)
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
        roi_a = roi_data["roi_a"]
        occupy_threshold = params["occupy_area_threshold"]
        release_threshold = params["release_area_threshold"]
        min_on = params["min_on_frames"]
        min_off = params["min_off_frames"]

        state = self.IDLE
        on_counter = 0          # 连续满足进入条件的帧数
        off_counter = 0         # 连续满足离开条件的帧数
        event_start = -1        # 当前事件开始帧
        current_occ_ratios = [] # 当前事件的遮挡比例记录

        events = []

        # 逐帧遍历
        total = search_end - search_start + 1
        cap.set(1, search_start)

        for local_idx in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            global_frame = search_start + local_idx

            metrics = self._metrics.compute(frame, roi_a, background_bgr)
            occ_ratio = metrics.get("occlusion_area_ratio", 0.0)

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
                        off_counter = 0
                else:
                    # 未满足, 回到 IDLE (噪声/尾巴)
                    state = self.IDLE
                    on_counter = 0

            elif state == self.OCCUPIED:
                current_occ_ratios.append(occ_ratio)
                if occ_ratio <= release_threshold:
                    state = self.CANDIDATE_OFF
                    off_counter = 1
                else:
                    off_counter = 0

            elif state == self.CANDIDATE_OFF:
                current_occ_ratios.append(occ_ratio)
                if occ_ratio >= occupy_threshold:
                    # 遮挡再次升高, 回到 OCCUPIED (规范 13: CANDIDATE_OFF → OCCUPIED)
                    state = self.OCCUPIED
                    off_counter = 0
                elif occ_ratio <= release_threshold:
                    off_counter += 1
                    if off_counter >= min_off:
                        # 确认离开: 事件结束帧 = 离开前最后一帧
                        event_end = global_frame - off_counter
                        state = self.IDLE

                        # 生成事件
                        if event_end >= event_start:
                            avg_occ = (np.mean(current_occ_ratios[:-off_counter])
                                       if len(current_occ_ratios) > off_counter else occ_ratio)
                            max_occ = (max(current_occ_ratios[:-off_counter])
                                       if len(current_occ_ratios) > off_counter else occ_ratio)
                            confidence = min(1.0, avg_occ * 3.0)  # 简单置信度映射

                            events.append({
                                "start_frame": event_start,
                                "end_frame": event_end,
                                "avg_occ_ratio": float(avg_occ),
                                "max_occ_ratio": float(max_occ),
                                "confidence": float(confidence),
                            })

                        on_counter = 0
                        off_counter = 0
                        current_occ_ratios = []
                        event_start = -1
                else:
                    # 中间区域 (release < occ < occupy): 遮挡未真正离开也未重新进入
                    # 重置计数器, 避免累积误判提前结束事件 (规范 13.2)
                    off_counter = 0

        # 如果扫描结束时仍处于 OCCUPIED/CANDIDATE 状态, 收尾
        if state in (self.OCCUPIED, self.CANDIDATE_OFF) and event_start >= 0:
            event_end = search_start + total - 1
            if event_end >= event_start and current_occ_ratios:
                avg_occ = float(np.mean(current_occ_ratios))
                max_occ = float(max(current_occ_ratios))
                confidence = min(1.0, avg_occ * 3.0)
                events.append({
                    "start_frame": event_start,
                    "end_frame": event_end,
                    "avg_occ_ratio": avg_occ,
                    "max_occ_ratio": max_occ,
                    "confidence": confidence,
                })

        return events

    # ------------------------------------------------------------------
    # 后处理: 过滤过短事件 (规范 14.1)
    # ------------------------------------------------------------------
    def _filter_short_events(self, events, fps, params):
        """过滤时长 < min_event_duration_seconds 的事件"""
        min_dur = params["min_event_duration_seconds"]
        min_frames = int(min_dur * fps)
        return [e for e in events if (e["end_frame"] - e["start_frame"]) >= min_frames]

    # ------------------------------------------------------------------
    # 后处理: 合并相邻事件 (规范 14.2)
    # ------------------------------------------------------------------
    def _merge_adjacent_events(self, events, fps, params):
        """合并间隔 < merge_gap_seconds 的相邻事件"""
        if not events:
            return []

        merge_gap_frames = int(params["merge_gap_seconds"] * fps)
        events = sorted(events, key=lambda e: e["start_frame"])

        merged = [events[0]]
        for evt in events[1:]:
            prev = merged[-1]
            gap = evt["start_frame"] - prev["end_frame"]
            if gap <= merge_gap_frames:
                # 合并
                merged[-1] = {
                    "start_frame": prev["start_frame"],
                    "end_frame": evt["end_frame"],
                    "avg_occ_ratio": (prev["avg_occ_ratio"] + evt["avg_occ_ratio"]) / 2.0,
                    "max_occ_ratio": max(prev["max_occ_ratio"], evt["max_occ_ratio"]),
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
    # 进度通知
    # ------------------------------------------------------------------
    @staticmethod
    def _notify(callback, percent, message):
        if callback:
            callback(percent, message)

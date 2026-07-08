"""
耳标颜色辅助检测 (Phase 7+8+9)
可选的辅助工具, 不覆盖人工结果
提供耳标颜色候选 + 冲突处理 + A→B替换辅助 + 向后兼容
"""

import cv2
import numpy as np
import logging
from collections import defaultdict
from collections.abc import Callable

logger = logging.getLogger(__name__)


class IdentityAssist:
    """耳标颜色辅助检测 — 不要求全自动识别, 仅提供候选"""

    # Ear tag pixel conditions
    EAR_V_MIN = 80
    EAR_S_MIN = 30
    WARM_H_LOW = 10
    WARM_H_HIGH = 30

    # Blob filter
    EAR_AREA_MIN = 3
    EAR_AREA_MAX = 500

    # Proximity to dark mouse mask
    MOUSE_PROXIMITY_PX = 20

    # ROI Search = Core × 2.8
    SEARCH_SCALE = 2.8

    def __init__(self, debug: bool = False):
        self.debug = debug
        if debug:
            logger.setLevel(logging.DEBUG)
            if not logger.handlers:
                h = logging.StreamHandler()
                h.setFormatter(logging.Formatter('%(asctime)s [%(name)s] %(message)s', datefmt='%H:%M:%S'))
                logger.addHandler(h)

    # ------------------------------------------------------------------
    # analyze_segment: 统一入口 (带进度回调)
    # ------------------------------------------------------------------
    def analyze_segment(
        self,
        segment: dict,
        roi_core: dict,
        cap,  # cv2.VideoCapture
        fps: float,
        background_bgr: np.ndarray | None = None,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> dict:
        """
        分析单个 CountSegment 的颜色身份。

        :param segment: CountSegment dict (含 start_frame, end_frame, estimated_mouse_count)
        :param roi_core: ROI A 参数字典 {cx, cy, a, b, angle}
        :param cap: 已打开的 cv2.VideoCapture
        :param fps: 视频 FPS
        :param background_bgr: 背景帧 (当前未使用)
        :param progress_callback: 进度回调 (percent 0-100, message)
        :return: dict 包含:
            auto_mouse_colors, auto_mouse_ids, identity_confidence,
            identity_needs_review, identity_conflict, identity_method
        """
        self._notify(progress_callback, 0, "颜色识别: 开始扫描帧...")

        # 调用底层 detect_ear_tags
        result = self.detect_ear_tags(
            segment=segment,
            roi_core=roi_core,
            cap=cap,
            fps=fps,
            background_bgr=background_bgr,
        )

        # 统一 identity_method 为 "color_rule" 格式
        result["identity_method"] = "color_rule"

        self._notify(progress_callback, 100, "颜色识别完成")

        return result

    # ------------------------------------------------------------------
    # Color classification (rule-based)
    # ------------------------------------------------------------------
    @staticmethod
    def classify_color(h: float, s: float, v: float) -> str:
        """Classify pixel color from HSV values.
        Returns: 'red' | 'blue' | 'green' | 'white' | 'unknown'
        """
        # red: H<10 or H>170, S>50
        if (h < 10 or h > 170) and s > 50:
            return "red"
        # blue: 90≤H≤140, S>35
        if 90 <= h <= 140 and s > 35:
            return "blue"
        # green: 40≤H≤80, S>35
        if 40 <= h <= 80 and s > 35:
            return "green"
        # white: S<30, V>180
        if s < 30 and v > 180:
            return "white"
        return "unknown"

    # ------------------------------------------------------------------
    # Contour-level color classification (pixel voting)
    # ------------------------------------------------------------------
    @staticmethod
    def _classify_contour(px: np.ndarray) -> tuple[str, bool]:
        """
        逐像素颜色投票，返回多数颜色。
        解决色相循环均值崩塌导致的红色→蓝色误判。
        
        :param px: (N,3) HSV像素数组
        :return: (color, is_low_confidence)
                 color: 'red'|'blue'|'green'|'white'|'unknown'
                 is_low_confidence: True if secondary (relaxed) thresholds were used
        """
        if len(px) == 0:
            return "unknown", True
        
        h = px[:, 0]
        s = px[:, 1]
        v = px[:, 2]
        
        # Primary vote: standard thresholds (same as classify_color)
        red_mask   = ((h < 10) | (h > 170)) & (s > 50)
        blue_mask  = (h >= 90) & (h <= 140) & (s > 35)
        green_mask = (h >= 40) & (h <= 80) & (s > 35)
        white_mask = (s < 30) & (v > 180)
        
        counts = {
            "red":   int(np.sum(red_mask)),
            "blue":  int(np.sum(blue_mask)),
            "green": int(np.sum(green_mask)),
            "white": int(np.sum(white_mask)),
        }
        
        best_color = max(counts, key=counts.get)
        if counts[best_color] > 0:
            return best_color, False
        
        # Secondary fallback: relaxed thresholds for low-confidence detection
        red_mask2   = ((h < 10) | (h > 170)) & (s > 35)
        blue_mask2  = (h >= 85) & (h <= 145) & (s > 25)
        green_mask2 = (h >= 35) & (h <= 85) & (s > 25)
        white_mask2 = (s < 40) & (v > 160)
        
        counts2 = {
            "red":   int(np.sum(red_mask2)),
            "blue":  int(np.sum(blue_mask2)),
            "green": int(np.sum(green_mask2)),
            "white": int(np.sum(white_mask2)),
        }
        
        best_color2 = max(counts2, key=counts2.get)
        if counts2[best_color2] > 0:
            return best_color2, True
        
        return "unknown", True

    # ------------------------------------------------------------------
    # Main detection
    # ------------------------------------------------------------------
    def detect_ear_tags(
        self,
        segment: dict,
        roi_core: dict,
        cap,  # cv2.VideoCapture
        fps: float,
        background_bgr: np.ndarray | None = None,
    ) -> dict:
        """
        Detect ear tag color candidates for one CountSegment.

        :param segment: CountSegment with start_frame / end_frame / estimated_mouse_count
        :param roi_core: ROI A params {cx, cy, a, b, angle}
        :param cap: open cv2.VideoCapture
        :param fps: video FPS
        :param background_bgr: background frame (unused currently)
        :return: dict ready to merge into CountSegment:
            auto_mouse_colors, auto_mouse_ids, identity_confidence,
            identity_needs_review, identity_conflict, identity_method, identity_note
        """
        start_f = segment["start_frame"]
        end_f = segment["end_frame"]
        mid_f = (start_f + end_f) // 2

        # ---- ROI Search = Core × 2.8 ----
        roi_search = dict(roi_core)
        roi_search["a"] = roi_core["a"] * self.SEARCH_SCALE
        roi_search["b"] = roi_core["b"] * self.SEARCH_SCALE

        a_core = roi_core["a"]
        b_core = roi_core["b"]
        a_search = roi_search["a"]
        b_search = roi_search["b"]
        angle = roi_search.get("angle", 0.0)

        # ---- Three-round frame selection ----
        search_frames = []
        seen = set()

        # Round 1: dense around mid (±15)
        for f in range(max(start_f, mid_f - 15), min(end_f, mid_f + 15) + 1):
            if f not in seen:
                search_frames.append(f)
                seen.add(f)

        # Round 2: sparse extension (±60, step 2)
        for f in range(max(start_f, mid_f - 60), min(end_f, mid_f + 60) + 1, 2):
            if f not in seen:
                search_frames.append(f)
                seen.add(f)

        # Round 3: full segment (every 5 frames)
        for f in range(start_f, end_f + 1, 5):
            if f not in seen:
                search_frames.append(f)
                seen.add(f)

        # ---- Accumulate hits per color ----
        # {color: {"hit_count": int, "frames": [int], "confs": [float], "areas": [float]}}
        color_hits: dict[str, dict] = defaultdict(
            lambda: {"hit_count": 0, "frames": [], "confs": [], "areas": []}
        )

        total_search = len(search_frames)
        logger.info(f"扫描帧数: {total_search} (R1密集+R2扩展+R3全段)")

        for fi, frame_idx in enumerate(search_frames):
            cap.set(1, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            h, w = frame.shape[:2]
            cx, cy = roi_core["cx"], roi_core["cy"]

            # Crop search ROI
            x1 = max(0, int(cx - a_search))
            y1 = max(0, int(cy - b_search))
            x2 = min(w, int(cx + a_search))
            y2 = min(h, int(cy + b_search))
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            crop_h, crop_w = crop.shape[:2]
            cx_c = crop_w / 2.0
            cy_c = crop_h / 2.0

            # Create masks in crop coordinates
            search_ellipse = self._ellipse(crop_h, crop_w, cx_c, cy_c, a_search, b_search, angle)
            core_ellipse = self._ellipse(crop_h, crop_w, cx_c, cy_c, a_core, b_core, angle)

            # Annular region = search \ core
            annular = cv2.bitwise_and(search_ellipse, cv2.bitwise_not(core_ellipse))

            # HSV
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

            # Dark mouse mask: V < 80 (用于 blobs 靠近判定)
            _, mouse_mask = cv2.threshold(hsv[:, :, 2], self.EAR_V_MIN - 1, 255, cv2.THRESH_BINARY_INV)
            mouse_mask = cv2.bitwise_and(mouse_mask, mouse_mask, mask=search_ellipse)

            # Ear-tag pixel mask (V>80, S>30, H∉[10,30])
            ep_mask = self._ear_tag_pixel_mask(hsv)
            ep_mask = cv2.bitwise_and(ep_mask, ep_mask, mask=annular)

            # Connected components
            contours, _ = cv2.findContours(ep_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.EAR_AREA_MIN or area > self.EAR_AREA_MAX:
                    continue

                # Near mouse mask check
                if not self._contour_near_mask(cnt, mouse_mask):
                    continue

                # Dominant color (mean HSV inside contour)
                c_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
                cv2.drawContours(c_mask, [cnt], -1, 255, -1)
                px = hsv[c_mask > 0]
                if len(px) == 0:
                    continue

                color, low_conf = self._classify_contour(px)
                if color == "unknown":
                    continue

                # Inner ring check
                M = cv2.moments(cnt)
                inner = False
                if M["m00"] > 0:
                    ccx = M["m10"] / M["m00"]
                    ccy = M["m01"] / M["m00"]
                    inner = self._in_inner_ring(ccx, ccy, a_core, b_core, a_search, b_search, cx_c, cy_c)

                # Per-hit confidence sub-score
                hit_conf = self._hit_conf_part(color, area, inner)
                if low_conf:
                    hit_conf *= 0.5  # Penalize secondary-threshold classifications

                color_hits[color]["hit_count"] += 1
                color_hits[color]["frames"].append(frame_idx)
                color_hits[color]["confs"].append(hit_conf)
                color_hits[color]["areas"].append(area)

        # ---- Aggregate per-color confidence ----
        for color in color_hits:
            h = color_hits[color]
            base = 0.5
            if color in ("red", "blue", "green"):
                base += 0.2
            if 3 <= np.mean(h["areas"]) <= 300 if h["areas"] else False:
                base += 0.1
            if h["hit_count"] >= 3:
                base += 0.1
            avg_hc = np.mean(h["confs"]) if h["confs"] else 0.0
            h["confidence"] = min(0.95, base + avg_hc * 0.5)

        # 日志: 汇总每种颜色的命中
        for color in sorted(color_hits.keys()):
            h = color_hits[color]
            logger.info(f"颜色 '{color}': hits={h['hit_count']} frames={len(h['frames'])} "
                        f"avg_area={np.mean(h['areas']):.1f} conf={h['confidence']:.2f}")

        # ---- Conflict resolution ----
        target_count = segment.get("estimated_mouse_count", 1)
        if target_count is None or target_count <= 0:
            target_count = 1
        target_count = min(target_count, 2)  # cap=2

        colors = sorted(color_hits.keys(), key=lambda c: color_hits[c]["confidence"], reverse=True)
        found_count = len(colors)

        logger.info(f"冲突处理: target={target_count} found={found_count} colors={colors}")

        auto_mouse_colors: list[str] = []
        auto_mouse_ids: list[str] = []
        identity_needs_review = False
        identity_conflict = False
        identity_confidence = 0.0
        identity_method = "ear_tag_color"
        note = ""

        if found_count == 0:
            auto_mouse_colors = ["unknown"] * target_count
            auto_mouse_ids = [""] * target_count
            identity_needs_review = True
            identity_confidence = 0.3
            identity_method += "|no_detection"
            note = "未检测到任何耳标颜色"
            logger.warning(f"未检测到耳标颜色! 扫描了{total_search}帧, 环形区域像素条件: V>{self.EAR_V_MIN} S>{self.EAR_S_MIN} H∉[{self.WARM_H_LOW},{self.WARM_H_HIGH}] area={self.EAR_AREA_MIN}-{self.EAR_AREA_MAX}, 需靠近mouse_mask≤{self.MOUSE_PROXIMITY_PX}px")

        elif found_count == target_count:
            auto_mouse_colors = colors[:]
            auto_mouse_ids = [f"auto_{c}" for c in colors]
            identity_needs_review = False
            identity_conflict = False
            identity_confidence = float(np.mean([color_hits[c]["confidence"] for c in colors]))
            identity_method += "|exact_match"

        elif found_count < target_count:
            auto_mouse_colors = colors + ["unknown"] * (target_count - found_count)
            auto_mouse_ids = [f"auto_{c}" for c in colors] + [""] * (target_count - found_count)
            identity_needs_review = True
            identity_confidence = 0.4
            identity_method += "|insufficient"
            note = f"仅找到{found_count}种颜色, 需{target_count}种"

        else:  # found_count > target_count
            top = colors[:target_count]
            auto_mouse_colors = top
            auto_mouse_ids = [f"auto_{c}" for c in top]
            identity_needs_review = True
            identity_conflict = True
            identity_confidence = float(np.mean([color_hits[c]["confidence"] for c in top]))
            identity_method += "|conflict"
            note = f"检测到{found_count}种颜色({','.join(colors)}), 取top{target_count}"

        # ---- Phase 8: A→B swap detection ----
        if target_count == 1 and found_count >= 2:
            top2 = colors[:2]
            if self._temporal_separation(color_hits, top2, fps):
                identity_conflict = True
                prefix = note + "; " if note else ""
                note = f"{prefix}Possible A-to-B swap detected"
                identity_method += "|a_to_b"

        return {
            "auto_mouse_colors": auto_mouse_colors,
            "auto_mouse_ids": auto_mouse_ids,
            "identity_confidence": round(identity_confidence, 3),
            "identity_needs_review": identity_needs_review,
            "identity_conflict": identity_conflict,
            "identity_method": identity_method,
            "identity_note": note,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ellipse(h: int, w: int, cx: float, cy: float, a: float, b: float, ang: float = 0.0) -> np.ndarray:
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(m, (int(cx), int(cy)), (int(a), int(b)), ang, 0, 360, 255, -1)
        return m

    @classmethod
    def _ear_tag_pixel_mask(cls, hsv: np.ndarray) -> np.ndarray:
        """V>80, S>30, H∉[10,30]"""
        v = hsv[:, :, 2]
        s = hsv[:, :, 1]
        h = hsv[:, :, 0]

        v_ok = v > cls.EAR_V_MIN
        s_ok = s > cls.EAR_S_MIN
        h_ok = (h < cls.WARM_H_LOW) | (h > cls.WARM_H_HIGH)

        return (v_ok & s_ok & h_ok).astype(np.uint8) * 255

    @classmethod
    def _contour_near_mask(cls, cnt, mouse_mask: np.ndarray) -> bool:
        """True if any pixel of contour is within MOUSE_PROXIMITY_PX of the mouse mask."""
        h, w = mouse_mask.shape
        cnt_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)

        # Dilate the ear-tag contour by proximity — if it overlaps mouse_mask, it's near
        ksz = cls.MOUSE_PROXIMITY_PX * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        dilated = cv2.dilate(cnt_mask, kernel, iterations=1)
        overlap = cv2.bitwise_and(dilated, mouse_mask)
        return np.count_nonzero(overlap) > 0

    @staticmethod
    def _in_inner_ring(ccx: float, ccy: float,
                       a_core: float, b_core: float,
                       a_srch: float, b_srch: float,
                       cx_c: float, cy_c: float) -> bool:
        """True if the point lies in the inner 50% of the annular ring (norm space)."""
        if a_srch <= 0 or b_srch <= 0:
            return False
        dx = ccx - cx_c
        dy = ccy - cy_c
        r = np.sqrt((dx / a_srch) ** 2 + (dy / b_srch) ** 2)
        core_r = a_core / a_srch  # ~ 0.4
        inner_thresh = core_r + (1.0 - core_r) * 0.5
        return r <= inner_thresh

    @staticmethod
    def _hit_conf_part(color: str, area: float, inner: bool) -> float:
        """Per-blob confidence sub-score (0–0.4)."""
        s = 0.0
        if color in ("red", "blue", "green"):
            s += 0.2
        if 10 <= area <= 300:
            s += 0.1
        if inner:
            s += 0.1
        return s

    @staticmethod
    def _temporal_separation(color_hits: dict, top_colors: list[str], fps: float) -> bool:
        """True if two colors appear in clearly different time periods."""
        if len(top_colors) < 2:
            return False
        c1, c2 = top_colors[0], top_colors[1]
        f1 = color_hits[c1]["frames"]
        f2 = color_hits[c2]["frames"]
        if not f1 or not f2:
            return False

        r1 = (min(f1), max(f1))
        r2 = (min(f2), max(f2))

        # One entirely before the other
        if r1[1] < r2[0] or r2[1] < r1[0]:
            return True

        # Overlap < 20% of total span
        total_span = max(r1[1], r2[1]) - min(r1[0], r2[0])
        if total_span <= 0:
            return False
        overlap = max(0, min(r1[1], r2[1]) - max(r1[0], r2[0]))
        return overlap / total_span < 0.2

    @staticmethod
    def _notify(callback, percent, message):
        """安全调用进度回调"""
        if callback:
            callback(percent, message)


# ------------------------------------------------------------------
# Utility: apply result to a CountSegment dict (in-place)
# ------------------------------------------------------------------
def apply_identity_to_segment(segment: dict, id_result: dict) -> dict:
    """
    Apply IdentityAssist results to a CountSegment dict.
    Does NOT overwrite manually confirmed results.

    :param segment: CountSegment dict (modified in-place)
    :param id_result: result from IdentityAssist.detect_ear_tags()
    :return: same segment (for chaining)
    """
    if segment.get("count_status") == "confirmed":
        return segment

    segment["auto_mouse_colors"] = id_result["auto_mouse_colors"]
    segment["auto_mouse_ids"] = id_result["auto_mouse_ids"]
    segment["identity_confidence"] = id_result["identity_confidence"]
    segment["identity_needs_review"] = id_result["identity_needs_review"]
    segment["identity_conflict"] = id_result.get("identity_conflict",
                                                  segment.get("identity_conflict", False))
    segment["identity_method"] = id_result["identity_method"]

    id_note = id_result.get("identity_note", "")
    if id_note:
        old = segment.get("note", "")
        segment["note"] = f"{old}; {id_note}" if old else id_note

    if id_result["identity_needs_review"]:
        segment["needs_review"] = True

    return segment

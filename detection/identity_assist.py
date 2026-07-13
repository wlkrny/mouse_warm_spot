"""
耳标颜色辅助检测 (Phase 7+8+9)
可选的辅助工具, 不覆盖人工结果
提供耳标颜色候选 + 冲突处理 + A→B替换辅助 + 向后兼容
"""

import os
import cv2
import numpy as np
import logging
import math
import json
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from .models.classifier import hsv_rule_classify

logger = logging.getLogger(__name__)


class IdentityAssist:
    """耳标颜色辅助检测 — 不要求全自动识别, 仅提供候选"""

    # Ear tag pixel conditions
    EAR_V_MIN = 80

    # Blob filter
    EAR_AREA_MIN = 3
    EAR_AREA_MAX = 500

    # Proximity to dark mouse mask
    MOUSE_PROXIMITY_PX = 20

    # ROI Search = Core × 2.8
    SEARCH_SCALE = 2.8

    # AI 视觉请求预算：每个 segment 最多云视觉请求数（默认 3；0=禁用云视觉，仅 CNN/HSV）
    # 通过 MOUSE_COLOR_AI_MAX_REQUESTS_PER_SEGMENT 环境变量可配置
    MAX_VISION_REQUESTS_PER_SEGMENT: int = 3

    def __init__(self, debug: bool = False, color_model_path=None, use_cnn=None):
        self.debug = debug
        if debug:
            logger.setLevel(logging.DEBUG)
            if not logger.handlers:
                h = logging.StreamHandler()
                h.setFormatter(logging.Formatter('%(asctime)s [%(name)s] %(message)s', datefmt='%H:%M:%S'))
                logger.addHandler(h)

        # ---- 可选 ONNX 分类器（懒加载，默认自动发现模型）----
        self._color_model_path = color_model_path
        self._use_cnn_flag = use_cnn
        self._classifier = None  # 懒初始化

        # 读取 AI 视觉请求预算
        budget_str = os.environ.get("MOUSE_COLOR_AI_MAX_REQUESTS_PER_SEGMENT", "3").strip()
        try:
            self._max_vision_requests = int(budget_str)
        except ValueError:
            logger.warning(
                "Invalid MOUSE_COLOR_AI_MAX_REQUESTS_PER_SEGMENT=%r; using default 3", budget_str
            )
            self._max_vision_requests = 3
        if self._max_vision_requests < 0:
            self._max_vision_requests = 0

        # 每个 segment 的请求计数器（在 detect_ear_tags 入口重置）
        self._vision_request_count: int = 0

    def _get_classifier(self):
        """懒获取 EarTagClassifier 单例。"""
        if self._classifier is None:
            # 环境变量优先级最低，由构造参数覆盖
            if self._color_model_path is None:
                env_path = os.environ.get("MOUSE_COLOR_MODEL_PATH", "").strip()
                if env_path:
                    self._color_model_path = env_path
            from .models.classifier import EarTagClassifier
            self._classifier = EarTagClassifier(
                color_model_path=self._color_model_path,
                use_cnn=self._use_cnn_flag,
            )
        return self._classifier

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

        # 调用底层 detect_ear_tags（identity_method 由 detect_ear_tags 内部设置，
        # 可能是 ear_tag_color_cnn|... 或 ear_tag_color_rule|...，此处不再覆盖）
        result = self.detect_ear_tags(
            segment=segment,
            roi_core=roi_core,
            cap=cap,
            fps=fps,
            background_bgr=background_bgr,
        )

        self._notify(progress_callback, 100, "颜色识别完成")

        return result

    # ------------------------------------------------------------------
    # Color classification (rule-based)
    # ------------------------------------------------------------------
    @staticmethod
    def classify_color(h: float, s: float, v: float) -> str:
        """Classify pixel color from HSV values.
        Returns: 'red' | 'yellow' | 'blue' | 'green' | 'white' | 'unknown'
        """
        # red: H≤12 or H≥165, S>45 (tightened S from 50→45)
        if (h <= 12 or h >= 165) and s > 45:
            return "red"
        # yellow: 13≤H≤42, S>45 (NEW — OpenCV HSV yellow ~13-42)
        if 13 <= h <= 42 and s > 45:
            return "yellow"
        # blue: 90≤H≤135, S>50 (tightened: H 140→135, S 35→50)
        if 90 <= h <= 135 and s > 50:
            return "blue"
        # green: 48≤H≤80, S>50 (tightened: H 40→48, S 35→50)
        if 48 <= h <= 80 and s > 50:
            return "green"
        # white: S<35, V>170 (relaxed: S 30→35, V 180→170)
        if s < 35 and v > 170:
            return "white"
        return "unknown"

    # ------------------------------------------------------------------
    # Contour-level color classification (pixel voting)
    # ------------------------------------------------------------------
    @staticmethod
    def _classify_contour(px: np.ndarray) -> tuple[str, bool, float]:
        """
        逐像素颜色投票，返回多数颜色，加入主色优势判断。
        解决色相循环均值崩塌导致的红色→蓝色误判。

        委托给 detection.models.classifier.hsv_rule_classify，
        避免阈值逻辑重复。

        :param px: (N,3) HSV像素数组
        :return: (color, is_low_confidence, best_ratio)
        """
        return hsv_rule_classify(px)

    # ------------------------------------------------------------------
    # Context frame selection for VLM multi-frame input
    # ------------------------------------------------------------------
    @staticmethod
    def _select_context_frames(start_f: int, end_f: int) -> list[int]:
        """为 VLM 上下文输入选择最多 9 帧（头三/中三/尾三，帧间隔 5）。

        头三帧：start, start+5, start+10
        中间三帧：mid-5, mid, mid+5，其中 mid=(start+end)//2
        结尾三帧：end-10, end-5, end

        所有候选 clamp 到 [start_f, end_f]，去重后按时间顺序排列。
        足够长且不重叠的 segment 返回 9 帧；短 segment 因
        clamp/去重可能少于 9，最低可至 1 帧（单帧 segment）。
        """
        mid_f = (start_f + end_f) // 2
        candidates = [
            start_f, start_f + 5, start_f + 10,
            mid_f - 5, mid_f, mid_f + 5,
            end_f - 10, end_f - 5, end_f,
        ]
        seen = set()
        frames = []
        for f in candidates:
            f = max(start_f, min(end_f, f))
            if f not in seen:
                seen.add(f)
                frames.append(f)
        frames.sort()  # 保持时间顺序
        return frames

    @staticmethod
    def _crop_context_frame(frame: np.ndarray, roi_core: dict,
                            a_context: float, b_context: float) -> np.ndarray | None:
        """裁剪上下文大范围图像（Search ROI × 1.5 外圈）。

        以 ROI core 中心为中心，半轴为 a_context / b_context。
        边界 clamp，绝不产生空 crop。

        :returns: cropped BGR image, or None if frame is invalid
        """
        h, w = frame.shape[:2]
        cx, cy = roi_core["cx"], roi_core["cy"]
        x1 = max(0, int(cx - a_context))
        y1 = max(0, int(cy - b_context))
        x2 = min(w, int(cx + a_context))
        y2 = min(h, int(cy + b_context))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

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

        # ---- 每个 segment 重置 AI 视觉请求计数器 ----
        self._vision_request_count = 0
        budget = self._max_vision_requests
        vision_enabled = budget > 0
        if not vision_enabled:
            logger.info("AI 视觉已禁用 (MOUSE_COLOR_AI_MAX_REQUESTS_PER_SEGMENT=0)，仅使用 CNN/HSV 规则")

        # ---- ROI Search = Core × 2.8 ----
        roi_search = dict(roi_core)
        roi_search["a"] = roi_core["a"] * self.SEARCH_SCALE
        roi_search["b"] = roi_core["b"] * self.SEARCH_SCALE

        a_core = roi_core["a"]
        b_core = roi_core["b"]
        a_search = roi_search["a"]
        b_search = roi_search["b"]
        a_context = a_search * 1.5  # VLM context crop 半轴
        b_context = b_search * 1.5
        angle = roi_search.get("angle", 0.0)

        # ---- VLM 上下文多帧请求（每 segment 仅一次）----
        vlm_context_color: str | None = None
        vlm_context_colors: list[str] | None = None
        vlm_context_count: int | None = None
        vlm_context_confidence: float = 0.0
        vlm_thermometer_present = False
        _vision_used_any = False
        _vlm_context_raw_resp: dict | None = None

        if vision_enabled:
            clf_vlm = self._get_classifier()
            if clf_vlm.is_vision_available:
                # 1) 选择上下文帧
                context_frame_ids = self._select_context_frames(start_f, end_f)
                logger.info("VLM context frame selection: %d unique frames (start=%d mid=%d end=%d segment_len=%d)",
                            len(context_frame_ids), start_f, mid_f, end_f, end_f - start_f + 1)

                # 2) 读取并裁剪上下文帧
                context_crops = []
                valid_context_ids = []
                for fid in context_frame_ids:
                    cap.set(1, fid)
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    crop = self._crop_context_frame(frame, roi_core, a_context, b_context)
                    if crop is not None and crop.size > 0:
                        context_crops.append(crop)
                        valid_context_ids.append(fid)

                # 3) 调用 VLM（消耗 1 次预算）
                if context_crops:
                    # 在真实 provider 调用前预扣预算
                    self._vision_request_count += 1
                    logger.info(
                        "VLM context call %d/%d (%d valid crops, frame_ids=%s)",
                        self._vision_request_count, budget,
                        len(context_crops), valid_context_ids,
                    )
                    try:
                        vlm_result = clf_vlm.classify_segment_frames(context_crops)
                        vlm_count = vlm_result.get("mouse_count")
                        vlm_colors = vlm_result.get("colors")
                        vlm_conf = vlm_result.get("confidence", 0.0)
                        vlm_method = vlm_result.get("method", "ear_tag_color_vlm")
                        parse_status = vlm_result.get("parse_status", "ok")
                        vlm_thermometer_present = vlm_result.get("thermometer_present", False)
                        _vlm_context_raw_resp = vlm_result.get("raw_response")
                        # A known first color is required; mouse two may explicitly be unknown.
                        if (vlm_count in (1, 2) and isinstance(vlm_colors, list)
                                and len(vlm_colors) == vlm_count and vlm_colors[0] != "unknown"
                                and vlm_conf >= 0.3):
                            # An injected pre-segment API is legacy: retain its historic
                            # color-only fusion and local count. A provider's actual old
                            # JSON has parse_status='legacy' and is still a valid count=1.
                            vlm_context_count = None if parse_status == "legacy_provider" else vlm_count
                            vlm_context_colors = vlm_colors
                            vlm_context_color = vlm_colors[0]
                            vlm_context_confidence = vlm_conf
                            _vision_used_any = True
                            logger.info("VLM count override: local=%s ai=%s confidence=%.3f thermometer=%s",
                                        segment.get("estimated_mouse_count", 1), vlm_count, vlm_conf,
                                        vlm_thermometer_present)
                        else:
                            logger.info("VLM context invalid/unknown/low confidence; falling back to CNN/HSV")
                    except Exception as exc:
                        logger.warning(
                            "VLM context call FAILED: %s; budget consumed, falling back to CNN/HSV", exc
                        )
                        # Debug: 即使失败也尝试保存 crops
                        self._save_vlm_context_debug(
                            context_crops, valid_context_ids,
                            clf_vlm._vision_provider, a_context, b_context, roi_core,
                            error=str(exc),
                        )
                else:
                    logger.info("VLM context: no valid crops (frames=%r); skipping vision", context_frame_ids)
            else:
                logger.debug("VLM context: vision provider not available")
        # ---- 结束 VLM 上下文请求 ----

        # A valid thermometer indication must not let the device-contaminated
        # VLM count/colors override local evidence.  Keep `_vision_used_any` as
        # the accepted-VLM marker needed for the safety result below.
        if _vision_used_any and vlm_thermometer_present:
            vlm_context_color = None
            vlm_context_count = None
            vlm_context_colors = None

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

        # 跟踪是否 CNN 被使用（VLM 跟踪已在上方 VLM 上下文块处理）
        _cnn_used_any = False

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

            # Ear-tag pixel mask: color union (red|yellow|green|blue|white)
            ep_mask = self._build_tag_candidate_mask(hsv)
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

                # ---- 提取 BGR patch 用于可选 CNN 推理 ----
                # 取轮廓包围盒作为 patch（维持 2D 图像结构）
                ys, xs = np.where(c_mask > 0)
                patch_bgr = None
                if len(ys) > 0:
                    y1p, y2p = ys.min(), ys.max() + 1
                    x1p, x2p = xs.min(), xs.max() + 1
                    if y2p > y1p and x2p > x1p:
                        patch_bgr = crop[y1p:y2p, x1p:x2p].copy()

                # ---- 分类: CNN → HSV 规则回退 ----
                # VLM 上下文已在 segment 级别处理，此处禁用 per-contour AI 视觉
                clf = self._get_classifier()

                # 永远禁用 per-contour 视觉调用（VLM 上下文已消耗段级预算）
                allow_vision = False

                # CNN/HSV 路径（allow_vision=False 确保不会触发网络调用）
                if clf.is_available:
                    resolved_color, resolved_method = clf.classify(
                        patch_bgr, px, allow_vision=False
                    )

                    if resolved_method == "ear_tag_color_cnn":
                        _cnn_used_any = True
                        if resolved_color != "unknown":
                            color = resolved_color
                            low_conf = False
                            best_ratio = 0.5
                        else:
                            color, low_conf, best_ratio = self._classify_contour(px)
                    else:
                        color, low_conf, best_ratio = self._classify_contour(px)
                else:
                    color, low_conf, best_ratio = self._classify_contour(px)

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

                # Blue/green quality penalty: avg S <55 or ratio <0.45 → 50% reduction
                if color in ("blue", "green"):
                    avg_s = float(np.mean(px[:, 1]))
                    if avg_s < 55:
                        hit_conf *= 0.5
                    if best_ratio < 0.45:
                        hit_conf *= 0.5

                color_hits[color]["hit_count"] += 1
                color_hits[color]["frames"].append(frame_idx)
                color_hits[color]["confs"].append(hit_conf)
                color_hits[color]["areas"].append(area)

        # ---- Aggregate per-color confidence ----
        total_search = max(1, len(search_frames))
        for color in color_hits:
            h = color_hits[color]

            # 唯一帧覆盖率 (时间持久性)
            unique_frames = len(set(h["frames"]))
            frame_ratio = unique_frames / total_search

            # 每命中平均质量
            avg_hc = np.mean(h["confs"]) if h["confs"] else 0.0

            score = 0.0

            # (1) 基础分: 0.35
            score += 0.35

            # (2) 已知颜色加成: 0.15
            if color in ("red", "yellow", "blue", "green"):
                score += 0.15

            # (3) 面积质量: 0.08
            if h["areas"]:
                area_mean = np.mean(h["areas"])
                if 3 <= area_mean <= 300:
                    score += 0.08

            # (4) 时间持久性: 最多 +0.30, 按 frame_ratio 线性缩放
            score += 0.30 * frame_ratio

            # (5) 命中数量 (对数尺度): 最多 +0.22
            #     log10(3)≈0.48, log10(1302)≈3.11; ×0.07 → 0.034~0.218
            hit_log = math.log10(max(1, h["hit_count"]))
            score += min(0.22, hit_log * 0.07)

            # (6) 每命中质量: 最多 +0.12 (avg_hc max ~0.4 × 0.3)
            score += 0.30 * avg_hc

            # Cap 从 0.95 提升至 0.99
            h["confidence"] = round(min(0.99, score), 4)

        # 日志: 汇总每种颜色的命中
        for color in sorted(color_hits.keys()):
            h = color_hits[color]
            logger.info(f"颜色 '{color}': hits={h['hit_count']} frames={len(h['frames'])} "
                        f"avg_area={np.mean(h['areas']):.1f} conf={h['confidence']:.2f}")

        # ---- VLM 上下文结果融合 ----
        # 仅当 context VLM 返回有效、置信度达阈值、非 unknown 颜色时才作为 VLM 成功结果。
        # VLM 颜色直接 boost 到最高置信度，确保不会被 CNN/HSV 多色候选覆盖。
        # 多鼠场景：VLM 颜色仅用于主要颜色决策，第二颜色仍由 CNN/HSV 提供。
        if _vision_used_any and vlm_context_color:
            if vlm_context_color not in color_hits:
                # VLM 发现 CNN/HSV 未检测到的颜色 → 添加高置信度条目
                color_hits[vlm_context_color] = {
                    "hit_count": max(1, int(total_search * 0.3)),
                    "frames": [],
                    "confs": [],
                    "areas": [],
                    "confidence": 0.99,
                }
                logger.info(f"VLM fusion: added '{vlm_context_color}' (not found by CNN/HSV)")
            else:
                # VLM 颜色已被 CNN/HSV 检测到 → boost 其置信度
                color_hits[vlm_context_color]["confidence"] = max(
                    color_hits[vlm_context_color].get("confidence", 0), 0.99
                )
                logger.info(f"VLM fusion: boosted '{vlm_context_color}' confidence to 0.99")
        elif _vision_used_any and not vlm_context_color and not vlm_thermometer_present:
            # 预算已消耗但 VLM 未给出有效颜色（unknown/低置信度）→ VLM 标志复位
            _vision_used_any = False
            logger.info("VLM fusion: resetting VLM flag (no valid color from context)")

        # ---- Conflict resolution ----
        target_count = segment.get("estimated_mouse_count", 1)
        if target_count is None or target_count <= 0:
            target_count = 1
        target_count = min(target_count, 2)  # cap=2

        ranked_colors = sorted(
            color_hits.keys(),
            key=lambda c: (
                color_hits[c]["confidence"],          # primary: confidence
                color_hits[c]["hit_count"],           # tiebreaker 1: 命中总数
                len(set(color_hits[c]["frames"])),    # tiebreaker 2: unique frames
            ),
            reverse=True,
        )

        # A successful context VLM answer is authoritative.  Do not encode this
        # priority as a confidence boost: HSV/CNN may legitimately tie at 0.99
        # and their secondary sort keys would then be able to displace the VLM.
        if _vision_used_any and vlm_context_color:
            ignored_rule_candidates = [
                color for color in ranked_colors if color != vlm_context_color
            ]
            colors = [vlm_context_color] + ignored_rule_candidates
            logger.info(
                "VLM priority override: selected=%s, ignored_rule_candidates=%s",
                vlm_context_color,
                ignored_rule_candidates,
            )
        else:
            colors = ranked_colors
        found_count = len(colors)

        # ---- 确定 identity_method 前缀 ----
        if _vision_used_any:
            method_prefix = "ear_tag_color_vlm"
        elif _cnn_used_any:
            method_prefix = "ear_tag_color_cnn"
        else:
            method_prefix = "ear_tag_color_rule"
        logger.info(f"冲突处理: target={target_count} found={found_count} colors={colors} method={method_prefix}")

        auto_mouse_colors: list[str] = []
        auto_mouse_ids: list[str] = []
        identity_needs_review = False
        identity_conflict = False
        identity_confidence = 0.0
        identity_method = method_prefix
        note = ""

        if _vision_used_any and vlm_context_color and target_count == 1:
            # A valid segment-level VLM result owns the single-mouse identity;
            # retain HSV/CNN candidates only as diagnostic evidence.
            auto_mouse_colors = [vlm_context_color]
            auto_mouse_ids = [f"auto_{vlm_context_color}"]
            identity_needs_review = False
            identity_conflict = False
            identity_confidence = vlm_context_confidence
            identity_method += "|priority_override"

        elif found_count == 0:
            auto_mouse_colors = ["unknown"] * target_count
            auto_mouse_ids = [""] * target_count
            identity_needs_review = True
            identity_confidence = 0.3
            identity_method += "|no_detection"
            note = "未检测到任何耳标颜色"
            logger.warning(f"未检测到耳标颜色! 扫描了{total_search}帧, 像素条件: 颜色联合筛选(red|yellow|green|blue|white) area={self.EAR_AREA_MIN}-{self.EAR_AREA_MAX}, 需靠近mouse_mask≤{self.MOUSE_PROXIMITY_PX}px")

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
            dropped = colors[target_count:]
            auto_mouse_colors = top
            auto_mouse_ids = [f"auto_{c}" for c in top]
            identity_needs_review = True
            identity_conflict = True
            identity_confidence = float(np.mean([color_hits[c]["confidence"] for c in top]))
            identity_method += "|conflict"
            note = (f"检测到{found_count}种颜色({','.join(colors)}), "
                    f"选中={top}, 丢弃={dropped}")
            # 丢弃颜色显著命中数时增强警告
            sig_dropped = [
                dc for dc in dropped
                if color_hits[dc]["hit_count"] > max(total_search * 0.15, 30)
            ]
            if sig_dropped:
                note += (
                    f"; ⚠ 丢弃颜色命中数显著: "
                    + ", ".join(f"{c}({color_hits[c]['hit_count']}hits)" for c in sig_dropped)
                )
            logger.warning(
                f"颜色冲突解决: 检测到{found_count}种颜色, "
                f"选中 {top} (conf={[color_hits[c]['confidence'] for c in top]}), "
                f"丢弃 {dropped} (conf={[color_hits[c]['confidence'] for c in dropped]}), "
                f"显著丢弃={sig_dropped if sig_dropped else '无'}"
            )

        # A valid segment VLM owns both count and ordered colors.  Do this after
        # local aggregation so invalid AI outcomes retain the old local behavior.
        if _vision_used_any and vlm_context_count and vlm_context_colors:
            target_count = vlm_context_count
            auto_mouse_colors = vlm_context_colors[:]
            auto_mouse_ids = [f"auto_{c}" if c != "unknown" else "" for c in auto_mouse_colors]
            identity_confidence = vlm_context_confidence
            identity_method = "ear_tag_color_vlm|priority_override|count_override"
            identity_conflict = False
            identity_needs_review = False
            notes = []
            if "unknown" in auto_mouse_colors:
                identity_needs_review = True
                notes.append("VLM 第二只鼠耳标颜色未知")
            if len(auto_mouse_colors) == 2 and auto_mouse_colors[0] == auto_mouse_colors[1]:
                identity_needs_review = True
                notes.append("VLM 两只鼠颜色重复，需人工复核")
            if notes:
                note = "; ".join(notes)

        # A valid VLM result explicitly reporting a thermometer/probe is an
        # interference signal, not a color/count answer.  Keep the local result
        # intact but force manual review and zero identity confidence.
        if _vision_used_any and vlm_thermometer_present:
            identity_confidence = 0.0
            identity_needs_review = True
            identity_conflict = False
            identity_method += "|thermometer_detected"
            note = "检测到测温器/探头，颜色识别置信度已置零，请人工复核"

        # ---- Phase 8: A→B swap detection ----
        if not (_vision_used_any and vlm_thermometer_present) and target_count == 1 and found_count >= 2:
            top2 = colors[:2]
            if self._temporal_separation(color_hits, top2, fps):
                identity_conflict = True
                prefix = note + "; " if note else ""
                note = f"{prefix}Possible A-to-B swap detected"
                identity_method += "|a_to_b"

        return {
            "target_count": target_count,
            "ai_mouse_count": vlm_context_count,
            "ai_colors": vlm_context_colors,
            "ai_confidence": vlm_context_confidence if _vision_used_any else 0.0,
            "ai_parse_status": "ok" if _vision_used_any else "fallback",
            "thermometer_present": bool(vlm_thermometer_present) if _vision_used_any else False,
            "auto_mouse_colors": auto_mouse_colors,
            "auto_mouse_ids": auto_mouse_ids,
            "identity_confidence": round(identity_confidence, 3),
            "identity_needs_review": identity_needs_review,
            "identity_conflict": identity_conflict,
            "identity_method": identity_method,
            "identity_note": note,
        }

    def _save_vlm_context_debug(self, context_crops: list[np.ndarray],
                                 frame_indices: list[int],
                                 vision_provider, a_context: float, b_context: float,
                                 roi_core: dict, response_json: dict | None = None,
                                 error: str | None = None):
        """MOUSE_COLOR_AI_SAVE_DEBUG=1 时，保存 VLM 上下文 debug 产物。

        保存：
          - 所有 context crops（文件名含排序和原视频 frame index）
          - response JSON（成功或 API/解析失败）
          - manifest JSON（provider/model 不含 key、帧索引、crop 算法/范围、结果/错误）

        严格不保存 Authorization header/API key，不写入大型 base64 payload。
        """
        if os.environ.get("MOUSE_COLOR_AI_SAVE_DEBUG", "").strip() not in ("1", "true", "yes", "on"):
            return
        try:
            out_dir = Path.cwd() / "debug_vision_output"
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            # 保存 context crops
            for i, (crop, fid) in enumerate(zip(context_crops, frame_indices)):
                fname = f"context_{ts}_{i:02d}_frame_{fid:06d}.jpg"
                cv2.imwrite(str(out_dir / fname), crop)

            # 保存 response JSON
            response_fname = None
            if response_json is not None:
                response_fname = f"response_{ts}.json"
                with open(out_dir / response_fname, "w", encoding="utf-8") as f:
                    json.dump(response_json, f, indent=2, ensure_ascii=False, default=str)

            # 保存 manifest JSON
            provider_name = getattr(vision_provider, '_method_name', 'unknown') if vision_provider else 'none'
            model_name = getattr(vision_provider, '_model', 'unknown') if vision_provider else 'none'

            manifest = {
                "provider": provider_name,
                "model": model_name,
                "timestamp": datetime.now().isoformat(),
                "frame_indices": frame_indices,
                "num_context_crops": len(context_crops),
                "crop_algorithm": "Search_ROI * 1.5 centered at ROI core (search_roi = core * SEARCH_SCALE)",
                "crop_range": {
                    "cx": roi_core["cx"],
                    "cy": roi_core["cy"],
                    "a_context": a_context,
                    "b_context": b_context,
                    "a_search": a_context / 1.5,
                    "b_search": b_context / 1.5,
                    "a_core": a_context / 1.5 / IdentityAssist.SEARCH_SCALE,
                    "b_core": b_context / 1.5 / IdentityAssist.SEARCH_SCALE,
                    "SEARCH_SCALE": IdentityAssist.SEARCH_SCALE,
                },
                "result": None,
                "error": None,
            }
            # 记录 finish_reason 与 parse_status（不含 API key）
            finish_reason = None
            parse_status = "skipped"
            if error:
                manifest["error"] = {"type": type(error).__name__ if hasattr(type(error), '__name__') else "str",
                                       "message": str(error)}
                parse_status = "error"
            elif response_json is not None:
                # 提取 finish_reason
                try:
                    choices = response_json.get("choices", [])
                    finish_reason = choices[0].get("finish_reason", "unknown") if choices else "unknown"
                except Exception:
                    finish_reason = "unknown"
                # 解析响应中可能的结果
                try:
                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                        from .models.vision_provider import _parse_segment_json
                        parsed = _parse_segment_json(content)
                        if parsed:
                            manifest["result"] = {"mouse_count": parsed["mouse_count"],
                                                  "colors": parsed["colors"],
                                                  "color": parsed["colors"][0],
                                                  "confidence": parsed["confidence"],
                                                  "thermometer_present": parsed["thermometer_present"]}
                            parse_status = "ok" if parsed["parse_status"] == "legacy" else parsed["parse_status"]
                        else:
                            manifest["result"] = {"raw_content": content[:200]}
                            parse_status = "no_json"
                except Exception:
                    parse_status = "parse_error"
                if manifest["result"] is None:
                    manifest["result"] = {"note": "response saved but parse failed"}
                    parse_status = "parse_failed"
            manifest["finish_reason"] = finish_reason
            manifest["parse_status"] = parse_status
            # Repeat parsed fields at stable top-level names for safe debug tooling.
            parsed_result = manifest.get("result") or {}
            manifest["ai_mouse_count"] = parsed_result.get("mouse_count")
            manifest["ai_colors"] = parsed_result.get("colors")
            manifest["confidence"] = parsed_result.get("confidence")
            manifest["thermometer_present"] = parsed_result.get("thermometer_present")
            # 记录 VLM max_tokens 上限用于截断诊断
            try:
                if vision_provider and hasattr(vision_provider, '_max_tokens'):
                    mt = vision_provider._max_tokens
                    if isinstance(mt, int):
                        manifest["max_tokens"] = mt
                    else:
                        manifest["max_tokens"] = None
                else:
                    manifest["max_tokens"] = None
            except Exception:
                manifest["max_tokens"] = None

            manifest_fname = f"manifest_{ts}.json"
            with open(out_dir / manifest_fname, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

            logger.info(
                "VLM context debug saved: %d crops + %s + %s",
                len(context_crops),
                response_fname or "no_response",
                manifest_fname,
            )
        except Exception as exc:
            logger.debug("VLM context debug save failed: %s", exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ellipse(h: int, w: int, cx: float, cy: float, a: float, b: float, ang: float = 0.0) -> np.ndarray:
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(m, (int(cx), int(cy)), (int(a), int(b)), ang, 0, 360, 255, -1)
        return m

    @classmethod
    def _build_tag_candidate_mask(cls, hsv: np.ndarray) -> np.ndarray:
        """Color union pixel mask: any pixel matching at least one color's H+S+V condition.

        Colors (vectorized):
          red:    (H≤12 or H≥165) AND S>45 AND V>50
          yellow: 13≤H≤42 AND S>45 AND V>70
          green:  48≤H≤80 AND S>50 AND V>50
          blue:   90≤H≤135 AND S>50 AND V>50
          white:  S<35 AND V>170
        """
        v = hsv[:, :, 2].astype(np.float32)
        s = hsv[:, :, 1].astype(np.float32)
        h = hsv[:, :, 0].astype(np.float32)

        red_mask    = ((h <= 12) | (h >= 165)) & (s > 45) & (v > 50)
        yellow_mask = (h >= 13) & (h <= 42) & (s > 45) & (v > 70)
        green_mask  = (h >= 48) & (h <= 80) & (s > 50) & (v > 50)
        blue_mask   = (h >= 90) & (h <= 135) & (s > 50) & (v > 50)
        white_mask  = (s < 35) & (v > 170)

        union = red_mask | yellow_mask | green_mask | blue_mask | white_mask
        return union.astype(np.uint8) * 255

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
        if color in ("red", "yellow", "blue", "green"):
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

    # Keep canonical automatic count in sync before GUI/CSV consumers inspect it.
    target_count = id_result.get("target_count")
    if target_count in (1, 2):
        segment["estimated_mouse_count"] = target_count
        segment["mouse_count"] = target_count

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

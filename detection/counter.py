"""
小鼠数量估计引擎 (MouseCounter)
实现改进.md：拆分两个mask、碎片合并、保守综合计数、ROI A最小重叠、调试信息
"""

import cv2
import numpy as np


class MouseCounter:
    """在 ROI C 内估计小鼠数量的引擎"""

    # ---- cap=2 硬约束 ----
    MAX_WARM_SPOT_OCCUPANTS = 2

    # ---- occupancy_mask 参数 (宽松, 用于占据检测; 此处保留供参考) ----
    DARK_V_THRESHOLD = 80
    BG_DIFF_THRESHOLD = 30

    # ---- counting_mask 参数 (更保守, 用于数量估计) ----
    STRONG_DARK_V_THRESHOLD = 55
    DARK_CANDIDATE_V_THRESHOLD = 95
    COUNTING_DIFF_THRESHOLD = 25

    # ---- 形态学参数 ----
    COUNTING_KERNEL_CLOSE = (9, 9)

    # ---- 过滤参数 ----
    MIN_BODY_AREA = 50
    MAX_ASPECT_RATIO = 5.0
    MIN_ASPECT_RATIO = 0.2

    # ---- ROI A 接触最小重叠 ----
    ROIA_MIN_OVERLAP_PX = 20
    ROIA_MIN_OVERLAP_RATIO = 0.02
    # A component may reach the Core boundary through a tiny segmentation gap.
    # This tolerance is deliberately small and applies only to already filtered
    # foreground components; it never turns the Core mask itself into foreground.
    CORE_CONTACT_TOLERANCE_PX = 3

    # ---- 碎片合并距离阈值 ----
    MERGE_DISTANCE_RATIO = 0.25     # 边界距离 < 单鼠宽度 × 0.25
    MERGE_CENTER_DIST_RATIO = 0.6   # 中心距 < 单鼠宽度 × 0.6

    def __init__(self, single_mouse_area_ref: float = None):
        """
        :param single_mouse_area_ref: 单只小鼠典型前景面积(像素), None 则从自动采样估计(取中位数)
        """
        self._single_mouse_area_ref = single_mouse_area_ref
        self._area_samples: list[float] = []  # 自动采样 (仅单blob帧)
        self._area_p50: float | None = None
        self._area_p95: float | None = None
        self._count_area_refs: dict[int, float] = {}  # {1: 800, 2: 1600, ...} 用户标记的多鼠参考面积
        # 多帧面积样本 (Phase 1+2): {1: [...], 2: [...], 3: [...], 4: [...]}
        self._count_area_samples: dict[int, list[float]] = {1: [], 2: []}

    # ------------------------------------------------------------------
    # 主入口: 估计当前帧小鼠数量
    # ------------------------------------------------------------------
    def estimate_count(
        self,
        frame_bgr: np.ndarray,
        roi_core: dict,
        roi_count: dict | None,
        background_bgr: np.ndarray | None = None,
    ) -> dict:
        """
        估计当前帧中参与占据暖点的小鼠数量

        :param frame_bgr:     当前帧 BGR 图像
        :param roi_core:      ROI Core 参数字典 {cx, cy, a, b, angle} (内圈, 用于精查)
        :param roi_count:     ROI Count 参数字典 {cx, cy, a, b, angle} (外圈, 用于计数区域)
                              为 None 时自动使用 roi_core × 1.8
        :param background_bgr: 背景帧 BGR 图像
        :return: dict:
            estimated_mouse_count   int (0-2)
            count_by_blob           int
            count_by_area           int
            total_mouse_area        float
            largest_blob_area       float
            core_touching_blob_count int
            count_confidence        float (0-1)
            blob_count              int
            area_ratio              float  # total_mouse_area / single_mouse_area_ref
            debug_blobs             list   # 每个有效blob的调试信息
            decision_reason         str    # 最终判定原因
        """
        # ---- 1. 截取 ROI Count 区域 (计数用外圈) ----
        h, w = frame_bgr.shape[:2]
        if roi_count is None:
            # fallback: 使用 roi_core × 1.8
            roi_count = {
                "cx": roi_core["cx"], "cy": roi_core["cy"],
                "a": roi_core["a"] * 1.8, "b": roi_core["b"] * 1.8,
                "angle": roi_core.get("angle", 0.0),
            }

        cx_count, cy_count = roi_count["cx"], roi_count["cy"]
        c_a = roi_count["a"]
        c_b = roi_count["b"]
        x1 = max(0, int(cx_count - c_a))
        y1 = max(0, int(cy_count - c_b))
        x2 = min(w, int(cx_count + c_a))
        y2 = min(h, int(cy_count + c_b))

        if x2 <= x1 or y2 <= y1:
            return self._empty_result()

        roi_c_crop = frame_bgr[y1:y2, x1:x2]
        crop_h, crop_w = roi_c_crop.shape[:2]

        no_bg = background_bgr is None
        if not no_bg:
            bh, bw = background_bgr.shape[:2]
            if y2 > bh or x2 > bw or y1 >= bh or x1 >= bw:
                # 背景帧尺寸小于坐标范围，回退到 dark-only
                no_bg = True
            else:
                bg_crop = background_bgr[y1:y2, x1:x2]
                if bg_crop.size == 0:
                    no_bg = True

        # ---- 2. 创建椭圆 mask ----
        cx_crop = crop_w / 2.0
        cy_crop = crop_h / 2.0
        angle = roi_count.get("angle", 0.0)

        # ROI Count mask (外圈, 计数区域)
        roi_count_mask = self._create_ellipse_mask(crop_h, crop_w, cx_crop, cy_crop, c_a, c_b, angle)
        # ROI Core mask (内圈, 用于精查接触判定)
        a_core = roi_core["a"]
        b_core = roi_core["b"]
        roi_core_mask = self._create_ellipse_mask(crop_h, crop_w, cx_crop, cy_crop, a_core, b_core, angle)
        roi_core_pixel_count = np.count_nonzero(roi_core_mask)

        # ---- 3. 灰度与absdiff ----
        if not no_bg:
            roi_c_gray = cv2.cvtColor(roi_c_crop, cv2.COLOR_BGR2GRAY)
            bg_gray = cv2.cvtColor(bg_crop, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(roi_c_gray, bg_gray)

        # ---- 4. occupancy_mask: dark OR diff (宽松, 仅保留供调试; 此处不使用) ----
        # roi_c_hsv = cv2.cvtColor(roi_c_crop, cv2.COLOR_BGR2HSV)
        # v_channel = roi_c_hsv[:, :, 2]
        # _, dark_mask = cv2.threshold(v_channel, self.DARK_V_THRESHOLD - 1, 255, cv2.THRESH_BINARY_INV)
        # _, diff_mask_occ = cv2.threshold(diff, self.BG_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        # occupancy_mask = cv2.bitwise_or(dark_mask, diff_mask_occ)
        # occupancy_mask = cv2.bitwise_and(occupancy_mask, occupancy_mask, mask=roi_c_mask)

        # ---- 5. counting_mask: 更保守 ----
        roi_c_hsv = cv2.cvtColor(roi_c_crop, cv2.COLOR_BGR2HSV)
        v_channel = roi_c_hsv[:, :, 2]

        # strong_dark: V < 55
        _, strong_dark = cv2.threshold(
            v_channel, self.STRONG_DARK_V_THRESHOLD - 1, 255, cv2.THRESH_BINARY_INV)

        if no_bg:
            # 无背景 fallback: 仅使用 strong_dark (不用diff)
            counting_mask = strong_dark.copy()
        else:
            # dark_candidate: V < 95
            _, dark_candidate = cv2.threshold(
                v_channel, self.DARK_CANDIDATE_V_THRESHOLD - 1, 255, cv2.THRESH_BINARY_INV)
            # diff_gray: absdiff > 25
            _, diff_gray = cv2.threshold(diff, self.COUNTING_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

            counting_mask = cv2.bitwise_or(strong_dark, cv2.bitwise_and(dark_candidate, diff_gray))
        counting_mask = cv2.bitwise_and(counting_mask, counting_mask, mask=roi_count_mask)

        # ---- 6. 形态学: 更大闭运算合并碎片 ----
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, self.COUNTING_KERNEL_CLOSE)
        cleaned = cv2.morphologyEx(counting_mask, cv2.MORPH_OPEN, kernel_open)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)

        # ---- 7. 连通区域分析 ----
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cleaned, connectivity=8
        )

        # ---- 8. 过滤面积/长宽比 (合并前) ----
        pre_blobs = []
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            left = stats[label_id, cv2.CC_STAT_LEFT]
            top = stats[label_id, cv2.CC_STAT_TOP]
            blob_w = stats[label_id, cv2.CC_STAT_WIDTH]
            blob_h = stats[label_id, cv2.CC_STAT_HEIGHT]

            if area < self.MIN_BODY_AREA:
                continue

            if blob_w > 0 and blob_h > 0:
                aspect_ratio = float(blob_w) / float(blob_h)
                if aspect_ratio > self.MAX_ASPECT_RATIO or aspect_ratio < self.MIN_ASPECT_RATIO:
                    continue

            pre_blobs.append({
                "area": area,
                "cx": centroids[label_id, 0],
                "cy": centroids[label_id, 1],
                "width": blob_w,
                "height": blob_h,
                "left": left,
                "top": top,
                "label_id": label_id,
            })

        # ---- 9. Keep only foreground components connected to ROI Core. ----
        # This happens *before* nearby-fragment merging: an independent mouse in
        # ROI Count must not be merged into a Core occupant merely by proximity.
        core_connected_pre_blobs = []
        ignored_outer_blob_count = 0
        for b in pre_blobs:
            b["touching_core"] = self._blob_touches_roi_a(
                labels, b["label_id"], roi_core_mask, cleaned,
                b["left"], b["top"], b["width"], b["height"],
                roi_core_pixel_count, self.CORE_CONTACT_TOLERANCE_PX
            )
            if b["touching_core"]:
                core_connected_pre_blobs.append(b)
            else:
                ignored_outer_blob_count += 1

        # ---- 10. Merge only Core-connected fragments. ----
        valid_blobs = self._merge_nearby_blobs(core_connected_pre_blobs)

        # ---- 11. 统计 ----
        blob_count = len(valid_blobs)
        core_touching = valid_blobs
        core_touching_blob_count = len(core_touching)
        count_by_blob = core_touching_blob_count if core_touching_blob_count >= 1 else 0
        count_by_blob = min(MouseCounter.MAX_WARM_SPOT_OCCUPANTS, count_by_blob)

        # 面积: 只用接触ROI A的blob
        total_mouse_area = sum(b["area"] for b in core_touching)
        largest_blob_area = max((b["area"] for b in core_touching), default=0)

        # ---- 12. 面积比计算 ----
        area_ref = self._get_area_reference()
        area_ratio = None
        has_area_ref = area_ref is not None and area_ref > 0 and total_mouse_area > 0
        if has_area_ref:
            area_ratio = total_mouse_area / area_ref
            count_by_area = self._area_threshold_count(area_ratio)
        elif total_mouse_area > 0:
            count_by_area = count_by_blob  # fallback
        else:
            count_by_area = 0

        # ---- 13. 自动面积采样 (仅单blob帧) ----
        if len(valid_blobs) == 1 and largest_blob_area >= self.MIN_BODY_AREA:
            self._area_samples.append(largest_blob_area)

        # ---- 14. 保守综合计数 ----
        estimated_mouse_count, count_confidence = self._compute_final_count(
            count_by_blob, area_ratio, has_area_ref,
            total_mouse_area, self._get_count_area_ref_strategy(), self._get_area_reference()
        )

        # ---- 15. 调试信息 ----
        debug_blobs = []
        for b in valid_blobs:
            ar = float(b["width"]) / float(b["height"]) if b["height"] > 0 else 0.0
            debug_blobs.append({
                "area": int(b["area"]),
                "bbox": [int(b["left"]), int(b["top"]),
                         int(b["width"]), int(b["height"])],
                "aspect_ratio": round(ar, 2),
                "touching_roia": b["touching_core"],
            })

        # ---- 无背景 fallback: 降低置信度 ----
        if no_bg:
            count_confidence *= 0.5

        decision_reason = self._build_decision_reason(
            count_by_blob, count_by_area,
            area_ratio, has_area_ref,
            estimated_mouse_count
        )
        if no_bg:
            decision_reason += "|no_background_fallback"

        return {
            "estimated_mouse_count": estimated_mouse_count,
            "count_by_blob": count_by_blob,
            "count_by_area": count_by_area,
            "total_mouse_area": float(total_mouse_area),
            "largest_blob_area": float(largest_blob_area),
            "core_touching_blob_count": core_touching_blob_count,
            "core_connected_blob_count": core_touching_blob_count,
            "ignored_outer_blob_count": ignored_outer_blob_count,
            "core_connected_area": float(total_mouse_area),
            "count_confidence": count_confidence,
            "blob_count": blob_count,
            # 新增调试字段
            "area_ratio": round(area_ratio, 3) if area_ratio is not None else 0.0,
            "debug_blobs": debug_blobs,
            "decision_reason": decision_reason,
            "count_area_refs": dict(self._count_area_refs),
            # Phase 1+2: 无背景 fallback 标记
            "fallback_mode": no_bg,
            "background_used": not no_bg,
        }

    # ------------------------------------------------------------------
    # 保守综合计数 (核心)
    # ------------------------------------------------------------------
    @staticmethod
    def _area_threshold_count(area_ratio: float) -> int:
        """面积保守阈值映射 (cap=2: 0/1/2 三档)"""
        if area_ratio < 0.3:
            return 0
        elif area_ratio < 1.7:
            return 1
        else:
            return 2

    def _get_count_area_ref_strategy(self) -> dict[int, float]:
        """
        获取各数量的参考面积策略:
        - count=1: P80 from _count_area_samples[1]
        - count=2: median from _count_area_samples[2]
        """
        return dict(self._count_area_refs)

    @staticmethod
    def _compute_final_count(count_by_blob: int, area_ratio: float | None,
                             has_area_ref: bool,
                             total_mouse_area: float = 0,
                             count_area_refs: dict[int, float] | None = None,
                             single_mouse_area_ref: float | None = None) -> tuple[int, float]:
        """
        保守综合计数 (cap=2 硬约束)
        向前兼容: 旧数据中 count=3/4 降级为 2

        :return: (estimated_mouse_count, count_confidence)
        """
        if count_by_blob == 0:
            return 0, 0.9
        if not has_area_ref or area_ratio is None:
            return min(count_by_blob, MouseCounter.MAX_WARM_SPOT_OCCUPANTS), 0.6

        # ---- 多鼠参考面积匹配 ----
        if count_by_blob == 1 and count_area_refs and len(count_area_refs) >= 1:
            best_count, best_deviation = MouseCounter._find_best_area_match(
                total_mouse_area, area_ratio, count_area_refs, single_mouse_area_ref
            )
            # cap=2: 降级旧数据 3/4 → 2
            best_count = min(best_count, MouseCounter.MAX_WARM_SPOT_OCCUPANTS)
            if best_deviation < 0.35:
                conf = min(0.8, max(0.3, 1.0 - best_deviation))
                return best_count, conf

        area_count = MouseCounter._area_threshold_count(area_ratio)

        # 关键规则: 只有一个身体簇时，最多判2只，低置信
        if count_by_blob == 1:
            if area_ratio < 1.7:
                return 1, 0.85
            else:
                return 2, 0.35  # 低置信 — 单个大团判2只

        # 两个及以上簇: cap=2
        result = max(count_by_blob, area_count)
        result = min(result, MouseCounter.MAX_WARM_SPOT_OCCUPANTS)
        return result, 0.75 if result <= 2 else 0.5

    @staticmethod
    def _find_best_area_match(
        total_mouse_area: float,
        area_ratio: float,
        count_area_refs: dict[int, float],
        single_mouse_area_ref: float | None = None,
    ) -> tuple[int, float]:
        """
        遍历所有可用参考面积, 找偏差最小的参考数量
        :return: (best_count, best_deviation)
        """
        best_count = 1
        # count=1 的偏差: 基于 single_mouse_area_ref 或 area_ratio
        if 1 in count_area_refs and count_area_refs[1] > 0:
            best_deviation = abs(total_mouse_area - count_area_refs[1]) / count_area_refs[1]
        elif single_mouse_area_ref is not None and single_mouse_area_ref > 0:
            best_deviation = abs(area_ratio - 1.0)
        else:
            best_deviation = abs(area_ratio - 1.0)

        for ref_count, ref_area in count_area_refs.items():
            if ref_count == 1 or ref_area <= 0:
                continue
            deviation = abs(total_mouse_area - ref_area) / ref_area
            if deviation < best_deviation:
                best_deviation = deviation
                best_count = ref_count

        return best_count, best_deviation

    @staticmethod
    def _build_decision_reason(count_by_blob: int, count_by_area: int,
                                area_ratio: float | None, has_area_ref: bool,
                                final_count: int) -> str:
        """构建判定原因字符串 (cap=2)"""
        if final_count == 0:
            return "no_mouse"
        if count_by_blob == 1 and has_area_ref and area_ratio is not None:
            if area_ratio < 1.7:
                return "single_cluster_normal"
            else:
                return "single_cluster_large_area"
        if count_by_blob >= 2:
            return "multi_clusters_confirmed"
        return f"blob_{count_by_blob}_area_{count_by_area}"

    # ------------------------------------------------------------------
    # 碎片合并
    # ------------------------------------------------------------------
    def _merge_nearby_blobs(self, blobs: list[dict]) -> list[dict]:
        """
        合并距离近的相邻blob
        规则: 边界距离 < 单鼠宽度×0.25 或 中心距 < 单鼠宽度×0.6
        """
        if len(blobs) <= 1:
            return blobs

        # 估计单鼠宽度
        area_ref = self._get_area_reference()
        if area_ref is None or area_ref <= 0:
            single_width = 20.0  # 默认
        else:
            single_width = np.sqrt(area_ref)

        merge_dist = single_width * self.MERGE_DISTANCE_RATIO
        center_dist = single_width * self.MERGE_CENTER_DIST_RATIO

        n = len(blobs)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        for i in range(n):
            for j in range(i + 1, n):
                xi, yi = blobs[i]["cx"], blobs[i]["cy"]
                xj, yj = blobs[j]["cx"], blobs[j]["cy"]
                cd = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

                li, ti, wi, hi = (blobs[i]["left"], blobs[i]["top"],
                                  blobs[i]["width"], blobs[i]["height"])
                lj, tj, wj, hj = (blobs[j]["left"], blobs[j]["top"],
                                  blobs[j]["width"], blobs[j]["height"])

                dx = max(0.0, float(max(li, lj)) - float(min(li + wi, lj + wj)))
                dy = max(0.0, float(max(ti, tj)) - float(min(ti + hi, tj + hj)))
                bbox_dist = np.sqrt(dx * dx + dy * dy)

                if bbox_dist < merge_dist or cd < center_dist:
                    union(i, j)

        # 按组聚合
        groups: dict[int, list[dict]] = {}
        for i in range(n):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(blobs[i])

        merged = []
        for group in groups.values():
            if len(group) == 1:
                merged.append(group[0])
            else:
                total_a = sum(b["area"] for b in group)
                w_sum = float(total_a) if total_a > 0 else 1.0
                cx = sum(b["cx"] * b["area"] for b in group) / w_sum
                cy = sum(b["cy"] * b["area"] for b in group) / w_sum
                lefts = [b["left"] for b in group]
                tops = [b["top"] for b in group]
                rights = [b["left"] + b["width"] for b in group]
                bottoms = [b["top"] + b["height"] for b in group]
                new_left = min(lefts)
                new_top = min(tops)
                new_w = max(rights) - new_left
                new_h = max(bottoms) - new_top
                merged.append({
                    "area": total_a,
                    "touching_core": any(b["touching_core"] for b in group),
                    "cx": cx,
                    "cy": cy,
                    "width": new_w,
                    "height": new_h,
                    "left": new_left,
                    "top": new_top,
                })
        return merged

    # ------------------------------------------------------------------
    # ROI A 接触检查 (改进: 最小重叠面积)
    # ------------------------------------------------------------------
    @staticmethod
    def _blob_touches_roi_a(labels, label_id, roi_a_mask, mask_img,
                            left, top, blob_w, blob_h,
                            roi_a_pixel_count: int = 0,
                            tolerance_px: int = 0) -> bool:
        """Return whether one foreground component is connected to ROI Core.

        A substantial direct overlap uses the established overlap thresholds.
        Otherwise a component may reach a *small* dilated Core boundary.  The
        component is still selected by its own label, so neither the Core mask
        nor unrelated outer blobs can become foreground by this operation.
        """
        label_crop = labels[top:top + blob_h, left:left + blob_w]
        roi_a_crop = roi_a_mask[top:top + blob_h, left:left + blob_w]
        blob_pixels = label_crop == label_id
        overlap_px = int(np.count_nonzero(blob_pixels & (roi_a_crop > 0)))
        if overlap_px >= MouseCounter.ROIA_MIN_OVERLAP_PX:
            if roi_a_pixel_count <= 0 or overlap_px / roi_a_pixel_count >= MouseCounter.ROIA_MIN_OVERLAP_RATIO:
                return True
        if tolerance_px <= 0:
            return False
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tolerance_px * 2 + 1,) * 2)
        expanded_core = cv2.dilate(roi_a_mask, k)
        expanded_crop = expanded_core[top:top + blob_h, left:left + blob_w]
        return bool(np.any(blob_pixels & (expanded_crop > 0)))

    # ------------------------------------------------------------------
    # 多鼠参考面积 (标记 [2][3][4] 帧时存储)
    # ------------------------------------------------------------------
    def set_count_area_ref(self, count: int, area: float):
        """存储指定数量的参考总面积 (用于计数时比对)"""
        if area > 0:
            self._count_area_refs[count] = area

    def get_count_area_refs(self) -> dict[int, float]:
        """获取所有用户标记的多鼠参考面积"""
        return dict(self._count_area_refs)

    # ------------------------------------------------------------------
    # 单鼠面积校准
    # ------------------------------------------------------------------
    def calibrate_from_samples(self, area_samples: list[float]):
        """
        从多个面积样本校准, 取 P80

        :param area_samples: 单鼠前景面积样本列表
        """
        if len(area_samples) >= 3:
            self._single_mouse_area_ref = float(np.percentile(area_samples, 80))
            self._area_p50 = float(np.percentile(area_samples, 50))
            self._area_p95 = float(np.percentile(area_samples, 95))

    def calibrate_from_frame(self, frame_bgr, roi_core, roi_count,
                             background_bgr) -> float | None:
        """
        从指定帧校准单鼠面积 (取最大blob面积)

        :return: 估计的单鼠面积 (像素), 或 None
        """
        result = self.estimate_count(frame_bgr, roi_core, roi_count, background_bgr)
        largest_blob = result.get("largest_blob_area", 0)
        if largest_blob > self.MIN_BODY_AREA:
            self._single_mouse_area_ref = largest_blob
            return largest_blob
        return None

    def set_single_mouse_area(self, area: float):
        """手动设置单鼠面积参考"""
        self._single_mouse_area_ref = area

    def get_single_mouse_area(self) -> float | None:
        """获取当前单鼠面积参考"""
        return self._get_area_reference()

    # ------------------------------------------------------------------
    # 多帧面积样本管理 (Phase 1+2)
    # ------------------------------------------------------------------
    def add_count_area_sample(self, count: int, area: float):
        """追加指定数量的面积样本并立即更新参考值"""
        if count in self._count_area_samples and area > 0:
            self._count_area_samples[count].append(area)
            self._recompute_count_area_ref(count)

    def remove_last_count_area_sample(self, count: int) -> float | None:
        """删除最后一个面积样本"""
        if count in self._count_area_samples and self._count_area_samples[count]:
            removed = self._count_area_samples[count].pop()
            self._recompute_count_area_ref(count)
            return removed
        return None

    def clear_all_count_area_samples(self):
        """清空所有面积样本"""
        for k in list(self._count_area_samples.keys()):
            self._count_area_samples[k] = []
        self._count_area_refs = {}
        self._single_mouse_area_ref = None

    def get_count_area_samples(self, count: int) -> list[float]:
        """获取指定数量的面积样本列表"""
        return list(self._count_area_samples.get(count, []))

    def _recompute_count_area_ref(self, count: int):
        """从样本重算单一数量的参考值 (count=1用P80, count=2用median)"""
        samples = self._count_area_samples.get(count, [])
        if not samples:
            self._count_area_refs.pop(count, None)
            if count == 1:
                self._single_mouse_area_ref = None
            return
        if count == 1:
            p80 = float(np.percentile(samples, 80))
            self._count_area_refs[1] = p80
            self._single_mouse_area_ref = p80
        else:
            median = float(np.median(samples))
            self._count_area_refs[count] = median

    def clone(self) -> 'MouseCounter':
        """深拷贝 (用于检测线程快照)"""
        import copy as _copy
        new = MouseCounter.__new__(MouseCounter)
        new._single_mouse_area_ref = self._single_mouse_area_ref
        new._area_samples = list(self._area_samples)
        new._area_p50 = self._area_p50
        new._area_p95 = self._area_p95
        new._count_area_refs = dict(self._count_area_refs)
        new._count_area_samples = {k: list(v) for k, v in self._count_area_samples.items()}
        # 向前兼容: clone 时只保留 cap=2 范围内的键
        for k in list(new._count_area_samples.keys()):
            if k > MouseCounter.MAX_WARM_SPOT_OCCUPANTS:
                del new._count_area_samples[k]
        return new

    def get_area_stats(self) -> dict:
        """获取面积统计: P50/P80/P95/样本数"""
        if self._area_p50 is not None:
            return {
                "p50": self._area_p50,
                "p80": self._single_mouse_area_ref,
                "p95": self._area_p95,
            }
        ref = self._get_area_reference()
        if ref is not None:
            return {"p50": ref, "p80": ref, "p95": ref}
        return {}

    def _get_area_reference(self) -> float | None:
        """获取面积参考: 手动设置优先, 否则自动估计"""
        if self._single_mouse_area_ref is not None and self._single_mouse_area_ref > 0:
            return self._single_mouse_area_ref
        if len(self._area_samples) >= 5:
            return float(np.median(self._area_samples))
        return None

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------
    @staticmethod
    def _create_ellipse_mask(h, w, cx, cy, a, b, angle_deg=0):
        """在裁剪图像上创建椭圆 mask"""
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(
            mask,
            center=(int(cx), int(cy)),
            axes=(int(a), int(b)),
            angle=angle_deg,
            startAngle=0,
            endAngle=360,
            color=255,
            thickness=-1,
        )
        return mask

    @staticmethod
    def _empty_result() -> dict:
        """无有效前景时的空结果 (0只小鼠)"""
        return {
            "error": True,
            "computed": False,
            "estimated_mouse_count": 0,
            "count_by_blob": 0,
            "count_by_area": 0,
            "total_mouse_area": 0.0,
            "largest_blob_area": 0.0,
            "core_touching_blob_count": 0,
            "core_connected_blob_count": 0,
            "ignored_outer_blob_count": 0,
            "core_connected_area": 0.0,
            "count_confidence": 0.0,
            "blob_count": 0,
            "area_ratio": 0.0,
            "debug_blobs": [],
            "decision_reason": "no_data",
            "count_area_refs": {},
            "fallback_mode": False,
            "background_used": False,
        }

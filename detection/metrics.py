"""
遮挡检测指标计算模块
实现规范 9.2 节定义的各项单帧检测指标
"""

import cv2
import numpy as np


class DetectionMetrics:
    """单帧遮挡检测指标计算"""

    # 暖点颜色 HSV 范围 (规范定义)
    WARM_H_LOW, WARM_H_HIGH = 10, 30
    WARM_S_LOW, WARM_S_HIGH = 20, 150
    WARM_V_LOW, WARM_V_HIGH = 100, 230

    # 深色像素阈值: V < 80
    DARK_V_THRESHOLD = 80

    # 遮挡面积比例权重 (规范 9.2.5)
    W_WARM = 0.35   # warm_color_ratio 权重
    W_DARK = 0.25   # dark_pixel_ratio 权重
    W_DIFF = 0.25   # background_diff_score 权重
    W_BLOB = 0.15   # largest_dark_blob_area_ratio 权重

    # 判定阈值 (规范)
    OCCUPY_THRESHOLD = 0.20

    def __init__(self):
        """初始化检测指标计算器"""
        self._background_bgr = None          # 背景帧 BGR 图像
        self._background_hsv = None          # 背景帧 HSV 图像
        self._background_warm_mask = None     # 背景帧暖色像素 mask
        self._background_warm_count = 0       # 背景帧暖色像素数
        self._background_roi_a_bgr = None    # 背景帧 ROI A 区域 BGR
        self._current_roi_a_bgr = None       # 当前帧 ROI A 区域 BGR
        self._current_roi_a_hsv = None       # 当前帧 ROI A 区域 HSV

    # ------------------------------------------------------------------
    # 背景设置
    # ------------------------------------------------------------------
    def set_background(self, frame_bgr: np.ndarray):
        """
        设置空场背景帧（整帧 BGR 图像）
        :param frame_bgr: 背景帧 BGR 图像
        """
        self._background_bgr = frame_bgr.copy()
        self._background_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # ------------------------------------------------------------------
    # ROI mask 工具
    # ------------------------------------------------------------------
    @staticmethod
    def create_ellipse_mask(shape, cx, cy, a, b, angle_deg=0):
        """
        创建椭圆 mask
        :param shape: (height, width) 图像尺寸
        :param cx, cy: 椭圆中心
        :param a, b: 半长轴 / 半短轴
        :param angle_deg: 旋转角度（度）
        :return: uint8 mask (0/255)
        """
        mask = np.zeros(shape[:2], dtype=np.uint8)
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
    def create_ellipse_mask_on_crop(crop_h, crop_w, a, b, angle_deg=0):
        """
        在裁剪图像上创建椭圆 mask (椭圆中心 = 裁剪图像中心)
        :param crop_h, crop_w: 裁剪图像尺寸
        :param a, b: 半长轴 / 半短轴
        :param angle_deg: 旋转角度
        :return: uint8 mask (0/255) 与裁剪图像同尺寸
        """
        mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        cx_crop = crop_w / 2.0
        cy_crop = crop_h / 2.0
        cv2.ellipse(
            mask,
            center=(int(cx_crop), int(cy_crop)),
            axes=(int(a), int(b)),
            angle=angle_deg,
            startAngle=0,
            endAngle=360,
            color=255,
            thickness=-1,
        )
        return mask

    # ------------------------------------------------------------------
    # 暖色像素检测
    # ------------------------------------------------------------------
    @staticmethod
    def warm_pixel_mask(hsv_image: np.ndarray) -> np.ndarray:
        """
        检测暖色像素 (暖点颜色范围)
        H=[10,30], S=[20,150], V=[100,230]
        :param hsv_image: HSV 图像
        :return: 二值 mask (0/255)
        """
        lower = np.array(
            [DetectionMetrics.WARM_H_LOW, DetectionMetrics.WARM_S_LOW, DetectionMetrics.WARM_V_LOW],
            dtype=np.uint8,
        )
        upper = np.array(
            [DetectionMetrics.WARM_H_HIGH, DetectionMetrics.WARM_S_HIGH, DetectionMetrics.WARM_V_HIGH],
            dtype=np.uint8,
        )
        mask = cv2.inRange(hsv_image, lower, upper)
        return mask

    @staticmethod
    def dark_pixel_mask(hsv_image: np.ndarray) -> np.ndarray:
        """
        检测深色像素 (V < 80)
        :param hsv_image: HSV 图像
        :return: 二值 mask (0/255)
        """
        v_channel = hsv_image[:, :, 2]
        _, mask = cv2.threshold(v_channel, DetectionMetrics.DARK_V_THRESHOLD - 1, 255, cv2.THRESH_BINARY_INV)
        return mask

    # ------------------------------------------------------------------
    # 裁剪 ROI A 相关方法
    # ------------------------------------------------------------------
    def _get_roi_a_bounding_rect(self, cx, cy, a, b):
        """计算 ROI A 椭圆的外接矩形 (整数像素坐标)"""
        x1 = max(0, int(cx - a))
        y1 = max(0, int(cy - b))
        x2 = int(cx + a)
        y2 = int(cy + b)
        return x1, y1, x2, y2

    def _crop_roi_a(self, frame_bgr, cx, cy, a, b):
        """从整帧中裁剪 ROI A 外接矩形区域"""
        x1, y1, x2, y2 = self._get_roi_a_bounding_rect(cx, cy, a, b)
        h, w = frame_bgr.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None, x1, y1, x2, y2
        crop = frame_bgr[y1:y2, x1:x2].copy()
        return crop, x1, y1, x2, y2

    # ------------------------------------------------------------------
    # 主计算入口
    # ------------------------------------------------------------------
    def compute(
        self,
        frame_bgr: np.ndarray,
        roi_a_params: dict,
        background_frame_bgr: np.ndarray = None,
    ) -> dict:
        """
        计算当前帧在 ROI A 内的所有检测指标

        :param frame_bgr:     当前帧 BGR 图像
        :param roi_a_params:  ROI A 参数字典:
            cx: 椭圆中心 x
            cy: 椭圆中心 y
            a:  半长轴 (像素)
            b:  半短轴 (像素)
            angle: 旋转角度 (度)
        :param background_frame_bgr: 背景帧 BGR 图像 (可选; 若为 None 则使用 set_background 设置的背景)
        :return: dict:
            warm_color_ratio          暖点颜色保留比例
            dark_pixel_ratio          深色像素比例
            background_diff_score     背景差异分数
            largest_dark_blob_area_ratio  最大深色连通区域面积比
            occlusion_area_ratio      综合遮挡面积比例
            is_occupied               是否为疑似占据 (bool)
        """
        # 解析 ROI A 参数
        cx = roi_a_params["cx"]
        cy = roi_a_params["cy"]
        a = roi_a_params["a"]
        b = roi_a_params["b"]
        angle = roi_a_params.get("angle", 0.0)

        # 确保背景可用
        bg_bgr = background_frame_bgr if background_frame_bgr is not None else self._background_bgr
        if bg_bgr is None:
            # 无背景帧时，返回默认值
            return self._empty_result("背景帧未设置")

        # 裁剪 ROI A 区域
        crop, x1, y1, x2, y2 = self._crop_roi_a(frame_bgr, cx, cy, a, b)
        bg_crop, _, _, _, _ = self._crop_roi_a(bg_bgr, cx, cy, a, b)
        if crop is None or bg_crop is None or crop.shape != bg_crop.shape:
            return self._empty_result("ROI A 裁剪区域无效")

        crop_h, crop_w = crop.shape[:2]

        # 创建椭圆 mask (在裁剪坐标系中)
        ellipse_mask = self.create_ellipse_mask_on_crop(crop_h, crop_w, a, b, angle)
        roi_pixel_count = np.count_nonzero(ellipse_mask)
        if roi_pixel_count == 0:
            return self._empty_result("ROI A 内无有效像素")

        # 转换为 HSV
        crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        bg_crop_hsv = cv2.cvtColor(bg_crop, cv2.COLOR_BGR2HSV)

        # --- 9.2.1 暖点颜色保留比例 ---
        cur_warm_mask = self.warm_pixel_mask(crop_hsv)
        cur_warm_in_roi = cv2.bitwise_and(cur_warm_mask, cur_warm_mask, mask=ellipse_mask)
        cur_warm_count = np.count_nonzero(cur_warm_in_roi)

        bg_warm_mask = self.warm_pixel_mask(bg_crop_hsv)
        bg_warm_in_roi = cv2.bitwise_and(bg_warm_mask, bg_warm_mask, mask=ellipse_mask)
        bg_warm_count = np.count_nonzero(bg_warm_in_roi)

        if bg_warm_count > 0:
            warm_color_ratio = cur_warm_count / bg_warm_count
        else:
            warm_color_ratio = 0.0  # 背景无暖色像素，异常

        # --- 9.2.2 深色像素比例 ---
        cur_dark_mask = self.dark_pixel_mask(crop_hsv)
        cur_dark_in_roi = cv2.bitwise_and(cur_dark_mask, cur_dark_mask, mask=ellipse_mask)
        dark_pixel_count = np.count_nonzero(cur_dark_in_roi)
        dark_pixel_ratio = dark_pixel_count / roi_pixel_count

        # --- 9.2.3 背景差异分数 ---
        # 归一化 MSE: MSE / (MSE + 1)
        # 只在椭圆 mask 内计算
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        bg_crop_gray = cv2.cvtColor(bg_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = crop_gray - bg_crop_gray
        # 只在 mask 内计算 MSE
        diff_masked = diff[ellipse_mask > 0]
        mse = np.mean(diff_masked ** 2) / 255.0  # 归一化到 [0,1] 范围
        background_diff_score = mse / (mse + 1.0)

        # --- 9.2.4 最大深色连通区域面积比 ---
        dark_in_roi_only = cv2.bitwise_and(cur_dark_mask, cur_dark_mask, mask=ellipse_mask)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            dark_in_roi_only, connectivity=8
        )
        # stats[0] 是背景 (label=0), 忽略
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_area = np.max(areas)
        else:
            largest_area = 0
        largest_dark_blob_area_ratio = largest_area / roi_pixel_count

        # --- 9.2.5 综合遮挡面积比例 ---
        occlusion_area_ratio = (
            self.W_WARM * (1.0 - warm_color_ratio)
            + self.W_DARK * dark_pixel_ratio
            + self.W_DIFF * background_diff_score
            + self.W_BLOB * largest_dark_blob_area_ratio
        )
        # 钳制到 [0, 1]
        occlusion_area_ratio = max(0.0, min(1.0, occlusion_area_ratio))

        # --- 判定 ---
        is_occupied = occlusion_area_ratio >= self.OCCUPY_THRESHOLD

        return {
            "warm_color_ratio": warm_color_ratio,
            "dark_pixel_ratio": dark_pixel_ratio,
            "background_diff_score": background_diff_score,
            "largest_dark_blob_area_ratio": largest_dark_blob_area_ratio,
            "occlusion_area_ratio": occlusion_area_ratio,
            "is_occupied": is_occupied,
            "roi_pixel_count": roi_pixel_count,
            "cur_warm_count": cur_warm_count,
            "bg_warm_count": bg_warm_count,
            "dark_pixel_count": dark_pixel_count,
            "largest_dark_area": largest_area,
        }

    @staticmethod
    def _empty_result(reason: str = "") -> dict:
        """返回空结果 (未计算)"""
        return {
            "warm_color_ratio": 0.0,
            "dark_pixel_ratio": 0.0,
            "background_diff_score": 0.0,
            "largest_dark_blob_area_ratio": 0.0,
            "occlusion_area_ratio": 0.0,
            "is_occupied": False,
            "roi_pixel_count": 0,
            "cur_warm_count": 0,
            "bg_warm_count": 0,
            "dark_pixel_count": 0,
            "largest_dark_area": 0,
            "reason": reason,
        }

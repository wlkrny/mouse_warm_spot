"""
暖点 ROI 放大窗口
实现规范 6.3 节：独立可停靠窗口,实时显示当前帧 ROI A 区域的 2x 放大图像
"""

import cv2
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QSizePolicy,
)


class ZoomWidget(QWidget):
    """暖点 ROI 放大组件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # 放大图像显示标签
        self._image_label = QLabel("等待 ROI 设置...")
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setMinimumSize(180, 180)
        self._image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._image_label.setStyleSheet("background-color: #1e1e1e; border: 1px solid #555;")
        layout.addWidget(self._image_label)

        # 状态文字标签
        self._status_label = QLabel("状态: 等待数据")
        self._status_label.setAlignment(Qt.AlignCenter)
        self._status_label.setStyleSheet("color: #aaa; font-size: 12px;")
        layout.addWidget(self._status_label)

        # 缩放倍数
        self._zoom_factor = 2.0

    # ------------------------------------------------------------------
    # 更新放大图像
    # ------------------------------------------------------------------
    def update_roi_crop(
        self,
        crop_bgr: np.ndarray | None,
        roi_a: dict | None = None,
        is_occupied: bool = False,
        occlusion_ratio: float = 0.0,
        dark_pixel_ratio: float = 0.0,
    ):
        """
        更新放大窗口显示
        :param crop_bgr: ROI A 裁剪区域 BGR 图像
        :param roi_a: ROI A 参数 (用于画边界)
        :param is_occupied: 当前帧遮挡判定
        :param occlusion_ratio: 遮挡面积比例
        :param dark_pixel_ratio: 深色像素比例 (规范 6.3 第6项)
        """
        if crop_bgr is None:
            self._image_label.setText("无 ROI 数据")
            self._status_label.setText("状态: 等待 ROI 设置")
            self._status_label.setStyleSheet("color: #aaa; font-size: 12px;")
            return

        h, w = crop_bgr.shape[:2]
        new_w = int(w * self._zoom_factor)
        new_h = int(h * self._zoom_factor)

        if new_w <= 0 or new_h <= 0:
            self._image_label.setText("ROI 区域过小")
            return

        # 放大
        zoomed = cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(zoomed, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, new_w, new_h, rgb.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # 在 pixmap 上绘制 ROI 边界
        if roi_a is not None:
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            pen = QPen(QColor(0, 255, 0), 2, Qt.SolidLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)

            # ROI A 椭圆在裁剪图上的坐标
            a_scaled = roi_a["a"] * self._zoom_factor
            b_scaled = roi_a["b"] * self._zoom_factor
            cx_scaled = new_w / 2.0
            cy_scaled = new_h / 2.0
            painter.drawEllipse(
                int(cx_scaled - a_scaled),
                int(cy_scaled - b_scaled),
                int(a_scaled * 2),
                int(b_scaled * 2),
            )
            painter.end()

        # 缩放显示以适应标签
        scaled_pixmap = pixmap.scaled(
            self._image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled_pixmap)

        # 更新状态文字 (规范 6.3 节: 显示遮挡判定 + 深色像素比例)
        if is_occupied:
            status_text = f"[!] 疑似占据 | 遮挡比: {occlusion_ratio:.1%} | 深色: {dark_pixel_ratio:.1%}"
            self._status_label.setStyleSheet(
                "color: #ff4444; font-size: 13px; font-weight: bold;"
            )
        else:
            status_text = f"[v] 未占据 | 遮挡比: {occlusion_ratio:.1%} | 深色: {dark_pixel_ratio:.1%}"
            self._status_label.setStyleSheet(
                "color: #44ff44; font-size: 13px; font-weight: bold;"
            )
        self._status_label.setText(status_text)

    def clear(self):
        """清空放大窗口"""
        self._image_label.setText("等待 ROI 设置...")
        self._status_label.setText("状态: 等待数据")
        self._status_label.setStyleSheet("color: #aaa; font-size: 12px;")

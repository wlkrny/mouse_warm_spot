"""
视频显示与 ROI 叠加控件
实现规范 6.2 节 (视频主显示区) 和 6.4 节 (ROI 绘制)
"""

import cv2
import numpy as np
import json
import os

from PySide6.QtCore import Qt, QTimer, Signal, QRectF, QPointF
from PySide6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QFont,
    QMouseEvent, QWheelEvent, QKeyEvent, QPaintEvent, QResizeEvent,
)
from PySide6.QtWidgets import QWidget, QMessageBox


class VideoWidget(QWidget):
    """视频显示控件：播放 + ROI 叠加 + 鼠标交互绘制 ROI"""

    # ---- 信号 ----
    frame_changed = Signal(int, object)        # frame_idx, frame_bgr
    roi_changed = Signal(object)               # roi_data dict
    background_set = Signal(object)            # background_frame_bgr

    # ---- ROI 绘制模式 ----
    MODE_NONE = 0
    MODE_DRAWING = 1       # 正在拖拽绘制

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

        # ---- 视频状态 ----
        self._cap: cv2.VideoCapture | None = None
        self._video_path: str = ""
        self._total_frames: int = 0
        self._fps: float = 0.0
        self._frame_w: int = 0
        self._frame_h: int = 0
        self._duration_sec: float = 0.0
        self._current_frame_idx: int = 0
        self._current_frame_bgr: np.ndarray | None = None

        # ---- 播放控制 ----
        self._playing = False
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._on_play_tick)

        # ---- ROI 数据 ----
        # ROI A: {cx, cy, a, b, angle}  椭圆参数 (相对于视频原始坐标)
        self.roi_a: dict | None = None
        self.buffer_roi_scale: float = 1.8  # ROI B 外扩倍数
        self.roi_c_scale: float = 1.6       # ROI C 外扩倍数 (用于计数, 改进.md 6 节)
        self._show_roi_c: bool = True       # ROI C 显示开关

        # ---- ROI 绘制状态 ----
        self._roi_mode = self.MODE_NONE
        self._roi_start: QPointF | None = None   # 鼠标按下位置 (widget 坐标)
        self._roi_current: QPointF | None = None  # 当前鼠标位置 (widget 坐标)
        self._drawing_roi_rect: QRectF | None = None

        # ---- 校准帧 (0/1/2/3/4只小鼠参考帧) ----
        # _calibrated_frames[0] = 0只(空场背景), _calibrated_frames[1] = 1只(面积参考)
        self._calibrated_frames: dict[int, 'np.ndarray'] = {}
        self._calibrated_frame_idx: dict[int, int] = {}

        # ---- 显示缩放 ----
        self._scale: float = 1.0

        # ---- 交互状态 ----
        self._is_occupied: bool = False  # 当前帧是否疑似占据

        # ---- 循环播放 ----
        self._loop_start_frame: int = -1     # 循环开始帧 (-1 = 未激活)
        self._loop_end_frame: int = -1       # 循环结束帧 (-1 = 未激活)
        self._loop_active: bool = False

    # ==================================================================
    # 视频文件操作
    # ==================================================================
    def open_video(self, path: str) -> bool:
        """打开视频文件"""
        self._release_video()

        if not os.path.isfile(path):
            QMessageBox.warning(self, "错误", f"视频文件不存在:\n{path}")
            return False

        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            QMessageBox.warning(self, "错误", f"无法打开视频文件:\n{path}")
            self._cap = None
            return False

        self._video_path = path
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        if self._fps <= 0:
            self._fps = 30.0
        self._frame_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._duration_sec = self._total_frames / self._fps if self._fps > 0 else 0.0
        self._current_frame_idx = 0

        # 读取首帧
        self._seek_and_read(0)
        self.update()
        return True

    def _release_video(self):
        """释放视频资源"""
        self._playing = False
        self._play_timer.stop()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._current_frame_bgr = None

    def _seek_and_read(self, frame_idx: int):
        """跳转到指定帧并读取"""
        if self._cap is None:
            return
        frame_idx = max(0, min(self._total_frames - 1, frame_idx))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._cap.read()
        if ret:
            self._current_frame_idx = frame_idx
            self._current_frame_bgr = frame
            self.frame_changed.emit(frame_idx, frame)
        else:
            # 读取失败,尝试重新定位
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self._cap.read()
            if ret:
                self._current_frame_idx = frame_idx
                self._current_frame_bgr = frame
                self.frame_changed.emit(frame_idx, frame)
        self.update()

    # ==================================================================
    # 播放控制
    # ==================================================================
    def play(self):
        """播放"""
        if self._cap is None or self._playing:
            return
        if self._current_frame_idx >= self._total_frames - 1:
            self._seek_and_read(0)
        self._playing = True
        interval_ms = int(1000.0 / self._fps) if self._fps > 0 else 33
        self._play_timer.start(interval_ms)

    def pause(self):
        """暂停"""
        self._playing = False
        self._play_timer.stop()

    def toggle_play_pause(self):
        """切换播放/暂停"""
        if self._playing:
            self.pause()
        else:
            self.play()

    def _on_play_tick(self):
        """播放定时器回调"""
        if self._cap is None:
            return

        # 循环播放检测：到达 loop_end_frame 时自动跳回 loop_start_frame
        if self._loop_active and self._loop_end_frame > 0:
            if self._current_frame_idx >= self._loop_end_frame:
                self._seek_and_read(self._loop_start_frame)
                return

        ret, frame = self._cap.read()
        if ret:
            self._current_frame_idx += 1
            self._current_frame_bgr = frame
            self.frame_changed.emit(self._current_frame_idx, frame)
            self.update()
        else:
            # 到达视频末尾
            self.pause()

    def next_frame(self):
        """下一帧 (→)"""
        if self._cap is None:
            return
        self._seek_and_read(self._current_frame_idx + 1)

    def prev_frame(self):
        """上一帧 (←)"""
        if self._cap is None:
            return
        self._seek_and_read(self._current_frame_idx - 1)

    def jump_frames(self, delta: int):
        """跳转 delta 帧 (A/D 键: ±10 帧)"""
        if self._cap is None:
            return
        self._seek_and_read(self._current_frame_idx + delta)

    def jump_to_frame(self, frame_idx: int):
        """跳转到指定帧号"""
        if self._cap is None:
            return
        self._seek_and_read(frame_idx)

    def jump_to_time_sec(self, time_sec: float):
        """跳转到指定时间 (秒)"""
        if self._cap is None or self._fps <= 0:
            return
        frame_idx = int(time_sec * self._fps)
        self._seek_and_read(frame_idx)

    # ==================================================================
    # 校准帧 (0只=背景, 1只=面积参考, 2-4只=可选参考)
    # ==================================================================
    def mark_calibration_frame(self, mouse_count: int):
        """标记当前帧为指定小鼠数量的参考帧"""
        if self._current_frame_bgr is None:
            QMessageBox.warning(self, "提示", "当前无视频帧，请先打开视频")
            return
        self._calibrated_frames[mouse_count] = self._current_frame_bgr.copy()
        self._calibrated_frame_idx[mouse_count] = self._current_frame_idx
        if mouse_count == 0:
            self.background_set.emit(self._current_frame_bgr)  # 仅 count=0 触发背景更新
        self.update()

    def unmark_calibration_frame(self, mouse_count: int):
        """删除指定数量的校准帧"""
        self._calibrated_frames.pop(mouse_count, None)
        self._calibrated_frame_idx.pop(mouse_count, None)
        self.update()

    def has_background(self) -> bool:
        """是否已标记0只帧(作为背景)"""
        return 0 in self._calibrated_frames

    def get_background(self) -> 'np.ndarray | None':
        """获取0只帧作为背景"""
        return self._calibrated_frames.get(0)

    def get_calibrated_frame(self, mouse_count: int) -> 'np.ndarray | None':
        """获取指定数量的校准帧"""
        return self._calibrated_frames.get(mouse_count)

    def has_calibrated_count(self, mouse_count: int) -> bool:
        """是否已标记指定数量的校准帧"""
        return mouse_count in self._calibrated_frames

    # ==================================================================
    # ROI 操作
    # ==================================================================
    def has_roi(self) -> bool:
        return self.roi_a is not None

    def get_roi_data(self) -> dict | None:
        """获取 ROI A 数据; 同时附带 ROI B 和 ROI C"""
        if self.roi_a is None:
            return None
        return {
            "roi_a": dict(self.roi_a),
            "buffer_roi_scale": self.buffer_roi_scale,
            "roi_c_scale": self.roi_c_scale,
        }

    def clear_roi(self):
        """删除 ROI"""
        self.roi_a = None
        self.roi_changed.emit(None)
        self.update()

    def save_roi(self, path: str):
        """保存 ROI 坐标为 JSON"""
        roi_data = self.get_roi_data()
        if roi_data is None:
            QMessageBox.warning(self, "提示", "未设置 ROI，请先绘制暖点核心 ROI")
            return
        data = {
            "roi_a": roi_data["roi_a"],
            "buffer_roi_scale": self.buffer_roi_scale,
            "roi_c_scale": self.roi_c_scale,
            "video_path": self._video_path,
            "frame_width": self._frame_w,
            "frame_height": self._frame_h,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_roi(self, path: str) -> bool:
        """加载 ROI 坐标 JSON"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.roi_a = data["roi_a"]
            self.buffer_roi_scale = data.get("buffer_roi_scale", 1.8)
            self.roi_c_scale = data.get("roi_c_scale", 1.6)
            self.roi_changed.emit(self.roi_a)
            self.update()
            return True
        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载 ROI 文件失败:\n{e}")
            return False

    # ==================================================================
    # 坐标转换: widget 坐标 ↔ 视频原始坐标
    # ==================================================================
    def _widget_to_video(self, wx: float, wy: float) -> tuple[float, float]:
        """将 widget 坐标转换为视频原始坐标"""
        if self._current_frame_bgr is None:
            return wx, wy
        # 计算缩放后图像在 widget 中的偏移
        img_w = int(self._frame_w * self._scale)
        img_h = int(self._frame_h * self._scale)
        offset_x = (self.width() - img_w) / 2.0
        offset_y = (self.height() - img_h) / 2.0
        vx = (wx - offset_x) / self._scale
        vy = (wy - offset_y) / self._scale
        return vx, vy

    def _video_to_widget(self, vx: float, vy: float) -> tuple[float, float]:
        """将视频原始坐标转换为 widget 坐标"""
        img_w = int(self._frame_w * self._scale)
        img_h = int(self._frame_h * self._scale)
        offset_x = (self.width() - img_w) / 2.0
        offset_y = (self.height() - img_h) / 2.0
        wx = vx * self._scale + offset_x
        wy = vy * self._scale + offset_y
        return wx, wy

    # ==================================================================
    # 绘制
    # ==================================================================
    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 背景
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        if self._current_frame_bgr is None:
            painter.setPen(QColor(150, 150, 150))
            painter.setFont(QFont("Microsoft YaHei", 14))
            painter.drawText(self.rect(), Qt.AlignCenter, "拖放视频文件到此处 或 通过菜单打开")
            return

        # 转换 BGR → RGB QImage
        frame_rgb = cv2.cvtColor(self._current_frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 缩放
        scaled_w = int(w * self._scale)
        scaled_h = int(h * self._scale)
        offset_x = (self.width() - scaled_w) / 2.0
        offset_y = (self.height() - scaled_h) / 2.0

        # 绘制视频帧
        target_rect = QRectF(offset_x, offset_y, scaled_w, scaled_h)
        painter.drawImage(target_rect, qimg)

        # 绘制 ROI
        if self.roi_a is not None:
            self._draw_roi(painter, offset_x, offset_y)

        # 绘制检测状态叠加 (规范 6.2 第15项)
        if self.roi_a is not None:
            self._draw_detection_status(painter, offset_x, offset_y)

        # 绘制正在拖拽的 ROI
        if self._roi_mode == self.MODE_DRAWING and self._drawing_roi_rect is not None:
            pen = QPen(QColor(0, 255, 0), 2, Qt.SolidLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(self._drawing_roi_rect)

    def _draw_roi(self, painter: QPainter, offset_x: float, offset_y: float):
        """在视频帧上叠加 ROI A, ROI B 和 ROI C"""
        cx, cy, a, b = self.roi_a["cx"], self.roi_a["cy"], self.roi_a["a"], self.roi_a["b"]
        angle = self.roi_a.get("angle", 0.0)

        # 转换到 widget 坐标
        wcx = cx * self._scale + offset_x
        wcy = cy * self._scale + offset_y
        wa = a * self._scale
        wb = b * self._scale

        # ROI C (蓝色虚线, 计数用, 改进.md 6 节) - 可通过开关控制
        if self._show_roi_c:
            pen_c = QPen(QColor(80, 160, 255), 1.5, Qt.DashLine)
            painter.setPen(pen_c)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(wcx, wcy), int(wa * self.roi_c_scale), int(wb * self.roi_c_scale))

        # ROI B (黄色虚线)
        pen_b = QPen(QColor(255, 255, 0), 2, Qt.DashLine)
        painter.setPen(pen_b)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QPointF(wcx, wcy), int(wa * self.buffer_roi_scale), int(wb * self.buffer_roi_scale))

        # ROI A (绿色实线)
        pen_a = QPen(QColor(0, 255, 0), 2.5, Qt.SolidLine)
        painter.setPen(pen_a)
        painter.drawEllipse(QPointF(wcx, wcy), int(wa), int(wb))

        # 标签文字
        font = QFont("Microsoft YaHei", 9)
        painter.setFont(font)
        if self._show_roi_c:
            painter.setPen(QColor(80, 160, 255))
            painter.drawText(
                QPointF(wcx - wa * self.roi_c_scale, wcy - wb * self.roi_c_scale - 4),
                "ROI C",
            )
        painter.setPen(QColor(255, 255, 0))
        painter.drawText(
            QPointF(wcx - wa * self.buffer_roi_scale, wcy - wb * self.buffer_roi_scale - 4),
            "ROI B",
        )
        painter.setPen(QColor(0, 255, 0))
        painter.drawText(QPointF(wcx - wa, wcy - wb - 4), "ROI A")

    def _draw_detection_status(self, painter: QPainter, offset_x: float, offset_y: float):
        """在视频帧右上角叠加检测状态文字 (规范 6.2 第15项)"""
        scaled_w = int(self._frame_w * self._scale)
        # 半透明背景框
        text = "[!] 疑似占据" if self._is_occupied else "[v] 未占据"
        font = QFont("Microsoft YaHei", 12, QFont.Bold)
        painter.setFont(font)
        fm = painter.fontMetrics()
        text_w = fm.horizontalAdvance(text) + 20
        text_h = fm.height() + 12
        bg_x = offset_x + scaled_w - text_w - 10
        bg_y = offset_y + 10

        # 半透明背景
        if self._is_occupied:
            painter.fillRect(int(bg_x), int(bg_y), int(text_w), int(text_h), QColor(200, 40, 40, 180))
            painter.setPen(QColor(255, 255, 255))
        else:
            painter.fillRect(int(bg_x), int(bg_y), int(text_w), int(text_h), QColor(40, 160, 40, 180))
            painter.setPen(QColor(255, 255, 255))
        painter.drawText(int(bg_x + 10), int(bg_y + fm.ascent() + 6), text)

    # ==================================================================
    # 鼠标事件 - ROI 绘制
    # ==================================================================
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self._current_frame_bgr is not None:
            self._roi_mode = self.MODE_DRAWING
            self._roi_start = event.position()
            self._roi_current = event.position()
            self._drawing_roi_rect = QRectF(self._roi_start, self._roi_current)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._roi_mode == self.MODE_DRAWING:
            self._roi_current = event.position()
            self._drawing_roi_rect = QRectF(self._roi_start, self._roi_current).normalized()
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self._roi_mode == self.MODE_DRAWING:
            self._roi_mode = self.MODE_NONE
            rect = QRectF(self._roi_start, event.position()).normalized()
            if rect.width() > 5 and rect.height() > 5:
                # 将 widget 坐标转为视频坐标
                vx1, vy1 = self._widget_to_video(rect.left(), rect.top())
                vx2, vy2 = self._widget_to_video(rect.right(), rect.bottom())
                cx = (vx1 + vx2) / 2.0
                cy = (vy1 + vy2) / 2.0
                a = abs(vx2 - vx1) / 2.0   # 半长轴
                b = abs(vy2 - vy1) / 2.0   # 半短轴
                self.roi_a = {"cx": cx, "cy": cy, "a": a, "b": b, "angle": 0.0}
                self.roi_changed.emit(self.roi_a)
            self._drawing_roi_rect = None
            self._roi_start = None
            self._roi_current = None
            self.update()
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        """滚轮缩放"""
        delta = event.angleDelta().y() / 120.0
        self._scale = max(0.1, min(5.0, self._scale + delta * 0.1))
        self.update()

    # ==================================================================
    # 键盘事件 - 播放/帧控制
    # ==================================================================
    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key == Qt.Key_Space:
            self.toggle_play_pause()
        elif key == Qt.Key_Right:
            self.next_frame()
        elif key == Qt.Key_Left:
            self.prev_frame()
        elif key == Qt.Key_D:
            self.jump_frames(10)    # 前进 10 帧
        elif key == Qt.Key_A:
            self.jump_frames(-10)   # 后退 10 帧
        else:
            super().keyPressEvent(event)

    # ==================================================================
    # 辅助属性
    # ==================================================================
    @property
    def current_frame_idx(self) -> int:
        return self._current_frame_idx

    @property
    def current_frame_bgr(self) -> np.ndarray | None:
        return self._current_frame_bgr

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def duration_sec(self) -> float:
        return self._duration_sec

    @property
    def current_time_sec(self) -> float:
        if self._fps > 0:
            return self._current_frame_idx / self._fps
        return 0.0

    @property
    def video_path(self) -> str:
        return self._video_path

    @property
    def playing(self) -> bool:
        return self._playing

    def get_roi_a_crop(self) -> np.ndarray | None:
        """获取当前帧 ROI A 裁剪区域 (用于放大窗口)"""
        if self._current_frame_bgr is None or self.roi_a is None:
            return None
        cx, cy = self.roi_a["cx"], self.roi_a["cy"]
        a, b = self.roi_a["a"], self.roi_a["b"]
        x1 = max(0, int(cx - a))
        y1 = max(0, int(cy - b))
        x2 = min(self._frame_w, int(cx + a))
        y2 = min(self._frame_h, int(cy + b))
        if x2 <= x1 or y2 <= y1:
            return None
        return self._current_frame_bgr[y1:y2, x1:x2].copy()

    def get_roi_a_size(self) -> tuple[int, int] | None:
        """获取 ROI A 裁剪区域尺寸"""
        if self.roi_a is None:
            return None
        a, b = self.roi_a["a"], self.roi_a["b"]
        return int(a * 2), int(b * 2)

    # ==================================================================
    # 循环播放
    # ==================================================================
    def set_loop_range(self, start_frame: int, end_frame: int):
        """
        设置循环播放范围
        :param start_frame: 循环开始帧
        :param end_frame: 循环结束帧
        """
        self._loop_start_frame = max(0, start_frame)
        self._loop_end_frame = min(self._total_frames - 1, end_frame)
        self._loop_active = True

    def clear_loop(self):
        """清除循环播放范围"""
        self._loop_start_frame = -1
        self._loop_end_frame = -1
        self._loop_active = False

    @property
    def loop_active(self) -> bool:
        return self._loop_active

    @property
    def loop_start_frame(self) -> int:
        return self._loop_start_frame

    @property
    def loop_end_frame(self) -> int:
        return self._loop_end_frame

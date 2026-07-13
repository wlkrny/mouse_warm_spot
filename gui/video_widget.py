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
        # ROI Core: {cx, cy, a, b, angle}  暖点铜片精确范围 (内圈, 绿实线)
        self.roi_core: dict | None = None
        # ROI Count: {cx, cy, a, b, angle}  两只鼠范围 (外圈, 蓝虚线, 默认=Core×1.8)
        self.roi_count: dict | None = None
        self._edit_mode: str = "core"       # "core" | "count"  当前编辑哪个椭圆

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

        # ---- Phase 10: Debug 视图数据 ----
        self._debug_data: dict = {}

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
        return self.roi_core is not None

    def set_edit_mode(self, mode: str):
        """切换当前编辑的椭圆: "core" 或 "count" """
        if mode in ("core", "count"):
            self._edit_mode = mode
            self.update()

    def get_roi_data(self) -> dict | None:
        """获取 ROI 数据: roi_core + roi_count"""
        if self.roi_core is None:
            return None
        return {
            "roi_core": dict(self.roi_core),
            "roi_count": dict(self.roi_count) if self.roi_count else None,
        }

    def clear_roi(self):
        """删除 ROI"""
        self.roi_core = None
        self.roi_count = None
        self.roi_changed.emit(None)
        self.update()

    def save_roi(self, path: str):
        """保存 ROI 坐标为 JSON (同时保存 core + count)"""
        if self.roi_core is None:
            QMessageBox.warning(self, "提示", "未设置 ROI，请先绘制暖点核心 ROI")
            return
        data = {
            "roi_core": dict(self.roi_core),
            "roi_count": dict(self.roi_count) if self.roi_count else None,
            "video_path": self._video_path,
            "frame_width": self._frame_w,
            "frame_height": self._frame_h,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_roi(self, path: str) -> bool:
        """加载 ROI 坐标 JSON (向后兼容旧格式)"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 新格式: roi_core + roi_count
            if "roi_core" in data:
                self.roi_core = data["roi_core"]
                if data.get("roi_count"):
                    self.roi_count = data["roi_count"]
                else:
                    # roi_count 缺失 → 默认 = core × 1.8
                    self.roi_count = self._scale_ellipse(self.roi_core, 1.8)
            # 向后兼容旧格式: roi_a → roi_core, 自动生成 roi_count = roi_a × 2.0
            elif "roi_a" in data:
                self.roi_core = data["roi_a"]
                old_scale = data.get("roi_c_scale", 2.0)
                self.roi_count = self._scale_ellipse(self.roi_core, old_scale)
            else:
                raise ValueError("ROI文件中未找到 roi_core 或 roi_a 字段")
            self.roi_changed.emit(self.roi_core)
            self.update()
            return True
        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载 ROI 文件失败:\n{e}")
            return False

    @staticmethod
    def _scale_ellipse(ellipse: dict, scale: float) -> dict:
        """按倍率缩放椭圆 (保持中心和角度不变)"""
        return {
            "cx": ellipse["cx"],
            "cy": ellipse["cy"],
            "a": ellipse["a"] * scale,
            "b": ellipse["b"] * scale,
            "angle": ellipse.get("angle", 0.0),
        }

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
        if self.roi_core is not None:
            self._draw_roi(painter, offset_x, offset_y)

        # 绘制检测状态叠加 (规范 6.2 第15项)
        if self.roi_core is not None:
            self._draw_detection_status(painter, offset_x, offset_y)

        # Phase 10: Debug 视图叠加
        self._draw_debug_overlays(painter, offset_x, offset_y)

        # 绘制正在拖拽的 ROI
        if self._roi_mode == self.MODE_DRAWING and self._drawing_roi_rect is not None:
            pen = QPen(QColor(0, 255, 0), 2, Qt.SolidLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(self._drawing_roi_rect)

    def _draw_roi(self, painter: QPainter, offset_x: float, offset_y: float):
        """在视频帧上叠加 ROI Core (绿实线) 和 ROI Count (蓝虚线)"""
        font = QFont("Microsoft YaHei", 9)
        painter.setFont(font)

        # ---- ROI Count (外圈, 蓝虚线) ----
        if self.roi_count is not None:
            cx, cy = self.roi_count["cx"], self.roi_count["cy"]
            a, b = self.roi_count["a"], self.roi_count["b"]
            wcx = cx * self._scale + offset_x
            wcy = cy * self._scale + offset_y
            wa = a * self._scale
            wb = b * self._scale

            pen_count = QPen(QColor(80, 160, 255), 1.5, Qt.DashLine)
            painter.setPen(pen_count)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(wcx, wcy), int(wa), int(wb))

            painter.setPen(QColor(80, 160, 255))
            painter.drawText(QPointF(wcx - wa, wcy - wb - 4), "ROI Count")

        # ---- ROI Core (内圈, 绿实线, 编辑中时加粗) ----
        cx, cy, a, b = self.roi_core["cx"], self.roi_core["cy"], self.roi_core["a"], self.roi_core["b"]
        wcx = cx * self._scale + offset_x
        wcy = cy * self._scale + offset_y
        wa = a * self._scale
        wb = b * self._scale

        core_width = 3.5 if self._edit_mode == "core" else 2.5
        pen_core = QPen(QColor(0, 255, 0), core_width, Qt.SolidLine)
        painter.setPen(pen_core)
        painter.drawEllipse(QPointF(wcx, wcy), int(wa), int(wb))

        painter.setPen(QColor(0, 255, 0))
        edit_hint = " [编辑中]" if self._edit_mode == "core" else ""
        painter.drawText(QPointF(wcx - wa, wcy - wb - 4), f"ROI Core{edit_hint}")

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
                new_ellipse = {"cx": cx, "cy": cy, "a": a, "b": b, "angle": 0.0}

                if self._edit_mode == "core":
                    self.roi_core = new_ellipse
                    # 首次绘制 core 时, 自动生成 count = core × 1.8
                    if self.roi_count is None:
                        self.roi_count = self._scale_ellipse(new_ellipse, 1.8)
                    self.roi_changed.emit(self.roi_core)
                else:
                    # edit_mode == "count"
                    self.roi_count = new_ellipse
                    self.roi_changed.emit(self.roi_core)
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
        elif key == Qt.Key_L:
            self.jump_frames(10)    # 快进 10 帧 (Phase 6)
        elif key == Qt.Key_J:
            self.jump_frames(-10)   # 快退 10 帧 (Phase 6)
        else:
            event.ignore()  # 未处理的键传播到 MainWindow

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
        """获取当前帧 ROI Core 裁剪区域 (用于放大窗口)"""
        if self._current_frame_bgr is None or self.roi_core is None:
            return None
        cx, cy = self.roi_core["cx"], self.roi_core["cy"]
        a, b = self.roi_core["a"], self.roi_core["b"]
        x1 = max(0, int(cx - a))
        y1 = max(0, int(cy - b))
        x2 = min(self._frame_w, int(cx + a))
        y2 = min(self._frame_h, int(cy + b))
        if x2 <= x1 or y2 <= y1:
            return None
        return self._current_frame_bgr[y1:y2, x1:x2].copy()

    def get_roi_a_size(self) -> tuple[int, int] | None:
        """获取 ROI Core 裁剪区域尺寸"""
        if self.roi_core is None:
            return None
        a, b = self.roi_core["a"], self.roi_core["b"]
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

    # ==================================================================
    # Phase 10: Debug 视图
    # ==================================================================
    def set_debug_data(self, data: dict):
        """设置 debug overlay 数据, 空 dict 清除"""
        self._debug_data = data
        self.update()

    def _draw_debug_overlays(self, painter: QPainter, offset_x: float, offset_y: float):
        """Phase 10: 绘制 debug 视图 (mask, 连通区, occ_score, count)"""
        if not self._debug_data or self._current_frame_bgr is None:
            return

        masks = self._debug_data.get("masks")
        if masks is None:
            return

        crop_rect = masks.get("crop_rect")
        if crop_rect is None:
            return
        x1, y1, x2, y2 = crop_rect

        # 将 crop 坐标转换到 widget 坐标
        wx1 = x1 * self._scale + offset_x
        wy1 = y1 * self._scale + offset_y
        wx2 = x2 * self._scale + offset_x
        wy2 = y2 * self._scale + offset_y
        crop_w = int(wx2 - wx1)
        crop_h = int(wy2 - wy1)

        if crop_w <= 0 or crop_h <= 0:
            return

        # 绘制 dark mask 叠加 (红色半透明)
        dark_mask = masks.get("dark_mask")
        if dark_mask is not None and dark_mask.shape[0] > 0:
            dark_scaled = cv2.resize(dark_mask.astype(np.uint8), (crop_w, crop_h))
            dark_rgba = np.zeros((crop_h, crop_w, 4), dtype=np.uint8)
            dark_rgba[dark_scaled > 0] = [200, 50, 50, 80]
            qimg = QImage(dark_rgba.data, crop_w, crop_h, crop_w * 4,
                          QImage.Format_RGBA8888)
            painter.drawImage(QPointF(wx1, wy1), qimg)

        # 绘制 warm mask 叠加 (橙色)
        warm_mask = masks.get("warm_mask")
        if warm_mask is not None and warm_mask.shape[0] > 0:
            warm_scaled = cv2.resize(warm_mask.astype(np.uint8), (crop_w, crop_h))
            warm_rgba = np.zeros((crop_h, crop_w, 4), dtype=np.uint8)
            warm_rgba[warm_scaled > 0] = [255, 165, 0, 60]
            qimg = QImage(warm_rgba.data, crop_w, crop_h, crop_w * 4,
                          QImage.Format_RGBA8888)
            painter.drawImage(QPointF(wx1, wy1), qimg)

        # 绘制连通区边框 (blob bboxes)
        debug_blobs = self._debug_data.get("debug_blobs", [])
        if debug_blobs:
            pen = QPen(QColor(255, 255, 0), 2, Qt.SolidLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            for blob in debug_blobs:
                bbox = blob.get("bbox", [0, 0, 10, 10])
                bx = wx1 + bbox[0] * self._scale
                by = wy1 + bbox[1] * self._scale
                bw = bbox[2] * self._scale
                bh = bbox[3] * self._scale
                painter.drawRect(int(bx), int(by), int(bw), int(bh))

        # 绘制 ROI A crop 边框
        pen = QPen(QColor(0, 128, 255), 2, Qt.DotLine)
        painter.setPen(pen)
        painter.drawRect(int(wx1), int(wy1), crop_w, crop_h)

        # 显示 occ_score 和 count 文字
        occ_score = self._debug_data.get("occ_score", 0.0)
        count = self._debug_data.get("count", 0)
        count_conf = self._debug_data.get("count_confidence", 0.0)
        dark_ratio = self._debug_data.get("dark_pixel_ratio", 0.0)
        blob_count = self._debug_data.get("blob_count", 0)

        font = QFont("Monospace", 10, QFont.Bold)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255))

        text_x = int(wx1 + 5)
        text_y = int(wy2 + 15)
        line_h = 16
        painter.drawText(text_x, text_y, f"OCC:{occ_score:.3f} Dark:{dark_ratio:.3f}")
        painter.drawText(text_x, text_y + line_h,
                         f"Count:{count} Conf:{count_conf:.2f} Blobs:{blob_count}")

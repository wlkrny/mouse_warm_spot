"""
主窗口布局 — Phase 2: 完整检测+审核工作流
整合：视频控件 + 放大窗口 + 指标面板 + 事件列表 + 标注面板
实现规范 6 节完整布局
"""

import os
import time
import cv2
import numpy as np

from PySide6.QtCore import Qt, Signal, Slot, QThread, QTimer
from PySide6.QtGui import QAction, QKeySequence, QFont, QKeyEvent
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QLineEdit, QStatusBar,
    QFileDialog, QMessageBox, QDockWidget, QToolBar, QMenuBar, QMenu,
    QApplication, QSplitter, QSizePolicy, QProgressBar,
)

from .video_widget import VideoWidget
from .zoom_widget import ZoomWidget
from .metrics_panel import MetricsPanel
from .event_list_widget import EventListWidget
from .annotation_panel import AnnotationPanel
from .calibration_store import CalibrationStore, CalibrationSample
from detection.metrics import DetectionMetrics
from detection.engine import DetectionEngine
from detection.counter import MouseCounter


# =========================================================================
# 检测后台线程
# =========================================================================
class DetectionWorker(QThread):
    """后台检测线程, 不阻塞 UI"""

    progress_signal = Signal(int, str)       # percent, message
    finished_signal = Signal(list)            # events list
    error_signal = Signal(str)               # error message

    def __init__(self, video_path: str, roi_data: dict,
                 background_bgr: np.ndarray, metrics_calc: DetectionMetrics,
                 params: dict | None = None):
        super().__init__()
        self._video_path = video_path
        self._roi_data = roi_data
        self._background_bgr = background_bgr
        self._metrics_calc = metrics_calc
        self._params = params

    def run(self):
        try:
            cap = cv2.VideoCapture(self._video_path)
            if not cap.isOpened():
                self.error_signal.emit(f"无法打开视频: {self._video_path}")
                return

            engine = DetectionEngine(self._metrics_calc)
            events = engine.detect(
                cap=cap,
                roi_data=self._roi_data,
                background_bgr=self._background_bgr,
                params=self._params,
                progress_callback=lambda pct, msg: self.progress_signal.emit(pct, msg),
            )

            cap.release()
            self.finished_signal.emit(events)

        except Exception as e:
            self.error_signal.emit(str(e))


class DetectionWorkerWithCounting(QThread):
    """后台两层检测线程 (改进.md) — 不阻塞 UI"""

    progress_signal = Signal(int, str)       # percent, message
    finished_signal = Signal(list, list)      # episodes, segments
    error_signal = Signal(str)

    def __init__(self, video_path: str, roi_data: dict,
                 background_bgr: np.ndarray, metrics_calc: DetectionMetrics,
                 params: dict | None = None,
                 counter: MouseCounter | None = None):
        super().__init__()
        self._video_path = video_path
        self._roi_data = roi_data
        self._background_bgr = background_bgr
        self._metrics_calc = metrics_calc
        self._params = params
        self._counter = counter

    def run(self):
        try:
            cap = cv2.VideoCapture(self._video_path)
            if not cap.isOpened():
                self.error_signal.emit(f"无法打开视频: {self._video_path}")
                return

            engine = DetectionEngine(self._metrics_calc)
            episodes, segments = engine.detect_with_counting(
                cap=cap,
                roi_data=self._roi_data,
                background_bgr=self._background_bgr,
                params=self._params,
                progress_callback=lambda pct, msg: self.progress_signal.emit(pct, msg),
                counter=self._counter,
            )

            cap.release()
            self.finished_signal.emit(episodes, segments)

        except Exception as e:
            self.error_signal.emit(str(e))


# =========================================================================
# 主窗口
# =========================================================================
class MainWindow(QMainWindow):
    """主窗口 — 完整检测+审核工作流"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("小鼠暖点占据半自动标注系统")
        self.setMinimumSize(1400, 800)
        self.resize(1600, 950)

        # 检测指标计算器
        self._metrics_calc = DetectionMetrics()

        # 计数引擎 (改进.md)
        self._mouse_counter = MouseCounter()

        # 检测引擎相关
        self._detection_worker: DetectionWorker | DetectionWorkerWithCounting | None = None
        self._detection_start_time: float = 0.0
        self._detection_stats: dict = {}
        self._events: list[dict] = []       # 检测生成的事件/子片段
        self._episodes: list[dict] = []     # 占据大事件 (改进.md)
        self._count_segments: list[dict] = []  # 计数子片段 (改进.md)

        # 校准标记系统 (Phase 1+2: CalibrationStore 接管)
        self._calib_store = CalibrationStore()

        # 当前审核事件索引
        self._current_review_idx: int = -1

        # ---- 构建 UI ----
        self._setup_central()
        self._setup_docks()
        self._setup_menu_bar()
        self._setup_toolbar()
        self._setup_statusbar()

        # ---- 信号连接 ----
        self._video_widget.frame_changed.connect(self._on_frame_changed)
        self._video_widget.roi_changed.connect(self._on_roi_changed)
        self._video_widget.background_set.connect(self._on_background_set)

        self._event_list.segment_selected.connect(self._on_event_selected)
        self._event_list.segment_split_requested.connect(self._on_segment_split_requested)
        self._annotation_panel.annotation_changed.connect(self._on_annotation_changed)
        self._annotation_panel.review_action.connect(self._on_review_action)
        self._annotation_panel.count_confirmed.connect(self._on_count_confirmed)

        # ---- 初始状态 ----
        self._update_controls_state()

    # ==================================================================
    # 菜单栏
    # ==================================================================
    def _setup_menu_bar(self):
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu("文件(&F)")

        open_video_action = QAction("打开视频(&O)...", self)
        open_video_action.setShortcut(QKeySequence("Ctrl+O"))
        open_video_action.triggered.connect(self._on_open_video)
        file_menu.addAction(open_video_action)

        file_menu.addSeparator()

        save_roi_action = QAction("保存 ROI(&S)...", self)
        save_roi_action.triggered.connect(self._on_save_roi)
        file_menu.addAction(save_roi_action)

        load_roi_action = QAction("加载 ROI(&L)...", self)
        load_roi_action.triggered.connect(self._on_load_roi)
        file_menu.addAction(load_roi_action)

        file_menu.addSeparator()

        save_bg_action = QAction("保存背景帧...", self)
        save_bg_action.triggered.connect(self._on_save_background)
        file_menu.addAction(save_bg_action)

        load_bg_action = QAction("加载背景帧...", self)
        load_bg_action.triggered.connect(self._on_load_background)
        file_menu.addAction(load_bg_action)

        file_menu.addSeparator()

        export_md_action = QAction("导出Markdown统计表...", self)
        export_md_action.triggered.connect(self._on_export_markdown)
        file_menu.addAction(export_md_action)

        file_menu.addSeparator()

        exit_action = QAction("退出(&X)", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 视图菜单
        view_menu = menubar.addMenu("视图(&V)")
        if hasattr(self, '_zoom_dock'):
            view_menu.addAction(self._zoom_dock.toggleViewAction())
        if hasattr(self, '_metrics_dock'):
            view_menu.addAction(self._metrics_dock.toggleViewAction())
        if hasattr(self, '_annotation_dock'):
            view_menu.addAction(self._annotation_dock.toggleViewAction())

    # ==================================================================
    # 工具栏
    # ==================================================================
    def _setup_toolbar(self):
        toolbar = QToolBar("主工具栏")
        toolbar.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        # 打开视频
        self._btn_open = QPushButton("打开视频")
        self._btn_open.clicked.connect(self._on_open_video)
        self._btn_open.setMinimumHeight(30)
        toolbar.addWidget(self._btn_open)

        toolbar.addSeparator()

        # 播放控制
        self._btn_play = QPushButton("> 播放")
        self._btn_play.clicked.connect(self._on_toggle_play)
        self._btn_play.setMinimumHeight(30)
        self._btn_play.setMinimumWidth(70)
        toolbar.addWidget(self._btn_play)

        self._btn_prev = QPushButton("<<")
        self._btn_prev.setToolTip("后退 10 帧 (A)")
        self._btn_prev.clicked.connect(lambda: self._video_widget.jump_frames(-10))
        self._btn_prev.setMinimumHeight(30)
        toolbar.addWidget(self._btn_prev)

        self._btn_prev1 = QPushButton("<")
        self._btn_prev1.setToolTip("上一帧 (←)")
        self._btn_prev1.clicked.connect(self._video_widget.prev_frame)
        self._btn_prev1.setMinimumHeight(30)
        toolbar.addWidget(self._btn_prev1)

        self._btn_next1 = QPushButton(">")
        self._btn_next1.setToolTip("下一帧 (→)")
        self._btn_next1.clicked.connect(self._video_widget.next_frame)
        self._btn_next1.setMinimumHeight(30)
        toolbar.addWidget(self._btn_next1)

        self._btn_next = QPushButton(">>")
        self._btn_next.setToolTip("前进 10 帧 (D)")
        self._btn_next.clicked.connect(lambda: self._video_widget.jump_frames(10))
        self._btn_next.setMinimumHeight(30)
        toolbar.addWidget(self._btn_next)

        toolbar.addSeparator()

        # 校准帧标记按钮 [0] [1] [2] [3] [4] (左键追加, 右键菜单撤回/清空)
        self._calib_btns: dict[int, QPushButton] = {}
        for count in range(5):
            btn = QPushButton(f"[{count}] 标记背景" if count == 0 else f"[{count}] 标记{count}只")
            btn.setToolTip(f"左键: 将当前帧追加为 {count} 只小鼠样本 | 右键: 撤回/清空")
            btn.setMinimumHeight(30)
            btn.clicked.connect(lambda checked, c=count: self._on_mark_calibration(c))
            btn.setContextMenuPolicy(Qt.CustomContextMenu)
            btn.customContextMenuRequested.connect(lambda pos, c=count: self._show_calib_context_menu(c, pos))
            btn.setStyleSheet(self._calib_btn_style(count, False))
            self._calib_btns[count] = btn
            toolbar.addWidget(btn)

        # 刷新计数
        self._btn_refresh = QPushButton("刷新计数")
        self._btn_refresh.setToolTip("强制重新计算小鼠数量 (R)")
        self._btn_refresh.clicked.connect(self._refresh_counter)
        self._btn_refresh.setMinimumHeight(30)
        toolbar.addWidget(self._btn_refresh)

        # ROI 操作
        self._btn_clear_roi = QPushButton("X 清除 ROI")
        self._btn_clear_roi.clicked.connect(self._video_widget.clear_roi)
        self._btn_clear_roi.setMinimumHeight(30)
        toolbar.addWidget(self._btn_clear_roi)

        # ROI C 开关 (改进.md 6 节)
        self._btn_toggle_roi_c = QPushButton("[C] ROI C 开")
        self._btn_toggle_roi_c.setToolTip("切换 ROI C (计数区域) 显示")
        self._btn_toggle_roi_c.setCheckable(True)
        self._btn_toggle_roi_c.setChecked(True)
        self._btn_toggle_roi_c.clicked.connect(self._on_toggle_roi_c)
        self._btn_toggle_roi_c.setMinimumHeight(30)
        toolbar.addWidget(self._btn_toggle_roi_c)

        # 自动检测
        toolbar.addSeparator()
        self._btn_detect = QPushButton("自动检测全视频")
        self._btn_detect.setToolTip("运行全视频自动检测引擎")
        self._btn_detect.clicked.connect(self._on_detect_full_video)
        self._btn_detect.setMinimumHeight(30)
        self._btn_detect.setStyleSheet(
            "background-color: #336699; color: #fff; font-weight: bold; padding: 4px 14px;"
        )
        toolbar.addWidget(self._btn_detect)



    # ==================================================================
    # 中央区域: 左侧视频+事件列表 | 右侧停靠窗口
    # ==================================================================
    def _setup_central(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 水平分割: 左侧 vs 右侧 (右侧用 dock 处理)
        # 左侧 = 视频 (上) + 事件列表 (下)
        self._video_widget = VideoWidget()

        self._event_list = EventListWidget()

        # 垂直分割器
        self._left_splitter = QSplitter(Qt.Vertical)
        self._left_splitter.addWidget(self._video_widget)
        self._left_splitter.addWidget(self._event_list)
        self._left_splitter.setStretchFactor(0, 3)
        self._left_splitter.setStretchFactor(1, 1)
        # 默认分割比例
        self._left_splitter.setSizes([600, 200])

        layout.addWidget(self._left_splitter, 1)

        # 底部: 进度条 + 帧信息
        bottom_row = QHBoxLayout()
        bottom_row.setContentsMargins(8, 4, 8, 4)
        bottom_row.setSpacing(8)

        # 时间滑块 (仅保留, 虚拟)
        self._time_slider = QSlider(Qt.Horizontal)
        self._time_slider.setRange(0, 0)
        self._time_slider.setTracking(True)
        self._time_slider.valueChanged.connect(self._on_slider_changed)
        self._time_slider.sliderReleased.connect(self._on_slider_released)
        bottom_row.addWidget(self._time_slider, 1)

        # 帧/时间信息
        self._lbl_frame_info = QLabel("帧: -- / -- | 时间: --:--:--.-- / --:--:--.-- | FPS: --")
        self._lbl_frame_info.setStyleSheet(
            "color: #ddd; font-family: monospace; font-size: 12px;"
        )
        bottom_row.addWidget(self._lbl_frame_info)

        # 检测进度条 (初始隐藏)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setMaximumWidth(200)
        self._progress_bar.setVisible(False)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #333; border: 1px solid #555;
                border-radius: 3px; text-align: center; color: #ddd;
            }
            QProgressBar::chunk {
                background-color: #336699; border-radius: 2px;
            }
        """)
        bottom_row.addWidget(self._progress_bar)

        layout.addLayout(bottom_row)

    # ==================================================================
    # 右侧停靠窗口: 放大窗口 + 指标面板 + 标注面板
    # ==================================================================
    def _setup_docks(self):
        # 暖点放大窗口
        self._zoom_widget = ZoomWidget()
        self._zoom_dock = QDockWidget("暖点放大", self)
        self._zoom_dock.setWidget(self._zoom_widget)
        self._zoom_dock.setMinimumWidth(250)
        self.addDockWidget(Qt.RightDockWidgetArea, self._zoom_dock)

        # 检测指标面板
        self._metrics_panel = MetricsPanel()
        self._metrics_dock = QDockWidget("检测指标", self)
        self._metrics_dock.setWidget(self._metrics_panel)
        self._metrics_dock.setMinimumWidth(280)
        self.addDockWidget(Qt.RightDockWidgetArea, self._metrics_dock)

        # 标注面板
        self._annotation_panel = AnnotationPanel()
        self._annotation_dock = QDockWidget("标注面板", self)
        self._annotation_dock.setWidget(self._annotation_panel)
        self._annotation_dock.setMinimumWidth(260)
        self.addDockWidget(Qt.RightDockWidgetArea, self._annotation_dock)

        # 垂直排列右侧三个 dock: 放大(上) → 指标(中) → 标注(下)
        self.splitDockWidget(self._zoom_dock, self._metrics_dock, Qt.Vertical)
        self.splitDockWidget(self._metrics_dock, self._annotation_dock, Qt.Vertical)

    # ==================================================================
    # 状态栏
    # ==================================================================
    def _setup_statusbar(self):
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._statusbar.showMessage("就绪 - 请打开视频文件, 绘制 ROI, 设置背景帧后开始检测")

    # ==================================================================
    # 信号处理
    # ==================================================================
    @Slot(int, object)
    def _on_frame_changed(self, frame_idx: int, frame_bgr: np.ndarray):
        """视频帧变化时更新所有联动控件"""
        # 更新滑块
        total = self._video_widget.total_frames
        self._time_slider.blockSignals(True)
        self._time_slider.setRange(0, max(0, total - 1))
        self._time_slider.setValue(frame_idx)
        self._time_slider.blockSignals(False)

        # 更新帧信息
        self._update_frame_info_label()

        # 更新播放按钮
        if self._video_widget.playing:
            self._btn_play.setText("|| 暂停")
        else:
            self._btn_play.setText("> 播放")

        # 更新放大窗口
        roi_data = self._video_widget.get_roi_data()
        if roi_data is not None:
            crop = self._video_widget.get_roi_a_crop()
            roi_a = roi_data["roi_a"]
            occ_ratio = getattr(self, '_last_occ_ratio', 0.0)
            is_occupied = getattr(self, '_last_is_occupied', False)
            dark_ratio = getattr(self, '_last_dark_ratio', 0.0)
            self._zoom_widget.update_roi_crop(crop, roi_a, is_occupied, occ_ratio, dark_ratio)

        # 更新检测指标
        self._recompute_metrics(frame_bgr)

        # 循环播放到达边界时更新事件列表高亮
        if self._video_widget.loop_active and self._events:
            self._check_loop_boundary(frame_idx)

    @Slot(object)
    def _on_roi_changed(self, roi_data):
        """ROI 变化"""
        if roi_data is None:
            self._zoom_widget.clear()
            self._metrics_panel.update_metrics(None)
        elif self._video_widget.current_frame_bgr is not None:
            self._recompute_metrics(self._video_widget.current_frame_bgr)
        self._update_controls_state()

    @Slot(object)
    def _on_background_set(self, bg_bgr):
        """背景帧设置 (count=0 时由 mark_calibration_frame 触发)"""
        self._metrics_calc.set_background(bg_bgr)
        idx = self._video_widget._calibrated_frame_idx.get(0, -1)
        self._statusbar.showMessage(
            f"暖点空场背景已标记 (帧 #{idx})"
        )
        if self._video_widget.current_frame_bgr is not None:
            self._recompute_metrics(self._video_widget.current_frame_bgr)
        self._update_controls_state()

    def _recompute_metrics(self, frame_bgr: np.ndarray):
        """计算并更新指标 — 拆分遮挡和计数; 无背景时遮挡指标为None, 计数仍用dark-only fallback"""
        roi_data = self._video_widget.get_roi_data()
        if roi_data is None:
            self._metrics_panel.update_metrics(None)
            self._zoom_widget.clear()
            return

        bg = self._get_effective_background()

        # ---- 遮挡指标: 无背景 → None ----
        if bg is None:
            self._metrics_panel.update_metrics(None)
            self._last_is_occupied = False
            self._last_occ_ratio = 0.0
            self._last_dark_ratio = 0.0
        else:
            metrics = self._metrics_calc.compute(frame_bgr, roi_data["roi_a"], bg)
            self._last_is_occupied = metrics.get("is_occupied", False)
            self._last_occ_ratio = metrics.get("occlusion_area_ratio", 0.0)
            self._last_dark_ratio = metrics.get("dark_pixel_ratio", 0.0)
            self._video_widget._is_occupied = self._last_is_occupied
            self._metrics_panel.update_metrics(metrics)

        # ---- 计数指标: 无背景时用 dark-only fallback 仍然计算 ----
        roi_c_scale = roi_data.get("roi_c_scale", 1.6)
        count_data = self._mouse_counter.estimate_count(
            frame_bgr, roi_data["roi_a"], roi_c_scale, bg,
        )
        self._metrics_panel.update_count_metrics(count_data)

        crop = self._video_widget.get_roi_a_crop()
        if crop is not None:
            self._zoom_widget.update_roi_crop(
                crop, roi_data["roi_a"], self._last_is_occupied,
                self._last_occ_ratio, self._last_dark_ratio,
            )

    # ==================================================================
    # 事件列表交互
    # ==================================================================
    @Slot(int)
    def _on_event_selected(self, event_idx: int):
        """点击事件列表 → 跳转并循环播放 (规范 第5步)"""
        seg = self._event_list.get_segment(event_idx)
        if seg is None:
            return

        self._current_review_idx = event_idx

        # 使用审核范围 (前 review_padding_seconds + 后 review_padding_seconds)
        start_f = seg.get("review_start_frame", seg.get("start_frame", 0))
        end_f = seg.get("review_end_frame", seg.get("end_frame", 0))

        # 设置循环播放范围
        self._video_widget.clear_loop()
        self._video_widget.set_loop_range(start_f, end_f)

        # 跳转到审核开始帧
        self._video_widget.jump_to_frame(start_f)

        # 如果当前暂停, 自动播放
        if not self._video_widget.playing:
            self._video_widget.play()

        # 更新标注面板
        self._annotation_panel.set_event(event_idx, seg)

        self._statusbar.showMessage(
            f"审核事件 #{seg.get('segment_id', event_idx+1)}: "
            f"帧 {start_f}-{end_f} (循环播放中)"
        )

    @Slot(int, list)
    def _on_annotation_changed(self, event_idx: int, mouse_ids: list):
        """标注面板小鼠选择变化"""
        self._event_list.update_segment_mouse_ids(event_idx, mouse_ids)

    @Slot(str, int)
    def _on_review_action(self, action_name: str, event_idx: int):
        """标注面板操作回调"""
        seg = self._event_list.get_segment(event_idx)
        if seg is None:
            return

        if action_name == "confirm":
            # 标记确认 → 绿色
            self._event_list.update_segment_status(event_idx, "confirmed")
            self._annotation_panel.set_event(event_idx, self._event_list.get_segment(event_idx))
            # 跳到下一个 pending
            next_idx = self._event_list.goto_next_pending(event_idx)
            if next_idx >= 0:
                self._event_list.select_row(next_idx)
                self._on_event_selected(next_idx)
            else:
                self._statusbar.showMessage("所有事件已审核完毕!")

        elif action_name == "reject":
            # 标记误检 → 红色
            self._event_list.update_segment_status(event_idx, "rejected")
            self._annotation_panel.set_event(event_idx, self._event_list.get_segment(event_idx))
            # 跳到下一个
            next_idx = self._event_list.goto_next_pending(event_idx)
            if next_idx >= 0:
                self._event_list.select_row(next_idx)
                self._on_event_selected(next_idx)
            else:
                self._statusbar.showMessage("所有事件已审核完毕!")

        elif action_name == "save":
            # 仅保存当前标注
            self._event_list.update_segment_mouse_ids(
                event_idx, self._annotation_panel.get_mouse_ids()
            )
            self._statusbar.showMessage(f"事件 #{seg.get('segment_id')} 标注已保存")

        elif action_name == "save_next":
            # 保存并跳到下一个 pending
            self._event_list.update_segment_mouse_ids(
                event_idx, self._annotation_panel.get_mouse_ids()
            )
            # 当前事件状态设为 modified
            if seg.get("status") == "pending":
                self._event_list.update_segment_status(event_idx, "modified")
            self._annotation_panel.set_event(event_idx, self._event_list.get_segment(event_idx))

            next_idx = self._event_list.goto_next_pending(event_idx)
            if next_idx >= 0:
                self._event_list.select_row(next_idx)
                self._on_event_selected(next_idx)
            else:
                self._statusbar.showMessage("所有事件已审核完毕!")

        elif action_name == "clear":
            # 清空选择已在 annotation_changed 中处理
            pass

    # ==================================================================
    # 自动检测
    # ==================================================================
    def _on_detect_full_video(self):
        """启动全视频自动检测"""
        # 前置检查
        if self._video_widget._cap is None:
            QMessageBox.warning(self, "提示", "请先打开视频文件")
            return
        if not self._video_widget.has_roi():
            QMessageBox.warning(self, "提示", "请先在视频上绘制暖点核心 ROI (鼠标拖拽椭圆)")
            return
        if not self._video_widget.has_background():
            QMessageBox.warning(self, "提示", "请先标记0只小鼠帧作为空场背景 (点击 [0] 标记0只)")
            return

        # 禁用按钮
        self._btn_detect.setEnabled(False)
        self._btn_detect.setText("... 检测中...")

        # 显示进度条
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)

        self._detection_start_time = time.time()

        # 启动后台线程 (两层检测)
        roi_data = self._video_widget.get_roi_data()
        bg = self._video_widget.get_background()
        video_path = self._video_widget.video_path

        self._detection_worker = DetectionWorkerWithCounting(
            video_path=video_path,
            roi_data=roi_data,
            background_bgr=bg,
            metrics_calc=self._metrics_calc,
            counter=self._mouse_counter,
        )
        self._detection_worker.progress_signal.connect(self._on_detection_progress)
        self._detection_worker.finished_signal.connect(self._on_detection_finished_with_counting)
        self._detection_worker.error_signal.connect(self._on_detection_error)
        self._detection_worker.start()

    @Slot(int, str)
    def _on_detection_progress(self, percent: int, message: str):
        """检测进度回调"""
        self._progress_bar.setValue(percent)
        self._statusbar.showMessage(f"[检测 {percent}%] {message}")

    @Slot(list, list)
    def _on_detection_finished_with_counting(self, episodes: list[dict], segments: list[dict]):
        """两层检测完成 (改进.md)"""
        elapsed = time.time() - self._detection_start_time

        # 恢复按钮
        self._btn_detect.setEnabled(True)
        self._btn_detect.setText("自动检测全视频")
        self._progress_bar.setVisible(False)

        self._episodes = episodes
        self._count_segments = segments

        # 将 segments 作为事件列表显示
        for i, seg in enumerate(segments):
            seg.setdefault("segment_id", f"{seg.get('episode_id', '')}-{chr(65 + i)}")

        self._events = segments  # 使用 segments 作为事件列表

        # 刷新事件列表
        self._event_list.clear()
        self._event_list.set_events(segments)

        self._statusbar.showMessage(
            f"检测完成! 耗时 {elapsed:.1f}秒 | "
            f"{len(episodes)} 个占据大事件, "
            f"{len(segments)} 个计数子片段 | "
            f"pending={sum(1 for s in segments if s.get('count_status', 'pending') == 'pending')}"
        )

        # 如果有事件, 自动选中第一个
        if segments:
            self._event_list.select_row(0)

    @Slot(list)
    def _on_detection_finished(self, events: list[dict]):
        """检测完成 (原始单层检测, 保留向后兼容)"""
        elapsed = time.time() - self._detection_start_time

        # 恢复按钮
        self._btn_detect.setEnabled(True)
        self._btn_detect.setText("自动检测全视频")
        self._progress_bar.setVisible(False)

        # 初始化事件属性
        for i, evt in enumerate(events):
            evt.setdefault("status", "pending")
            evt.setdefault("mouse_ids", [])
            evt.setdefault("note", "")
            evt["segment_id"] = i + 1

        self._events = events

        # 刷新事件列表
        self._event_list.clear()
        self._event_list.set_events(events)

        # 统计信息
        total_triggers = self._count_total_pending(events)
        self._statusbar.showMessage(
            f"检测完成! 耗时 {elapsed:.1f}秒 | "
            f"共生成 {len(events)} 个候选事件 | "
            f"状态: pending={sum(1 for e in events if e.get('status')=='pending')}"
        )

        # 如果有事件, 自动选中第一个
        if events:
            self._event_list.select_row(0)

    @Slot(str)
    def _on_detection_error(self, error_msg: str):
        """检测错误"""
        self._btn_detect.setEnabled(True)
        self._btn_detect.setText("自动检测全视频")
        self._progress_bar.setVisible(False)
        QMessageBox.critical(self, "检测错误", f"自动检测失败:\n{error_msg}")
        self._statusbar.showMessage(f"检测失败: {error_msg}")

    # ==================================================================
    # UI 辅助
    # ==================================================================
    @staticmethod
    def _count_total_pending(events: list) -> int:
        return sum(1 for e in events if e.get("status") == "pending")

    def _update_frame_info_label(self):
        """更新帧信息标签"""
        total = self._video_widget.total_frames
        idx = self._video_widget.current_frame_idx
        time_cur_sec = self._video_widget.current_time_sec
        time_total_sec = self._video_widget.duration_sec
        fps = self._video_widget.fps

        cur_str = self._format_time(time_cur_sec)
        total_str = self._format_time(time_total_sec)

        self._lbl_frame_info.setText(
            f"帧: {idx} / {total} | "
            f"时间: {cur_str} / {total_str} | "
            f"FPS: {fps:.2f}"
        )

    def _check_loop_boundary(self, frame_idx: int):
        """当循环播放到达结束帧时, 更新状态"""
        pass  # 循环逻辑在 video_widget._on_play_tick 中处理

    # ==================================================================
    # 动作回调
    # ==================================================================
    def _on_open_video(self):
        """打开视频"""
        path, _ = QFileDialog.getOpenFileName(
            self, "打开视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv *.flv);;所有文件 (*.*)",
        )
        if path:
            success = self._video_widget.open_video(path)
            if success:
                self._update_controls_state()
                self._update_frame_info_label()
                self._event_list.clear()
                self._events = []
                self._annotation_panel.set_event(-1, None)
                self._statusbar.showMessage(f"已打开: {os.path.basename(path)}")
            else:
                self._statusbar.showMessage("视频打开失败")

    def _on_toggle_play(self):
        """播放/暂停"""
        self._video_widget.toggle_play_pause()
        if self._video_widget.playing:
            self._btn_play.setText("|| 暂停")
        else:
            self._btn_play.setText("> 播放")

    def _on_save_roi(self):
        """保存 ROI"""
        path, _ = QFileDialog.getSaveFileName(
            self, "保存 ROI 坐标", "roi.json", "JSON 文件 (*.json)"
        )
        if path:
            self._video_widget.save_roi(path)
            self._statusbar.showMessage(f"ROI 已保存: {path}")

    def _on_load_roi(self):
        """加载 ROI"""
        path, _ = QFileDialog.getOpenFileName(
            self, "加载 ROI 坐标", "", "JSON 文件 (*.json)"
        )
        if path:
            self._video_widget.load_roi(path)
            self._statusbar.showMessage(f"ROI 已加载: {path}")

    def _on_save_background(self):
        """保存背景帧 PNG"""
        bg = self._video_widget.get_background()
        if bg is None:
            QMessageBox.warning(self, "提示", "请先设置背景帧")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存背景帧", "background.png", "PNG 图像 (*.png)"
        )
        if path:
            cv2.imwrite(path, bg)
            self._statusbar.showMessage(f"背景帧已保存: {path}")

    def _on_load_background(self):
        """加载背景帧"""
        path, _ = QFileDialog.getOpenFileName(
            self, "加载背景帧", "", "PNG 图像 (*.png);;所有文件 (*.*)"
        )
        if path:
            bg = cv2.imread(path)
            if bg is not None:
                self._video_widget._calibrated_frames[0] = bg
                self._video_widget._calibrated_frame_idx[0] = -2
                self._video_widget.background_set.emit(bg)
                self._on_background_set(bg)
                self._statusbar.showMessage(f"背景帧已加载: {path}")
            else:
                QMessageBox.warning(self, "错误", "无法读取背景帧图像")

    def _on_slider_changed(self, value: int):
        """滑块拖动"""
        if self._video_widget._cap is not None:
            self._video_widget.jump_to_frame(value)

    def _on_slider_released(self):
        """滑块释放"""
        value = self._time_slider.value()
        self._video_widget.jump_to_frame(value)

    # ==================================================================
    # 键盘事件 — 快捷键映射 (规范 7 节)
    # ==================================================================
    def keyPressEvent(self, event: QKeyEvent):
        """主窗口键盘事件, 转发到各控件"""
        key = event.key()

        # Space: 播放/暂停
        if key == Qt.Key_Space:
            self._video_widget.toggle_play_pause()
            return

        # ← → 单帧
        if key == Qt.Key_Left:
            self._video_widget.prev_frame()
            return
        if key == Qt.Key_Right:
            self._video_widget.next_frame()
            return

        # A/D ±10 帧
        if key == Qt.Key_A:
            self._video_widget.jump_frames(-10)
            return
        if key == Qt.Key_D:
            self._video_widget.jump_frames(10)
            return

        # J/L 上下事件
        if key == Qt.Key_J:
            self._goto_prev_event()
            return
        if key == Qt.Key_L:
            self._goto_next_event()
            return

        # Enter: 保存并下一个
        if key in (Qt.Key_Return, Qt.Key_Enter):
            self._on_review_action("confirm", self._current_review_idx)
            return

        # X: 标记误检
        if key == Qt.Key_X:
            self._on_review_action("reject", self._current_review_idx)
            return

        # C: 清空选择
        if key == Qt.Key_C:
            self._annotation_panel._on_clear()
            return

        # S: 保存
        if key == Qt.Key_S:
            self._on_review_action("save", self._current_review_idx)
            return

        # 1/2/3/4: 小鼠切换
        if key == Qt.Key_1:
            self._annotation_panel._on_mouse_toggle(1)
            return
        if key == Qt.Key_2:
            self._annotation_panel._on_mouse_toggle(2)
            return
        if key == Qt.Key_3:
            self._annotation_panel._on_mouse_toggle(3)
            return
        if key == Qt.Key_4:
            self._annotation_panel._on_mouse_toggle(4)
            return

        # Ctrl+1/2/3/4: 确认数量 (改进.md 13 节)
        if event.modifiers() == Qt.ControlModifier:
            if key == Qt.Key_1:
                self._on_count_confirmed(self._current_review_idx, 1)
                return
            if key == Qt.Key_2:
                self._on_count_confirmed(self._current_review_idx, 2)
                return
            if key == Qt.Key_3:
                self._on_count_confirmed(self._current_review_idx, 3)
                return
            if key == Qt.Key_4:
                self._on_count_confirmed(self._current_review_idx, 4)
                return

        # R: 强制刷新计数器
        if key == Qt.Key_R:
            self._refresh_counter()
            return

        super().keyPressEvent(event)

    def _goto_prev_event(self):
        """跳转到上一个事件"""
        if not self._events:
            return
        new_idx = self._event_list.goto_prev_event(self._current_review_idx)
        if new_idx != self._current_review_idx:
            self._event_list.select_row(new_idx)
            self._on_event_selected(new_idx)

    def _goto_next_event(self):
        """跳转到下一个事件"""
        if not self._events:
            return
        new_idx = self._event_list.goto_next_event(self._current_review_idx)
        if new_idx != self._current_review_idx:
            self._event_list.select_row(new_idx)
            self._on_event_selected(new_idx)

    # ==================================================================
    # 计数与拆分相关回调 (改进.md)
    # ==================================================================
    @Slot(int, int)
    def _on_count_confirmed(self, event_idx: int, count: int):
        """确认数量 (Ctrl+1~4)"""
        seg = self._event_list.get_segment(event_idx)
        if seg is None:
            return
        seg["confirmed_mouse_count"] = count
        seg["count_status"] = "confirmed"
        self._event_list._refresh_table()
        self._annotation_panel.set_event(event_idx, self._event_list.get_segment(event_idx))
        self._statusbar.showMessage(f"片段 #{seg.get('segment_id')} 数量确认: {count} 只")

    @Slot(int, int)
    def _on_segment_split_requested(self, idx: int, frame: int):
        """拆分请求 (来自事件列表右键菜单)"""
        if not (0 <= idx < len(self._event_list.get_segments())):
            return
        if frame < 0:
            frame = self._video_widget.current_frame_idx
        self._event_list.split_segment_at_frame(idx, frame)
        self._events = self._event_list.get_segments()
        seg_id = self._events[idx].get('segment_id', '?') if idx < len(self._events) else '?'
        self._statusbar.showMessage(f"片段 #{seg_id} 已拆分")

    def _on_split_at_current_frame(self):
        """在当前帧拆分当前片段 (K 键, 改进.md 13 节)"""
        if self._current_review_idx < 0:
            return
        seg = self._event_list.get_segment(self._current_review_idx)
        if seg is None:
            return

        cur_frame = self._video_widget.current_frame_idx
        if cur_frame <= seg["start_frame"] or cur_frame >= seg["end_frame"]:
            self._statusbar.showMessage("当前帧不在片段范围内, 无法拆分")
            return

        self._event_list.split_segment_at_frame(self._current_review_idx, cur_frame)
        self._events = self._event_list.get_segments()
        self._statusbar.showMessage(
            f"片段 #{seg.get('segment_id')} 在帧 {cur_frame} 处拆分"
        )


    def _on_toggle_roi_c(self):
        """切换 ROI C 显示"""
        self._video_widget._show_roi_c = self._btn_toggle_roi_c.isChecked()
        if self._video_widget._show_roi_c:
            self._btn_toggle_roi_c.setText("[C] ROI C 开")
        else:
            self._btn_toggle_roi_c.setText("[C] ROI C 关")
        self._video_widget.update()

    def _on_mark_calibration(self, mouse_count: int):
        """左键永远追加样本到 CalibrationStore, 绝不撤回"""
        frame = self._video_widget.current_frame_bgr
        frame_idx = self._video_widget.current_frame_idx
        if frame is None:
            return

        # 1. 追加样本到 CalibrationStore
        sample = CalibrationSample(
            mouse_count=mouse_count, frame_idx=frame_idx,
            frame_bgr=frame.copy(),
        )
        self._calib_store.add_sample(sample)

        # 2. 如果是 [0], 立刻更新背景 + 自动重测所有 pending 样本
        if mouse_count == 0:
            bg = self._calib_store.latest_background()
            self._video_widget._calibrated_frames[0] = bg
            self._video_widget._calibrated_frame_idx[0] = frame_idx
            self._video_widget.background_set.emit(bg)
            self._metrics_calc.set_background(bg)
            self._remeasure_all_pending_samples()  # 关键!
            self._rebuild_counter_references()
            self._statusbar.showMessage(f"背景已标记 (帧#{frame_idx}), 已重测所有pending样本")
        else:
            # 3. 尝试测量当前样本面积
            self._try_measure_sample(sample)
            # 4. 重建 MouseCounter 参考值
            self._rebuild_counter_references()
            n = self._calib_store.count(mouse_count)
            v = sum(1 for s in self._calib_store.get_samples(mouse_count) if s.valid)
            self._statusbar.showMessage(
                f"已标记 {mouse_count} 只样本 ×{v}/{n} (帧#{frame_idx})"
            )

        # 5. 更新按钮显示
        self._update_calib_button_labels()
        # 6. 延迟刷新计数
        QTimer.singleShot(10, lambda: self._refresh_counter())
        self._update_controls_state()

    # ==================================================================
    # 校准辅助方法 (Phase 1+2)
    # ==================================================================
    def _try_measure_sample(self, sample: CalibrationSample):
        """如果有背景+ROI+前景, 计算面积并标记 valid"""
        if not self._calib_store.has_background():
            sample.valid = False
            sample.reason = "no_background"
            return
        roi_data = self._video_widget.get_roi_data()
        if roi_data is None:
            sample.valid = False
            sample.reason = "no_roi"
            return
        bg = self._calib_store.latest_background()
        try:
            count_result = self._mouse_counter.estimate_count(
                sample.frame_bgr, roi_data["roi_a"],
                roi_data.get("roi_c_scale", 1.6), bg,
            )
            area = count_result.get("total_mouse_area", 0)
            sample.measured_area = area
            if area > 0:
                sample.valid = True
                sample.reason = ""
            else:
                sample.valid = False
                sample.reason = "zero_area"
        except Exception:
            sample.valid = False
            sample.reason = "measure_error"

    def _remeasure_all_pending_samples(self):
        """遍历所有 not valid 的 1-4 样本, 重新测量"""
        for count in range(1, 5):
            for sample in self._calib_store.get_samples(count):
                if not sample.valid:
                    self._try_measure_sample(sample)

    def _rebuild_counter_references(self):
        """清空 MouseCounter 样本 → 遍历 CalibrationStore 中 valid 样本 → add_count_area_sample"""
        self._mouse_counter.clear_all_count_area_samples()
        for count in range(1, 5):
            for sample in self._calib_store.get_samples(count):
                if sample.valid and sample.measured_area is not None and sample.measured_area > 0:
                    self._mouse_counter.add_count_area_sample(count, sample.measured_area)

    def _get_effective_background(self) -> 'np.ndarray | None':
        """返回 CalibrationStore.latest_background() 或 None"""
        return self._calib_store.latest_background()

    def _update_calib_button_labels(self):
        """按钮文字改为 "[N]背景×n" 或 "[N] N只×v/n" 格式, 有样本无背景用黄色"""
        has_bg = self._calib_store.has_background()
        for count in range(5):
            n = self._calib_store.count(count)
            btn = self._calib_btns[count]
            if count == 0:
                label = f"[0]背景×{n}" if n > 0 else "[0]标记背景"
            else:
                v = sum(1 for s in self._calib_store.get_samples(count) if s.valid)
                label = f"[{count}] {count}只×{v}/{n}" if n > 0 else f"[{count}]标记{count}只"
            btn.setText(label)
            if count > 0 and n > 0 and not has_bg:
                btn.setStyleSheet(self._calib_btn_style(count, True, warning=True))
            elif n > 0:
                btn.setStyleSheet(self._calib_btn_style(count, True))
            else:
                btn.setStyleSheet(self._calib_btn_style(count, False))

    # ==================================================================
    # 右键上下文菜单
    # ==================================================================
    def _show_calib_context_menu(self, count: int, pos):
        """校准按钮右键菜单: 撤回最后一个样本 / 清空该数量所有样本"""
        menu = QMenu(self)
        undo_action = menu.addAction("撤回最后一个样本")
        clear_action = menu.addAction("清空该数量所有样本")
        action = menu.exec_(self._calib_btns[count].mapToGlobal(pos))
        if action == undo_action:
            self._undo_last_sample(count)
        elif action == clear_action:
            self._clear_samples(count)

    def _undo_last_sample(self, count: int):
        """撤回最后一个样本"""
        removed = self._calib_store.remove_last_sample(count)
        if removed is None:
            self._statusbar.showMessage(f"没有 {count} 只样本可撤回")
            return
        if count == 0:
            # 同步清除 VideoWidget 背景
            self._video_widget._calibrated_frames.pop(0, None)
            self._video_widget._calibrated_frame_idx.pop(0, None)
            bg = self._calib_store.latest_background()
            if bg is not None:
                self._video_widget._calibrated_frames[0] = bg
                self._metrics_calc.set_background(bg)
                self._video_widget.background_set.emit(bg)
            self._remeasure_all_pending_samples()
        self._rebuild_counter_references()
        self._update_calib_button_labels()
        QTimer.singleShot(10, lambda: self._refresh_counter())
        self._statusbar.showMessage(f"已撤回 {count} 只样本 (帧#{removed.frame_idx})")

    def _clear_samples(self, count: int):
        """清空该数量所有样本"""
        self._calib_store.clear_samples(count)
        if count == 0:
            self._video_widget._calibrated_frames.pop(0, None)
            self._video_widget._calibrated_frame_idx.pop(0, None)
        self._rebuild_counter_references()
        self._update_calib_button_labels()
        QTimer.singleShot(10, lambda: self._refresh_counter())
        self._statusbar.showMessage(f"已清空 {count} 只所有样本")

    @staticmethod
    def _calib_btn_style(count: int, marked: bool, warning: bool = False) -> str:
        """校准按钮样式: 已标记用绿色边框高亮, 有样本无背景用黄色警告"""
        if warning:
            return f"""
                QPushButton {{
                    background-color: #5a5a2a; color: #ffc;
                    border: 2px solid #cccc44; border-radius: 4px;
                    padding: 4px 8px; font-weight: bold;
                }}
            """
        if marked:
            return f"""
                QPushButton {{
                    background-color: #2a5a2a; color: #8f8;
                    border: 2px solid #44cc44; border-radius: 4px;
                    padding: 4px 8px; font-weight: bold;
                }}
            """
        else:
            return f"""
                QPushButton {{
                    background-color: #3a3a3a; color: #ccc;
                    border: 1px solid #666; border-radius: 4px;
                    padding: 4px 8px;
                }}
                QPushButton:hover {{ background-color: #4a4a4a; }}
            """

    # ==================================================================
    # 计数器刷新 (R 键, 改进.md) — 支持无背景 fallback
    # ==================================================================
    def _refresh_counter(self):
        """强制刷新当前帧的计数器 (R 键快捷键)"""
        frame = self._video_widget.current_frame_bgr
        roi_data = self._video_widget.get_roi_data()
        bg = self._get_effective_background()

        if frame is None or roi_data is None:
            self._statusbar.showMessage("计数刷新: 请先设置 ROI")
            return

        count_data = self._mouse_counter.estimate_count(
            frame, roi_data["roi_a"],
            roi_data.get("roi_c_scale", 1.6),
            bg,
        )
        self._metrics_panel.update_count_metrics(count_data)

        count = count_data.get("estimated_mouse_count", 0)
        conf = count_data.get("count_confidence", 0.0)
        fallback = ""
        if count_data.get("fallback_mode"):
            fallback = " (无背景fallback)"
        self._statusbar.showMessage(
            f"计数刷新: 估计 {count} 只小鼠 (置信度 {conf:.0%}){fallback}"
        )

    # ==================================================================
    # Markdown 统计表导出
    # ==================================================================
    def _on_export_markdown(self):
        """导出 Markdown 统计表"""
        confirmed = self._event_list.get_confirmed_segments()
        if not confirmed:
            QMessageBox.information(self, "提示", "没有已确认的事件, 请先确认事件后再导出。")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "导出Markdown统计表", "mouse_occupancy_report.md",
            "Markdown 文件 (*.md);;所有文件 (*.*)"
        )
        if not path:
            return

        # 辅助: 格式化时间
        def fmt_time(sec: float) -> str:
            if sec < 0:
                sec = 0.0
            h = int(sec // 3600)
            m = int((sec % 3600) // 60)
            s = int(sec % 60)
            ms = int((sec - int(sec)) * 100)
            if h > 0:
                return f"{h:02d}:{m:02d}:{s:02d}.{ms:02d}"
            return f"{m:02d}:{s:02d}.{ms:02d}"

        # Part A: 事件明细
        lines = []
        lines.append("# 小鼠暖点占据统计报告")
        lines.append("")
        lines.append("## 事件明细")
        lines.append("")
        lines.append("| 编号 | 开始时间 | 结束时间 | 时长(秒) | 小鼠编号 | 数量 |")
        lines.append("|------|----------|----------|----------|----------|------|")

        for seg in confirmed:
            sid = seg.get("segment_id", "--")
            st = fmt_time(seg.get("start_time", 0.0))
            et = fmt_time(seg.get("end_time", 0.0))
            dur = seg.get("duration", 0.0)
            mouse_ids = seg.get("mouse_ids", [])
            mouse_str = ",".join(str(m) for m in mouse_ids) if mouse_ids else "--"
            count = seg.get("confirmed_mouse_count", len(mouse_ids) if mouse_ids else "--")
            lines.append(f"| {sid} | {st} | {et} | {dur:.2f} | {mouse_str} | {count} |")

        lines.append("")

        # Part B: 每只小鼠汇总
        mouse_stats: dict[int, dict] = {}
        for seg in confirmed:
            dur = seg.get("duration", 0.0)
            for mid in seg.get("mouse_ids", []):
                if mid not in mouse_stats:
                    mouse_stats[mid] = {"total_duration": 0.0, "event_count": 0}
                mouse_stats[mid]["total_duration"] += dur
                mouse_stats[mid]["event_count"] += 1

        lines.append("## 小鼠汇总")
        lines.append("")
        lines.append("| 小鼠编号 | 总占据时长(秒) | 事件次数 |")
        lines.append("|----------|----------------|----------|")
        if mouse_stats:
            for mid in sorted(mouse_stats.keys()):
                ms = mouse_stats[mid]
                lines.append(f"| {mid} | {ms['total_duration']:.2f} | {ms['event_count']} |")
        else:
            lines.append("| -- | -- | -- |")

        lines.append("")

        # Part C: 暖点汇总
        total_duration = sum(seg.get("duration", 0.0) for seg in confirmed)
        total_events = len(confirmed)
        multi_mouse_events = sum(
            1 for seg in confirmed if len(seg.get("mouse_ids", [])) > 1
        )

        lines.append("## 暖点汇总")
        lines.append("")
        lines.append("| 暖点编号 | 总被占据时长(秒) | 总事件数 | 多鼠事件数 |")
        lines.append("|----------|------------------|----------|------------|")
        lines.append(f"| ROI-1 | {total_duration:.2f} | {total_events} | {multi_mouse_events} |")
        lines.append("")

        # 写入文件
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            self._statusbar.showMessage(f"Markdown 统计表已导出: {path}")
        except Exception as e:
            QMessageBox.critical(self, "导出错误", f"导出失败:\n{e}")
            self._statusbar.showMessage(f"导出失败: {e}")

    # ==================================================================
    # 工具方法
    # ==================================================================
    def _update_controls_state(self):
        """更新控件可用性"""
        has_video = self._video_widget._cap is not None
        has_bg = self._video_widget.has_background()
        has_roi = self._video_widget.has_roi()
        self._btn_play.setEnabled(has_video)
        self._btn_prev.setEnabled(has_video)
        self._btn_prev1.setEnabled(has_video)
        self._btn_next1.setEnabled(has_video)
        self._btn_next.setEnabled(has_video)
        self._btn_clear_roi.setEnabled(has_video)
        self._btn_detect.setEnabled(has_video and has_bg and has_roi)
        self._btn_toggle_roi_c.setEnabled(has_video)
        # 刷新按钮: 无背景也可刷新 (用 fallback)
        self._btn_refresh.setEnabled(has_video and has_roi)
        for btn in self._calib_btns.values():
            btn.setEnabled(has_video)
        self._time_slider.setEnabled(has_video)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """格式化时间为 HH:MM:SS.ms"""
        if seconds < 0:
            seconds = 0.0
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 100)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:02d}"

    @staticmethod
    def _count_total_pending(events: list) -> int:
        return sum(1 for e in events if e.get("status") == "pending")

"""
主窗口布局 — Phase 2: 完整检测+审核工作流
整合：视频控件 + 放大窗口 + 指标面板 + 事件列表 + 标注面板
实现规范 6 节完整布局
"""

import os
import time
import traceback
import cv2
import numpy as np

from PySide6.QtCore import Qt, Signal, Slot, QThread, QTimer
from PySide6.QtGui import QAction, QKeySequence, QFont, QKeyEvent
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QLineEdit, QStatusBar,
    QFileDialog, QMessageBox, QDockWidget, QToolBar, QMenuBar, QMenu,
    QApplication, QSplitter, QSizePolicy, QProgressBar, QProgressDialog, QInputDialog, QScrollArea,
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
from detection.identity_assist import IdentityAssist, apply_identity_to_segment
from export.csv_exporter import CsvExporter


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
        cap = None
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

            self.finished_signal.emit(events)

        except Exception as e:
            self.error_signal.emit(f"{e}\n{traceback.format_exc()}")
        finally:
            if cap is not None:
                cap.release()


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
        cap = None
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

            self.finished_signal.emit(episodes, segments)

        except Exception as e:
            self.error_signal.emit(f"{e}\n{traceback.format_exc()}")
        finally:
            if cap is not None:
                cap.release()


class ColorIdentifyWorker(QThread):
    """后台颜色识别线程 — 不阻塞 UI"""
    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, segment, roi_core, video_path, fps, parent=None):
        super().__init__(parent)
        self.segment = segment
        self.roi_core = roi_core
        self._video_path = video_path
        self.fps = fps
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        cap = None
        try:
            cap = cv2.VideoCapture(self._video_path)
            if not cap.isOpened():
                self.error.emit(f"无法打开视频: {self._video_path}")
                return
            if self._cancelled:
                return
            assist = IdentityAssist(debug=True)
            result = assist.analyze_segment(
                segment=self.segment, roi_core=self.roi_core,
                cap=cap, fps=self.fps,
                progress_callback=lambda pct, msg: self.progress.emit(pct, msg),
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")
        finally:
            if cap is not None:
                cap.release()


# =========================================================================
# 主窗口
# =========================================================================
class BatchColorIdentifyWorker(QThread):
    """串行识别多个 segment；所有界面更新由主线程的 slot 完成。"""
    item_started = Signal(int, int, int)  # index, ordinal, total
    item_progress = Signal(int, int, int, str)  # index, ordinal, percent, message
    item_finished = Signal(int, dict)
    item_error = Signal(int, str)
    finished_batch = Signal(bool)

    def __init__(self, items, roi_core, video_path, fps, parent=None):
        super().__init__(parent)
        self._items = items
        self._roi_core = roi_core
        self._video_path = video_path
        self._fps = fps
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        cap = None
        try:
            cap = cv2.VideoCapture(self._video_path)
            if not cap.isOpened():
                self.item_error.emit(-1, f"无法打开视频: {self._video_path}")
                return
            total = len(self._items)
            for ordinal, (index, segment) in enumerate(self._items, 1):
                if self._cancelled:
                    break
                self.item_started.emit(index, ordinal, total)
                try:
                    result = IdentityAssist(debug=True).analyze_segment(
                        segment=segment, roi_core=self._roi_core, cap=cap, fps=self._fps,
                        progress_callback=lambda pct, msg, i=index, o=ordinal:
                            self.item_progress.emit(i, o, pct, msg),
                    )
                    if not self._cancelled:
                        self.item_finished.emit(index, result)
                except Exception as exc:
                    self.item_error.emit(index, str(exc))
            self.finished_batch.emit(self._cancelled)
        except Exception as exc:
            self.item_error.emit(-1, str(exc))
        finally:
            if cap is not None:
                cap.release()


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

        # 身份识别线程（单项或串行批量，避免并发云请求）
        self._color_worker = None
        self._batch_color_worker = None
        self._batch_color_progress = None

        # 校准标记系统 (Phase 1+2: CalibrationStore 接管)
        self._calib_store = CalibrationStore()

        # 当前审核事件索引
        self._current_review_idx: int = -1

        # Phase 10: Debug view
        self._debug_view_enabled: bool = False
        self._debug_data: dict = {}

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

        export_csv_action = QAction("导出CSV(&E)...", self)
        export_csv_action.setShortcut(QKeySequence("Ctrl+E"))
        export_csv_action.triggered.connect(self._on_export_csv)
        file_menu.addAction(export_csv_action)

        file_menu.addSeparator()

        save_project_action = QAction("保存项目(&P)...", self)
        save_project_action.setShortcut(QKeySequence("Ctrl+S"))
        save_project_action.triggered.connect(self._on_save_project)
        file_menu.addAction(save_project_action)

        load_project_action = QAction("加载项目(&J)...", self)
        load_project_action.triggered.connect(self._on_load_project)
        file_menu.addAction(load_project_action)

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

        view_menu.addSeparator()

        # Phase 10: Debug 视图开关
        self._debug_view_action = QAction("Debug 视图(&D)", self)
        self._debug_view_action.setCheckable(True)
        self._debug_view_action.setChecked(False)
        self._debug_view_action.setShortcut(QKeySequence("Ctrl+D"))
        self._debug_view_action.triggered.connect(self._on_toggle_debug_view)
        view_menu.addAction(self._debug_view_action)

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
        self._btn_prev.setToolTip("快退 10 帧 (J)")
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
        self._btn_next.setToolTip("快进 10 帧 (L)")
        self._btn_next.clicked.connect(lambda: self._video_widget.jump_frames(10))
        self._btn_next.setMinimumHeight(30)
        toolbar.addWidget(self._btn_next)

        toolbar.addSeparator()

        # 校准帧标记按钮 [0] [1] [2] (cap=2: 左键追加, 右键菜单撤回/清空)
        self._calib_btns: dict[int, QPushButton] = {}
        for count in range(3):
            btn = QPushButton(f"[{count}] 标记背景" if count == 0 else f"[{count}] 标记{count}只")
            btn.setToolTip(f"左键: 将当前帧追加为 {count} 只小鼠样本 | 右键: 撤回/清空")
            btn.setMinimumHeight(30)
            btn.clicked.connect(lambda checked, c=count: self._on_mark_calibration(c))
            btn.setContextMenuPolicy(Qt.CustomContextMenu)
            btn.customContextMenuRequested.connect(lambda pos, c=count: self._show_calib_context_menu(c, pos))
            btn.setStyleSheet(self._calib_btn_style(count, False))
            self._calib_btns[count] = btn
            toolbar.addWidget(btn)

        # ROI 操作
        self._btn_clear_roi = QPushButton("X 清除 ROI")
        self._btn_clear_roi.clicked.connect(self._video_widget.clear_roi)
        self._btn_clear_roi.setMinimumHeight(30)
        toolbar.addWidget(self._btn_clear_roi)

        # 编辑模式切换: 内圈(Core) / 外圈(Count)
        self._btn_edit_mode = QPushButton("编辑内圈")
        self._btn_edit_mode.setToolTip("切换当前编辑的椭圆: 内圈(暖点铜片)/外圈(两只鼠范围)")
        self._btn_edit_mode.setCheckable(True)
        self._btn_edit_mode.setChecked(True)  # 默认编辑内圈
        self._btn_edit_mode.clicked.connect(self._on_toggle_edit_mode)
        self._btn_edit_mode.setMinimumHeight(30)
        toolbar.addWidget(self._btn_edit_mode)

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

        # 颜色识别（可选择当前事件或全部事件）
        self._btn_color = QPushButton("颜色识别")
        self._btn_color.setToolTip("识别当前事件或串行识别全部事件")
        self._btn_color.clicked.connect(self._on_color_identify)
        self._btn_color.setMinimumHeight(30)
        self._btn_color.setEnabled(False)
        toolbar.addWidget(self._btn_color)


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
        # 暗色滚动区域样式
        scroll_style = (
            "QScrollArea { background-color: #2a2a2a; border: none; }"
            "QScrollArea > QWidget > QWidget { background-color: #2a2a2a; }"
        )

        # 暖点放大窗口
        self._zoom_widget = ZoomWidget()
        zoom_scroll = QScrollArea()
        zoom_scroll.setWidgetResizable(True)
        zoom_scroll.setWidget(self._zoom_widget)
        zoom_scroll.setStyleSheet(scroll_style)
        self._zoom_dock = QDockWidget("暖点放大", self)
        self._zoom_dock.setWidget(zoom_scroll)
        self._zoom_dock.setMinimumWidth(250)
        self.addDockWidget(Qt.RightDockWidgetArea, self._zoom_dock)

        # 检测指标面板
        self._metrics_panel = MetricsPanel()
        metrics_scroll = QScrollArea()
        metrics_scroll.setWidgetResizable(True)
        metrics_scroll.setWidget(self._metrics_panel)
        metrics_scroll.setStyleSheet(scroll_style)
        self._metrics_dock = QDockWidget("检测指标", self)
        self._metrics_dock.setWidget(metrics_scroll)
        self._metrics_dock.setMinimumWidth(280)
        self.addDockWidget(Qt.RightDockWidgetArea, self._metrics_dock)

        # 标注面板
        self._annotation_panel = AnnotationPanel()
        annot_scroll = QScrollArea()
        annot_scroll.setWidgetResizable(True)
        annot_scroll.setWidget(self._annotation_panel)
        annot_scroll.setStyleSheet(scroll_style)
        self._annotation_dock = QDockWidget("标注面板", self)
        self._annotation_dock.setWidget(annot_scroll)
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
            roi_core = roi_data["roi_core"]
            occ_ratio = getattr(self, '_last_occ_ratio', 0.0)
            is_occupied = getattr(self, '_last_is_occupied', False)
            dark_ratio = getattr(self, '_last_dark_ratio', 0.0)
            self._zoom_widget.update_roi_crop(crop, roi_core, is_occupied, occ_ratio, dark_ratio)

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
            metrics = self._metrics_calc.compute(frame_bgr, roi_data["roi_core"], bg)
            self._last_is_occupied = metrics.get("is_occupied", False)
            self._last_occ_ratio = metrics.get("occlusion_area_ratio", 0.0)
            self._last_dark_ratio = metrics.get("dark_pixel_ratio", 0.0)
            self._video_widget._is_occupied = self._last_is_occupied
            self._metrics_panel.update_metrics(metrics)

        # ---- 计数指标: 无背景时用 dark-only fallback 仍然计算 ----
        count_data = self._mouse_counter.estimate_count(
            frame_bgr, roi_data["roi_core"], roi_data.get("roi_count"), bg,
        )
        self._metrics_panel.update_count_metrics(count_data)

        # Phase 10: Debug 视图 - 收集中间 mask
        if self._debug_view_enabled and bg is not None:
            debug_masks = self._metrics_calc.compute_debug_masks(
                frame_bgr, roi_data["roi_core"], bg
            )
            debug_data = {
                "masks": debug_masks,
                "occ_score": metrics.get("occlusion_area_ratio", 0.0) if metrics else 0.0,
                "count": count_data.get("estimated_mouse_count", 0),
                "count_confidence": count_data.get("count_confidence", 0.0),
                "blob_count": count_data.get("blob_count", 0),
                "dark_pixel_ratio": metrics.get("dark_pixel_ratio", 0.0) if metrics else 0.0,
                "debug_blobs": count_data.get("debug_blobs", []),
            }
            self._video_widget.set_debug_data(debug_data)
        elif not self._debug_view_enabled:
            self._video_widget.set_debug_data({})

        crop = self._video_widget.get_roi_a_crop()
        if crop is not None:
            self._zoom_widget.update_roi_crop(
                crop, roi_data["roi_core"], self._last_is_occupied,
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
            # Phase 6: 确认逻辑 — count=1选1个mouse_id, count=2选2个, count=0可标误检
            est_count = seg.get("estimated_mouse_count")
            mouse_ids = self._annotation_panel.get_mouse_ids()
            if est_count == 0:
                # count=0 → 标记误检
                self._event_list.update_segment_status(event_idx, "rejected")
                seg["confirmed_mouse_count"] = 0
                seg["count_status"] = "rejected"
                self._event_list._refresh_table()
                self._annotation_panel.set_event(event_idx, self._event_list.get_segment(event_idx))
                self._statusbar.showMessage(f"片段 #{seg.get('segment_id')} count=0, 已标记为误检")
            elif est_count is not None and est_count >= 1 and len(mouse_ids) < est_count:
                # 小鼠数量不足
                self._statusbar.showMessage(
                    f"确认失败: count={est_count} 需要选择 {est_count} 只小鼠, "
                    f"当前仅选 {len(mouse_ids)} 只"
                )
                return
            else:
                # 正常确认
                seg["mouse_ids"] = sorted(mouse_ids)
                seg["confirmed_mouse_count"] = len(mouse_ids)
                self._event_list.update_segment_status(event_idx, "confirmed")
                seg["count_status"] = "confirmed"
                self._event_list._refresh_table()
                self._annotation_panel.set_event(event_idx, self._event_list.get_segment(event_idx))
                self._statusbar.showMessage(
                    f"片段 #{seg.get('segment_id')} 已确认: "
                    f"count={len(mouse_ids)}, 小鼠={mouse_ids}"
                )
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

        if self._detection_worker is not None and self._detection_worker.isRunning():
            QMessageBox.warning(self, "检测进行中", "请等待当前检测完成")
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

        counter_snapshot = self._mouse_counter.clone() if self._mouse_counter else None
        self._detection_worker = DetectionWorkerWithCounting(
            video_path=video_path,
            roi_data=roi_data,
            background_bgr=bg,
            metrics_calc=self._metrics_calc,
            counter=counter_snapshot,
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

        # 启用身份辅助按钮

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

    def _on_save_project(self):
        """保存项目状态到 .mwp.json 文件"""
        import json

        path, _ = QFileDialog.getSaveFileName(
            self, "保存项目", "", "项目文件 (*.mwp.json)"
        )
        if not path:
            return

        try:
            # --- 收集数据 ---
            video_path = self._video_widget.video_path

            # ROI 数据 (get_roi_data 返回 dict, 已可直接序列化)
            roi_data = self._video_widget.get_roi_data()

            # 校准样本 (不保存 frame_bgr)
            calibration = []
            for count in range(3):
                for sample in self._calib_store.get_samples(count):
                    calibration.append({
                        "mouse_count": sample.mouse_count,
                        "frame_idx": sample.frame_idx,
                    })

            # 事件/片段 (转换内部 numpy 值)
            segments = self._event_list.get_segments()
            serializable_segments = []
            for seg in segments:
                clean_seg = {}
                for k, v in seg.items():
                    if isinstance(v, np.ndarray):
                        clean_seg[k] = v.tolist()
                    elif isinstance(v, (np.integer,)):
                        clean_seg[k] = int(v)
                    elif isinstance(v, (np.floating,)):
                        clean_seg[k] = float(v)
                    elif isinstance(v, list):
                        clean_seg[k] = [
                            x.tolist() if isinstance(x, np.ndarray)
                            else int(x) if isinstance(x, (np.integer,))
                            else float(x) if isinstance(x, (np.floating,))
                            else x
                            for x in v
                        ]
                    else:
                        clean_seg[k] = v
                serializable_segments.append(clean_seg)

            data = {
                "version": 1,
                "video_path": video_path or "",
                "roi_data": roi_data,
                "calibration": calibration,
                "segments": serializable_segments,
            }

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            self._statusbar.showMessage(f"项目已保存: {path}")

        except Exception as e:
            QMessageBox.critical(self, "保存错误", f"项目保存失败:\n{e}")

    def _on_load_project(self):
        """从 .mwp.json 文件加载项目状态"""
        import json

        path, _ = QFileDialog.getOpenFileName(
            self, "加载项目", "", "项目文件 (*.mwp.json)"
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "加载错误", f"无法读取项目文件:\n{e}")
            return

        # 版本校验
        version = data.get("version", 0)
        if version > 1:
            QMessageBox.warning(
                self, "版本不兼容",
                f"项目文件版本 {version} 高于当前支持版本 1，可能无法正确加载。"
            )

        try:
            # --- 恢复视频 ---
            video_path = data.get("video_path", "")
            video_opened = False
            if video_path and os.path.exists(video_path):
                video_opened = self._video_widget.open_video(video_path)
                if not video_opened:
                    self._statusbar.showMessage(f"警告: 无法打开视频 {video_path}")
            elif video_path:
                self._statusbar.showMessage(f"警告: 视频文件不存在 {video_path}")

            # --- 恢复 ROI ---
            roi_data = data.get("roi_data")
            if roi_data:
                if "roi_core" in roi_data and roi_data["roi_core"]:
                    self._video_widget.roi_core = roi_data["roi_core"]
                if "roi_count" in roi_data and roi_data["roi_count"]:
                    self._video_widget.roi_count = roi_data["roi_count"]
                self._video_widget.roi_changed.emit(
                    self._video_widget.roi_core
                )
                self._video_widget.update()

            # --- 恢复校准样本 (无 frame_bgr, 标记 invalid) ---
            calib_data = data.get("calibration", [])
            if calib_data:
                # 清空现有校准数据
                for count in range(3):
                    self._calib_store.clear_samples(count)
                self._video_widget._calibrated_frames.clear()
                self._video_widget._calibrated_frame_idx.clear()

                for entry in calib_data:
                    sample = CalibrationSample(
                        mouse_count=entry["mouse_count"],
                        frame_idx=entry["frame_idx"],
                        frame_bgr=np.zeros((1, 1, 3), dtype=np.uint8),
                    )
                    sample.valid = False
                    sample.reason = "loaded_without_frame"
                    self._calib_store.add_sample(sample)

                    # 同步背景帧引用
                    if entry["mouse_count"] == 0:
                        self._video_widget._calibrated_frame_idx[0] = entry["frame_idx"]

            # --- 恢复事件/片段 ---
            segments = data.get("segments", [])
            if segments:
                self._events = segments
                self._count_segments = segments
                self._event_list.clear()
                self._event_list.set_events(segments)
                if segments:
                    self._event_list.select_row(0)

            self._update_controls_state()
            self._update_frame_info_label()
            self._update_calib_button_labels()
            self._statusbar.showMessage(
                f"项目已加载: {path}"
                + (" | 校准样本需重新标记背景后重测" if calib_data else "")
            )

        except Exception as e:
            QMessageBox.critical(self, "加载错误", f"项目加载失败:\n{e}")

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

        # ↑↓ 上下箭头: 跳转上一个/下一个事件并自动播放
        if key == Qt.Key_Up:
            self._goto_prev_event()
            return
        if key == Qt.Key_Down:
            self._goto_next_event()
            return

        # J/L 快退/进 (Phase 6: 原 A/D ±10 帧移至此)
        if key == Qt.Key_J:
            self._video_widget.jump_frames(-10)
            return
        if key == Qt.Key_L:
            self._video_widget.jump_frames(10)
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

        # Shift+1/2: 确认数量 (改进.md 13 节; cap=2)
        if event.modifiers() == Qt.ShiftModifier:
            if key == Qt.Key_1:
                self._on_count_confirmed(self._current_review_idx, 1)
                return
            if key == Qt.Key_2:
                self._on_count_confirmed(self._current_review_idx, 2)
                return

        # R: 强制刷新计数器
        if key == Qt.Key_R:
            self._refresh_counter()
            return

        # K: 拆分片段 (Phase 6)
        if key == Qt.Key_K:
            self._on_split_at_current_frame()
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
        """确认数量 (Shift+1~2)"""
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









    def _on_toggle_edit_mode(self):
        """切换编辑模式: core ↔ count"""
        if self._btn_edit_mode.isChecked():
            self._video_widget.set_edit_mode("core")
            self._btn_edit_mode.setText("编辑内圈")
        else:
            self._video_widget.set_edit_mode("count")
            self._btn_edit_mode.setText("编辑外圈")

    def _on_toggle_debug_view(self, checked: bool):
        """Phase 10: Debug 视图开关"""
        self._debug_view_enabled = checked
        if checked:
            self._statusbar.showMessage("Debug 视图已启用: ROI边框/深色mask/暖色mask/前景mask/连通区/occ/count")
        else:
            self._statusbar.showMessage("Debug 视图已关闭")
            self._video_widget.set_debug_data({})
        # 立即刷新当前帧
        if self._video_widget.current_frame_bgr is not None:
            self._recompute_metrics(self._video_widget.current_frame_bgr)
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
                sample.frame_bgr, roi_data["roi_core"],
                roi_data.get("roi_count"), bg,
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
        for count in range(1, 3):
            for sample in self._calib_store.get_samples(count):
                if not sample.valid:
                    self._try_measure_sample(sample)

    def _rebuild_counter_references(self):
        """清空 MouseCounter 样本 → 遍历 CalibrationStore 中 valid 样本 → add_count_area_sample (cap=2)"""
        self._mouse_counter.clear_all_count_area_samples()
        for count in range(1, 3):
            for sample in self._calib_store.get_samples(count):
                if sample.valid and sample.measured_area is not None and sample.measured_area > 0:
                    self._mouse_counter.add_count_area_sample(count, sample.measured_area)

    def _get_effective_background(self) -> 'np.ndarray | None':
        """返回 CalibrationStore.latest_background() 或 None"""
        return self._calib_store.latest_background()

    def _update_calib_button_labels(self):
        """按钮文字改为 "[N]背景×n" 或 "[N] N只×v/n" 格式, 有样本无背景用黄色"""
        has_bg = self._calib_store.has_background()
        for count in range(3):
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
            frame, roi_data["roi_core"],
            roi_data.get("roi_count"),
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
    # 颜色识别
    # ==================================================================
    def _on_color_identify(self):
        """选择当前事件或全部事件；批量路径严格串行。"""
        if self._color_worker and self._color_worker.isRunning() or self._batch_color_worker and self._batch_color_worker.isRunning():
            QMessageBox.information(self, "颜色识别", "颜色识别正在进行中")
            return
        roi_data = self._video_widget.get_roi_data()
        if roi_data is None or not self._video_widget.video_path:
            QMessageBox.warning(self, "提示", "请先打开视频并绘制 ROI")
            return
        box = QMessageBox(self)
        box.setWindowTitle("颜色识别范围")
        box.setText("请选择颜色识别范围：")
        current = box.addButton("当前事件", QMessageBox.AcceptRole)
        all_events = box.addButton("全部事件", QMessageBox.ActionRole)
        cancel = box.addButton("取消", QMessageBox.RejectRole)
        has_current = self._event_list.get_segment(self._current_review_idx) is not None
        current.setEnabled(has_current)
        if not has_current:
            box.setInformativeText("未选中当前事件；仍可识别全部事件。")
        box.exec()
        if box.clickedButton() is current and has_current:
            self._start_single_color_identify(self._current_review_idx, roi_data)
        elif box.clickedButton() is all_events:
            self._start_batch_color_identify(roi_data)

    def _start_single_color_identify(self, seg_idx, roi_data):
        seg = self._event_list.get_segment(seg_idx)
        if not seg:
            return
        progress = QProgressDialog(f"颜色识别: 正在分析片段 #{seg.get('segment_id', '?')}...", "取消", 0, 100, self)
        progress.setWindowTitle("颜色识别")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        self._color_worker = ColorIdentifyWorker(seg, roi_data["roi_core"], self._video_widget.video_path, self._video_widget.fps, self)
        self._color_worker.progress.connect(lambda pct, msg: (progress.setLabelText(msg), progress.setValue(pct)))
        self._color_worker.finished.connect(lambda result: self._on_color_finished(progress, seg_idx, result))
        self._color_worker.error.connect(lambda err: self._on_color_error(progress, err))
        progress.canceled.connect(self._color_worker.cancel)
        self._color_worker.start()

    def _apply_color_result(self, index, result):
        seg = self._event_list.get_segment(index)
        if seg is None:
            return
        apply_identity_to_segment(seg, result)
        self._event_list._update_row(index)
        if index == self._current_review_idx:
            self._annotation_panel.set_event(index, seg)
        return seg

    def _on_color_finished(self, progress, index, result):
        """保留当前事件的人工颜色→鼠号确认；测温器结果直接安全写回。"""
        progress.close()
        seg = self._event_list.get_segment(index)
        if seg is None:
            return
        if result.get("thermometer_present"):
            self._apply_color_result(index, result)
            QMessageBox.warning(self, "颜色识别", "检测到测温器/探头，颜色识别置信度已置零，请人工复核。")
            self._statusbar.showMessage(f"颜色识别完成: 事件 #{seg.get('segment_id')}，置信度 0.00")
            return
        valid_colors = [c for c in result.get("auto_mouse_colors", []) if c != "unknown"]
        if not valid_colors:
            QMessageBox.information(self, "颜色识别", "未检测到有效耳标颜色。")
            return
        mouse_ids = []
        for color in valid_colors:
            mid, ok = QInputDialog.getInt(self, f"识别到 {color}",
                                           f"检测到耳标颜色: {color}\n请选择对应的鼠号 (1-4):",
                                           value=1, minValue=1, maxValue=4, step=1)
            if ok:
                mouse_ids.append(mid)
        if mouse_ids:
            self._apply_color_result(index, result)
            seg["mouse_ids"] = mouse_ids
            seg["mouse_count"] = len(mouse_ids)
            seg["count_status"] = "confirmed"
            self._event_list.update_segment_mouse_ids(index, mouse_ids)
            self._event_list.update_segment_status(index, "confirmed")
            if index == self._current_review_idx:
                self._annotation_panel.set_event(index, seg)
        self._statusbar.showMessage(f"颜色识别完成: 事件 #{seg.get('segment_id')}，置信度 {result.get('identity_confidence', 0):.2f}")

    def _start_batch_color_identify(self, roi_data):
        items = [(i, seg) for i, seg in enumerate(self._event_list.get_segments())
                 if seg.get("start_frame") is not None and seg.get("end_frame") is not None
                 and seg["end_frame"] >= seg["start_frame"]]
        if not items:
            QMessageBox.information(self, "颜色识别", "没有可处理的非空事件。")
            return
        progress = QProgressDialog("颜色识别: 准备中...", "取消", 0, 100, self)
        progress.setWindowTitle("全部事件颜色识别")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        self._batch_color_progress = progress
        self._batch_color_worker = BatchColorIdentifyWorker(items, roi_data["roi_core"], self._video_widget.video_path, self._video_widget.fps, self)
        self._batch_color_worker.item_started.connect(lambda idx, i, n: progress.setLabelText(f"当前 {i}/{n}: 事件 #{self._event_list.get_segment(idx).get('segment_id')}"))
        self._batch_color_worker.item_progress.connect(lambda idx, i, pct, msg: (progress.setLabelText(f"当前 {i}/{len(items)}: 事件 #{self._event_list.get_segment(idx).get('segment_id')} — {msg}"), progress.setValue(pct)))
        self._batch_color_worker.item_finished.connect(self._on_batch_color_finished)
        self._batch_color_worker.item_error.connect(self._on_batch_color_error)
        self._batch_color_worker.finished_batch.connect(self._on_batch_color_done)
        progress.canceled.connect(self._batch_color_worker.cancel)
        self._batch_color_worker.start()

    @Slot(int, dict)
    def _on_batch_color_finished(self, index, result):
        seg = self._apply_color_result(index, result)
        if seg:
            self._statusbar.showMessage(f"颜色识别完成: 事件 #{seg.get('segment_id')}，置信度 {result.get('identity_confidence', 0):.2f}")

    @Slot(int, str)
    def _on_batch_color_error(self, index, error_msg):
        label = "视频" if index < 0 else f"事件 #{self._event_list.get_segment(index).get('segment_id')}"
        self._statusbar.showMessage(f"颜色识别失败 ({label})，将继续: {error_msg.splitlines()[0]}")

    @Slot(bool)
    def _on_batch_color_done(self, cancelled):
        if self._batch_color_progress:
            self._batch_color_progress.close()
        self._statusbar.showMessage("全部事件颜色识别已取消；已完成结果已保留。" if cancelled else "全部事件颜色识别完成。")

    def _on_color_error(self, progress, error_msg):
        progress.close()
        QMessageBox.critical(self, "颜色识别错误", f"识别失败:\n{error_msg}")
        self._statusbar.showMessage(f"颜色识别失败: {error_msg}")

    # ==================================================================
    # 计数刷新 (R键)
    # ==================================================================
    def _on_export_csv(self):
        """导出 CSV (以 CountSegment 为单位)"""
        segments = self._event_list.get_segments()
        if not segments:
            QMessageBox.information(self, "提示", "还没有检测事件, 请先运行自动检测。")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "导出CSV", "mouse_segments.csv",
            "CSV 文件 (*.csv);;所有文件 (*.*)"
        )
        if not path:
            return

        try:
            video_file = self._video_widget.video_path
            fps = self._video_widget.fps
            roi_data = self._video_widget.get_roi_data()

            CsvExporter.export_segments(
                segments=segments,
                video_file=video_file,
                fps=fps,
                output_path=path,
                roi_data=roi_data,
            )
            self._statusbar.showMessage(f"CSV 已导出: {path}")
        except Exception as e:
            QMessageBox.critical(self, "导出错误", f"CSV 导出失败:\n{e}")
            self._statusbar.showMessage(f"CSV 导出失败: {e}")

    # ==================================================================
    # Markdown 完整视频时间线报表导出
    # ==================================================================
    def _on_export_markdown(self):
        """导出 Markdown 完整视频时间线报表"""
        segments = self._event_list.get_segments()
        if not segments:
            QMessageBox.information(self, "提示", "还没有检测事件, 请先运行自动检测。")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "导出Markdown时间线报表", "mouse_timeline_report.md",
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

        # 状态映射
        STATUS_MAP = {
            "pending": "待审核",
            "confirmed": "已确认",
            "rejected": "误检",
            "manual": "人工新增",
            "modified": "已修改",
        }

        # 按 start_time 排序
        sorted_segs = sorted(segments, key=lambda s: s.get("start_time", 0.0))

        # 确定视频总时长: 取 VideoWidget.duration_sec, 未知时用最后片段结束时间
        video_duration = self._video_widget.duration_sec
        if video_duration <= 0:
            video_duration = max(
                (s.get("end_time", 0.0) for s in sorted_segs), default=0.0
            )

        # ================================================================
        # Part A: 完整时间线 (所有时间段全覆盖)
        # ================================================================
        lines = []
        lines.append("# 小鼠暖点占据完整时间线报告")
        lines.append("")
        lines.append("## 完整时间线")
        lines.append("")
        lines.append("| 编号 | 开始时间 | 结束时间 | 时长(s) | 状态 | 估计数量 | 确认数量 | 小鼠 | 置信度 | 备注 |")
        lines.append("|------|----------|----------|---------|------|----------|----------|------|--------|------|")

        cursor = 0.0
        for seg in sorted_segs:
            st = seg.get("start_time", 0.0)
            et = seg.get("end_time", 0.0)

            # 前面有空隙 (>0.01s) → 插入未占据行
            if st - cursor > 0.01:
                gap_dur = st - cursor
                lines.append(
                    f"| -- | {fmt_time(cursor)} | {fmt_time(st)} | {gap_dur:.2f} "
                    f"| 未占据 | -- | -- | -- | -- | |"
                )

            # 片段行
            sid = seg.get("segment_id", "--")
            status_cn = STATUS_MAP.get(seg.get("status", "pending"), seg.get("status", "pending"))

            # 估计数量
            est_count = seg.get("estimated_mouse_count")
            est_str = str(est_count) if est_count is not None else "--"

            # 确认数量
            conf_count = seg.get("confirmed_mouse_count")
            conf_str = str(conf_count) if conf_count is not None else "--"

            # 小鼠
            mouse_ids = seg.get("mouse_ids", [])
            mouse_str = ",".join(str(m) for m in mouse_ids) if mouse_ids else "--"

            # 置信度
            confidence = seg.get("count_confidence", 0.0)
            conf_val_str = f"{confidence:.2f}" if confidence > 0 else "--"

            # 备注
            remarks = []
            if confidence > 0 and confidence < 0.5:
                remarks.append("低置信")
            if seg.get("modified_by_user"):
                remarks.append("已修改")
            note = seg.get("count_note", seg.get("note", ""))
            if note:
                remarks.append(str(note))
            remarks_str = "，".join(remarks)

            dur = seg.get("duration", seg["end_time"] - seg["start_time"])

            lines.append(
                f"| {sid} | {fmt_time(st)} | {fmt_time(et)} | {dur:.2f} | {status_cn} | "
                f"{est_str} | {conf_str} | {mouse_str} | {conf_val_str} | {remarks_str} |"
            )

            cursor = et

        # 结尾空隙
        if video_duration - cursor > 0.01:
            gap_dur = video_duration - cursor
            lines.append(
                f"| -- | {fmt_time(cursor)} | {fmt_time(video_duration)} | {gap_dur:.2f} "
                f"| 未占据 | -- | -- | -- | -- | 视频结束 |"
            )

        lines.append("")

        # ================================================================
        # Part B: 小鼠汇总 (仅 confirmed 状态片段)
        # ================================================================
        confirmed = [s for s in sorted_segs if s.get("status") == "confirmed"]

        mouse_stats: dict[int, dict] = {}
        for seg in confirmed:
            dur = seg.get("duration", seg["end_time"] - seg["start_time"])
            for mid in seg.get("mouse_ids", []):
                if mid not in mouse_stats:
                    mouse_stats[mid] = {"total_duration": 0.0, "event_count": 0}
                mouse_stats[mid]["total_duration"] += dur
                mouse_stats[mid]["event_count"] += 1

        lines.append("## 小鼠汇总 (仅已确认)")
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

        # ================================================================
        # Part C: 暖点汇总
        # ================================================================
        total_occupied = sum(
            seg.get("duration", seg["end_time"] - seg["start_time"])
            for seg in sorted_segs
        )
        total_unoccupied = video_duration - total_occupied
        total_events = len(sorted_segs)
        multi_mouse = sum(1 for seg in sorted_segs if len(seg.get("mouse_ids", [])) > 1)
        confirmed_count = len(confirmed)

        lines.append("## 暖点汇总")
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("|------|------|")
        lines.append(f"| 视频总时长 | {fmt_time(video_duration)} |")
        lines.append(f"| 总占据时长 | {total_occupied:.2f}s |")
        lines.append(f"| 总未占据时长 | {total_unoccupied:.2f}s |")
        lines.append(f"| 全部事件数 | {total_events} |")
        lines.append(f"| 已确认事件数 | {confirmed_count} |")
        lines.append(f"| 多鼠事件数 | {multi_mouse} |")
        lines.append("")

        # ================================================================
        # Part D: 校准摘要
        # ================================================================
        calib = self._calib_store
        has_calib = calib is not None and (
            calib.count(0) > 0
            or any(calib.count(c) > 0 for c in range(1, 3))
        )

        lines.append("## 校准摘要")
        lines.append("")

        if not has_calib:
            lines.append("未校准")
        else:
            lines.append("| 样本类型 | 样本数 | 有效数 | 参考面积 (px²) |")
            lines.append("|----------|--------|--------|-----------------|")
            for count in range(3):
                n = calib.count(count)
                samples = calib.get_samples(count)
                v = sum(1 for s in samples if s.valid)

                # 参考面积: 有效样本面积的中位数
                valid_areas = [
                    s.measured_area
                    for s in samples
                    if s.valid and s.measured_area is not None and s.measured_area > 0
                ]
                if valid_areas:
                    ref_area = sorted(valid_areas)[len(valid_areas) // 2]
                    area_str = f"{ref_area:.0f}"
                else:
                    area_str = "--"

                label = "背景 (0只)" if count == 0 else f"{count} 只小鼠"
                lines.append(f"| {label} | {n} | {v} | {area_str} |")

        lines.append("")

        # 写入文件
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            self._statusbar.showMessage(f"时间线报表已导出: {path}")
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
        self._btn_color.setEnabled(has_video)
        self._btn_edit_mode.setEnabled(has_video)
        # 刷新按钮: 无背景也可刷新 (用 fallback)
        for c, btn in self._calib_btns.items():
            btn.setEnabled(has_video and c <= 2)  # cap=2: [3]/[4] 永久禁用
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

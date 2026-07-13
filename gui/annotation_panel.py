"""
标注面板
实现规范 6.9 节：人工标注面板
支持多选小鼠编号 (1/2/3/4)、标记确认/误检、保存、快捷键绑定
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QKeyEvent
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QSizePolicy,
)


class AnnotationPanel(QWidget):
    """标注面板"""

    # 信号
    annotation_changed = Signal(int, list)    # event_idx, mouse_ids
    review_action = Signal(str, int)          # action_name, event_idx
    # action_name: "confirm", "reject", "save", "save_next", "clear"
    count_confirmed = Signal(int, int)        # event_idx, count (1-2)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(260)

        # 当前状态
        self._current_event_idx: int = -1
        self._mouse_ids: list[int] = []
        self._mouse_buttons: dict[int, QPushButton] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # 标题
        title = QLabel("标注面板")
        title.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #eee;")
        layout.addWidget(title)

        # 分隔线
        layout.addWidget(self._sep())

        # 事件信息区
        self._lbl_event_id = QLabel("事件: --")
        self._lbl_event_id.setStyleSheet("color: #ccc; font-size: 12px;")
        layout.addWidget(self._lbl_event_id)

        self._lbl_time_range = QLabel("时间: --")
        self._lbl_time_range.setStyleSheet("color: #ccc; font-size: 11px; font-family: monospace;")
        layout.addWidget(self._lbl_time_range)

        self._lbl_duration = QLabel("时长: --")
        self._lbl_duration.setStyleSheet("color: #ccc; font-size: 11px;")
        layout.addWidget(self._lbl_duration)

        self._lbl_status = QLabel("状态: --")
        self._lbl_status.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(self._lbl_status)

        self._lbl_mouse = QLabel("小鼠: --")
        self._lbl_mouse.setStyleSheet("color: #ffcc44; font-size: 12px; font-weight: bold;")
        layout.addWidget(self._lbl_mouse)

        # 分隔线
        layout.addWidget(self._sep())

        # 小鼠选择区
        lbl_select = QLabel("小鼠标注 (多选, toggle):")
        lbl_select.setStyleSheet("color: #aaa; font-size: 10px;")
        layout.addWidget(lbl_select)

        mouse_row = QHBoxLayout()
        for mid in range(1, 5):
            btn = QPushButton(f"鼠 {mid}")
            btn.setCheckable(True)
            btn.setMinimumHeight(36)
            btn.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
            btn.setToolTip(f"快捷键: {mid}")
            btn.clicked.connect(lambda checked, m=mid: self._on_mouse_toggle(m))
            btn.setStyleSheet(self._mouse_btn_style(mid, False))
            self._mouse_buttons[mid] = btn
            mouse_row.addWidget(btn)
        layout.addLayout(mouse_row)

        # 分隔线
        layout.addWidget(self._sep())

        # ---- 数量确认区 (改进.md 11.3, 13 节; cap=2) ----
        lbl_count = QLabel("确认数量 (Shift+1~2):")
        lbl_count.setStyleSheet("color: #aaa; font-size: 10px;")
        layout.addWidget(lbl_count)

        count_row = QHBoxLayout()
        self._count_buttons: dict[int, QPushButton] = {}
        for c in range(1, 3):
            btn = QPushButton(f"{c} 只")
            btn.setCheckable(False)
            btn.setMinimumHeight(30)
            btn.setFont(QFont("Microsoft YaHei", 10))
            btn.setToolTip(f"确认数量为 {c} 只 (Shift+{c})")
            btn.clicked.connect(lambda checked, cnt=c: self._on_count_confirm(cnt))
            btn.setStyleSheet(self._count_btn_style())
            self._count_buttons[c] = btn
            count_row.addWidget(btn)
        layout.addLayout(count_row)

        self._lbl_est_count = QLabel("估计数量: --")
        self._lbl_est_count.setStyleSheet("color: #aaa; font-size: 10px;")
        layout.addWidget(self._lbl_est_count)

        self._lbl_count_conf = QLabel("计数置信度: --")
        self._lbl_count_conf.setStyleSheet("color: #aaa; font-size: 10px;")
        layout.addWidget(self._lbl_count_conf)

        # 分隔线
        layout.addWidget(self._sep())

        # ---- 身份识别结果显示区 (颜色识别) ----
        self._lbl_identity_title = QLabel("颜色识别结果:")
        self._lbl_identity_title.setStyleSheet("color: #aaa; font-size: 10px;")
        layout.addWidget(self._lbl_identity_title)

        self._lbl_auto_colors = QLabel("候选颜色: --")
        self._lbl_auto_colors.setStyleSheet("color: #88ccff; font-size: 11px; font-family: monospace;")
        layout.addWidget(self._lbl_auto_colors)

        self._lbl_auto_ids = QLabel("自动ID: --")
        self._lbl_auto_ids.setStyleSheet("color: #88ccff; font-size: 10px; font-family: monospace;")
        layout.addWidget(self._lbl_auto_ids)

        self._lbl_id_conf = QLabel("识别置信度: --")
        self._lbl_id_conf.setStyleSheet("color: #aaa; font-size: 10px;")
        layout.addWidget(self._lbl_id_conf)

        self._lbl_id_method = QLabel("识别方法: --")
        self._lbl_id_method.setStyleSheet("color: #666; font-size: 9px;")
        layout.addWidget(self._lbl_id_method)

        self._lbl_id_conflict = QLabel("")
        self._lbl_id_conflict.setStyleSheet("color: #ff6644; font-size: 11px; font-weight: bold;")
        self._lbl_id_conflict.setVisible(False)
        layout.addWidget(self._lbl_id_conflict)

        self._lbl_id_review = QLabel("")
        self._lbl_id_review.setStyleSheet("color: #ffcc44; font-size: 10px;")
        self._lbl_id_review.setVisible(False)
        layout.addWidget(self._lbl_id_review)

        # 分隔线
        layout.addWidget(self._sep())

        # 操作按钮区
        btn_row1 = QHBoxLayout()

        self._btn_clear = QPushButton("清空 (C)")
        self._btn_clear.setToolTip("清空当前小鼠标注")
        self._btn_clear.clicked.connect(self._on_clear)
        self._btn_clear.setMinimumHeight(30)
        self._btn_clear.setStyleSheet(self._action_btn_style("#666"))
        btn_row1.addWidget(self._btn_clear)

        self._btn_save = QPushButton("保存 (S)")
        self._btn_save.setToolTip("保存当前标注 (S 键)")
        self._btn_save.clicked.connect(self._on_save)
        self._btn_save.setMinimumHeight(30)
        self._btn_save.setStyleSheet(self._action_btn_style("#336699"))
        btn_row1.addWidget(self._btn_save)

        layout.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()

        self._btn_confirm = QPushButton("[v] 确认 (Enter)")
        self._btn_confirm.setToolTip("标记确认并跳到下一个事件 (Enter)")
        self._btn_confirm.clicked.connect(self._on_confirm)
        self._btn_confirm.setMinimumHeight(30)
        self._btn_confirm.setStyleSheet(self._action_btn_style("#338833"))
        btn_row2.addWidget(self._btn_confirm)

        layout.addLayout(btn_row2)

        btn_row3 = QHBoxLayout()

        self._btn_reject = QPushButton("[x] 误检 (X)")
        self._btn_reject.setToolTip("标记误检")
        self._btn_reject.clicked.connect(self._on_reject)
        self._btn_reject.setMinimumHeight(30)
        self._btn_reject.setStyleSheet(self._action_btn_style("#883333"))
        btn_row3.addWidget(self._btn_reject)

        self._btn_save_next = QPushButton("保存并下一个")
        self._btn_save_next.setToolTip("保存标注并跳到下一个 pending 事件")
        self._btn_save_next.clicked.connect(self._on_save_next)
        self._btn_save_next.setMinimumHeight(30)
        self._btn_save_next.setStyleSheet(self._action_btn_style("#3366aa"))
        btn_row3.addWidget(self._btn_save_next)

        layout.addLayout(btn_row3)

        layout.addStretch()

    # ------------------------------------------------------------------
    # 数据处理
    # ------------------------------------------------------------------
    def set_event(self, event_idx: int, event_data: dict | None):
        """
        设置当前事件
        :param event_idx: 事件索引
        :param event_data: 事件数据 dict
        """
        self._current_event_idx = event_idx

        if event_data is None:
            self._lbl_event_id.setText("事件: --")
            self._lbl_time_range.setText("时间: --")
            self._lbl_duration.setText("时长: --")
            self._lbl_status.setText("状态: --")
            self._lbl_mouse.setText("小鼠: --")
            self._lbl_est_count.setText("估计数量: --")
            self._lbl_count_conf.setText("计数置信度: --")
            self._lbl_count_conf.setStyleSheet("color: #aaa; font-size: 10px;")
            self._lbl_auto_colors.setText("候选颜色: --")
            self._lbl_auto_ids.setText("自动ID: --")
            self._lbl_id_conf.setText("识别置信度: --")
            self._lbl_id_method.setText("识别方法: --")
            self._lbl_id_conflict.setText("")
            self._lbl_id_conflict.setVisible(False)
            self._lbl_id_review.setText("")
            self._lbl_id_review.setVisible(False)
            self._set_mouse_ids([])
            for btn in self._count_buttons.values():
                btn.setEnabled(False)
            return

        seg_id = event_data.get("segment_id", "--")
        start = event_data.get("start_time", 0.0)
        end = event_data.get("end_time", 0.0)
        duration = event_data.get("duration", 0.0)
        status = event_data.get("status", "pending")
        mouse_ids = event_data.get("mouse_ids", [])

        self._lbl_event_id.setText(f"事件: #{seg_id}")
        self._lbl_time_range.setText(f"时间: {self._fmt_time(start)} - {self._fmt_time(end)}")
        self._lbl_duration.setText(f"时长: {duration:.2f} 秒")

        status_labels = {
            "pending": "未审核 (黄色)", "confirmed": "已确认 (绿色)",
            "rejected": "误检 (红色)", "manual": "人工新增 (蓝色)", "modified": "已修改 (紫色)"
        }
        self._lbl_status.setText(f"状态: {status_labels.get(status, status)}")

        # 更新计数信息 (改进.md 11.3 节)
        est_count = event_data.get("estimated_mouse_count")
        self._lbl_est_count.setText(f"估计数量: {est_count if est_count is not None else '--'}")
        count_conf = event_data.get("count_confidence", 0.0)
        if count_conf > 0:
            if count_conf < 0.7:
                self._lbl_count_conf.setText(f"计数置信度: {count_conf:.2f} [!] 低")
                self._lbl_count_conf.setStyleSheet("color: #ff8800; font-size: 10px;")
            else:
                self._lbl_count_conf.setText(f"计数置信度: {count_conf:.2f}")
                self._lbl_count_conf.setStyleSheet("color: #44cc44; font-size: 10px;")
        else:
            self._lbl_count_conf.setText("计数置信度: --")
            self._lbl_count_conf.setStyleSheet("color: #aaa; font-size: 10px;")

        # 按钮启用状态
        has_seg = event_idx >= 0
        for btn in self._count_buttons.values():
            btn.setEnabled(has_seg)

        self._set_mouse_ids(mouse_ids)

        # ---- 身份识别结果显示 ----
        auto_colors = event_data.get("auto_mouse_colors", [])
        auto_ids = event_data.get("auto_mouse_ids", [])
        id_conf = event_data.get("identity_confidence", 0.0)
        id_method = event_data.get("identity_method", "")
        id_conflict = event_data.get("identity_conflict", False)
        id_needs_review = event_data.get("identity_needs_review", False)

        if auto_colors:
            self._lbl_auto_colors.setText(f"候选颜色: {', '.join(auto_colors)}")
            if id_conf > 0.6:
                self._lbl_auto_colors.setStyleSheet("color: #44cc44; font-size: 11px; font-family: monospace;")
            elif id_conf > 0.3:
                self._lbl_auto_colors.setStyleSheet("color: #88ccff; font-size: 11px; font-family: monospace;")
            else:
                self._lbl_auto_colors.setStyleSheet("color: #ffcc44; font-size: 11px; font-family: monospace;")
        else:
            self._lbl_auto_colors.setText("候选颜色: --")
            self._lbl_auto_colors.setStyleSheet("color: #888; font-size: 11px; font-family: monospace;")

        if auto_ids:
            self._lbl_auto_ids.setText(f"自动ID: {', '.join(auto_ids)}")
        else:
            self._lbl_auto_ids.setText("自动ID: --")

        if id_conf > 0:
            if id_conf >= 0.7:
                self._lbl_id_conf.setText(f"识别置信度: {id_conf:.2f} (高)")
                self._lbl_id_conf.setStyleSheet("color: #44cc44; font-size: 10px;")
            elif id_conf >= 0.4:
                self._lbl_id_conf.setText(f"识别置信度: {id_conf:.2f} (中)")
                self._lbl_id_conf.setStyleSheet("color: #ffcc44; font-size: 10px;")
            else:
                self._lbl_id_conf.setText(f"识别置信度: {id_conf:.2f} (低)")
                self._lbl_id_conf.setStyleSheet("color: #ff6644; font-size: 10px;")
        else:
            if "thermometer_detected" in id_method:
                self._lbl_id_conf.setText("识别置信度: 0.00 (测温器干扰)")
                self._lbl_id_conf.setStyleSheet("color: #ff6644; font-size: 10px;")
            else:
                self._lbl_id_conf.setText("识别置信度: --")
                self._lbl_id_conf.setStyleSheet("color: #aaa; font-size: 10px;")

        if id_method:
            self._lbl_id_method.setText(f"识别方法: {id_method}")
        else:
            self._lbl_id_method.setText("识别方法: --")

        if id_conflict:
            self._lbl_id_conflict.setText("[!] 身份冲突 — 存在多种颜色候选")
            self._lbl_id_conflict.setVisible(True)
        else:
            self._lbl_id_conflict.setText("")
            self._lbl_id_conflict.setVisible(False)

        if id_needs_review:
            self._lbl_id_review.setText("[!] 检测到测温器/探头，请人工复核" if "thermometer_detected" in id_method else "[!] 建议人工审核")
            self._lbl_id_review.setVisible(True)
        else:
            self._lbl_id_review.setText("")
            self._lbl_id_review.setVisible(False)

    def _set_mouse_ids(self, mouse_ids: list[int]):
        """设置小鼠选择状态"""
        self._mouse_ids = sorted(set(mouse_ids)) if mouse_ids else []

        # 更新按钮状态
        for mid in range(1, 5):
            btn = self._mouse_buttons[mid]
            is_checked = mid in self._mouse_ids
            btn.setChecked(is_checked)
            btn.setStyleSheet(self._mouse_btn_style(mid, is_checked))

        # 更新标签
        if self._mouse_ids:
            self._lbl_mouse.setText(f"小鼠: {'; '.join(str(m) for m in self._mouse_ids)}")
            self._lbl_mouse.setStyleSheet("color: #ffcc44; font-size: 12px; font-weight: bold;")
        else:
            self._lbl_mouse.setText("小鼠: --")
            self._lbl_mouse.setStyleSheet("color: #aaa; font-size: 12px;")

    def get_mouse_ids(self) -> list[int]:
        """获取当前小鼠标注"""
        return list(self._mouse_ids)

    # ------------------------------------------------------------------
    # 交互回调
    # ------------------------------------------------------------------
    def _on_mouse_toggle(self, mouse_id: int):
        """切换小鼠选择"""
        if mouse_id in self._mouse_ids:
            self._mouse_ids.remove(mouse_id)
        else:
            self._mouse_ids.append(mouse_id)
            self._mouse_ids.sort()

        # 更新按钮样式
        for mid in range(1, 5):
            btn = self._mouse_buttons[mid]
            btn.setStyleSheet(self._mouse_btn_style(mid, mid in self._mouse_ids))

        if self._mouse_ids:
            self._lbl_mouse.setText(f"小鼠: {'; '.join(str(m) for m in self._mouse_ids)}")
            self._lbl_mouse.setStyleSheet("color: #ffcc44; font-size: 12px; font-weight: bold;")
        else:
            self._lbl_mouse.setText("小鼠: --")
            self._lbl_mouse.setStyleSheet("color: #aaa; font-size: 12px;")

        if self._current_event_idx >= 0:
            self.annotation_changed.emit(self._current_event_idx, list(self._mouse_ids))

    def _on_clear(self):
        """清空选择"""
        self._set_mouse_ids([])
        if self._current_event_idx >= 0:
            self.annotation_changed.emit(self._current_event_idx, [])
        self.review_action.emit("clear", self._current_event_idx)

    def _on_save(self):
        """保存当前标注"""
        if self._current_event_idx >= 0:
            self.review_action.emit("save", self._current_event_idx)

    def _on_save_next(self):
        """保存并跳到下一个 pending"""
        if self._current_event_idx >= 0:
            self.review_action.emit("save_next", self._current_event_idx)

    def _on_confirm(self):
        """标记确认"""
        if self._current_event_idx >= 0:
            self.review_action.emit("confirm", self._current_event_idx)

    def _on_reject(self):
        """标记误检"""
        if self._current_event_idx >= 0:
            self.review_action.emit("reject", self._current_event_idx)

    def _on_count_confirm(self, count: int):
        """确认数量 (Shift+1~2) (改进.md 11.3 节)"""
        if self._current_event_idx >= 0:
            self.count_confirmed.emit(self._current_event_idx, count)

    # ------------------------------------------------------------------
    # 样式
    # ------------------------------------------------------------------
    @staticmethod
    def _sep():
        """分隔线"""
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #555;")
        return sep

    @staticmethod
    def _mouse_btn_style(mouse_id: int, checked: bool) -> str:
        """小鼠标注按钮样式"""
        colors = {
            1: "#4488cc",
            2: "#44aa66",
            3: "#cc8844",
            4: "#aa4488",
        }
        c = colors.get(mouse_id, "#888")
        if checked:
            return f"""
                QPushButton {{
                    background-color: {c}; color: #fff;
                    border: 2px solid #fff; border-radius: 6px;
                    font-weight: bold;
                }}
                QPushButton:hover {{ background-color: {c}; opacity: 0.8; }}
            """
        else:
            return f"""
                QPushButton {{
                    background-color: #3a3a3a; color: #aaa;
                    border: 1px solid #666; border-radius: 6px;
                }}
                QPushButton:hover {{ background-color: {c}; color: #fff; }}
            """

    @staticmethod
    def _action_btn_style(color: str) -> str:
        return f"""
            QPushButton {{
                background-color: {color}; color: #eee;
                border: 1px solid #666; border-radius: 4px;
                padding: 4px 8px;
            }}
            QPushButton:hover {{ background-color: {color}; border-color: #999; }}
        """

    @staticmethod
    def _count_btn_style() -> str:
        """数量确认按钮样式 (统一暗色)"""
        return """
            QPushButton {
                background-color: #3a3a3a; color: #ccc;
                border: 1px solid #555; border-radius: 3px;
                padding: 4px 12px; font-size: 12px;
            }
            QPushButton:checked {
                background-color: #555; color: #fff;
                border: 1px solid #888;
            }
            QPushButton:hover { background-color: #4a4a4a; }
        """

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        """格式化时间为 MM:SS.ms"""
        if seconds < 0:
            seconds = 0.0
        m = int(seconds // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 100)
        return f"{m:02d}:{s:02d}.{ms:02d}"

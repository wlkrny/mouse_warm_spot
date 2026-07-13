"""
事件列表控件
实现规范 6.8 节：候选事件列表
显示自动检测生成的事件, 支持点击跳转、右键菜单、颜色标记状态
"""

from PySide6.QtCore import QEvent, Qt, Signal, QPoint
from PySide6.QtGui import QColor, QBrush, QFont, QAction
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QMenu, QAbstractItemView, QLabel,
)


class EventListWidget(QWidget):
    """事件列表控件"""

    # 信号
    segment_selected = Signal(int)             # segment_idx
    segment_status_changed = Signal(int, str)  # segment_idx, new_status
    segments_merged = Signal(int, int)         # source_idx, target_idx
    segment_split_requested = Signal(int, int) # segment_idx, frame_idx

    # 列定义 (添加计数相关列, 改进.md 11.2 节)
    COL_ID = 0
    COL_START = 1
    COL_END = 2
    COL_DURATION = 3
    COL_EST_COUNT = 4      # 估计数量
    COL_CONF_COUNT = 5     # 确认数量
    COL_MOUSE = 6          # 小鼠编号
    COL_CONFIDENCE = 7     # 数量置信度
    COL_STATUS = 8         # 状态
    COL_COUNT = 9

    HEADERS = ["#", "开始时间", "结束时间", "时长(s)", "估计数量", "确认数量", "小鼠", "置信度", "状态"]

    # 状态颜色映射
    STATUS_COLORS = {
        "pending": QColor(180, 180, 40),    # 黄色 - 未审核
        "confirmed": QColor(60, 180, 60),   # 绿色 - 已确认
        "rejected": QColor(220, 60, 60),    # 红色 - 误检
        "manual": QColor(60, 120, 220),     # 蓝色 - 人工新增
        "modified": QColor(160, 60, 200),   # 紫色 - 人工修改
    }

    STATUS_LABELS = {
        "pending": "未审核",
        "confirmed": "已确认",
        "rejected": "误检",
        "manual": "人工新增",
        "modified": "已修改",
    }

    # 计数状态颜色映射 (改进.md 11.1 节)
    COUNT_STATUS_COLORS = {
        "pending": QColor(180, 180, 40),    # 浅黄
        "confirmed": QColor(60, 180, 60),   # 绿色
        "rejected": QColor(220, 60, 60),    # 红色
        "manual": QColor(60, 120, 220),     # 蓝色
        "modified": QColor(160, 60, 200),   # 紫色
    }

    # 橙色: 计数低置信度 (改进.md 11.1 节)
    LOW_CONFIDENCE_COLOR = QColor(255, 140, 0)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 150)

        # 数据存储: list[dict]
        self._segments: list[dict] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # 标题
        title = QLabel("事件列表")
        title.setStyleSheet("color: #ddd; font-size: 13px; font-weight: bold;")
        layout.addWidget(title)

        # 表格
        self._table = QTableWidget()
        self._table.setColumnCount(self.COL_COUNT)
        self._table.setHorizontalHeaderLabels(self.HEADERS)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setContextMenuPolicy(Qt.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._on_context_menu)
        # QTableWidget 会在内部消费按键；通过事件过滤器将非导航快捷键
        # 交给 MainWindow，确保选中事件列表后 C / Shift+数字等仍然有效。
        self._table.installEventFilter(self)

        # 表头自适应
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(self.COL_ID, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.COL_START, QHeaderView.Stretch)
        header.setSectionResizeMode(self.COL_END, QHeaderView.Stretch)
        header.setSectionResizeMode(self.COL_DURATION, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.COL_EST_COUNT, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.COL_CONF_COUNT, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.COL_MOUSE, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.COL_CONFIDENCE, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.COL_STATUS, QHeaderView.ResizeToContents)

        # 颜色交替行 (暗色主题下覆盖系统默认白色交替色)
        from PySide6.QtGui import QPalette
        pal = self._table.palette()
        pal.setColor(QPalette.Base, QColor(0x2a, 0x2a, 0x2a))
        pal.setColor(QPalette.AlternateBase, QColor(0x33, 0x33, 0x33))
        pal.setColor(QPalette.Text, QColor(0xdd, 0xdd, 0xdd))
        self._table.setPalette(pal)

        # 样式
        self._table.setStyleSheet("""
            QTableWidget {
                background-color: #2a2a2a;
                color: #ddd;
                gridline-color: #444;
                border: 1px solid #555;
                font-size: 11px;
            }
            QTableWidget::item:selected {
                background-color: #336699;
            }
            QHeaderView::section {
                background-color: #3a3a3a;
                color: #ccc;
                border: 1px solid #555;
                padding: 3px;
                font-weight: bold;
            }
        """)

        # 信号连接 — 使用 currentCellChanged 统一处理鼠标点击和键盘上下箭头导航
        self._table.itemDoubleClicked.connect(self._on_double_click)
        self._table.currentCellChanged.connect(self._on_current_cell_changed)

        layout.addWidget(self._table)

    # ------------------------------------------------------------------
    # 数据操作
    # ------------------------------------------------------------------
    def set_events(self, events: list[dict]):
        """
        设置事件列表数据 (兼容 OccupancyEpisode 和 CountSegment 格式)
        自动检测并适配两种格式
        :param events: 事件或子片段列表
        """
        self._segments = []
        for i, evt in enumerate(events):
            seg = dict(evt)  # 浅拷贝保留所有原始字段
            # 确保 UI 需要的字段有默认值
            seg.setdefault("segment_id", i + 1)
            seg.setdefault("start_time", evt.get("start_time_sec", 0.0))
            seg.setdefault("end_time", evt.get("end_time_sec", 0.0))
            seg.setdefault("duration", evt.get("duration_sec", 0.0))
            seg.setdefault("review_start_frame", evt.get("start_frame", 0))
            seg.setdefault("review_end_frame", evt.get("end_frame", 0))
            seg.setdefault("review_start_time", evt.get("start_time", 0.0))
            seg.setdefault("review_end_time", evt.get("end_time", 0.0))
            seg.setdefault("status", evt.get("count_status", "pending"))
            seg.setdefault("confidence", evt.get("count_confidence", 0.0))
            seg.setdefault("is_short_event", False)
            seg.setdefault("estimated_mouse_count", None)
            seg.setdefault("confirmed_mouse_count", None)
            seg.setdefault("detected_by", "auto")
            seg.setdefault("modified_by_user", False)
            self._segments.append(seg)

        self._refresh_table()

    def get_segments(self) -> list[dict]:
        """获取所有片段数据"""
        return self._segments

    def get_segment(self, index: int) -> dict | None:
        """获取指定索引的片段"""
        if 0 <= index < len(self._segments):
            return self._segments[index]
        return None

    def update_segment_status(self, index: int, status: str):
        """更新指定片段的状态"""
        if 0 <= index < len(self._segments):
            self._segments[index]["status"] = status
            self._update_row(index)
            self.segment_status_changed.emit(index, status)

    def update_segment_mouse_ids(self, index: int, mouse_ids: list):
        """更新指定片段的小鼠编号"""
        if 0 <= index < len(self._segments):
            self._segments[index]["mouse_ids"] = sorted(mouse_ids)
            self._update_row(index)

    def update_segment_boundary(self, index: int, **kwargs):
        """更新指定片段的时间边界"""
        if 0 <= index < len(self._segments):
            seg = self._segments[index]
            for key, value in kwargs.items():
                seg[key] = value
            self._refresh_table()

    def merge_segments(self, idx1: int, idx2: int):
        """合并两个相邻片段"""
        if idx1 == idx2:
            return
        i1, i2 = min(idx1, idx2), max(idx1, idx2)
        if i2 >= len(self._segments):
            return
        s1 = self._segments[i1]
        s2 = self._segments[i2]

        # 合并
        merged = dict(s1)
        merged["end_time"] = s2["end_time"]
        merged["end_frame"] = s2["end_frame"]
        merged["review_end_time"] = s2["review_end_time"]
        merged["review_end_frame"] = s2["review_end_frame"]
        merged["duration"] = merged["end_time"] - merged["start_time"]
        merged["confidence"] = max(s1["confidence"], s2["confidence"])
        merged["avg_occ_ratio"] = (s1["avg_occ_ratio"] + s2["avg_occ_ratio"]) / 2.0
        merged["max_occ_ratio"] = max(s1["max_occ_ratio"], s2["max_occ_ratio"])
        merged["status"] = "modified"
        # 合并小鼠编号
        merged["mouse_ids"] = sorted(set(s1.get("mouse_ids", []) + s2.get("mouse_ids", [])))
        merged["status"] = "modified" if s1["status"] != "pending" else "pending"

        self._segments[i1] = merged
        del self._segments[i2]
        self._reindex()
        self._refresh_table()
        self.segments_merged.emit(i1, i2)

    def delete_segment(self, index: int):
        """删除指定片段"""
        if 0 <= index < len(self._segments):
            del self._segments[index]
            self._reindex()
            self._refresh_table()

    def clear(self):
        """清空所有事件"""
        self._segments = []
        self._table.setRowCount(0)

    # ------------------------------------------------------------------
    # 表格刷新
    # ------------------------------------------------------------------
    def _refresh_table(self):
        """完全重建表格"""
        self._table.setRowCount(len(self._segments))
        for row_idx, seg in enumerate(self._segments):
            self._update_row(row_idx)

    def _update_row(self, row_idx: int):
        """更新单行 — Phase 6: 行颜色状态"""
        if row_idx < 0 or row_idx >= len(self._segments):
            return
        seg = self._segments[row_idx]

        # ---- Phase 6: 行背景颜色优先级 ----
        status = seg.get("count_status", seg.get("status", "pending"))
        needs_review = seg.get("needs_review", False)
        identity_conflict = seg.get("identity_conflict", False)
        is_false_positive = seg.get("is_possible_false_positive", False)
        est_count = seg.get("estimated_mouse_count")
        count_confidence = seg.get("count_confidence", 0.0)

        # 行颜色判定 (优先级从高到低)
        if status == "rejected" or identity_conflict:
            row_color = QColor(220, 60, 60)     # 红: 误检/冲突
        elif count_confidence > 0 and count_confidence < 0.5:
            row_color = QColor(255, 140, 0)     # 橙: 低置信
        elif needs_review:
            row_color = QColor(255, 200, 50)    # 黄: 待审核/需审核
        elif status == "confirmed":
            row_color = QColor(60, 180, 60)     # 绿: 已确认
        elif status == "modified":
            row_color = QColor(160, 60, 200)    # 紫: 已修改
        elif status == "manual":
            row_color = QColor(60, 120, 220)    # 蓝: 人工新增
        elif est_count == 0:
            row_color = QColor(150, 150, 150)   # 灰: count=0
        else:
            row_color = QColor(180, 180, 40)    # 默认: 黄/待确认

        # Phase 6: 设置行前景色
        for col in range(self.COL_COUNT):
            item = self._table.item(row_idx, col)
            if item is not None:
                item.setForeground(QBrush(QColor(0xdd, 0xdd, 0xdd)))

        # Phase 6: 行背景提示色 (通过状态列展示)
        color = row_color

        # 工具提示: 显示额外标记
        tooltip_parts = []
        if seg.get("needs_review"):
            tooltip_parts.append("需审核")
        if "thermometer_detected" in seg.get("identity_method", ""):
            tooltip_parts.append("检测到测温器/探头：身份置信度已置零")
        if seg.get("identity_conflict"):
            tooltip_parts.append("身份冲突")
        if seg.get("is_possible_false_positive"):
            tooltip_parts.append("可能误检")
        if seg.get("is_short_event"):
            tooltip_parts.append("短事件")
        tooltip = "\n".join(tooltip_parts) if tooltip_parts else ""

        # ID
        id_item = QTableWidgetItem(str(seg["segment_id"]))
        id_item.setTextAlignment(Qt.AlignCenter)
        if tooltip:
            id_item.setToolTip(tooltip)
        self._table.setItem(row_idx, self.COL_ID, id_item)

        # 开始时间 (MM:SS.ms)
        start_item = QTableWidgetItem(self._fmt_time(seg["start_time"]))
        start_item.setTextAlignment(Qt.AlignCenter)
        self._table.setItem(row_idx, self.COL_START, start_item)

        # 结束时间
        end_item = QTableWidgetItem(self._fmt_time(seg["end_time"]))
        end_item.setTextAlignment(Qt.AlignCenter)
        self._table.setItem(row_idx, self.COL_END, end_item)

        # 时长
        dur_item = QTableWidgetItem(f"{seg['duration']:.2f}")
        dur_item.setTextAlignment(Qt.AlignCenter)
        # Phase 3: 短事件橙色标记
        if seg.get("is_short_event"):
            dur_item.setForeground(QBrush(self.LOW_CONFIDENCE_COLOR))
            font = dur_item.font()
            font.setBold(True)
            dur_item.setFont(font)
        self._table.setItem(row_idx, self.COL_DURATION, dur_item)

        # 估计数量 (改进.md 11.2 节)
        est_count = seg.get("estimated_mouse_count")
        if est_count is not None:
            est_item = QTableWidgetItem(str(est_count))
        else:
            est_item = QTableWidgetItem("--")
        est_item.setTextAlignment(Qt.AlignCenter)
        self._table.setItem(row_idx, self.COL_EST_COUNT, est_item)

        # 确认数量
        conf_count = seg.get("confirmed_mouse_count")
        if conf_count is not None:
            conf_count_item = QTableWidgetItem(str(conf_count))
        else:
            conf_count_item = QTableWidgetItem("--")
        conf_count_item.setTextAlignment(Qt.AlignCenter)
        self._table.setItem(row_idx, self.COL_CONF_COUNT, conf_count_item)

        # 小鼠编号
        mouse_str = "; ".join(str(mid) for mid in seg.get("mouse_ids", [])) or "--"
        mouse_item = QTableWidgetItem(mouse_str)
        mouse_item.setTextAlignment(Qt.AlignCenter)
        if not seg.get("mouse_ids"):
            mouse_item.setForeground(QBrush(QColor(120, 120, 120)))
        self._table.setItem(row_idx, self.COL_MOUSE, mouse_item)

        # 数量置信度 (改进.md 11.2 节: 低置信度用橙色标记)
        if count_confidence > 0:
            conf_val_item = QTableWidgetItem(f"{count_confidence:.2f}")
        else:
            conf_val_item = QTableWidgetItem("--")
        conf_val_item.setTextAlignment(Qt.AlignCenter)
        if 0 < count_confidence < 0.7:
            conf_val_item.setForeground(QBrush(self.LOW_CONFIDENCE_COLOR))
        self._table.setItem(row_idx, self.COL_CONFIDENCE, conf_val_item)

        # 状态 — Phase 6: 使用 row_color 统一显示
        count_status = seg.get("count_status")
        display_status = seg.get("count_status", seg.get("status", "pending"))
        if display_status in ("confirmed", "rejected"):
            status_text = "已审核"
        else:
            status_text = "未审核"
        status_item = QTableWidgetItem(status_text)
        status_item.setTextAlignment(Qt.AlignCenter)
        status_item.setForeground(QBrush(color))
        if tooltip:
            status_item.setToolTip(tooltip)
        self._table.setItem(row_idx, self.COL_STATUS, status_item)

    # ------------------------------------------------------------------
    # 交互
    # ------------------------------------------------------------------
    def _on_double_click(self, item: QTableWidgetItem):
        """双击跳转到事件 (兼容保留)"""
        row = item.row()
        if 0 <= row < len(self._segments):
            self._table.selectRow(row)
            self.segment_selected.emit(row)

    def _on_current_cell_changed(self, currentRow: int, currentColumn: int,
                                  previousRow: int, previousColumn: int):
        """当前行变化时触发 — 统一处理鼠标点击和键盘上下箭头导航"""
        if currentRow >= 0 and currentRow < len(self._segments) and currentRow != previousRow:
            self.segment_selected.emit(currentRow)

    def select_row(self, index: int):
        """选中指定行"""
        if 0 <= index < self._table.rowCount():
            self._table.selectRow(index)
            self._table.scrollToItem(self._table.item(index, 0))

    def get_confirmed_segments(self) -> list[dict]:
        """返回所有 confirmed 状态的片段"""
        return [s for s in self._segments if s.get("count_status") == "confirmed"]

    def get_current_row(self) -> int:
        """获取当前选中行索引"""
        selected = self._table.selectedItems()
        if selected:
            return selected[0].row()
        return -1

    def goto_next_pending(self, current_idx: int) -> int:
        """查找下一个 pending 事件索引"""
        for i in range(current_idx + 1, len(self._segments)):
            if self._segments[i]["status"] == "pending":
                return i
        return -1

    def goto_prev_event(self, current_idx: int) -> int:
        """跳到上一个事件"""
        if current_idx > 0:
            return current_idx - 1
        return 0

    def goto_next_event(self, current_idx: int) -> int:
        """跳到下一个事件"""
        if current_idx < len(self._segments) - 1:
            return current_idx + 1
        return current_idx

    # ------------------------------------------------------------------
    # 键盘事件 — 传播快捷键到主窗口
    # ------------------------------------------------------------------
    def eventFilter(self, watched, event):
        """让事件列表获得焦点时，主窗口快捷键仍可用。

        ``QTableWidget`` 会在自己的事件处理阶段消费字母和数字键，单靠
        EventListWidget.keyPressEvent() 无法收到这些事件。仅保留表格导航键
        给表格本身，其余按键直接交由顶层 MainWindow 处理。
        """
        if watched is self._table and event.type() == QEvent.KeyPress:
            navigation_keys = {
                Qt.Key_Up, Qt.Key_Down, Qt.Key_PageUp, Qt.Key_PageDown,
                Qt.Key_Home, Qt.Key_End,
            }
            if event.key() not in navigation_keys:
                window = self.window()
                if window is not self:
                    window.keyPressEvent(event)
                    return True
        return super().eventFilter(watched, event)

    # ------------------------------------------------------------------
    # 右键菜单
    # ------------------------------------------------------------------
    def _on_context_menu(self, pos: QPoint):
        """右键菜单 (改进.md: 拆分片段)"""
        row = self._table.rowAt(pos.y())
        if row < 0 or row >= len(self._segments):
            return

        menu = QMenu(self)

        # 拆分片段 (改进.md 13 节: K 键)
        split_action = menu.addAction("拆分片段")
        split_action.triggered.connect(lambda: self.segment_split_requested.emit(row, -1))

        menu.addSeparator()

        # 合并到上一个
        if row > 0:
            merge_prev = menu.addAction(f"合并到上一个 (#{self._segments[row-1]['segment_id']})")
            merge_prev.triggered.connect(lambda: self.merge_segments(row - 1, row))

        # 合并到下一个
        if row < len(self._segments) - 1:
            merge_next = menu.addAction(f"合并到下一个 (#{self._segments[row+1]['segment_id']})")
            merge_next.triggered.connect(lambda: self.merge_segments(row, row + 1))

        menu.addSeparator()

        # 删除
        del_action = menu.addAction("删除事件")
        del_action.triggered.connect(lambda: self.delete_segment(row))

        menu.exec(self._table.viewport().mapToGlobal(pos))

    def split_segment_at_frame(self, index: int, frame_idx: int):
        """在当前帧拆分指定片段 (改进.md 13 节: K 键)"""
        if not (0 <= index < len(self._segments)):
            return

        seg = self._segments[index]
        # frame_idx 必须在片段范围内
        if frame_idx <= seg["start_frame"] or frame_idx >= seg["end_frame"]:
            return

        fps = 30.0  # 近似; 实际时间从帧差推断
        if seg["end_frame"] > seg["start_frame"]:
            fps = (seg["end_frame"] - seg["start_frame"]) / max(seg["duration"], 0.001)

        # 前半段
        first_half = dict(seg)
        first_half["end_frame"] = frame_idx - 1
        first_half["end_time"] = (frame_idx - 1) / fps
        first_half["duration"] = first_half["end_time"] - first_half["start_time"]
        first_half["review_end_frame"] = first_half["end_frame"]
        first_half["review_end_time"] = first_half["end_time"]
        first_half["status"] = "modified"
        first_half["modified_by_user"] = True

        # 后半段
        second_half = dict(seg)
        second_half["start_frame"] = frame_idx
        second_half["start_time"] = frame_idx / fps
        second_half["duration"] = second_half["end_time"] - second_half["start_time"]
        second_half["review_start_frame"] = second_half["start_frame"]
        second_half["review_start_time"] = second_half["start_time"]
        second_half["status"] = "modified"
        second_half["modified_by_user"] = True

        # 替换: 删除原片段, 插入两个新片段
        self._segments[index] = first_half
        self._segments.insert(index + 1, second_half)
        self._reindex()
        self._refresh_table()
        self._table.selectRow(index)

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    def _reindex(self):
        """重新编排序号"""
        for i, seg in enumerate(self._segments):
            seg["segment_id"] = i + 1

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        """格式化时间为 MM:SS.ms"""
        if seconds < 0:
            seconds = 0.0
        m = int(seconds // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 100)
        return f"{m:02d}:{s:02d}.{ms:02d}"

    @property
    def count(self) -> int:
        return len(self._segments)

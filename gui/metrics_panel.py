"""
单帧检测指标面板
实现规范 9.2 节：实时显示当前帧各项遮挡检测指标
用 QProgressBar + 数值标签显示
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QProgressBar, QFrame, QSizePolicy,
)


class MetricsPanel(QWidget):
    """单帧检测指标可视化面板"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(280)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(6)

        # 标题
        title = QLabel("单帧遮挡检测指标")
        title.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        self._layout.addWidget(title)

        # 分隔线
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #555;")
        self._layout.addWidget(sep)

        # 各项指标
        self._metric_widgets = {}
        metrics_def = [
            ("warm_color_ratio", "暖点颜色保留比例", True),   # True = 越低越可疑
            ("dark_pixel_ratio", "深色像素比例", False),       # False = 越高越可疑
            ("background_diff_score", "背景差异分数", False),
            ("largest_dark_blob_area_ratio", "最大深色连通区域面积比", False),
            ("occlusion_area_ratio", "综合遮挡面积比例", False),
        ]
        for key, name, is_warm in metrics_def:
            self._metric_widgets[key] = self._add_metric_row(name, is_warm)

        # 分隔线
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet("color: #555;")
        self._layout.addWidget(sep2)

        # ---- 计数指标 (改进.md) ----
        count_title = QLabel("小鼠计数指标")
        count_title.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        count_title.setAlignment(Qt.AlignCenter)
        count_title.setStyleSheet("color: #aaa;")
        self._layout.addWidget(count_title)

        # 估计数量 (大号数字)
        self._lbl_est_count = QLabel("--")
        self._lbl_est_count.setFont(QFont("Arial", 32, QFont.Bold))
        self._lbl_est_count.setAlignment(Qt.AlignCenter)
        self._lbl_est_count.setMinimumHeight(50)
        self._lbl_est_count.setStyleSheet(
            "color: #ffcc44; background-color: #222; border-radius: 6px; padding: 6px;"
        )
        self._layout.addWidget(self._lbl_est_count)

        # count_by_blob / count_by_area 标签
        count_info_row = QHBoxLayout()
        self._lbl_count_blob = QLabel("连通区: --")
        self._lbl_count_blob.setStyleSheet("color: #ccc; font-size: 10px;")
        count_info_row.addWidget(self._lbl_count_blob)
        self._lbl_count_area = QLabel("面积: --")
        self._lbl_count_area.setStyleSheet("color: #ccc; font-size: 10px;")
        count_info_row.addWidget(self._lbl_count_area)
        self._layout.addLayout(count_info_row)

        self._lbl_count_conf = QLabel("置信度: --")
        self._lbl_count_conf.setStyleSheet("color: #aaa; font-size: 10px;")
        self._lbl_count_conf.setAlignment(Qt.AlignCenter)
        self._layout.addWidget(self._lbl_count_conf)

        # ---- 调试信息 (改进.md) ----
        debug_title = QLabel("调试信息")
        debug_title.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        debug_title.setAlignment(Qt.AlignCenter)
        debug_title.setStyleSheet("color: #777; margin-top: 4px;")
        self._layout.addWidget(debug_title)

        self._lbl_area_ratio = QLabel("面积比: --")
        self._lbl_area_ratio.setStyleSheet("color: #999; font-size: 9px; font-family: monospace;")
        self._lbl_area_ratio.setWordWrap(True)
        self._layout.addWidget(self._lbl_area_ratio)

        self._lbl_area_refs = QLabel("参考: --")
        self._lbl_area_refs.setStyleSheet("color: #999; font-size: 9px; font-family: monospace;")
        self._lbl_area_refs.setWordWrap(True)
        self._layout.addWidget(self._lbl_area_refs)

        self._lbl_decision_reason = QLabel("判定原因: --")
        self._lbl_decision_reason.setStyleSheet("color: #999; font-size: 9px; font-family: monospace;")
        self._lbl_decision_reason.setWordWrap(True)
        self._layout.addWidget(self._lbl_decision_reason)

        self._lbl_blob_debug = QLabel("")
        self._lbl_blob_debug.setStyleSheet("color: #888; font-size: 8px; font-family: monospace;")
        self._lbl_blob_debug.setWordWrap(True)
        self._lbl_blob_debug.setMinimumHeight(20)
        self._layout.addWidget(self._lbl_blob_debug)

        # 分隔线
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.HLine)
        sep3.setStyleSheet("color: #555;")
        self._layout.addWidget(sep3)

        # 判定结果标签
        self._judgment_label = QLabel("当前帧判定: --")
        self._judgment_label.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        self._judgment_label.setAlignment(Qt.AlignCenter)
        self._judgment_label.setMinimumHeight(40)
        self._judgment_label.setStyleSheet(
            "background-color: #2a2a2a; border-radius: 6px; padding: 8px;"
        )
        self._layout.addWidget(self._judgment_label)

        # 原因
        self._reason_label = QLabel("")
        self._reason_label.setAlignment(Qt.AlignCenter)
        self._reason_label.setStyleSheet("color: #999; font-size: 10px;")
        self._reason_label.setWordWrap(True)
        self._layout.addWidget(self._reason_label)

        self._layout.addStretch()

    def _add_metric_row(self, name: str, is_warm: bool) -> dict:
        """添加一行指标显示: 标签 + QProgressBar + 数值标签"""
        row = QHBoxLayout()
        row.setSpacing(4)

        # 指标名称标签
        label = QLabel(name)
        label.setMinimumWidth(140)
        label.setStyleSheet("color: #ccc; font-size: 11px;")
        row.addWidget(label)

        # 进度条
        bar = QProgressBar()
        bar.setRange(0, 1000)  # 0.0% ~ 100.0%, 精度 0.1%
        bar.setValue(0)
        bar.setTextVisible(False)
        bar.setFixedHeight(14)
        bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row.addWidget(bar)

        # 数值标签
        value_label = QLabel("--")
        value_label.setMinimumWidth(50)
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        value_label.setStyleSheet("color: #eee; font-size: 11px; font-family: monospace;")
        row.addWidget(value_label)

        self._layout.addLayout(row)

        return {
            "bar": bar,
            "value_label": value_label,
            "is_warm": is_warm,  # True for warm_color_ratio (high=good)
        }

    def update_metrics(self, metrics: dict | None):
        """
        更新所有指标显示
        :param metrics: DetectionMetrics.compute() 返回的 dict, 或 None
        """
        if metrics is None:
            self._clear_all()
            return

        # 如果有错误原因
        reason = metrics.get("reason", "")
        if reason:
            self._reason_label.setText(f"[!] {reason}")
        else:
            self._reason_label.setText("")

        # 更新每个指标
        for key, widgets in self._metric_widgets.items():
            value = metrics.get(key, 0.0)
            pct = int(value * 1000)  # 转为千分比
            pct = max(0, min(1000, pct))

            bar: QProgressBar = widgets["bar"]
            value_label: QLabel = widgets["value_label"]
            is_warm = widgets["is_warm"]

            bar.setValue(pct)

            # 颜色: 暖点颜色比例高=好(绿), 其他指标低=好(绿)
            if is_warm:
                # warm_color_ratio: 高 = 好 (绿)
                if value >= 0.8:
                    color = "#44bb44"
                elif value >= 0.5:
                    color = "#dddd44"
                else:
                    color = "#ff4444"
            else:
                # 其他指标: 低 = 好 (绿)
                if value <= 0.15:
                    color = "#44bb44"
                elif value <= 0.35:
                    color = "#dddd44"
                else:
                    color = "#ff4444"

            value_label.setText(f"{value:.1%}")
            value_label.setStyleSheet(f"color: {color}; font-size: 11px; font-family: monospace;")

            # 进度条颜色
            bar.setStyleSheet(
                f"""
                QProgressBar {{
                    background-color: #333;
                    border: 1px solid #555;
                    border-radius: 3px;
                }}
                QProgressBar::chunk {{
                    background-color: {color};
                    border-radius: 2px;
                }}
                """
            )

        # 判定结果
        is_occupied = metrics.get("is_occupied", False)
        occ_ratio = metrics.get("occlusion_area_ratio", 0.0)
        if is_occupied:
            self._judgment_label.setText(f"[!] 疑似占据 (遮挡比: {occ_ratio:.1%})")
            self._judgment_label.setStyleSheet(
                "background-color: #4a2020; color: #ff6666; border-radius: 6px; padding: 8px;"
                "font-weight: bold;"
            )
        else:
            self._judgment_label.setText(f"[v] 未占据 (遮挡比: {occ_ratio:.1%})")
            self._judgment_label.setStyleSheet(
                "background-color: #204020; color: #66ff66; border-radius: 6px; padding: 8px;"
                "font-weight: bold;"
            )

    def update_count_metrics(self, count_data: dict | None):
        """
        更新小鼠计数指标 (改进.md)
        :param count_data: MouseCounter.estimate_count() 返回的 dict
        """
        if count_data is None:
            self._lbl_est_count.setText("--")
            self._lbl_est_count.setStyleSheet(
                "color: #ffcc44; background-color: #222; border-radius: 6px; padding: 6px;"
            )
            self._lbl_count_blob.setText("连通区: --")
            self._lbl_count_area.setText("面积: --")
            self._lbl_count_conf.setText("置信度: --")
            self._lbl_count_conf.setStyleSheet("color: #aaa; font-size: 10px;")
            self._lbl_area_ratio.setText("面积比: --")
            self._lbl_decision_reason.setText("判定原因: --")
            self._lbl_blob_debug.setText("")
            return

        est_count = count_data.get("estimated_mouse_count", 0)
        if est_count is None:
            est_count = 0

        if est_count == 0:
            self._lbl_est_count.setText("0")
            self._lbl_est_count.setStyleSheet(
                "color: #888; background-color: #222; border-radius: 6px; padding: 6px;"
            )
        else:
            self._lbl_est_count.setText(str(est_count))
            self._lbl_est_count.setStyleSheet(
                "color: #ffcc44; background-color: #222; border-radius: 6px; padding: 6px;"
            )

        count_by_blob = count_data.get("count_by_blob", "--")
        count_by_area = count_data.get("count_by_area", "--")
        self._lbl_count_blob.setText(f"连通区: {count_by_blob}")
        self._lbl_count_area.setText(f"面积: {count_by_area}")

        count_conf = count_data.get("count_confidence", 0.0)
        if count_conf > 0:
            if count_conf < 0.5:
                self._lbl_count_conf.setText(f"置信度: {count_conf:.2f} [!] 低")
                self._lbl_count_conf.setStyleSheet("color: #ff8800; font-size: 10px; font-weight: bold;")
            else:
                self._lbl_count_conf.setText(f"置信度: {count_conf:.2f}")
                self._lbl_count_conf.setStyleSheet("color: #44cc44; font-size: 10px;")
        else:
            self._lbl_count_conf.setText("置信度: --")
            self._lbl_count_conf.setStyleSheet("color: #aaa; font-size: 10px;")

        # ---- 调试信息 (改进.md) ----
        area_ratio = count_data.get("area_ratio", 0.0)
        self._lbl_area_ratio.setText(f"面积比: {area_ratio:.2f} (总前/单鼠参)")

        # 显示各数量参考面积
        count_area_refs = count_data.get("count_area_refs", {})
        if count_area_refs:
            ref_parts = [f"{c}={int(a)}" for c, a in sorted(count_area_refs.items())]
            self._lbl_area_refs.setText(f"参考: {' '.join(ref_parts)}")
        else:
            self._lbl_area_refs.setText("参考: --")

        decision_reason = count_data.get("decision_reason", "--")
        self._lbl_decision_reason.setText(f"判定原因: {decision_reason}")

        # blob 调试信息
        debug_blobs = count_data.get("debug_blobs", [])
        if debug_blobs:
            lines = []
            for i, b in enumerate(debug_blobs):
                bbox = b.get("bbox", [])
                area = b.get("area", 0)
                ar = b.get("aspect_ratio", 0)
                touching = b.get("touching_roia", False)
                lines.append(
                    f"B{i}: area={area} ar={ar:.1f} roi={'Y' if touching else 'N'}"
                )
            self._lbl_blob_debug.setText(" | ".join(lines))
        else:
            self._lbl_blob_debug.setText("")

    def _clear_all(self):
        """清空所有指标显示"""
        for key, widgets in self._metric_widgets.items():
            widgets["bar"].setValue(0)
            widgets["value_label"].setText("--")
            widgets["value_label"].setStyleSheet("color: #eee; font-size: 11px; font-family: monospace;")
        self._judgment_label.setText("当前帧判定: --")
        self._judgment_label.setStyleSheet(
            "background-color: #2a2a2a; color: #aaa; border-radius: 6px; padding: 8px;"
        )
        self._reason_label.setText("")
        # 清除计数指标
        self._lbl_est_count.setText("--")
        self._lbl_count_blob.setText("连通区: --")
        self._lbl_count_area.setText("面积: --")
        self._lbl_count_conf.setText("置信度: --")
        self._lbl_count_conf.setStyleSheet("color: #aaa; font-size: 10px;")
        self._lbl_area_ratio.setText("面积比: --")
        self._lbl_decision_reason.setText("判定原因: --")
        self._lbl_area_refs.setText("参考: --")
        self._lbl_blob_debug.setText("")

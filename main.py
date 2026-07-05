#!/usr/bin/env python3
"""
小鼠暖点占据半自动标注系统 - Phase 1 原型
入口文件

用法:
    python main.py
"""

import sys
import os

# 确保当前目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from gui.main_window import MainWindow


def main():
    """应用入口"""
    # 高 DPI 支持
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("MouseWarmSpot")
    app.setApplicationVersion("0.1.0-phase1")

    # 设置全局样式 (暗色主题)
    app.setStyleSheet("""
        QMainWindow {
            background-color: #2b2b2b;
        }
        QMenuBar {
            background-color: #333;
            color: #ddd;
        }
        QMenuBar::item:selected {
            background-color: #555;
        }
        QMenu {
            background-color: #333;
            color: #ddd;
        }
        QMenu::item:selected {
            background-color: #555;
        }
        QToolBar {
            background-color: #353535;
            border-bottom: 1px solid #555;
            spacing: 4px;
            padding: 2px;
        }
        QPushButton {
            background-color: #444;
            color: #eee;
            border: 1px solid #666;
            border-radius: 4px;
            padding: 4px 10px;
        }
        QPushButton:hover {
            background-color: #555;
        }
        QPushButton:disabled {
            background-color: #333;
            color: #777;
        }
        QSlider::groove:horizontal {
            height: 8px;
            background: #444;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            width: 16px;
            margin: -4px 0;
            background: #888;
            border-radius: 8px;
        }
        QSlider::sub-page:horizontal {
            background: #4488cc;
            border-radius: 4px;
        }
        QLineEdit {
            background-color: #444;
            color: #eee;
            border: 1px solid #666;
            border-radius: 4px;
            padding: 2px 6px;
        }
        QDockWidget {
            titlebar-close-icon: none;
            color: #ddd;
        }
        QDockWidget::title {
            background-color: #3a3a3a;
            padding: 4px;
            border-bottom: 1px solid #555;
        }
        QStatusBar {
            background-color: #353535;
            color: #aaa;
        }
    """)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

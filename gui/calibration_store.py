"""
校准标记存储系统 — 多帧样本管理
Phase 1+2: CalibrationStore 接管 VideoWidget 的校准帧管理
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class CalibrationSample:
    """单个校准标记样本"""
    mouse_count: int          # 0-4
    frame_idx: int            # 标记时的视频帧号
    frame_bgr: np.ndarray | None = None   # 帧图像 (深拷贝)
    measured_area: float | None = None    # 测量到的前景总面积
    valid: bool = False                   # 测量是否有效
    reason: str = ""                      # 无效原因


class CalibrationStore:
    """多帧校准样本存储, 管理 0-4 只小鼠的标记样本"""

    def __init__(self):
        self.samples: dict[int, list[CalibrationSample]] = {
            0: [], 1: [], 2: [], 3: [], 4: [],
        }

    def add_sample(self, sample: CalibrationSample):
        """追加一个样本"""
        self.samples[sample.mouse_count].append(sample)

    def remove_last_sample(self, mouse_count: int) -> CalibrationSample | None:
        """撤回最后一个样本"""
        if self.samples.get(mouse_count):
            return self.samples[mouse_count].pop()
        return None

    def clear_samples(self, mouse_count: int):
        """清空指定数量的所有样本"""
        self.samples[mouse_count] = []

    def get_samples(self, mouse_count: int) -> list[CalibrationSample]:
        """获取指定数量的样本列表"""
        return self.samples.get(mouse_count, [])

    def count(self, mouse_count: int) -> int:
        """获取指定数量的样本数"""
        return len(self.samples.get(mouse_count, []))

    def has_background(self) -> bool:
        """是否有背景 (0只) 样本"""
        return len(self.samples.get(0, [])) > 0

    def latest_background(self) -> np.ndarray | None:
        """获取最新的背景帧图像"""
        samples = self.samples.get(0, [])
        return samples[-1].frame_bgr if samples else None

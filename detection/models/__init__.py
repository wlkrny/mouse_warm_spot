"""
detection.models — 可选 AI 推理模型
包含:
  - 基于 ONNX 的耳标颜色分类器（轻量 CNN）
  - AI 视觉模型 provider（Kimi/Minimax）
  - HSV 规则回退
"""

from .classifier import EarTagClassifier, CATEGORIES, hsv_rule_classify
from .vision_provider import (
    VisionProvider,
    VisionProviderError,
    VisionProviderFactory,
    KimiVisionProvider,
    MinimaxVisionProvider,
    OpenRouterVisionProvider,
    _bgr_to_jpeg_data_uri,
    _parse_color_json,
)

__all__ = [
    "EarTagClassifier",
    "CATEGORIES",
    "hsv_rule_classify",
    "VisionProvider",
    "VisionProviderError",
    "VisionProviderFactory",
    "KimiVisionProvider",
    "MinimaxVisionProvider",
    "OpenRouterVisionProvider",
    "_bgr_to_jpeg_data_uri",
    "_parse_color_json",
]
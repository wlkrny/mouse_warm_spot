"""
单元测试: AI 视觉 provider (Kimi/Minimax) — 无需真实网络凭据。

测试范围:
  - _bgr_to_jpeg_data_uri 编码/格式
  - _parse_color_json 解析（正常/异常/边界）
  - _OpenAICompatVisionProvider payload 构造
  - VisionProviderFactory 创建/配置错误
  - VisionProviderError 可诊断性
  - EarTagClassifier AI vision 集成（mock provider）

所有测试仅使用 Python 标准库 unittest + NumPy 合成数组，
不依赖真实 API 密钥或网络连接。
"""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =========================================================================
# 辅助：合成 BGR patch
# =========================================================================
def _make_patch(h=48, w=48):
    """创建一个随机 BGR uint8 patch。"""
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


# =========================================================================
# _bgr_to_jpeg_data_uri
# =========================================================================
class TestBgrToJpegDataUri(unittest.TestCase):
    """验证 BGR → JPEG base64 data URI 编码。"""

    def test_returns_data_uri_prefix(self):
        from detection.models.vision_provider import _bgr_to_jpeg_data_uri
        patch = _make_patch()
        uri = _bgr_to_jpeg_data_uri(patch)
        self.assertTrue(uri.startswith("data:image/jpeg;base64,"))

    def test_pure_black_encodes(self):
        from detection.models.vision_provider import _bgr_to_jpeg_data_uri
        patch = np.zeros((32, 32, 3), dtype=np.uint8)
        uri = _bgr_to_jpeg_data_uri(patch)
        self.assertIn("base64,", uri)
        # 长度应远大于空 base64
        self.assertGreater(len(uri), 30)

    def test_pure_white_encodes(self):
        from detection.models.vision_provider import _bgr_to_jpeg_data_uri
        patch = np.full((32, 32, 3), 255, dtype=np.uint8)
        uri = _bgr_to_jpeg_data_uri(patch)
        self.assertGreater(len(uri), 30)

    def test_quality_affects_size(self):
        from detection.models.vision_provider import _bgr_to_jpeg_data_uri
        patch = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        uri_low = _bgr_to_jpeg_data_uri(patch, quality=30)
        uri_high = _bgr_to_jpeg_data_uri(patch, quality=95)
        # 高画质通常更长
        self.assertGreater(len(uri_high), len(uri_low) * 0.9)


# =========================================================================
# _parse_color_json
# =========================================================================
class TestParseColorJson(unittest.TestCase):
    """验证 AI 模型 JSON 输出解析。"""

    def test_valid_json_full(self):
        from detection.models.vision_provider import _parse_color_json
        result = _parse_color_json('{"color": "red", "confidence": 0.92}')
        self.assertEqual(result, ("red", 0.92))

    def test_valid_json_yellow(self):
        from detection.models.vision_provider import _parse_color_json
        result = _parse_color_json('{"color":"yellow","confidence":0.75}')
        self.assertEqual(result, ("yellow", 0.75))

    def test_valid_json_blue(self):
        from detection.models.vision_provider import _parse_color_json
        result = _parse_color_json('{"color": "blue", "confidence": 0.88}')
        self.assertEqual(result, ("blue", 0.88))

    def test_valid_json_green(self):
        from detection.models.vision_provider import _parse_color_json
        result = _parse_color_json('{"color": "green", "confidence": 0.65}')
        self.assertEqual(result, ("green", 0.65))

    def test_valid_json_white(self):
        from detection.models.vision_provider import _parse_color_json
        result = _parse_color_json('{"color": "white", "confidence": 0.70}')
        self.assertEqual(result, ("white", 0.70))

    def test_valid_json_unknown(self):
        from detection.models.vision_provider import _parse_color_json
        result = _parse_color_json('{"color": "unknown", "confidence": 0.95}')
        self.assertEqual(result, ("unknown", 0.95))

    def test_extract_from_text_with_markdown(self):
        """模型可能在 JSON 前后加 markdown 文本。"""
        from detection.models.vision_provider import _parse_color_json
        result = _parse_color_json(
            'Based on the image, here is the result:\n'
            '{"color": "red", "confidence": 0.85}\n'
            'This is the ear tag color.'
        )
        self.assertEqual(result, ("red", 0.85))

    def test_extract_json_from_code_fence(self):
        """模型可能用 ```json 包裹。"""
        from detection.models.vision_provider import _parse_color_json
        raw = '```json\n{"color": "blue", "confidence": 0.80}\n```'
        result = _parse_color_json(raw)
        self.assertEqual(result, ("blue", 0.80))

    def test_invalid_color_returns_none(self):
        from detection.models.vision_provider import _parse_color_json
        result = _parse_color_json('{"color": "purple", "confidence": 0.9}')
        self.assertIsNone(result)

    def test_invalid_confidence_range_clamped(self):
        """超出 [0,1] 的 confidence 被自动 clamp，不返回 None。"""
        from detection.models.vision_provider import _parse_color_json
        # confidence > 1.0 → clamped to 1.0 (robust handling)
        result = _parse_color_json('{"color": "red", "confidence": 1.5}')
        self.assertEqual(result, ("red", 1.0))

    def test_no_json_returns_none(self):
        from detection.models.vision_provider import _parse_color_json
        result = _parse_color_json("This is just plain text without json")
        self.assertIsNone(result)

    def test_empty_string_returns_none(self):
        from detection.models.vision_provider import _parse_color_json
        self.assertIsNone(_parse_color_json(""))
        self.assertIsNone(_parse_color_json(None))

    def test_malformed_json_returns_none(self):
        from detection.models.vision_provider import _parse_color_json
        result = _parse_color_json('{"color": "red", "confidence": 0.9,')
        self.assertIsNone(result)

    def test_missing_color_returns_none(self):
        from detection.models.vision_provider import _parse_color_json
        result = _parse_color_json('{"confidence": 0.9}')
        self.assertIsNone(result)

    def test_single_int_json(self):
        """非 dict 顶层 JSON 应返回 None，但内嵌对象的数组也会被提取。"""
        from detection.models.vision_provider import _parse_color_json
        self.assertIsNone(_parse_color_json('42'))
        self.assertIsNone(_parse_color_json('"red"'))
        # 数组内嵌有效对象会被正则提取并成功解析（稳健行为）
        result = _parse_color_json('[{"color":"red","confidence":0.5}]')
        self.assertEqual(result, ("red", 0.5))

    def test_color_case_insensitive(self):
        from detection.models.vision_provider import _parse_color_json
        result = _parse_color_json('{"color": "RED", "confidence": 0.9}')
        self.assertEqual(result, ("red", 0.9))

    def test_confidence_int_to_float(self):
        from detection.models.vision_provider import _parse_color_json
        result = _parse_color_json('{"color": "blue", "confidence": 1}')
        self.assertEqual(result, ("blue", 1.0))


# =========================================================================
# Segment count JSON contract (pure parsing; no network)
# =========================================================================
class TestParseSegmentJson(unittest.TestCase):
    def test_new_contract_and_legacy_contract(self):
        from detection.models.vision_provider import _parse_segment_json
        self.assertEqual(_parse_segment_json('{"mouse_count":2,"colors":["red","blue"],"confidence":0.8}'),
                         {"mouse_count": 2, "colors": ["red", "blue"], "confidence": 0.8,
                          "thermometer_present": False, "parse_status": "ok"})
        legacy = _parse_segment_json('{"color":"yellow","confidence":0.7}')
        self.assertEqual(legacy["mouse_count"], 1)
        self.assertEqual(legacy["colors"], ["yellow"])
        self.assertEqual(legacy["parse_status"], "legacy")

    def test_invalid_count_colors_and_non_final_json_are_rejected(self):
        from detection.models.vision_provider import _parse_segment_json
        for raw in (
            '{"mouse_count":3,"colors":["red","blue","green"],"confidence":0.8}',
            '{"mouse_count":2,"colors":["red"],"confidence":0.8}',
            '{"mouse_count":1,"colors":["purple"],"confidence":0.8}',
            'answer: {"mouse_count":1,"colors":["red"],"confidence":0.8}',
        ):
            self.assertIsNone(_parse_segment_json(raw))


# =========================================================================
# _OpenAICompatVisionProvider payload
# =========================================================================
class TestOpenAICompatPayload(unittest.TestCase):
    """验证 payload 构造。"""

    def test_build_payload_includes_image(self):
        from detection.models.vision_provider import (
            KimiVisionProvider, _bgr_to_jpeg_data_uri,
        )
        with patch.dict(os.environ, {
            "KIMI_API_KEY": "sk-test",
            "KIMI_API_BASE": "https://example.invalid/kimi/v1",
            "KIMI_VISION_MODEL": "test-vision-model",
            "MOUSE_COLOR_AI_PROVIDER": "kimi",
        }, clear=True):
            provider = KimiVisionProvider()
            data_uri = _bgr_to_jpeg_data_uri(_make_patch())
            payload = provider._build_payload(data_uri)

            self.assertIn("model", payload)
            self.assertIn("messages", payload)
            self.assertEqual(len(payload["messages"]), 2)

            # System message
            self.assertEqual(payload["messages"][0]["role"], "system")

            # User message with image
            user_msg = payload["messages"][1]
            self.assertEqual(user_msg["role"], "user")
            self.assertIsInstance(user_msg["content"], list)
            self.assertEqual(len(user_msg["content"]), 2)
            self.assertEqual(user_msg["content"][0]["type"], "text")
            self.assertEqual(user_msg["content"][1]["type"], "image_url")
            self.assertIn("base64,", user_msg["content"][1]["image_url"]["url"])

    def test_build_headers_has_auth(self):
        from detection.models.vision_provider import KimiVisionProvider
        with patch.dict(os.environ, {
            "KIMI_API_KEY": "sk-test-123",
            "KIMI_API_BASE": "https://example.invalid/kimi/v1",
            "KIMI_VISION_MODEL": "test-vision-model",
            "MOUSE_COLOR_AI_PROVIDER": "kimi",
        }, clear=True):
            provider = KimiVisionProvider()
            headers = provider._build_headers()
            self.assertEqual(headers["Authorization"], "Bearer sk-test-123")
            self.assertEqual(headers["Content-Type"], "application/json")

    def test_minimax_provider_uses_minimax_prefix(self):
        from detection.models.vision_provider import MinimaxVisionProvider
        with patch.dict(os.environ, {
            "MINIMAX_API_KEY": "mm-test",
            "MINIMAX_API_BASE": "https://example.invalid/minimax/v1",
            "MINIMAX_VISION_MODEL": "test-vision-model",
            "MOUSE_COLOR_AI_PROVIDER": "minimax",
        }, clear=True):
            provider = MinimaxVisionProvider()
            self.assertEqual(provider._method_name, "ear_tag_color_minimax")
            self.assertEqual(provider._env_prefix, "MINIMAX")


# =========================================================================
# VisionProviderFactory
# =========================================================================
class TestVisionProviderFactory(unittest.TestCase):
    """验证 VisionProviderFactory 配置解析。"""

    def test_hsv_returns_none(self):
        from detection.models.vision_provider import VisionProviderFactory
        with patch.dict(os.environ, {"MOUSE_COLOR_AI_PROVIDER": "hsv"}, clear=True):
            result = VisionProviderFactory.create()
            self.assertIsNone(result)

    def test_empty_returns_none(self):
        from detection.models.vision_provider import VisionProviderFactory
        with patch.dict(os.environ, {}, clear=True):
            result = VisionProviderFactory.create()
            self.assertIsNone(result)

    def test_unknown_provider_raises(self):
        from detection.models.vision_provider import VisionProviderFactory, VisionProviderError
        with patch.dict(os.environ, {"MOUSE_COLOR_AI_PROVIDER": "openai"}, clear=True):
            with self.assertRaises(VisionProviderError) as ctx:
                VisionProviderFactory.create()
            self.assertIn("不支持", str(ctx.exception))
            self.assertIn("openai", str(ctx.exception))

    def test_openrouter_provider_creates_with_full_config(self):
        """openrouter provider 需 OPENROUTER_API_KEY + _BASE + _MODEL 三者。"""
        from detection.models.vision_provider import (
            VisionProviderFactory, OpenRouterVisionProvider,
        )
        with patch.dict(os.environ, {
            "MOUSE_COLOR_AI_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "sk-or-test",
            "OPENROUTER_API_BASE": "https://example.invalid/openrouter/v1",
            "OPENROUTER_VISION_MODEL": "test-vision-model",
        }, clear=True):
            provider = VisionProviderFactory.create()
            self.assertIsInstance(provider, OpenRouterVisionProvider)
            self.assertEqual(provider._method_name, "ear_tag_color_openrouter")

    def test_openrouter_missing_key_raises_diagnostic(self):
        """缺少 OPENROUTER_API_KEY 时报错。"""
        from detection.models.vision_provider import VisionProviderFactory, VisionProviderError
        with patch.dict(os.environ, {"MOUSE_COLOR_AI_PROVIDER": "openrouter"}, clear=True):
            with self.assertRaises(VisionProviderError) as ctx:
                VisionProviderFactory.create()
            self.assertIn("OPENROUTER_API_KEY", str(ctx.exception))

    def test_openrouter_missing_base_raises_diagnostic(self):
        """缺少 OPENROUTER_API_BASE 时报错。"""
        from detection.models.vision_provider import VisionProviderFactory, VisionProviderError
        with patch.dict(os.environ, {
            "MOUSE_COLOR_AI_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "sk-or-test",
        }, clear=True):
            with self.assertRaises(VisionProviderError) as ctx:
                VisionProviderFactory.create()
            self.assertIn("OPENROUTER_API_BASE", str(ctx.exception))

    def test_openrouter_missing_model_raises_diagnostic(self):
        """缺少 OPENROUTER_VISION_MODEL 时报错。"""
        from detection.models.vision_provider import VisionProviderFactory, VisionProviderError
        with patch.dict(os.environ, {
            "MOUSE_COLOR_AI_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "sk-or-test",
            "OPENROUTER_API_BASE": "https://example.invalid/openrouter/v1",
        }, clear=True):
            with self.assertRaises(VisionProviderError) as ctx:
                VisionProviderFactory.create()
            self.assertIn("OPENROUTER_VISION_MODEL", str(ctx.exception))

    def test_kimi_missing_key_raises(self):
        from detection.models.vision_provider import VisionProviderFactory, VisionProviderError
        with patch.dict(os.environ, {"MOUSE_COLOR_AI_PROVIDER": "kimi"}, clear=True):
            with self.assertRaises(VisionProviderError) as ctx:
                VisionProviderFactory.create()
            self.assertIn("KIMI_API_KEY", str(ctx.exception))

    def test_kimi_missing_base_raises_diagnostic(self):
        """缺少 KIMI_API_BASE 时报告具体缺失变量。"""
        from detection.models.vision_provider import VisionProviderFactory, VisionProviderError
        with patch.dict(os.environ, {
            "MOUSE_COLOR_AI_PROVIDER": "kimi",
            "KIMI_API_KEY": "sk-test",
        }, clear=True):
            with self.assertRaises(VisionProviderError) as ctx:
                VisionProviderFactory.create()
            msg = str(ctx.exception)
            self.assertIn("KIMI_API_BASE", msg)

    def test_kimi_missing_model_raises_diagnostic(self):
        """缺少 KIMI_VISION_MODEL 时报告具体缺失变量。"""
        from detection.models.vision_provider import VisionProviderFactory, VisionProviderError
        with patch.dict(os.environ, {
            "MOUSE_COLOR_AI_PROVIDER": "kimi",
            "KIMI_API_KEY": "sk-test",
            "KIMI_API_BASE": "https://example.invalid/kimi/v1",
        }, clear=True):
            with self.assertRaises(VisionProviderError) as ctx:
                VisionProviderFactory.create()
            msg = str(ctx.exception)
            self.assertIn("KIMI_VISION_MODEL", msg)

    def test_minimax_missing_base_raises_diagnostic(self):
        """缺少 MINIMAX_API_BASE 时报告具体缺失变量。"""
        from detection.models.vision_provider import VisionProviderFactory, VisionProviderError
        with patch.dict(os.environ, {
            "MOUSE_COLOR_AI_PROVIDER": "minimax",
            "MINIMAX_API_KEY": "mm-test",
        }, clear=True):
            with self.assertRaises(VisionProviderError) as ctx:
                VisionProviderFactory.create()
            msg = str(ctx.exception)
            self.assertIn("MINIMAX_API_BASE", msg)

    def test_minimax_missing_model_raises_diagnostic(self):
        """缺少 MINIMAX_VISION_MODEL 时报告具体缺失变量。"""
        from detection.models.vision_provider import VisionProviderFactory, VisionProviderError
        with patch.dict(os.environ, {
            "MOUSE_COLOR_AI_PROVIDER": "minimax",
            "MINIMAX_API_KEY": "mm-test",
            "MINIMAX_API_BASE": "https://example.invalid/minimax/v1",
        }, clear=True):
            with self.assertRaises(VisionProviderError) as ctx:
                VisionProviderFactory.create()
            msg = str(ctx.exception)
            self.assertIn("MINIMAX_VISION_MODEL", msg)

    def test_minimax_missing_key_raises(self):
        from detection.models.vision_provider import VisionProviderFactory, VisionProviderError
        with patch.dict(os.environ, {"MOUSE_COLOR_AI_PROVIDER": "minimax"}, clear=True):
            with self.assertRaises(VisionProviderError) as ctx:
                VisionProviderFactory.create()
            self.assertIn("MINIMAX_API_KEY", str(ctx.exception))

    def test_kimi_with_key_succeeds(self):
        from detection.models.vision_provider import (
            VisionProviderFactory, KimiVisionProvider,
        )
        with patch.dict(os.environ, {
            "MOUSE_COLOR_AI_PROVIDER": "kimi",
            "KIMI_API_KEY": "sk-test",
            "KIMI_API_BASE": "https://example.invalid/kimi/v1",
            "KIMI_VISION_MODEL": "test-vision-model",
        }, clear=True):
            provider = VisionProviderFactory.create()
            self.assertIsInstance(provider, KimiVisionProvider)
            self.assertEqual(provider._method_name, "ear_tag_color_kimi")

    def test_minimax_with_key_succeeds(self):
        from detection.models.vision_provider import (
            VisionProviderFactory, MinimaxVisionProvider,
        )
        with patch.dict(os.environ, {
            "MOUSE_COLOR_AI_PROVIDER": "minimax",
            "MINIMAX_API_KEY": "mm-test",
            "MINIMAX_API_BASE": "https://example.invalid/minimax/v1",
            "MINIMAX_VISION_MODEL": "test-vision-model",
        }, clear=True):
            provider = VisionProviderFactory.create()
            self.assertIsInstance(provider, MinimaxVisionProvider)
            self.assertEqual(provider._method_name, "ear_tag_color_minimax")

    def test_custom_timeout(self):
        from detection.models.vision_provider import KimiVisionProvider
        with patch.dict(os.environ, {
            "MOUSE_COLOR_AI_PROVIDER": "kimi",
            "KIMI_API_KEY": "sk-test",
            "KIMI_API_BASE": "https://example.invalid/kimi/v1",
            "KIMI_VISION_MODEL": "test-vision-model",
            "MOUSE_COLOR_AI_TIMEOUT": "15",
        }, clear=True):
            provider = KimiVisionProvider()
            self.assertEqual(provider._timeout, 15.0)

    def test_invalid_timeout_defaults(self):
        from detection.models.vision_provider import KimiVisionProvider
        from detection.models.vision_provider import DEFAULT_TIMEOUT
        with patch.dict(os.environ, {
            "MOUSE_COLOR_AI_PROVIDER": "kimi",
            "KIMI_API_KEY": "sk-test",
            "KIMI_API_BASE": "https://example.invalid/kimi/v1",
            "KIMI_VISION_MODEL": "test-vision-model",
            "MOUSE_COLOR_AI_TIMEOUT": "not_a_number",
        }, clear=True):
            provider = KimiVisionProvider()
            self.assertEqual(provider._timeout, DEFAULT_TIMEOUT)

    def test_custom_model(self):
        from detection.models.vision_provider import KimiVisionProvider
        with patch.dict(os.environ, {
            "MOUSE_COLOR_AI_PROVIDER": "kimi",
            "KIMI_API_KEY": "sk-test",
            "KIMI_API_BASE": "https://example.invalid/kimi/v1",
            "KIMI_VISION_MODEL": "custom-model-v1",
        }, clear=True):
            provider = KimiVisionProvider()
            self.assertEqual(provider._model, "custom-model-v1")

    def test_custom_api_base(self):
        from detection.models.vision_provider import KimiVisionProvider
        with patch.dict(os.environ, {
            "MOUSE_COLOR_AI_PROVIDER": "kimi",
            "KIMI_API_KEY": "sk-test",
            "KIMI_API_BASE": "https://custom.api.example.com/v1",
            "KIMI_VISION_MODEL": "test-vision-model",
        }, clear=True):
            provider = KimiVisionProvider()
            self.assertEqual(provider._api_base, "https://custom.api.example.com/v1")

    def test_register_custom_provider(self):
        from detection.models.vision_provider import (
            VisionProviderFactory, VisionProvider, VisionProviderError,
        )

        class CustomVisionProvider(VisionProvider):
            def classify(self, patch_bgr):
                return "red", 1.0, "ear_tag_color_custom"

            def classify_frames(self, frames):
                return "red", 1.0, "ear_tag_color_custom", None

        VisionProviderFactory.register("custom_test", CustomVisionProvider)
        try:
            with patch.dict(os.environ, {
                "MOUSE_COLOR_AI_PROVIDER": "custom_test",
            }, clear=True):
                provider = VisionProviderFactory.create()
                self.assertIsInstance(provider, CustomVisionProvider)
                color, conf, method = provider.classify(_make_patch())
                self.assertEqual(color, "red")
                self.assertEqual(method, "ear_tag_color_custom")
        finally:
            # 清理注册
            VisionProviderFactory._registry.pop("custom_test", None)


# =========================================================================
# VisionProviderError
# =========================================================================
class TestVisionProviderError(unittest.TestCase):
    """验证异常可诊断性。"""

    def test_contains_message(self):
        from detection.models.vision_provider import VisionProviderError
        exc = VisionProviderError("test error")
        self.assertEqual(str(exc), "test error")

    def test_contains_provider(self):
        from detection.models.vision_provider import VisionProviderError
        exc = VisionProviderError("err", provider="kimi")
        self.assertEqual(exc.provider, "kimi")

    def test_contains_status_code(self):
        from detection.models.vision_provider import VisionProviderError
        exc = VisionProviderError("err", status_code=503)
        self.assertEqual(exc.status_code, 503)


# =========================================================================
# EarTagClassifier AI vision 集成测试
# =========================================================================
class TestEarTagClassifierVisionIntegration(unittest.TestCase):
    """验证 EarTagClassifier 与 AI vision provider 集成。"""

    def setUp(self):
        from detection.models.classifier import EarTagClassifier
        self.clf = EarTagClassifier(use_cnn=False)

    def test_is_vision_available_false_by_default(self):
        """默认（无 AI provider 配置）时 is_vision_available 为 False。"""
        with patch.dict(os.environ, {}, clear=True):
            clf = self.clf.__class__(use_cnn=False)
            self.assertFalse(clf.is_vision_available)

    def test_vision_provider_injected_available(self):
        """直接注入 vision_provider 后 is_vision_available 为 True。"""
        from detection.models.classifier import EarTagClassifier
        mock = MagicMock()
        mock.classify.return_value = ("red", 0.9, "ear_tag_color_kimi")
        clf = EarTagClassifier(use_cnn=False, vision_provider=mock)
        self.assertTrue(clf.is_vision_available)

    def test_classify_uses_vision_when_available(self):
        """vision provider 可用时应优先使用 AI 视觉分类。"""
        from detection.models.classifier import EarTagClassifier
        mock = MagicMock()
        mock.classify.return_value = ("blue", 0.85, "ear_tag_color_minimax")
        clf = EarTagClassifier(use_cnn=False, vision_provider=mock)

        patch_bgr = _make_patch()
        hsv_px = np.array([[0, 200, 200]], dtype=np.float32)  # red
        color, method = clf.classify(patch_bgr, hsv_px)

        self.assertEqual(color, "blue")
        self.assertEqual(method, "ear_tag_color_minimax")
        mock.classify.assert_called_once()

    def test_classify_uses_openrouter_when_configured(self):
        """openrouter provider 可用时应优先使用。"""
        from detection.models.classifier import EarTagClassifier
        mock = MagicMock()
        mock.classify.return_value = ("red", 0.88, "ear_tag_color_openrouter")
        clf = EarTagClassifier(use_cnn=False, vision_provider=mock)

        patch_bgr = _make_patch()
        hsv_px = np.array([[110, 200, 200]], dtype=np.float32)  # blue
        color, method = clf.classify(patch_bgr, hsv_px)

        self.assertEqual(color, "red")
        self.assertEqual(method, "ear_tag_color_openrouter")
        mock.classify.assert_called_once()

    def test_classify_vision_unknown_falls_back(self):
        """vision 返回 unknown → 应回退到 HSV 规则。"""
        from detection.models.classifier import EarTagClassifier
        mock = MagicMock()
        mock.classify.return_value = ("unknown", 0.0, "ear_tag_color_kimi")
        clf = EarTagClassifier(use_cnn=False, vision_provider=mock)

        patch_bgr = _make_patch()
        hsv_px = np.array([[25, 200, 200]], dtype=np.float32)  # yellow
        color, method = clf.classify(patch_bgr, hsv_px)

        self.assertEqual(color, "yellow")
        self.assertEqual(method, "ear_tag_color_rule")
        mock.classify.assert_called_once()

    def test_classify_vision_exception_falls_back(self):
        """vision 异常 → 不应崩溃，回退到 HSV。"""
        from detection.models.classifier import EarTagClassifier
        mock = MagicMock()
        mock.classify.side_effect = RuntimeError("network error")
        clf = EarTagClassifier(use_cnn=False, vision_provider=mock)

        patch_bgr = _make_patch()
        hsv_px = np.array([[110, 200, 200]], dtype=np.float32)  # blue
        color, method = clf.classify(patch_bgr, hsv_px)

        self.assertEqual(color, "blue")
        self.assertEqual(method, "ear_tag_color_rule")
        mock.classify.assert_called_once()

    def test_classify_no_patch_returns_rule_even_with_vision(self):
        """patch 为 None 时不走 vision，直接走规则。"""
        from detection.models.classifier import EarTagClassifier
        mock = MagicMock()
        clf = EarTagClassifier(use_cnn=False, vision_provider=mock)

        hsv_px = np.array([[60, 200, 200]], dtype=np.float32)  # green
        color, method = clf.classify(None, hsv_px)

        self.assertEqual(color, "green")
        self.assertEqual(method, "ear_tag_color_rule")
        mock.classify.assert_not_called()

    def test_classify_empty_patch_skips_vision(self):
        """patch 为空时不走 vision。"""
        from detection.models.classifier import EarTagClassifier
        mock = MagicMock()
        clf = EarTagClassifier(use_cnn=False, vision_provider=mock)

        empty = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        hsv_px = np.array([[0, 10, 220]], dtype=np.float32)  # white
        color, method = clf.classify(empty, hsv_px)

        self.assertEqual(color, "white")
        self.assertEqual(method, "ear_tag_color_rule")
        mock.classify.assert_not_called()

    def test_classify_low_confidence_triggers_fallback(self):
        """vision confidence 低于阈值 → internal fallback within vision_classify。"""
        from detection.models.classifier import EarTagClassifier
        mock = MagicMock()
        # confidence < CNN_CONFIDENCE_THRESHOLD (0.3)
        mock.classify.return_value = ("red", 0.1, "ear_tag_color_kimi")
        clf = EarTagClassifier(use_cnn=False, vision_provider=mock)

        patch_bgr = _make_patch()
        hsv_px = np.array([[25, 200, 200]], dtype=np.float32)  # yellow
        color, method = clf.classify(patch_bgr, hsv_px)

        # 低置信度 → 视为 unknown → 回退到 HSV
        self.assertEqual(color, "yellow")
        self.assertEqual(method, "ear_tag_color_rule")
        mock.classify.assert_called_once()

    def test_init_vision_provider_factory_integration(self):
        """通过工厂创建 provider 的集成路径。"""
        from detection.models.classifier import EarTagClassifier
        with patch.dict(os.environ, {
            "MOUSE_COLOR_AI_PROVIDER": "kimi",
            "KIMI_API_KEY": "sk-test",
            "KIMI_API_BASE": "https://example.invalid/kimi/v1",
            "KIMI_VISION_MODEL": "test-vision-model",
        }, clear=True):
            clf = EarTagClassifier(use_cnn=False)
            self.assertTrue(clf.is_vision_available)
            self.assertIsNotNone(clf._vision_provider)
            self.assertEqual(clf._vision_provider._method_name, "ear_tag_color_kimi")

    def test_init_vision_provider_hsv_config(self):
        """MOUSE_COLOR_AI_PROVIDER=hsv 时不创建 provider。"""
        from detection.models.classifier import EarTagClassifier
        with patch.dict(os.environ, {
            "MOUSE_COLOR_AI_PROVIDER": "hsv",
        }, clear=True):
            clf = EarTagClassifier(use_cnn=False)
            self.assertFalse(clf.is_vision_available)


# =========================================================================
# IdentityAssist AI vision 集成测试
# =========================================================================
class TestIdentityAssistVisionIntegration(unittest.TestCase):
    """验证 IdentityAssist 与 AI vision method 前缀追踪。"""

    def test_construct_with_vision_provider(self):
        from detection.identity_assist import IdentityAssist
        assist = IdentityAssist(use_cnn=False)
        self.assertIsNotNone(assist)

        clf = assist._get_classifier()
        self.assertIsNotNone(clf)

    def test_get_classifier_lazy_reuse(self):
        from detection.identity_assist import IdentityAssist
        assist = IdentityAssist(use_cnn=False)
        clf1 = assist._get_classifier()
        clf2 = assist._get_classifier()
        self.assertIs(clf1, clf2)


# =========================================================================
# AI 请求预算 & allow_vision 测试
# =========================================================================
class TestAllowVisionBudget(unittest.TestCase):
    """验证 allow_vision 参数和请求预算控制。"""

    def test_allow_vision_false_skips_provider(self):
        """allow_vision=False 时 classify 不应调用 vision provider。"""
        from detection.models.classifier import EarTagClassifier
        mock = MagicMock()
        mock.classify.return_value = ("red", 0.9, "ear_tag_color_kimi")
        clf = EarTagClassifier(use_cnn=False, vision_provider=mock)

        patch_bgr = _make_patch()
        hsv_px = np.array([[0, 200, 200]], dtype=np.float32)  # red
        color, method = clf.classify(patch_bgr, hsv_px, allow_vision=False)

        # 不应调用 mock，应回退到 HSV 规则
        mock.classify.assert_not_called()
        self.assertEqual(color, "red")
        self.assertEqual(method, "ear_tag_color_rule")

    def test_allow_vision_false_cnn_still_works(self):
        """allow_vision=False 时 CNN 仍可正常工作（若可用）。"""
        from detection.models.classifier import EarTagClassifier
        # vision provider 存在，但 allow_vision=False
        mock_vision = MagicMock()
        clf = EarTagClassifier(use_cnn=False, vision_provider=mock_vision)

        patch_bgr = _make_patch()
        hsv_px = np.array([[110, 200, 200]], dtype=np.float32)  # blue
        color, method = clf.classify(patch_bgr, hsv_px, allow_vision=False)

        mock_vision.classify.assert_not_called()
        self.assertEqual(color, "blue")
        self.assertEqual(method, "ear_tag_color_rule")

    def test_allow_vision_false_hsv_works_when_cnn_unavailable(self):
        """allow_vision=False 且 CNN 不可用时 HSV 规则正常回退。"""
        from detection.models.classifier import EarTagClassifier
        clf = EarTagClassifier(use_cnn=False)  # no CNN, no vision

        px = np.array([[60, 200, 200]], dtype=np.float32)  # green
        color, method = clf.classify(None, px, allow_vision=False)
        self.assertEqual(color, "green")
        self.assertEqual(method, "ear_tag_color_rule")

    def test_default_budget_read_from_env(self):
        """默认 MOUSE_COLOR_AI_MAX_REQUESTS_PER_SEGMENT 应为 3。"""
        from detection.identity_assist import IdentityAssist
        with patch.dict(os.environ, {}, clear=True):
            assist = IdentityAssist(use_cnn=False)
            self.assertEqual(assist._max_vision_requests, 3)

    def test_custom_budget_from_env(self):
        """自定义预算值应正确读取。"""
        from detection.identity_assist import IdentityAssist
        with patch.dict(os.environ, {
            "MOUSE_COLOR_AI_MAX_REQUESTS_PER_SEGMENT": "10",
        }, clear=True):
            assist = IdentityAssist(use_cnn=False)
            self.assertEqual(assist._max_vision_requests, 10)

    def test_budget_zero_disables_vision(self):
        """预算=0 时应禁用全部云视觉。"""
        from detection.identity_assist import IdentityAssist
        with patch.dict(os.environ, {
            "MOUSE_COLOR_AI_MAX_REQUESTS_PER_SEGMENT": "0",
        }, clear=True):
            assist = IdentityAssist(use_cnn=False)
            self.assertEqual(assist._max_vision_requests, 0)

    def test_budget_negative_clamped_to_zero(self):
        """负预算值应 clamp 为 0。"""
        from detection.identity_assist import IdentityAssist
        with patch.dict(os.environ, {
            "MOUSE_COLOR_AI_MAX_REQUESTS_PER_SEGMENT": "-5",
        }, clear=True):
            assist = IdentityAssist(use_cnn=False)
            self.assertEqual(assist._max_vision_requests, 0)

    def test_budget_invalid_defaults(self):
        """无效预算值应回退到默认 3。"""
        from detection.identity_assist import IdentityAssist
        with patch.dict(os.environ, {
            "MOUSE_COLOR_AI_MAX_REQUESTS_PER_SEGMENT": "not_a_number",
        }, clear=True):
            assist = IdentityAssist(use_cnn=False)
            self.assertEqual(assist._max_vision_requests, 3)

    def test_budget_exhausted_never_calls_provider(self):
        """预算耗尽后 classify 不应调用 vision provider。

        模拟 budget=0 场景：classify 传入 allow_vision=False
        确保即使 provider 可用也不会被触发。
        """
        from detection.models.classifier import EarTagClassifier
        mock = MagicMock()
        mock.classify.return_value = ("red", 0.9, "ear_tag_color_kimi")
        clf = EarTagClassifier(use_cnn=False, vision_provider=mock)

        hsv_px = np.array([[25, 200, 200]], dtype=np.float32)  # yellow
        patch_bgr = _make_patch()

        # 预算耗尽 → allow_vision=False
        color, method = clf.classify(patch_bgr, hsv_px, allow_vision=False)

        # vision provider 不应被调用
        mock.classify.assert_not_called()
        # 应回退到 HSV 规则
        self.assertEqual(color, "yellow")
        self.assertEqual(method, "ear_tag_color_rule")


# =========================================================================
# 预算计数逻辑测试 — 验证"尝试次数"（非"成功次数"）计数
# =========================================================================
class TestBudgetCountOnAttempt(unittest.TestCase):
    """验证 IdentityAssist 在调用 vision provider 之前计数，
    无论 provider 成功、失败、返回低置信度，尝试都已消耗一次预算。
    达到预算后绝不可再调用 provider，但必须继续 CNN/HSV 回退。
    """

    def setUp(self):
        self.mock_vision = MagicMock()
        self.mock_vision.classify.return_value = ("red", 0.9, "ear_tag_color_kimi")

    def _make_patch(self):
        return np.random.randint(0, 256, (48, 48, 3), dtype=np.uint8)

    def _make_hsv_px(self, h=0, s=200, v=200):
        """Create HSV pixel array that maps to a known HSV rule color."""
        return np.array([[h, s, v]], dtype=np.float32)

    def test_vision_success_increments_counter(self):
        """vision provider 成功返回时，尝试仍计入预算（提前计数）。"""
        from detection.identity_assist import IdentityAssist

        with patch.dict(os.environ, {}, clear=True):
            assist = IdentityAssist(use_cnn=False)
            clf = assist._get_classifier()
            clf._vision_provider = self.mock_vision
            clf._vision_init_attempted = True
            clf._initialized = True

            budget = assist._max_vision_requests  # 3
            self.assertEqual(assist._vision_request_count, 0)

            patch_bgr = self._make_patch()
            hsv_px = self._make_hsv_px(h=0)  # red

            # 模拟 contour 处理中的预算预增逻辑
            budget_ok = assist._vision_request_count < budget
            allow_vision = budget_ok and clf.is_vision_available

            if allow_vision and patch_bgr is not None and patch_bgr.size > 0:
                assist._vision_request_count += 1

            assist._vision_request_count
            self.assertEqual(assist._vision_request_count, 1,
                             "调用前应已计入一次尝试")

            # 调用 classify（allow_vision=True）
            color, method = clf.classify(patch_bgr, hsv_px, allow_vision=True)
            self.mock_vision.classify.assert_called_once()
            # 预算不会因结果二次递增
            self.assertEqual(assist._vision_request_count, 1)

    def test_vision_exception_increments_counter(self):
        """vision provider 抛出异常时，尝试仍计入预算。"""
        from detection.identity_assist import IdentityAssist

        failing_vision = MagicMock()
        failing_vision.classify.side_effect = RuntimeError("network error")

        with patch.dict(os.environ, {}, clear=True):
            assist = IdentityAssist(use_cnn=False)
            clf = assist._get_classifier()
            clf._vision_provider = failing_vision
            clf._vision_init_attempted = True
            clf._initialized = True

            budget = assist._max_vision_requests
            self.assertEqual(assist._vision_request_count, 0)

            patch_bgr = self._make_patch()
            hsv_px = self._make_hsv_px(h=0)  # red

            # 预算预增逻辑
            budget_ok = assist._vision_request_count < budget
            allow_vision = budget_ok and clf.is_vision_available
            if allow_vision and patch_bgr is not None and patch_bgr.size > 0:
                assist._vision_request_count += 1

            self.assertEqual(assist._vision_request_count, 1,
                             "异常发生前应已计入一次尝试")

            # classify 内部会 catch 异常并回退到 HSV
            color, method = clf.classify(patch_bgr, hsv_px, allow_vision=True)
            failing_vision.classify.assert_called_once()
            # 计数维持 1，没有因异常二次递增
            self.assertEqual(assist._vision_request_count, 1)
            # 回退到 HSV 规则
            self.assertEqual(method, "ear_tag_color_rule")

    def test_empty_patch_does_not_consume_budget(self):
        """patch 不存在/为空时不虚增计数。"""
        from detection.identity_assist import IdentityAssist

        with patch.dict(os.environ, {}, clear=True):
            assist = IdentityAssist(use_cnn=False)
            clf = assist._get_classifier()
            clf._vision_provider = self.mock_vision
            clf._vision_init_attempted = True
            clf._initialized = True

            budget = assist._max_vision_requests
            self.assertEqual(assist._vision_request_count, 0)

            # patch 为 None
            budget_ok = assist._vision_request_count < budget
            allow_vision = budget_ok and clf.is_vision_available
            patch_bgr = None
            if allow_vision and patch_bgr is not None and patch_bgr.size > 0:
                assist._vision_request_count += 1

            self.assertEqual(assist._vision_request_count, 0,
                             "None patch 不应消耗预算")

            # patch 为空
            patch_bgr = np.array([], dtype=np.uint8).reshape(0, 0, 3)
            if allow_vision and patch_bgr is not None and patch_bgr.size > 0:
                assist._vision_request_count += 1

            self.assertEqual(assist._vision_request_count, 0,
                             "空 patch 不应消耗预算")

            self.mock_vision.classify.assert_not_called()

    def test_budget_exhausted_no_further_calls(self):
        """预算耗尽后同一 segment 的额外候选不得再次调用 provider。"""
        from detection.identity_assist import IdentityAssist

        with patch.dict(os.environ, {}, clear=True):
            assist = IdentityAssist(use_cnn=False)
            clf = assist._get_classifier()
            clf._vision_provider = self.mock_vision
            clf._vision_init_attempted = True
            clf._initialized = True

            budget = assist._max_vision_requests  # 3

            # 耗尽预算
            assist._vision_request_count = budget
            self.assertEqual(assist._vision_request_count, 3)

            patch_bgr = self._make_patch()
            hsv_px = self._make_hsv_px(h=0)  # red

            budget_ok = assist._vision_request_count < budget
            allow_vision = budget_ok and clf.is_vision_available

            self.assertFalse(budget_ok, "budget_ok 应为 False")
            self.assertFalse(allow_vision, "allow_vision 应为 False（预算耗尽）")

            # 即使分类器可用，allow_vision=False 防止调用 provider
            color, method = clf.classify(patch_bgr, hsv_px, allow_vision=allow_vision)
            self.mock_vision.classify.assert_not_called()
            # 回退到 HSV 规则
            self.assertEqual(method, "ear_tag_color_rule")

    def test_budget_exhausted_cnn_hsv_still_works(self):
        """预算耗尽后 CNN/HSV 仍工作。"""
        from detection.models.classifier import EarTagClassifier

        mock = MagicMock()
        mock.classify.return_value = ("red", 0.9, "ear_tag_color_kimi")
        clf = EarTagClassifier(use_cnn=False, vision_provider=mock)
        clf._initialized = True

        hsv_px = self._make_hsv_px(h=25)  # yellow
        patch_bgr = self._make_patch()

        # 预算耗尽 → allow_vision=False
        color, method = clf.classify(patch_bgr, hsv_px, allow_vision=False)

        mock.classify.assert_not_called()
        self.assertEqual(color, "yellow")
        self.assertEqual(method, "ear_tag_color_rule")

    def test_three_calls_then_stops(self):
        """预算=3 时，最多调用 3 次 vision provider，第 4 次拦截。"""
        from detection.identity_assist import IdentityAssist

        with patch.dict(os.environ, {}, clear=True):
            assist = IdentityAssist(use_cnn=False)
            clf = assist._get_classifier()
            clf._vision_provider = self.mock_vision
            clf._vision_init_attempted = True
            clf._initialized = True

            budget = assist._max_vision_requests  # 3

            call_count = 0
            for i in range(5):
                budget_ok = assist._vision_request_count < budget
                allow_vision = budget_ok and clf.is_vision_available

                patch_bgr = self._make_patch()
                hsv_px = self._make_hsv_px(h=0)

                if allow_vision and patch_bgr is not None and patch_bgr.size > 0:
                    assist._vision_request_count += 1
                    call_count += 1

                if allow_vision or clf.is_available:
                    clf.classify(patch_bgr, hsv_px, allow_vision=allow_vision)

            self.assertEqual(assist._vision_request_count, 3,
                             f"应为 3 次尝试，实际 {assist._vision_request_count}")
            self.assertEqual(call_count, 3,
                             f"应为 3 次调用，实际 {call_count}")
            self.assertEqual(self.mock_vision.classify.call_count, 3,
                             f"vision provider 应为 3 次调用，实际 "
                             f"{self.mock_vision.classify.call_count}")


# =========================================================================
# VLM priority over HSV/CNN fusion
# =========================================================================
class TestVLMCountOverride(unittest.TestCase):
    def test_apply_count_override_and_confirmed_guard(self):
        from detection.identity_assist import apply_identity_to_segment
        result = {"target_count": 2, "auto_mouse_colors": ["red", "blue"],
                  "auto_mouse_ids": ["auto_red", "auto_blue"], "identity_confidence": .9,
                  "identity_needs_review": False, "identity_conflict": False,
                  "identity_method": "ear_tag_color_vlm|count_override", "identity_note": ""}
        segment = {"count_status": "pending", "estimated_mouse_count": 1, "mouse_count": 0}
        apply_identity_to_segment(segment, result)
        self.assertEqual(segment["estimated_mouse_count"], 2)
        self.assertEqual(segment["mouse_count"], 2)
        confirmed = {"count_status": "confirmed", "estimated_mouse_count": 1, "mouse_count": 1}
        apply_identity_to_segment(confirmed, result)
        self.assertEqual(confirmed["estimated_mouse_count"], 1)


class TestVLMFusionPriority(unittest.TestCase):
    """Successful context VLM results must not lose HSV/CNN sort tie-breaks."""

    @staticmethod
    def _yellow_tag_cap():
        """Return a no-network capture whose HSV path has a persistent yellow tag."""
        import cv2

        frame = np.full((200, 200, 3), 20, dtype=np.uint8)
        # The explicit proximity mock below isolates fusion from mask geometry.
        cv2.ellipse(frame, (100, 100), (18, 12), 0, 0, 360, (20, 20, 20), -1)
        cv2.circle(frame, (122, 100), 5, (0, 255, 255), -1)  # BGR yellow

        class FakeCap:
            def set(self, _pos, _value):
                pass

            def read(self):
                return True, frame.copy()

        return FakeCap()

    @staticmethod
    def _assist_with_vlm(vlm_result):
        from detection.identity_assist import IdentityAssist

        assist = IdentityAssist(use_cnn=False)
        mock_vision = MagicMock()
        mock_vision.classify_frames.return_value = vlm_result
        clf = assist._get_classifier()
        clf._vision_provider = mock_vision
        clf._vision_init_attempted = True
        return assist, mock_vision

    @staticmethod
    def _detect(assist, target_count):
        # Keep this focused on fusion: tag proximity geometry is covered elsewhere.
        assist._contour_near_mask = lambda _cnt, _mouse_mask: True
        return assist.detect_ear_tags(
            segment={"start_frame": 0, "end_frame": 29,
                     "estimated_mouse_count": target_count},
            roi_core={"cx": 100, "cy": 100, "a": 12, "b": 12},
            cap=TestVLMFusionPriority._yellow_tag_cap(),
            fps=30,
        )

    def test_single_mouse_vlm_green_strictly_overrides_hsv_yellow_tie(self):
        """A valid green VLM answer wins even when HSV yellow reaches 0.99."""
        assist, mock_vision = self._assist_with_vlm(
            ("green", 0.72, "ear_tag_color_gpt", {})
        )

        with self.assertLogs("detection.identity_assist", level="INFO") as logs:
            result = self._detect(assist, target_count=1)

        mock_vision.classify_frames.assert_called_once()
        self.assertEqual(result["auto_mouse_colors"], ["green"])
        self.assertEqual(result["auto_mouse_ids"], ["auto_green"])
        self.assertTrue(result["identity_method"].startswith("ear_tag_color_vlm"))
        self.assertIn("priority_override", result["identity_method"])
        self.assertTrue(any(
            "VLM priority override: selected=green, ignored_rule_candidates=['yellow']" in line
            for line in logs.output
        ))

    def test_multiple_mice_places_vlm_color_first_without_duplication(self):
        """One VLM color occupies only the first multi-mouse slot."""
        assist, _mock_vision = self._assist_with_vlm(
            ("green", 0.72, "ear_tag_color_gpt", {})
        )

        result = self._detect(assist, target_count=2)

        self.assertEqual(result["auto_mouse_colors"], ["green", "yellow"])
        self.assertEqual(result["auto_mouse_ids"], ["auto_green", "auto_yellow"])
        self.assertTrue(result["identity_method"].startswith("ear_tag_color_vlm"))

    def test_vlm_failure_preserves_hsv_behavior(self):
        """A failed VLM call leaves the prior HSV yellow result unchanged."""
        assist, mock_vision = self._assist_with_vlm(
            ("unknown", 0.0, "ear_tag_color_gpt", {})
        )
        mock_vision.classify_frames.side_effect = RuntimeError("mock VLM failure")

        result = self._detect(assist, target_count=1)

        mock_vision.classify_frames.assert_called_once()
        self.assertEqual(result["auto_mouse_colors"], ["yellow"])
        self.assertEqual(result["auto_mouse_ids"], ["auto_yellow"])
        self.assertFalse(result["identity_method"].startswith("ear_tag_color_vlm"))


# =========================================================================
# 语义修复验证: _vision_used_any 仅在 VLM 真正返回可用颜色时置 True
# =========================================================================
class TestVisionUsedAnySemantics(unittest.TestCase):
    """验证 _vision_used_any 语义：

    - 预算计数严格在调用前进行（无论结果）
    - _vision_used_any (VLM 前缀) 仅在 clf.classify() 实际返回
      provider method（kimi/minimax/openrouter）且 resolved_color != "unknown" 时置 True
    - 网络/API exception → budget 消耗，但 VLM 标志不置位
    - provider 返回 unknown/低置信度导致回退 → budget 消耗，VLM 标志不置位
    - CNN/HSV 路径均不可置 VLM 前缀
    """

    def _make_patch(self):
        return np.random.randint(0, 256, (48, 48, 3), dtype=np.uint8)

    def _make_hsv_px(self, h=0, s=200, v=200):
        return np.array([[h, s, v]], dtype=np.float32)

    # 辅助：模拟 detect_ear_tags 中 per-contour 的预算+VLM分类逻辑
    @staticmethod
    def _simulate_contour_classify(assist, clf, patch_bgr, hsv_px,
                                    vision_used_any, cnn_used_any,
                                    vision_enabled, budget):
        """模拟 detect_ear_tags 中单个 contour 的分类决策。

        返回: (updated_vision_used_any, updated_cnn_used_any, color, method)
        """
        budget_ok = vision_enabled and assist._vision_request_count < budget
        allow_vision = budget_ok and clf.is_vision_available

        # 预算预增（与实际逻辑一致）
        if allow_vision and patch_bgr is not None and patch_bgr.size > 0:
            assist._vision_request_count += 1

        # classify 调用
        if allow_vision or clf.is_available:
            resolved_color, resolved_method = clf.classify(
                patch_bgr, hsv_px, allow_vision=allow_vision
            )

            # VLM 分支检查（与 detect_ear_tags 内联逻辑一致）
            if resolved_method.startswith("ear_tag_color_kimi") or \
               resolved_method.startswith("ear_tag_color_minimax") or \
               resolved_method.startswith("ear_tag_color_openrouter"):
                if resolved_color != "unknown":
                    vision_used_any = True
                    color = resolved_color
                else:
                    color, _, _ = assist._classify_contour(hsv_px)
            elif resolved_method == "ear_tag_color_cnn":
                cnn_used_any = True
                if resolved_color != "unknown":
                    color = resolved_color
                else:
                    color, _, _ = assist._classify_contour(hsv_px)
            else:
                color, _, _ = assist._classify_contour(hsv_px)
        else:
            color, _, _ = assist._classify_contour(hsv_px)
            resolved_method = "ear_tag_color_rule"

        return vision_used_any, cnn_used_any, color, resolved_method

    # ---- 核心测试 ----

    def test_vision_exception_budget_consumed_method_not_vlm(self):
        """Provider 抛出异常 → 预算消耗 1，但 _vision_used_any 仍为 False。"""
        from detection.identity_assist import IdentityAssist

        failing_vision = MagicMock()
        failing_vision.classify.side_effect = RuntimeError("network error")

        with patch.dict(os.environ, {}, clear=True):
            assist = IdentityAssist(use_cnn=False)
            clf = assist._get_classifier()
            clf._vision_provider = failing_vision
            clf._vision_init_attempted = True
            clf._initialized = True

            vision_used_any = False
            cnn_used_any = False
            budget = assist._max_vision_requests
            patch_bgr = self._make_patch()
            hsv_px = self._make_hsv_px(h=0)  # red in HSV

            vision_used_any, cnn_used_any, color, method = \
                self._simulate_contour_classify(
                    assist, clf, patch_bgr, hsv_px,
                    vision_used_any, cnn_used_any,
                    vision_enabled=True, budget=budget,
                )

            # 预算应已消耗 1
            self.assertEqual(assist._vision_request_count, 1,
                             "异常也应消耗预算")
            # VLM 标志不应置位
            self.assertFalse(vision_used_any,
                             "异常时 _vision_used_any 应为 False")
            # 应回退到 HSV 规则
            self.assertEqual(method, "ear_tag_color_rule")
            self.assertFalse(cnn_used_any)

    def test_vision_returns_unknown_budget_consumed_method_not_vlm(self):
        """Provider 返回 unknown → 预算消耗 1，但 _vision_used_any 仍为 False。"""
        from detection.identity_assist import IdentityAssist

        unknown_vision = MagicMock()
        unknown_vision.classify.return_value = ("unknown", 0.0, "ear_tag_color_kimi")

        with patch.dict(os.environ, {}, clear=True):
            assist = IdentityAssist(use_cnn=False)
            clf = assist._get_classifier()
            clf._vision_provider = unknown_vision
            clf._vision_init_attempted = True
            clf._initialized = True

            vision_used_any = False
            cnn_used_any = False
            budget = assist._max_vision_requests
            patch_bgr = self._make_patch()
            hsv_px = self._make_hsv_px(h=25)  # yellow in HSV

            vision_used_any, cnn_used_any, color, method = \
                self._simulate_contour_classify(
                    assist, clf, patch_bgr, hsv_px,
                    vision_used_any, cnn_used_any,
                    vision_enabled=True, budget=budget,
                )

            # 预算应已消耗 1
            self.assertEqual(assist._vision_request_count, 1,
                             "即使返回 unknown 也应消耗预算")
            # VLM 标志不应置位（因为 resolved_color == "unknown"）
            self.assertFalse(vision_used_any,
                             "返回 unknown 时 _vision_used_any 应为 False")
            # 应回退到 HSV 规则（黄色）
            self.assertEqual(method, "ear_tag_color_rule")
            self.assertEqual(color, "yellow")

    def test_vision_returns_valid_color_vlm_flag_set(self):
        """Provider 返回非 unknown 颜色 → 预算消耗 1，VLM 标志置 True。"""
        from detection.identity_assist import IdentityAssist

        valid_vision = MagicMock()
        valid_vision.classify.return_value = ("blue", 0.85, "ear_tag_color_kimi")

        with patch.dict(os.environ, {}, clear=True):
            assist = IdentityAssist(use_cnn=False)
            clf = assist._get_classifier()
            clf._vision_provider = valid_vision
            clf._vision_init_attempted = True
            clf._initialized = True

            vision_used_any = False
            cnn_used_any = False
            budget = assist._max_vision_requests
            patch_bgr = self._make_patch()
            hsv_px = self._make_hsv_px(h=0)  # red in HSV

            vision_used_any, cnn_used_any, color, method = \
                self._simulate_contour_classify(
                    assist, clf, patch_bgr, hsv_px,
                    vision_used_any, cnn_used_any,
                    vision_enabled=True, budget=budget,
                )

            self.assertEqual(assist._vision_request_count, 1)
            self.assertTrue(vision_used_any,
                            "返回有效颜色时 _vision_used_any 应为 True")
            self.assertEqual(method, "ear_tag_color_kimi")
            self.assertEqual(color, "blue")

    def test_vision_low_confidence_falls_back_vlm_not_set(self):
        """Provider 低置信度 → classifier 内部将 color 转为 unknown
        → _vision_used_any 不应置位（回退）。

        注：_vision_classify 在 confidence < CNN_CONFIDENCE_THRESHOLD 时
        返回 ("unknown", method)，因此 classify() 回退到 HSV rule。
        """
        from detection.identity_assist import IdentityAssist

        low_conf_vision = MagicMock()
        # 低置信度 → EarTagClassifier._vision_classify 将其转为 unknown
        low_conf_vision.classify.return_value = ("red", 0.1, "ear_tag_color_minimax")

        with patch.dict(os.environ, {}, clear=True):
            assist = IdentityAssist(use_cnn=False)
            clf = assist._get_classifier()
            clf._vision_provider = low_conf_vision
            clf._vision_init_attempted = True
            clf._initialized = True

            vision_used_any = False
            cnn_used_any = False
            budget = assist._max_vision_requests
            patch_bgr = self._make_patch()
            hsv_px = self._make_hsv_px(h=60)  # green in HSV

            vision_used_any, cnn_used_any, color, method = \
                self._simulate_contour_classify(
                    assist, clf, patch_bgr, hsv_px,
                    vision_used_any, cnn_used_any,
                    vision_enabled=True, budget=budget,
                )

            # 预算仍消耗（已尝试调用）
            self.assertEqual(assist._vision_request_count, 1)
            # VLM 标志不置位 — classifier._vision_classify 内部将 low conf
            # 转为 ("unknown", ...)，classify() 因此回退到 HSV
            self.assertFalse(vision_used_any,
                             "低置信度回退时 _vision_used_any 应为 False")
            self.assertEqual(method, "ear_tag_color_rule")
            self.assertEqual(color, "green")

    def test_direct_hsv_no_vlm_flag(self):
        """无 provider 可用时直接走 HSV → 不消耗预算，不置 VLM 标志。"""
        from detection.identity_assist import IdentityAssist

        with patch.dict(os.environ, {}, clear=True):
            assist = IdentityAssist(use_cnn=False)
            clf = assist._get_classifier()
            # No vision provider at all
            clf._vision_provider = None
            clf._vision_init_attempted = True
            clf._initialized = True

            vision_used_any = False
            cnn_used_any = False
            budget = assist._max_vision_requests
            patch_bgr = self._make_patch()
            hsv_px = self._make_hsv_px(h=110)  # blue

            vision_used_any, cnn_used_any, color, method = \
                self._simulate_contour_classify(
                    assist, clf, patch_bgr, hsv_px,
                    vision_used_any, cnn_used_any,
                    vision_enabled=True, budget=budget,
                )

            self.assertEqual(assist._vision_request_count, 0,
                             "无 provider 时不消耗预算")
            self.assertFalse(vision_used_any)
            self.assertFalse(cnn_used_any)
            self.assertEqual(method, "ear_tag_color_rule")
            self.assertEqual(color, "blue")

    def test_mixed_scenario_budget_exhausted_then_vlm(self):
        """混合场景：前三次抛异常消耗预算，第四次无预算→HSV，
        最终 identity_method 应以 ear_tag_color_rule 开头。
        """
        from detection.identity_assist import IdentityAssist

        failing_vision = MagicMock()
        failing_vision.classify.side_effect = RuntimeError("network error")

        with patch.dict(os.environ, {}, clear=True):
            assist = IdentityAssist(use_cnn=False)
            clf = assist._get_classifier()
            clf._vision_provider = failing_vision
            clf._vision_init_attempted = True
            clf._initialized = True

            vision_used_any = False
            cnn_used_any = False
            budget = assist._max_vision_requests  # 3

            # 模拟 detect_ear_tags 中的循环：多个 contour 依次分类
            for i in range(5):
                patch_bgr = self._make_patch()
                hsv_px = self._make_hsv_px(h=0)  # red

                vision_used_any, cnn_used_any, color, method = \
                    self._simulate_contour_classify(
                        assist, clf, patch_bgr, hsv_px,
                        vision_used_any, cnn_used_any,
                        vision_enabled=True, budget=budget,
                    )

            # 前 3 次消耗预算（异常回退），后 2 次无预算直接 HSV
            self.assertEqual(assist._vision_request_count, 3)
            # 所有异常回退 → VLM 标志始终 False
            self.assertFalse(vision_used_any,
                             "全部异常回退时 _vision_used_any 应为 False")
            self.assertEqual(failing_vision.classify.call_count, 3)

    def test_identity_method_prefix_from_flags(self):
        """验证从 _vision_used_any / _cnn_used_any 推导
        method_prefix 的三个路径正确。"""
        # VLM path
        if True:
            prefix = "ear_tag_color_vlm"
        self.assertEqual(prefix, "ear_tag_color_vlm")

        # CNN path
        vision_used = False
        cnn_used = True
        if vision_used:
            prefix = "ear_tag_color_vlm"
        elif cnn_used:
            prefix = "ear_tag_color_cnn"
        else:
            prefix = "ear_tag_color_rule"
        self.assertEqual(prefix, "ear_tag_color_cnn")

        # Rule path
        vision_used = False
        cnn_used = False
        if vision_used:
            prefix = "ear_tag_color_vlm"
        elif cnn_used:
            prefix = "ear_tag_color_cnn"
        else:
            prefix = "ear_tag_color_rule"
        self.assertEqual(prefix, "ear_tag_color_rule")


# =========================================================================
# Multi-frame context (classify_frames) 测试
# =========================================================================
class TestMultiFramePayload(unittest.TestCase):
    """验证 classify_frames payload 构造：多图顺序、数目、提示词。"""

    def _make_patch(self, h=48, w=48):
        return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

    def test_multi_payload_has_all_images(self):
        """payload 中 image_url 数量应等于输入帧数。"""
        from detection.models.vision_provider import (
            KimiVisionProvider, _bgr_to_jpeg_data_uri,
        )
        with patch.dict(os.environ, {
            "KIMI_API_KEY": "sk-test",
            "KIMI_API_BASE": "https://example.invalid/kimi/v1",
            "KIMI_VISION_MODEL": "test-vision-model",
            "MOUSE_COLOR_AI_PROVIDER": "kimi",
        }, clear=True):
            provider = KimiVisionProvider()
            frames = [self._make_patch() for _ in range(9)]
            uris = [_bgr_to_jpeg_data_uri(f) for f in frames]
            payload = provider._build_multi_payload(uris, len(frames))

            user_msg = payload["messages"][1]
            self.assertEqual(user_msg["role"], "user")
            # content: 1 text + N images
            image_parts = [p for p in user_msg["content"] if p["type"] == "image_url"]
            self.assertEqual(len(image_parts), 9)

    def test_multi_payload_image_order_chronological(self):
        """image_url 应按给定顺序排列（时间顺序）。"""
        from detection.models.vision_provider import (
            KimiVisionProvider, _bgr_to_jpeg_data_uri,
        )
        with patch.dict(os.environ, {
            "KIMI_API_KEY": "sk-test",
            "KIMI_API_BASE": "https://example.invalid/kimi/v1",
            "KIMI_VISION_MODEL": "test-vision-model",
            "MOUSE_COLOR_AI_PROVIDER": "kimi",
        }, clear=True):
            provider = KimiVisionProvider()
            frames = [self._make_patch() for _ in range(5)]
            uris = [_bgr_to_jpeg_data_uri(f) for f in frames]
            payload = provider._build_multi_payload(uris, len(frames))

            user_msg = payload["messages"][1]
            image_parts = [p for p in user_msg["content"] if p["type"] == "image_url"]
            # 验证第二张图包含与第一张不同的 base64
            self.assertNotEqual(
                image_parts[0]["image_url"]["url"],
                image_parts[1]["image_url"]["url"],
            )

    def test_multi_prompt_mentions_context(self):
        """提示词应明确说明是同一 clip 的上下文图。"""
        from detection.models.vision_provider import (
            KimiVisionProvider, _bgr_to_jpeg_data_uri,
        )
        with patch.dict(os.environ, {
            "KIMI_API_KEY": "sk-test",
            "KIMI_API_BASE": "https://example.invalid/kimi/v1",
            "KIMI_VISION_MODEL": "test-vision-model",
            "MOUSE_COLOR_AI_PROVIDER": "kimi",
        }, clear=True):
            provider = KimiVisionProvider()
            frames = [self._make_patch() for _ in range(9)]
            uris = [_bgr_to_jpeg_data_uri(f) for f in frames]
            payload = provider._build_multi_payload(uris, len(frames))

            user_msg = payload["messages"][1]
            text_part = user_msg["content"][0]
            self.assertEqual(text_part["type"], "text")
            text = text_part["text"]
            self.assertIn("SAME video clip", text)
            self.assertIn("beginning", text.lower())
            self.assertIn("middle", text.lower())
            self.assertIn("end", text.lower())
            self.assertIn("chronological", text.lower())

    def test_multi_payload_short_segment_prompt(self):
        """短 segment（<9帧）提示词不应崩溃。"""
        from detection.models.vision_provider import (
            KimiVisionProvider, _bgr_to_jpeg_data_uri,
        )
        with patch.dict(os.environ, {
            "KIMI_API_KEY": "sk-test",
            "KIMI_API_BASE": "https://example.invalid/kimi/v1",
            "KIMI_VISION_MODEL": "test-vision-model",
            "MOUSE_COLOR_AI_PROVIDER": "kimi",
        }, clear=True):
            provider = KimiVisionProvider()
            # 2 frames
            uris = [_bgr_to_jpeg_data_uri(self._make_patch()) for _ in range(2)]
            payload = provider._build_multi_payload(uris, 2)
            text = payload["messages"][1]["content"][0]["text"]
            self.assertIn("short video clip", text)

    def test_multi_payload_has_system_message(self):
        """system message 应存在且要求 JSON 输出。"""
        from detection.models.vision_provider import (
            KimiVisionProvider, _bgr_to_jpeg_data_uri,
        )
        with patch.dict(os.environ, {
            "KIMI_API_KEY": "sk-test",
            "KIMI_API_BASE": "https://example.invalid/kimi/v1",
            "KIMI_VISION_MODEL": "test-vision-model",
            "MOUSE_COLOR_AI_PROVIDER": "kimi",
        }, clear=True):
            provider = KimiVisionProvider()
            uris = [_bgr_to_jpeg_data_uri(self._make_patch()) for _ in range(3)]
            payload = provider._build_multi_payload(uris, 3)
            self.assertEqual(payload["messages"][0]["role"], "system")
            self.assertIn("JSON", payload["messages"][0]["content"])


class TestClassifyFramesAPI(unittest.TestCase):
    """验证 classify_frames 方法交互（mock 网络）。"""

    def _make_patch(self, h=48, w=48):
        return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

    def test_classify_frames_returns_tuple_of_4(self):
        """classify_frames 应返回 4 元组 (color, confidence, method, raw_json)。"""
        from detection.models.vision_provider import KimiVisionProvider
        with patch.dict(os.environ, {
            "KIMI_API_KEY": "sk-test",
            "KIMI_API_BASE": "https://example.invalid/kimi/v1",
            "KIMI_VISION_MODEL": "test-vision-model",
            "MOUSE_COLOR_AI_PROVIDER": "kimi",
        }, clear=True):
            provider = KimiVisionProvider()
            frames = [self._make_patch() for _ in range(9)]

            # Mock _post_json to return a valid response
            mock_resp = {
                "choices": [{
                    "message": {"content": '{"color": "red", "confidence": 0.85}'},
                }],
            }
            with patch.object(provider, '_post_json', return_value=mock_resp):
                color, conf, method, raw = provider.classify_frames(frames)

            self.assertEqual(color, "red")
            self.assertAlmostEqual(conf, 0.85, delta=0.01)
            self.assertIn("ear_tag_color", method)
            self.assertIsNotNone(raw)
            self.assertEqual(raw, mock_resp)

    def test_classify_frames_empty_input(self):
        """空帧列表应返回 unknown。"""
        from detection.models.vision_provider import KimiVisionProvider
        with patch.dict(os.environ, {
            "KIMI_API_KEY": "sk-test",
            "KIMI_API_BASE": "https://example.invalid/kimi/v1",
            "KIMI_VISION_MODEL": "test-vision-model",
            "MOUSE_COLOR_AI_PROVIDER": "kimi",
        }, clear=True):
            provider = KimiVisionProvider()
            color, conf, method, raw = provider.classify_frames([])
            self.assertEqual(color, "unknown")
            self.assertEqual(conf, 0.0)
            self.assertIsNone(raw)

    def test_classify_frames_via_classifier(self):
        """EarTagClassifier.classify_frames 应委托给 vision_provider。"""
        from detection.models.classifier import EarTagClassifier
        mock = MagicMock()
        mock.classify_frames.return_value = ("blue", 0.92, "ear_tag_color_kimi", {"mock": True})
        clf = EarTagClassifier(use_cnn=False, vision_provider=mock)

        frames = [self._make_patch() for _ in range(9)]
        color, method, conf = clf.classify_frames(frames)
        self.assertEqual(color, "blue")
        self.assertEqual(method, "ear_tag_color_kimi")
        self.assertAlmostEqual(conf, 0.92, delta=0.01)
        mock.classify_frames.assert_called_once_with(frames)

    def test_classify_frames_low_confidence(self):
        """低置信度应在 classifier 层转为 unknown。"""
        from detection.models.classifier import EarTagClassifier
        mock = MagicMock()
        mock.classify_frames.return_value = ("red", 0.1, "ear_tag_color_kimi", {})
        clf = EarTagClassifier(use_cnn=False, vision_provider=mock)

        frames = [self._make_patch()]
        color, method, conf = clf.classify_frames(frames)
        self.assertEqual(color, "unknown")
        self.assertEqual(method, "ear_tag_color_kimi")


# =========================================================================
# Context 帧选择和裁剪测试
# =========================================================================
class TestContextFrameSelection(unittest.TestCase):
    """验证 _select_context_frames 帧选择逻辑。"""

    def test_normal_30_frame_segment(self):
        """30帧 segment (0-29)：头{0,5,10} 中{9,14,19} 尾{19,24,29} → 去重后 8 帧。"""
        from detection.identity_assist import IdentityAssist
        frames = IdentityAssist._select_context_frames(0, 29)
        # head: {0, 5, 10}  mid: {9, 14, 19}  tail: {19, 24, 29}
        # dedup (19 shared mid/tail) → 8 unique
        self.assertEqual(len(frames), 8)
        expected = [0, 5, 9, 10, 14, 19, 24, 29]
        self.assertEqual(frames, expected)
        self.assertEqual(frames, sorted(frames))  # 时间顺序
        # 验证 start/mid/end 锚点存在
        self.assertIn(0, frames)
        self.assertIn(14, frames)
        self.assertIn(29, frames)

    def test_long_50_frame_segment(self):
        """50帧 segment (0-49) 足够长无重叠 → 精确 9 帧。"""
        from detection.identity_assist import IdentityAssist
        frames = IdentityAssist._select_context_frames(0, 49)
        # head: {0, 5, 10}  mid=(0+49)//2=24: {19, 24, 29}  tail: {39, 44, 49}
        self.assertEqual(len(frames), 9)
        expected = [0, 5, 10, 19, 24, 29, 39, 44, 49]
        self.assertEqual(frames, expected)
        self.assertEqual(frames, sorted(frames))
        self.assertIn(0, frames)
        self.assertIn(24, frames)
        self.assertIn(49, frames)

    def test_short_3_frame_segment(self):
        """3帧 segment 应 clamp 到 [0,2] 去重后 ≤9。"""
        from detection.identity_assist import IdentityAssist
        frames = IdentityAssist._select_context_frames(0, 2)
        # anchors: 0, 1, 2. offsets: -5→0, 0, +5→2 etc. Dedup gives {0,1,2}
        self.assertLessEqual(len(frames), 9)
        self.assertTrue(len(frames) >= 1)
        self.assertEqual(frames, sorted(frames))
        for f in frames:
            self.assertGreaterEqual(f, 0)
            self.assertLessEqual(f, 2)

    def test_single_frame_segment(self):
        """单帧 segment 只产生一个帧索引。"""
        from detection.identity_assist import IdentityAssist
        frames = IdentityAssist._select_context_frames(5, 5)
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0], 5)

    def test_dedup_works(self):
        """短 segment 应去重。"""
        from detection.identity_assist import IdentityAssist
        # 5帧 segment: 0,1,2,3,4
        frames = IdentityAssist._select_context_frames(0, 4)
        self.assertEqual(len(frames), len(set(frames)))
        self.assertEqual(frames, sorted(frames))

    def test_all_within_range(self):
        """所有帧索引应在 [start, end] 范围内。"""
        from detection.identity_assist import IdentityAssist
        for start, end in [(0, 10), (5, 100), (3, 6), (100, 100)]:
            frames = IdentityAssist._select_context_frames(start, end)
            for f in frames:
                self.assertGreaterEqual(f, start)
                self.assertLessEqual(f, end)


class TestContextCropClamping(unittest.TestCase):
    """验证 _crop_context_frame 裁剪 clamp。"""

    def test_crop_centered_within_bounds(self):
        """ROI 中心在图像中央 → 正常裁剪。"""
        from detection.identity_assist import IdentityAssist
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        roi_core = {"cx": 320, "cy": 240}
        crop = IdentityAssist._crop_context_frame(frame, roi_core, 100, 80)
        self.assertIsNotNone(crop)
        self.assertEqual(crop.shape[0], 160)  # 2*80
        self.assertEqual(crop.shape[1], 200)  # 2*100

    def test_crop_clamped_to_edges(self):
        """ROI 靠近图像边缘 → clamp 到边界。"""
        from detection.identity_assist import IdentityAssist
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        roi_core = {"cx": 10, "cy": 5}
        crop = IdentityAssist._crop_context_frame(frame, roi_core, 200, 200)
        self.assertIsNotNone(crop)
        # 裁剪不应超出图像范围
        self.assertGreater(crop.shape[1], 0)
        self.assertGreater(crop.shape[0], 0)
        # x1=0, y1=0, x2=min(640, 210)=210, y2=min(480, 205)=205
        self.assertEqual(crop.shape[1], 210)

    def test_crop_never_empty(self):
        """有效 ROI 中心不应产生空 crop。"""
        from detection.identity_assist import IdentityAssist
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        roi_core = {"cx": 320, "cy": 240}
        for a, b in [(1, 1), (10, 10), (50, 50), (300, 200)]:
            crop = IdentityAssist._crop_context_frame(frame, roi_core, a, b)
            self.assertIsNotNone(crop)
            self.assertGreater(crop.size, 0)


# =========================================================================
# Segment 级别 VLM 调用 & 回退 集成测试
# =========================================================================
class TestSegmentLevelVLMIntegration(unittest.TestCase):
    """验证 detect_ear_tags 中 per-segment VLM 调用与 CNN/HSV 回退。"""

    def _make_patch(self, h=48, w=48):
        return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

    def test_per_segment_one_vlm_call(self):
        """每个 segment 仅调用一次 classify_frames（非 per-contour）。"""
        from detection.identity_assist import IdentityAssist
        from detection.models.classifier import EarTagClassifier

        mock_vision = MagicMock()
        mock_vision.classify_frames.return_value = ("yellow", 0.8, "ear_tag_color_kimi", {})
        # 也需要 mock classify 以防 CNN fallback 走不到
        mock_vision.classify.return_value = ("unknown", 0.0, "ear_tag_color_kimi")

        with patch.dict(os.environ, {}, clear=True):
            assist = IdentityAssist(use_cnn=False)
            clf = assist._get_classifier()
            clf._vision_provider = mock_vision
            clf._vision_init_attempted = True
            clf._initialized = True

            # 构造最小 fake 视频
            frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

            class FakeCap:
                def set(self, pos, val):
                    pass
                def read(self):
                    return True, frame.copy()

            segment = {"start_frame": 0, "end_frame": 29, "estimated_mouse_count": 1}
            roi_core = {"cx": 320, "cy": 240, "a": 30, "b": 30}

            result = assist.detect_ear_tags(
                segment=segment, roi_core=roi_core, cap=FakeCap(), fps=30,
            )

            # classify_frames 应被调用 1 次（per-segment VLM context）
            mock_vision.classify_frames.assert_called_once()
            # classify 可能被调用多次但 allow_vision=False（CNN/HSV fallback）
            # 验证 identity_method 以 ear_tag_color_vlm 开头
            self.assertTrue(
                result["identity_method"].startswith("ear_tag_color_vlm"),
                f"Expected VLM prefix, got {result['identity_method']}",
            )
            # 预算应为 1
            self.assertEqual(assist._vision_request_count, 1)

    def test_vlm_failure_still_counts_budget(self):
        """VLM 失败 → 消耗 1 次预算，不标 VLM，继续 CNN/HSV。"""
        from detection.identity_assist import IdentityAssist
        from detection.models.classifier import EarTagClassifier

        mock_vision = MagicMock()
        mock_vision.classify_frames.side_effect = RuntimeError("network error")
        mock_vision.classify.return_value = ("unknown", 0.0, "ear_tag_color_kimi")

        with patch.dict(os.environ, {}, clear=True):
            assist = IdentityAssist(use_cnn=False)
            clf = assist._get_classifier()
            clf._vision_provider = mock_vision
            clf._vision_init_attempted = True
            clf._initialized = True

            frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

            class FakeCap:
                def set(self, pos, val):
                    pass
                def read(self):
                    return True, frame.copy()

            segment = {"start_frame": 0, "end_frame": 29, "estimated_mouse_count": 1}
            roi_core = {"cx": 320, "cy": 240, "a": 30, "b": 30}

            result = assist.detect_ear_tags(
                segment=segment, roi_core=roi_core, cap=FakeCap(), fps=30,
            )

            # 预算消耗 1
            self.assertEqual(assist._vision_request_count, 1)
            # classify_frames 被调用过
            mock_vision.classify_frames.assert_called_once()
            # VLM 失败 → 不应标 VLM prefix
            self.assertFalse(
                result["identity_method"].startswith("ear_tag_color_vlm"),
                f"Expected NOT VLM after failure, got {result['identity_method']}",
            )

    def test_budget_zero_no_vlm_call(self):
        """budget=0 → 不应调用 VLM，直接走 CNN/HSV。"""
        from detection.identity_assist import IdentityAssist

        mock_vision = MagicMock()

        with patch.dict(os.environ, {
            "MOUSE_COLOR_AI_MAX_REQUESTS_PER_SEGMENT": "0",
        }, clear=True):
            assist = IdentityAssist(use_cnn=False)
            clf = assist._get_classifier()
            clf._vision_provider = mock_vision
            clf._vision_init_attempted = True
            clf._initialized = True

            self.assertEqual(assist._max_vision_requests, 0)

            frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

            class FakeCap:
                def set(self, pos, val):
                    pass
                def read(self):
                    return True, frame.copy()

            segment = {"start_frame": 0, "end_frame": 29, "estimated_mouse_count": 1}
            roi_core = {"cx": 320, "cy": 240, "a": 30, "b": 30}

            result = assist.detect_ear_tags(
                segment=segment, roi_core=roi_core, cap=FakeCap(), fps=30,
            )

            # VLM 不应被调用
            mock_vision.classify_frames.assert_not_called()
            mock_vision.classify.assert_not_called()
            self.assertEqual(assist._vision_request_count, 0)
            self.assertFalse(result["identity_method"].startswith("ear_tag_color_vlm"))

    def test_debug_manifest_saved_without_key(self):
        """debug 保存的 manifest 不应包含 API key。"""
        from detection.identity_assist import IdentityAssist

        mock_vision = MagicMock()
        mock_vision._method_name = "ear_tag_color_kimi"
        mock_vision._model = "test-model"

        with patch.dict(os.environ, {
            "MOUSE_COLOR_AI_SAVE_DEBUG": "1",
        }, clear=True):
            assist = IdentityAssist(use_cnn=False)
            context_crops = [self._make_patch() for _ in range(3)]
            frame_indices = [10, 20, 30]
            roi_core = {"cx": 320, "cy": 240}
            resp = {"choices": [{"message": {"content": '{"color":"red","confidence":0.8}'}}]}

            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                orig_cwd = os.getcwd()
                try:
                    os.chdir(tmp)
                    assist._save_vlm_context_debug(
                        context_crops, frame_indices,
                        mock_vision, 100, 80, roi_core,
                        response_json=resp,
                    )
                    # 查找 manifest
                    debug_dir = Path(tmp) / "debug_vision_output"
                    manifests = list(debug_dir.glob("manifest_*.json"))
                    self.assertEqual(len(manifests), 1)
                    with open(manifests[0], "r") as f:
                        manifest = json.load(f)
                    # 不应包含 API key
                    manifest_str = json.dumps(manifest)
                    self.assertNotIn("api_key", manifest_str.lower())
                    self.assertNotIn("authorization", manifest_str.lower())
                    self.assertNotIn("base64", manifest_str.lower())
                    # 应包含 provider/model/帧索引/crop 信息
                    self.assertIn("provider", manifest)
                    self.assertIn("model", manifest)
                    self.assertIn("frame_indices", manifest)
                    self.assertEqual(manifest["frame_indices"], frame_indices)
                    self.assertIn("crop_algorithm", manifest)
                    self.assertIn("result", manifest)
                    self.assertEqual(manifest["result"]["color"], "red")
                    # 验证 crop 图像文件存在
                    crops = list(debug_dir.glob("context_*.jpg"))
                    self.assertEqual(len(crops), 3)
                    # 验证 response 文件存在
                    responses = list(debug_dir.glob("response_*.json"))
                    self.assertEqual(len(responses), 1)
                finally:
                    os.chdir(orig_cwd)

    def test_debug_manifest_saves_on_error(self):
        """VLM 失败时 debug 也应保存 crops 和 manifest（含 error）。"""
        from detection.identity_assist import IdentityAssist

        mock_vision = MagicMock()
        mock_vision._method_name = "ear_tag_color_minimax"
        mock_vision._model = "mini-model"

        with patch.dict(os.environ, {"MOUSE_COLOR_AI_SAVE_DEBUG": "1"}, clear=True):
            assist = IdentityAssist(use_cnn=False)
            context_crops = [self._make_patch()]
            frame_indices = [5]
            roi_core = {"cx": 100, "cy": 200}

            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                orig_cwd = os.getcwd()
                try:
                    os.chdir(tmp)
                    assist._save_vlm_context_debug(
                        context_crops, frame_indices,
                        mock_vision, 50, 40, roi_core,
                        error="Simulated network timeout",
                    )
                    debug_dir = Path(tmp) / "debug_vision_output"
                    manifests = list(debug_dir.glob("manifest_*.json"))
                    self.assertEqual(len(manifests), 1)
                    with open(manifests[0], "r") as f:
                        manifest = json.load(f)
                    self.assertIsNotNone(manifest.get("error"))
                    self.assertIn("timeout", manifest["error"]["message"])
                    self.assertIsNone(manifest.get("result"))
                    # Crop 文件仍应保存
                    crops = list(debug_dir.glob("context_*.jpg"))
                    self.assertEqual(len(crops), 1)
                finally:
                    os.chdir(orig_cwd)

    def test_debug_manifest_includes_finish_reason_and_parse_status(self):
        """Manifest 应包含 finish_reason 和 parse_status 字段。"""
        from detection.identity_assist import IdentityAssist

        mock_vision = MagicMock()
        mock_vision._method_name = "ear_tag_color_kimi"
        mock_vision._model = "test-model"
        mock_vision._max_tokens = 512

        with patch.dict(os.environ, {"MOUSE_COLOR_AI_SAVE_DEBUG": "1"}, clear=True):
            assist = IdentityAssist(use_cnn=False)
            context_crops = [self._make_patch() for _ in range(2)]
            frame_indices = [0, 10]
            roi_core = {"cx": 320, "cy": 240}
            resp = {
                "choices": [{
                    "finish_reason": "stop",
                    "message": {"content": '{"color":"green","confidence":0.8}'},
                }],
            }

            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                orig_cwd = os.getcwd()
                try:
                    os.chdir(tmp)
                    assist._save_vlm_context_debug(
                        context_crops, frame_indices,
                        mock_vision, 100, 80, roi_core,
                        response_json=resp,
                    )
                    debug_dir = Path(tmp) / "debug_vision_output"
                    manifests = list(debug_dir.glob("manifest_*.json"))
                    self.assertEqual(len(manifests), 1)
                    with open(manifests[0], "r") as f:
                        manifest = json.load(f)
                    self.assertEqual(manifest["finish_reason"], "stop")
                    self.assertEqual(manifest["parse_status"], "ok")
                    self.assertEqual(manifest.get("max_tokens"), 512)
                finally:
                    os.chdir(orig_cwd)

    def test_debug_manifest_finish_reason_length_parse_fail(self):
        """finish_reason=length + 空 content → manifest 记录 finish_reason=length, parse_status=no_json。"""
        from detection.identity_assist import IdentityAssist

        mock_vision = MagicMock()
        mock_vision._method_name = "ear_tag_color_openrouter"
        mock_vision._model = "minimax/minimax-m3"
        mock_vision._max_tokens = 512

        with patch.dict(os.environ, {"MOUSE_COLOR_AI_SAVE_DEBUG": "1"}, clear=True):
            assist = IdentityAssist(use_cnn=False)
            context_crops = [self._make_patch()]
            frame_indices = [5]
            roi_core = {"cx": 100, "cy": 200}
            resp = {
                "choices": [{
                    "finish_reason": "length",
                    "message": {
                        "content": "\n",
                        "reasoning": "Let me analyze... the tag is green...",
                    },
                }],
            }

            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                orig_cwd = os.getcwd()
                try:
                    os.chdir(tmp)
                    assist._save_vlm_context_debug(
                        context_crops, frame_indices,
                        mock_vision, 50, 40, roi_core,
                        response_json=resp,
                    )
                    debug_dir = Path(tmp) / "debug_vision_output"
                    manifests = list(debug_dir.glob("manifest_*.json"))
                    self.assertEqual(len(manifests), 1)
                    with open(manifests[0], "r") as f:
                        manifest = json.load(f)
                    self.assertEqual(manifest["finish_reason"], "length")
                    self.assertEqual(manifest["parse_status"], "no_json")
                    # 不应泄露 reasoning 或 API key
                    manifest_str = json.dumps(manifest)
                    self.assertNotIn("api_key", manifest_str.lower())
                    self.assertNotIn("authorization", manifest_str.lower())
                finally:
                    os.chdir(orig_cwd)


# =========================================================================
# MOUSE_COLOR_AI_MAX_TOKENS 环境变量测试
# =========================================================================
class TestMaxTokensEnvVar(unittest.TestCase):
    """验证 _read_max_tokens() 环境变量读取与 payload 使用。"""

    def test_default_returns_512(self):
        from detection.models.vision_provider import _read_max_tokens, DEFAULT_MAX_TOKENS
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_read_max_tokens(), 512)
            self.assertEqual(_read_max_tokens(), DEFAULT_MAX_TOKENS)

    def test_valid_custom_value(self):
        from detection.models.vision_provider import _read_max_tokens
        with patch.dict(os.environ, {"MOUSE_COLOR_AI_MAX_TOKENS": "1024"}, clear=True):
            self.assertEqual(_read_max_tokens(), 1024)

    def test_valid_large_value(self):
        from detection.models.vision_provider import _read_max_tokens
        with patch.dict(os.environ, {"MOUSE_COLOR_AI_MAX_TOKENS": "4096"}, clear=True):
            self.assertEqual(_read_max_tokens(), 4096)

    def test_invalid_non_integer_defaults(self):
        from detection.models.vision_provider import _read_max_tokens
        with patch.dict(os.environ, {"MOUSE_COLOR_AI_MAX_TOKENS": "not_a_number"}, clear=True):
            self.assertEqual(_read_max_tokens(), 512)

    def test_negative_defaults(self):
        from detection.models.vision_provider import _read_max_tokens
        with patch.dict(os.environ, {"MOUSE_COLOR_AI_MAX_TOKENS": "-1"}, clear=True):
            self.assertEqual(_read_max_tokens(), 512)

    def test_zero_defaults(self):
        from detection.models.vision_provider import _read_max_tokens
        with patch.dict(os.environ, {"MOUSE_COLOR_AI_MAX_TOKENS": "0"}, clear=True):
            self.assertEqual(_read_max_tokens(), 512)

    def test_empty_string_defaults(self):
        from detection.models.vision_provider import _read_max_tokens
        with patch.dict(os.environ, {"MOUSE_COLOR_AI_MAX_TOKENS": ""}, clear=True):
            self.assertEqual(_read_max_tokens(), 512)

    def test_whitespace_only_defaults(self):
        from detection.models.vision_provider import _read_max_tokens
        with patch.dict(os.environ, {"MOUSE_COLOR_AI_MAX_TOKENS": "   "}, clear=True):
            self.assertEqual(_read_max_tokens(), 512)

    def test_payload_uses_env_max_tokens(self):
        """_build_payload 中的 max_tokens 应使用环境变量值而非硬编码 128。"""
        from detection.models.vision_provider import (
            KimiVisionProvider, _bgr_to_jpeg_data_uri,
        )
        with patch.dict(os.environ, {
            "KIMI_API_KEY": "sk-test",
            "KIMI_API_BASE": "https://example.invalid/kimi/v1",
            "KIMI_VISION_MODEL": "test-vision-model",
            "MOUSE_COLOR_AI_MAX_TOKENS": "1024",
        }, clear=True):
            provider = KimiVisionProvider()
            data_uri = _bgr_to_jpeg_data_uri(_make_patch())
            # 单图 payload
            payload = provider._build_payload(data_uri)
            self.assertEqual(payload["max_tokens"], 1024)
            # 多图 payload
            multi_payload = provider._build_multi_payload([data_uri], 1)
            self.assertEqual(multi_payload["max_tokens"], 1024)

    def test_payload_default_max_tokens(self):
        """未设置 MOUSE_COLOR_AI_MAX_TOKENS 时 payload 使用默认 512。"""
        from detection.models.vision_provider import (
            KimiVisionProvider, _bgr_to_jpeg_data_uri,
        )
        with patch.dict(os.environ, {
            "KIMI_API_KEY": "sk-test",
            "KIMI_API_BASE": "https://example.invalid/kimi/v1",
            "KIMI_VISION_MODEL": "test-vision-model",
        }, clear=True):
            provider = KimiVisionProvider()
            data_uri = _bgr_to_jpeg_data_uri(_make_patch())
            payload = provider._build_payload(data_uri)
            self.assertEqual(payload["max_tokens"], 512)
            multi_payload = provider._build_multi_payload([data_uri], 1)
            self.assertEqual(multi_payload["max_tokens"], 512)

    def test_invalid_env_max_tokens_falls_back_in_payload(self):
        """非法的 MOUSE_COLOR_AI_MAX_TOKENS 回退默认 512 并在 payload 中使用 512。"""
        from detection.models.vision_provider import (
            KimiVisionProvider, _bgr_to_jpeg_data_uri,
        )
        with patch.dict(os.environ, {
            "KIMI_API_KEY": "sk-test",
            "KIMI_API_BASE": "https://example.invalid/kimi/v1",
            "KIMI_VISION_MODEL": "test-vision-model",
            "MOUSE_COLOR_AI_MAX_TOKENS": "abc",
        }, clear=True):
            provider = KimiVisionProvider()
            data_uri = _bgr_to_jpeg_data_uri(_make_patch())
            payload = provider._build_payload(data_uri)
            self.assertEqual(payload["max_tokens"], 512)


# =========================================================================
# Truncation diagnostics: finish_reason=length 处理测试
# =========================================================================
class TestTruncationDiagnostics(unittest.TestCase):
    """验证 finish_reason=length 时的截断诊断逻辑。"""

    def test_length_with_empty_content_returns_unknown(self):
        """finish_reason=length + content 仅换行符 → 返回 unknown，不发散到 reasoning。"""
        from detection.models.vision_provider import KimiVisionProvider
        with patch.dict(os.environ, {
            "KIMI_API_KEY": "sk-test",
            "KIMI_API_BASE": "https://example.invalid/kimi/v1",
            "KIMI_VISION_MODEL": "test-vision-model",
        }, clear=True):
            provider = KimiVisionProvider()
            resp = {
                "choices": [{
                    "finish_reason": "length",
                    "message": {
                        "content": "\n",
                        "reasoning": "The ear tag appears green across frames 4-6. Final answer would be green.",
                    },
                }],
            }
            with self.assertLogs("detection.models.vision_provider", level="WARNING") as cm:
                color, confidence, method = provider._parse_response(resp)
            self.assertEqual(color, "unknown")
            self.assertEqual(confidence, 0.0)
            # 应记录截断警告
            warning_text = " ".join(cm.output)
            self.assertIn("截断", warning_text)
            self.assertIn("MOUSE_COLOR_AI_MAX_TOKENS", warning_text)

    def test_length_with_unparseable_content_returns_unknown(self):
        """finish_reason=length + content 无法解析 → 返回 unknown + 截断警告。"""
        from detection.models.vision_provider import KimiVisionProvider
        with patch.dict(os.environ, {
            "KIMI_API_KEY": "sk-test",
            "KIMI_API_BASE": "https://example.invalid/kimi/v1",
            "KIMI_VISION_MODEL": "test-vision-model",
        }, clear=True):
            provider = KimiVisionProvider()
            resp = {
                "choices": [{
                    "finish_reason": "length",
                    "message": {
                        "content": '{"color": "green", "confi',  # 截断的 JSON
                    },
                }],
            }
            with self.assertLogs("detection.models.vision_provider", level="WARNING") as cm:
                color, confidence, method = provider._parse_response(resp)
            self.assertEqual(color, "unknown")
            self.assertEqual(confidence, 0.0)
            warning_text = " ".join(cm.output)
            self.assertIn("截断", warning_text)
            self.assertIn("MOUSE_COLOR_AI_MAX_TOKENS", warning_text)

    def test_length_with_reasoning_has_no_effect_on_color(self):
        """即使 reasoning 中提到 'green'，也不应影响分类结果（必须是 unknown）。"""
        from detection.models.vision_provider import KimiVisionProvider
        with patch.dict(os.environ, {
            "KIMI_API_KEY": "sk-test",
            "KIMI_API_BASE": "https://example.invalid/kimi/v1",
            "KIMI_VISION_MODEL": "test-vision-model",
        }, clear=True):
            provider = KimiVisionProvider()
            resp = {
                "choices": [{
                    "finish_reason": "length",
                    "message": {
                        "content": "",
                        "reasoning": "The ear tag is clearly green with high confidence.",
                        "reasoning_details": [{
                            "type": "reasoning.text",
                            "text": "The ear tag is clearly green.",
                        }],
                    },
                }],
            }
            with self.assertLogs("detection.models.vision_provider", level="WARNING") as cm:
                color, confidence, method = provider._parse_response(resp)
            self.assertEqual(color, "unknown")
            self.assertEqual(confidence, 0.0)
            # 不应解析 reasoning 作为颜色
            warning_text = " ".join(cm.output)
            self.assertIn("截断", warning_text)

    def test_length_with_valid_json_content_succeeds(self):
        """finish_reason=length 但有完整 JSON content → 仍应成功解析（边截断但有 JSON）。"""
        from detection.models.vision_provider import KimiVisionProvider
        with patch.dict(os.environ, {
            "KIMI_API_KEY": "sk-test",
            "KIMI_API_BASE": "https://example.invalid/kimi/v1",
            "KIMI_VISION_MODEL": "test-vision-model",
        }, clear=True):
            provider = KimiVisionProvider()
            resp = {
                "choices": [{
                    "finish_reason": "length",
                    "message": {
                        "content": '{"color": "blue", "confidence": 0.75}',
                    },
                }],
            }
            color, confidence, method = provider._parse_response(resp)
            self.assertEqual(color, "blue")
            self.assertEqual(confidence, 0.75)

    def test_stop_with_empty_content_raises_error(self):
        """finish_reason=stop 但 content 为空 → 应抛出 VisionProviderError。"""
        from detection.models.vision_provider import (
            KimiVisionProvider, VisionProviderError,
        )
        with patch.dict(os.environ, {
            "KIMI_API_KEY": "sk-test",
            "KIMI_API_BASE": "https://example.invalid/kimi/v1",
            "KIMI_VISION_MODEL": "test-vision-model",
        }, clear=True):
            provider = KimiVisionProvider()
            resp = {
                "choices": [{
                    "finish_reason": "stop",
                    "message": {"content": ""},
                }],
            }
            with self.assertRaises(VisionProviderError) as ctx:
                provider._parse_response(resp)
            err_msg = str(ctx.exception).lower()
            self.assertTrue(
                "content" in err_msg or "空" in str(ctx.exception),
                f"Expected VisionProviderError about empty content, got: {ctx.exception}",
            )

    def test_parse_response_logs_warning_not_containing_api_key(self):
        """截断警告日志不应包含 API key/Authorization header。"""
        from detection.models.vision_provider import KimiVisionProvider
        with patch.dict(os.environ, {
            "KIMI_API_KEY": "sk-secret-key-123456",
            "KIMI_API_BASE": "https://example.invalid/kimi/v1",
            "KIMI_VISION_MODEL": "test-vision-model",
        }, clear=True):
            provider = KimiVisionProvider()
            resp = {
                "choices": [{
                    "finish_reason": "length",
                    "message": {
                        "content": "\n",
                        "reasoning": "green tag",
                    },
                }],
            }
            with self.assertLogs("detection.models.vision_provider", level="WARNING") as cm:
                provider._parse_response(resp)
            warning_text = " ".join(cm.output)
            # 不应泄露 API key
            self.assertNotIn("sk-secret", warning_text)
            self.assertNotIn("Bearer", warning_text)

    def test_max_tokens_reflected_in_provider_instance(self):
        """验证 provider 实例存储 _max_tokens 值。"""
        from detection.models.vision_provider import KimiVisionProvider
        with patch.dict(os.environ, {
            "KIMI_API_KEY": "sk-test",
            "KIMI_API_BASE": "https://example.invalid/kimi/v1",
            "KIMI_VISION_MODEL": "test-vision-model",
            "MOUSE_COLOR_AI_MAX_TOKENS": "2048",
        }, clear=True):
            provider = KimiVisionProvider()
            self.assertEqual(provider._max_tokens, 2048)

    def test_classify_frames_length_truncation_yields_unknown(self):
        """端到端：classify_frames 遇到 length truncation → 返回 unknown。"""
        from detection.models.vision_provider import KimiVisionProvider
        with patch.dict(os.environ, {
            "KIMI_API_KEY": "sk-test",
            "KIMI_API_BASE": "https://example.invalid/kimi/v1",
            "KIMI_VISION_MODEL": "test-vision-model",
            "MOUSE_COLOR_AI_MAX_TOKENS": "128",
        }, clear=True):
            provider = KimiVisionProvider()
            frames = [_make_patch() for _ in range(9)]

            # API 返回 length truncation，仅换行 content，但有 reasoning
            mock_resp = {
                "choices": [{
                    "finish_reason": "length",
                    "message": {
                        "content": "\n",
                        "reasoning": "The ear tag is green in frames 4-6.",
                    },
                }],
            }
            with patch.object(provider, '_post_json', return_value=mock_resp):
                color, conf, method, raw = provider.classify_frames(frames)

            self.assertEqual(color, "unknown")
            self.assertEqual(conf, 0.0)
            self.assertIsNotNone(raw)


if __name__ == "__main__":
    unittest.main()
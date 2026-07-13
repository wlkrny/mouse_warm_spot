"""
单元测试: 耳标颜色分类器 & IdentityAssist 无模型构造

测试范围:
  - EarTagClassifier 规则回退（五种颜色 + 空输入）
  - EarTagClassifier._preprocess 形状/dtype
  - EarTagClassifier._rule_classify 精度
  - EarTagClassifier 无模型构造（onnxruntime 未安装时的行为）
  - IdentityAssist 无模型构造 & _classify_contour 兼容性

所有测试仅使用 Python 标准库 unittest + NumPy 合成数组，
不依赖 torch、onnxruntime、OpenCV 视频或真实模型权重。
"""

import unittest
import numpy as np
import sys
import os

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEarTagClassifierRuleFallback(unittest.TestCase):
    """验证 HSV 规则分类（五种颜色 + 空输入 + unknown）。"""

    @classmethod
    def setUpClass(cls):
        from detection.models.classifier import EarTagClassifier
        # 构造一个明确禁用 CNN 的分类器，确保走规则路径
        cls.clf = EarTagClassifier(use_cnn=False)

    # ------------------------------------------------------------------
    # 五种颜色
    # ------------------------------------------------------------------
    def test_red_pixel(self):
        """红色 HSV 像素应识别为 red。"""
        # OpenCV HSV: red ≈ (0, 200, 200) 或 (179, 200, 200)
        px = np.array([[0, 200, 200]], dtype=np.float32)
        color, method = self.clf.classify(None, px)
        self.assertEqual(color, "red")
        self.assertEqual(method, "ear_tag_color_rule")

    def test_yellow_pixel(self):
        """黄色 HSV 像素应识别为 yellow。"""
        # OpenCV HSV: yellow ≈ (25, 200, 200)
        px = np.array([[25, 200, 200]], dtype=np.float32)
        color, method = self.clf.classify(None, px)
        self.assertEqual(color, "yellow")
        self.assertEqual(method, "ear_tag_color_rule")

    def test_blue_pixel(self):
        """蓝色 HSV 像素应识别为 blue。"""
        px = np.array([[110, 200, 200]], dtype=np.float32)
        color, method = self.clf.classify(None, px)
        self.assertEqual(color, "blue")
        self.assertEqual(method, "ear_tag_color_rule")

    def test_green_pixel(self):
        """绿色 HSV 像素应识别为 green。"""
        px = np.array([[60, 200, 200]], dtype=np.float32)
        color, method = self.clf.classify(None, px)
        self.assertEqual(color, "green")
        self.assertEqual(method, "ear_tag_color_rule")

    def test_white_pixel(self):
        """白色 HSV 像素应识别为 white。"""
        px = np.array([[0, 10, 220]], dtype=np.float32)
        color, method = self.clf.classify(None, px)
        self.assertEqual(color, "white")
        self.assertEqual(method, "ear_tag_color_rule")

    # ------------------------------------------------------------------
    # 边界/空输入
    # ------------------------------------------------------------------
    def test_empty_pixels_returns_unknown(self):
        """空 HSV 数组应返回 unknown。"""
        color, method = self.clf.classify(None, np.array([], dtype=np.float32).reshape(0, 3))
        self.assertEqual(color, "unknown")
        self.assertEqual(method, "ear_tag_color_rule")

    def test_none_pixels_returns_unknown(self):
        """None HSV 输入应返回 unknown。"""
        color, method = self.clf.classify(None, None)
        self.assertEqual(color, "unknown")
        self.assertEqual(method, "ear_tag_color_rule")

    def test_ambiguous_pixel_returns_unknown(self):
        """不匹配任何已知颜色的像素应返回 unknown。"""
        # H=200, S=10, V=60 不匹配任何一级或二级阈值
        px = np.array([[200, 10, 60]], dtype=np.float32)
        color, method = self.clf.classify(None, px)
        self.assertEqual(color, "unknown")

    # ------------------------------------------------------------------
    # 多像素投票
    # ------------------------------------------------------------------
    def test_majority_vote_red(self):
        """多数红色像素 + 少数噪声 → red。"""
        px = np.array([
            [0, 200, 200],    # red
            [0, 200, 200],    # red
            [0, 200, 200],    # red
            [0, 200, 200],    # red
            [45, 40, 150],    # ambiguous
        ], dtype=np.float32)
        color, method = self.clf.classify(None, px)
        self.assertEqual(color, "red")

    def test_majority_vote_yellow(self):
        """多数黄色像素 → yellow。"""
        px = np.array([
            [25, 200, 200],
            [25, 200, 200],
            [25, 200, 200],
            [45, 40, 150],
        ], dtype=np.float32)
        color, method = self.clf.classify(None, px)
        self.assertEqual(color, "yellow")


class TestEarTagClassifierPreprocess(unittest.TestCase):
    """验证预处理输出形状和 dtype（使用默认 32×32 尺寸）。"""

    @classmethod
    def setUpClass(cls):
        from detection.models.classifier import EarTagClassifier
        # 构造一个禁用 CNN 的分类器实例，其 _model_h/_model_w 为默认 32×32
        cls.clf = EarTagClassifier(use_cnn=False)
        cls.clf._initialized = True  # 跳过懒初始化

    def test_preprocess_shape(self):
        """_preprocess 输出应为 (1, 3, 32, 32) float32（默认尺寸）。"""
        patch = np.random.randint(0, 256, (64, 48, 3), dtype=np.uint8)
        tensor = self.clf._preprocess(patch)
        self.assertEqual(tensor.shape, (1, 3, 32, 32))
        self.assertEqual(tensor.dtype, np.float32)

    def test_preprocess_value_range(self):
        """预处理结果应在 [0.0, 1.0] 范围内。"""
        patch = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        tensor = self.clf._preprocess(patch)
        self.assertGreaterEqual(tensor.min(), 0.0)
        self.assertLessEqual(tensor.max(), 1.0)

    def test_preprocess_pure_black(self):
        """全黑 patch → 全零张量附近。"""
        patch = np.zeros((40, 40, 3), dtype=np.uint8)
        tensor = self.clf._preprocess(patch)
        self.assertAlmostEqual(float(tensor.max()), 0.0, delta=0.01)

    def test_preprocess_pure_white(self):
        """全白 patch → 接近 1.0 的张量。"""
        patch = np.full((40, 40, 3), 255, dtype=np.uint8)
        tensor = self.clf._preprocess(patch)
        self.assertAlmostEqual(float(tensor.min()), 1.0, delta=0.01)

    def test_preprocess_custom_spatial_size(self):
        """自定义空间尺寸时，_preprocess 应输出对应尺寸。"""
        from detection.models.classifier import EarTagClassifier
        clf = EarTagClassifier(use_cnn=False)
        clf._initialized = True
        clf._model_h, clf._model_w = 48, 64
        patch = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        tensor = clf._preprocess(patch)
        self.assertEqual(tensor.shape, (1, 3, 48, 64))


class TestEarTagClassifierNoModel(unittest.TestCase):
    """验证无模型场景：onnxruntime 未安装时完全回退到规则。"""

    def test_use_cnn_false_always_rule(self):
        """use_cnn=False 时永远返回 ear_tag_color_rule method。"""
        from detection.models.classifier import EarTagClassifier
        clf = EarTagClassifier(use_cnn=False)
        px = np.array([[0, 200, 200]], dtype=np.float32)
        _, method = clf.classify(None, px)
        self.assertEqual(method, "ear_tag_color_rule")

    def test_nonexistent_model_path_rule_fallback(self):
        """指定不存在模型路径 → 回退规则。"""
        from detection.models.classifier import EarTagClassifier
        clf = EarTagClassifier(color_model_path="/nonexistent/path/model.onnx")
        px = np.array([[25, 200, 200]], dtype=np.float32)
        color, method = clf.classify(None, px)
        self.assertEqual(color, "yellow")
        self.assertEqual(method, "ear_tag_color_rule")

    def test_is_available_false_without_model(self):
        """无模型时 is_available 应为 False。"""
        from detection.models.classifier import EarTagClassifier
        clf = EarTagClassifier(color_model_path="/nonexistent/path/model.onnx")
        self.assertFalse(clf.is_available)

    def test_is_available_false_with_use_cnn_false(self):
        """use_cnn=False 时 is_available 应为 False。"""
        from detection.models.classifier import EarTagClassifier
        clf = EarTagClassifier(use_cnn=False)
        self.assertFalse(clf.is_available)

    def test_fallback_on_invalid_patch(self):
        """无效 patch（None 或空）应走规则路径且不崩溃。"""
        from detection.models.classifier import EarTagClassifier
        clf = EarTagClassifier(use_cnn=False)
        px = np.array([[60, 200, 200]], dtype=np.float32)  # green

        # None patch
        color, method = clf.classify(None, px)
        self.assertEqual(color, "green")
        self.assertEqual(method, "ear_tag_color_rule")

        # Empty patch (0 pixels)
        empty = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        color2, method2 = clf.classify(empty, px)
        self.assertEqual(color2, "green")


class TestIdentityAssistNoModel(unittest.TestCase):
    """验证 IdentityAssist 无模型构造兼容性。"""

    def test_construct_default(self):
        """默认构造不应崩溃。"""
        from detection.identity_assist import IdentityAssist
        assist = IdentityAssist()
        self.assertIsNotNone(assist)

    def test_construct_with_debug(self):
        """debug=True 构造不应崩溃。"""
        from detection.identity_assist import IdentityAssist
        assist = IdentityAssist(debug=True)
        self.assertIsNotNone(assist)
        self.assertTrue(assist.debug)

    def test_construct_with_model_path_none(self):
        """color_model_path=None 构造不应崩溃。"""
        from detection.identity_assist import IdentityAssist
        assist = IdentityAssist(color_model_path=None)
        clf = assist._get_classifier()
        self.assertIsNotNone(clf)
        # 无模型时 is_available 应为 False
        self.assertFalse(clf.is_available)

    def test_construct_with_use_cnn_false(self):
        """use_cnn=False 构造不应崩溃。"""
        from detection.identity_assist import IdentityAssist
        assist = IdentityAssist(use_cnn=False)
        self.assertIsNotNone(assist)

    def test_classify_contour_static_red(self):
        """_classify_contour 静态方法对红色像素返回正确结果。"""
        from detection.identity_assist import IdentityAssist
        px = np.array([
            [0, 200, 200],
            [0, 200, 200],
            [0, 200, 200],  # 3 red → meets min_pixels≥3
            [45, 40, 150],
        ], dtype=np.float32)
        color, low_conf, ratio = IdentityAssist._classify_contour(px)
        self.assertEqual(color, "red")
        # 主阈值通过 → low_conf=False
        self.assertFalse(low_conf)
        self.assertGreater(ratio, 0.35)

    def test_classify_contour_static_blue(self):
        """_classify_contour 对蓝色像素返回正确结果。"""
        from detection.identity_assist import IdentityAssist
        px = np.array([
            [110, 200, 200],
            [110, 200, 200],
            [110, 200, 200],
        ], dtype=np.float32)
        color, low_conf, ratio = IdentityAssist._classify_contour(px)
        self.assertEqual(color, "blue")

    def test_classify_contour_empty(self):
        """_classify_contour 空输入返回 unknown, low_conf=True, ratio=0.0。"""
        from detection.identity_assist import IdentityAssist
        px = np.array([], dtype=np.float32).reshape(0, 3)
        color, low_conf, ratio = IdentityAssist._classify_contour(px)
        self.assertEqual(color, "unknown")
        self.assertTrue(low_conf)
        self.assertEqual(ratio, 0.0)

    def test_analyze_segment_identity_method_not_overwritten(self):
        """analyze_segment 不应覆盖 detect_ear_tags 设置的 identity_method。

        由于 analyze_segment 需要真实视频，这里仅验证内部逻辑：
        构造一个最小 result dict 并直接调用 apply_identity_to_segment，
        确认 identity_method 被透传。
        """
        from detection.identity_assist import apply_identity_to_segment
        seg = {"count_status": "pending"}
        id_result = {
            "auto_mouse_colors": ["red"],
            "auto_mouse_ids": ["auto_red"],
            "identity_confidence": 0.85,
            "identity_needs_review": False,
            "identity_conflict": False,
            "identity_method": "ear_tag_color_cnn|exact_match",
            "identity_note": "",
        }
        apply_identity_to_segment(seg, id_result)
        self.assertEqual(seg["identity_method"], "ear_tag_color_cnn|exact_match")

    def test_segment_confirmed_not_overwritten(self):
        """已确认的 segment 不应被 apply_identity_to_segment 覆盖。"""
        from detection.identity_assist import apply_identity_to_segment
        seg = {
            "count_status": "confirmed",
            "mouse_ids": [1],
            "identity_method": "manual",
        }
        id_result = {
            "auto_mouse_colors": ["blue"],
            "auto_mouse_ids": ["auto_blue"],
            "identity_confidence": 0.9,
            "identity_needs_review": False,
            "identity_conflict": False,
            "identity_method": "ear_tag_color_rule|exact_match",
            "identity_note": "test",
        }
        apply_identity_to_segment(seg, id_result)
        # 不应覆盖已确认的 segment
        self.assertEqual(seg["count_status"], "confirmed")
        self.assertEqual(seg["identity_method"], "manual")
        self.assertEqual(seg["mouse_ids"], [1])


class TestCategories(unittest.TestCase):
    """验证类别常量。"""

    def test_categories_fixed_order(self):
        from detection.models.classifier import CATEGORIES
        self.assertEqual(CATEGORIES, ["red", "yellow", "blue", "green", "white", "unknown"])

    def test_categories_length(self):
        from detection.models.classifier import CATEGORIES
        self.assertEqual(len(CATEGORIES), 6)


class TestCNNInferenceWithFakeSession(unittest.TestCase):
    """覆盖 CNN 推理管线 — 使用 fake ONNX session（不依赖 onnxruntime）。

    测试范围:
      - 非 "input" 输入名（从元数据读取）
      - 32×32 静态形状元数据
      - 动态空间维度（回退 32×32）
      - logits → softmax
      - 已归一化概率（不重复 softmax）
      - 异常/错误类别数回退
      - 不可用 session 时不走 CNN
    """

    @staticmethod
    def _make_fake_meta(name="fake_input", shape=None):
        """构造一个 fake ONNX 输入/输出元数据对象。"""
        Meta = type("Meta", (), {})
        m = Meta()
        m.name = name
        m.shape = shape or [1, 3, 32, 32]
        return m

    @staticmethod
    def _make_fake_session(input_meta, output_meta, outputs_list):
        """构造一个 fake ONNX InferenceSession。

        :param input_meta:   get_inputs() 返回列表中的第一个元素
        :param output_meta:  get_outputs() 返回列表中的第一个元素
        :param outputs_list: session.run(...) 返回值
        """
        class FakeSession:
            def get_inputs(slf):
                return [input_meta]

            def get_outputs(slf):
                return [output_meta]

            def run(slf, output_names, feed_dict):
                # 校验输入键名正确传入
                slf.last_feed_keys = list(feed_dict.keys())
                return list(outputs_list)

        return FakeSession()

    # ------------------------------------------------------------------
    # 辅助：构造一个已注入 fake session 的分类器
    # ------------------------------------------------------------------
    def _classifier_with_fake(self, input_name="model_input",
                               input_shape=None, output_shape=None,
                               run_outputs=None):
        """构造 EarTagClassifier 并注入 fake session。"""
        from detection.models.classifier import EarTagClassifier

        clf = EarTagClassifier(use_cnn=False)
        clf._ort_available = True
        clf._initialized = True

        inp_shape = input_shape or [1, 3, 32, 32]
        out_shape = output_shape or [1, 6]

        inp_meta = self._make_fake_meta(input_name, inp_shape)
        out_meta = self._make_fake_meta("output", out_shape)

        sess = self._make_fake_session(inp_meta, out_meta, run_outputs or [])
        clf._ort_session = sess

        # 验证元数据
        ok = clf._validate_model_metadata()
        return clf, sess, ok

    # ------------------------------------------------------------------
    # 1. 非 "input" 输入名
    # ------------------------------------------------------------------
    def test_custom_input_name(self):
        """输入名从 session 元数据取得，不硬编码 'input'。"""
        clf, sess, ok = self._classifier_with_fake(
            input_name="my_custom_input",
            run_outputs=[np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)],
        )
        self.assertTrue(ok)
        self.assertEqual(clf._input_name, "my_custom_input")

        patch = np.random.randint(0, 256, (48, 48, 3), dtype=np.uint8)
        px = np.array([[0, 200, 200]], dtype=np.float32)
        color, method = clf.classify(patch, px)

        # 检查 feed_dict 键名正确
        self.assertIn("last_feed_keys", dir(sess))
        self.assertIn("my_custom_input", sess.last_feed_keys)
        self.assertNotIn("input", sess.last_feed_keys)
        # CNN 路径成功时应使用 cnn method
        self.assertEqual(method, "ear_tag_color_cnn")
        self.assertEqual(color, "red")

    # ------------------------------------------------------------------
    # 2. 32×32 静态元数据
    # ------------------------------------------------------------------
    def test_static_32x32_metadata(self):
        """静态 [1,3,32,32] 元数据应正确缓存。"""
        clf, sess, ok = self._classifier_with_fake(
            input_shape=[1, 3, 32, 32],
            run_outputs=[np.array([[0.1, 0.1, 0.1, 0.1, 0.5, 0.1]], dtype=np.float32)],
        )
        self.assertTrue(ok)
        self.assertEqual(clf._model_h, 32)
        self.assertEqual(clf._model_w, 32)

        patch = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        px = np.array([[0, 200, 200]], dtype=np.float32)
        color, method = clf.classify(patch, px)
        self.assertEqual(method, "ear_tag_color_cnn")

    # ------------------------------------------------------------------
    # 3. 动态空间维度 → 回退 32×32
    # ------------------------------------------------------------------
    def test_dynamic_spatial_dims_fallback_32x32(self):
        """动态维度 ['batch', 3, 'h', 'w'] 应回退到 32×32。"""
        clf, sess, ok = self._classifier_with_fake(
            input_shape=["batch", 3, "height", "width"],
            run_outputs=[np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)],
        )
        self.assertTrue(ok)
        self.assertEqual(clf._model_h, 32)
        self.assertEqual(clf._model_w, 32)

        patch = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        px = np.array([[0, 200, 200]], dtype=np.float32)
        color, method = clf.classify(patch, px)
        self.assertEqual(method, "ear_tag_color_cnn")

    def test_dynamic_batch_static_spatial(self):
        """静态空间 + 动态 batch: ['batch', 3, 64, 48] → 使用 64×48。"""
        clf, sess, ok = self._classifier_with_fake(
            input_shape=["batch", 3, 64, 48],
            run_outputs=[np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)],
        )
        self.assertTrue(ok)
        self.assertEqual(clf._model_h, 64)
        self.assertEqual(clf._model_w, 48)

    # ------------------------------------------------------------------
    # 4. 输出形状 (1,6) 和 (6,) 均支持
    # ------------------------------------------------------------------
    def test_output_shape_1x6(self):
        """输出 (1,6) 应正确 squeeze。"""
        from detection.models.classifier import EarTagClassifier
        # 构造 one-hot logits（高 logit → 高 softmax 概率）
        logits_2d = np.array([[0.1, 0.2, 0.1, 0.1, 0.1, 10.0]], dtype=np.float32)
        result = EarTagClassifier._safe_decode_output([logits_2d])
        self.assertIsNotNone(result)
        pred_idx, max_prob = result
        self.assertEqual(pred_idx, 5)  # "unknown"
        self.assertGreater(max_prob, 0.9)

    def test_output_shape_6(self):
        """输出 (6,) 应直接使用。"""
        from detection.models.classifier import EarTagClassifier
        logits_1d = np.array([0.1, 0.2, 0.1, 0.1, 0.1, 10.0], dtype=np.float32)
        result = EarTagClassifier._safe_decode_output([logits_1d])
        self.assertIsNotNone(result)
        pred_idx, max_prob = result
        self.assertEqual(pred_idx, 5)
        self.assertGreater(max_prob, 0.9)

    # ------------------------------------------------------------------
    # 5. logits → softmax
    # ------------------------------------------------------------------
    def test_softmax_on_logits(self):
        """非概率 logits 应施加 softmax。"""
        from detection.models.classifier import EarTagClassifier
        # 原始 logits 之和 != 1
        logits = np.array([1.0, 2.0, 3.0, 0.5, 0.3, 0.1], dtype=np.float32)
        result = EarTagClassifier._safe_decode_output([logits])
        self.assertIsNotNone(result)
        pred_idx, max_prob = result
        self.assertEqual(pred_idx, 2)  # "blue" (index 2)
        # 输出应和为 1（softmax 后）
        self.assertGreater(max_prob, 0.0)

    # ------------------------------------------------------------------
    # 6. 已归一化概率 → 不重复 softmax
    # ------------------------------------------------------------------
    def test_already_normalized_probs(self):
        """已归一化概率（和≈1，均在[0,1]）不应重复 softmax。"""
        from detection.models.classifier import EarTagClassifier
        probs = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.75], dtype=np.float32)
        result = EarTagClassifier._safe_decode_output([probs])
        self.assertIsNotNone(result)
        pred_idx, max_prob = result
        self.assertEqual(pred_idx, 5)  # "unknown"
        self.assertAlmostEqual(max_prob, 0.75, delta=0.01)

    # ------------------------------------------------------------------
    # 7. 异常输出 → 回退
    # ------------------------------------------------------------------
    def test_empty_output_returns_none(self):
        """空输出列表 → None。"""
        from detection.models.classifier import EarTagClassifier
        result = EarTagClassifier._safe_decode_output([])
        self.assertIsNone(result)

    def test_nan_output_returns_none(self):
        """NaN 输出 → None。"""
        from detection.models.classifier import EarTagClassifier
        arr = np.array([0.1, 0.1, 0.1, 0.1, 0.1, np.nan], dtype=np.float32)
        result = EarTagClassifier._safe_decode_output([arr])
        self.assertIsNone(result)

    def test_inf_output_returns_none(self):
        """Inf 输出 → None。"""
        from detection.models.classifier import EarTagClassifier
        arr = np.array([0.1, 0.1, 0.1, 0.1, 0.1, np.inf], dtype=np.float32)
        result = EarTagClassifier._safe_decode_output([arr])
        self.assertIsNone(result)

    def test_wrong_class_count_returns_none(self):
        """类别数 ≠ 6 → None。"""
        from detection.models.classifier import EarTagClassifier
        arr = np.array([0.1, 0.2, 0.3, 0.4, 0.0], dtype=np.float32)  # 5 类
        result = EarTagClassifier._safe_decode_output([arr])
        self.assertIsNone(result)

    def test_wrong_rank_output_returns_none(self):
        """非 1D 输出（如 (1,2,3)）→ None。"""
        from detection.models.classifier import EarTagClassifier
        arr = np.ones((1, 2, 3), dtype=np.float32)
        result = EarTagClassifier._safe_decode_output([arr])
        self.assertIsNone(result)

    def test_zero_class_output_returns_none(self):
        """零长度输出 → None。"""
        from detection.models.classifier import EarTagClassifier
        arr = np.array([], dtype=np.float32)
        result = EarTagClassifier._safe_decode_output([arr])
        self.assertIsNone(result)

    def test_non_numeric_output_returns_none(self):
        """非数值输出（如对象数组）→ None。"""
        from detection.models.classifier import EarTagClassifier
        arr = np.array(["a", "b", "c", "d", "e", "f"], dtype=object)
        result = EarTagClassifier._safe_decode_output([arr])
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # 8. 端到端：fake session + classify
    # ------------------------------------------------------------------
    def test_classify_with_fake_cnn_returns_cnn_method(self):
        """fake session CNN 推理成功 → method='ear_tag_color_cnn'。"""
        clf, sess, ok = self._classifier_with_fake(
            input_name="data",
            input_shape=[1, 3, 32, 32],
            run_outputs=[np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float32)],
        )
        self.assertTrue(ok)

        patch = np.random.randint(0, 256, (48, 48, 3), dtype=np.uint8)
        px = np.array([[0, 200, 200]], dtype=np.float32)
        color, method = clf.classify(patch, px)
        self.assertEqual(color, "white")
        self.assertEqual(method, "ear_tag_color_cnn")

    def test_classify_cnn_unknown_falls_back_to_hsv(self):
        """CNN 返回 unknown → 自动回退 HSV 规则。"""
        clf, sess, ok = self._classifier_with_fake(
            run_outputs=[np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)],
        )
        self.assertTrue(ok)

        # 提供有效 BGR patch 让 CNN 先尝试，同时提供黄色 HSV 像素用于回退
        patch = np.random.randint(0, 256, (48, 48, 3), dtype=np.uint8)
        px = np.array([[25, 200, 200]], dtype=np.float32)  # yellow
        color, method = clf.classify(patch, px)
        # CNN 返回 unknown（最高概率在第5类），classify 回退到 HSV rules
        self.assertEqual(color, "yellow")
        self.assertEqual(method, "ear_tag_color_rule")

    def test_classify_fake_cnn_exception_graceful(self):
        """CNN 推理异常 → 不崩溃，回退到 HSV 规则。"""
        from detection.models.classifier import EarTagClassifier

        # 构造一个会抛异常的 fake session
        class BadSession:
            def get_inputs(slf):
                return [TestCNNInferenceWithFakeSession._make_fake_meta("x", ["N", 3, 32, 32])]
            def get_outputs(slf):
                return [TestCNNInferenceWithFakeSession._make_fake_meta("y", ["N", 6])]
            def run(slf, *args, **kwargs):
                raise RuntimeError("simulated ONNX error")

        clf = EarTagClassifier(use_cnn=False)
        clf._ort_available = True
        clf._initialized = True
        clf._ort_session = BadSession()
        ok = clf._validate_model_metadata()
        self.assertTrue(ok)  # metadata 看起来合法

        patch = np.random.randint(0, 256, (48, 48, 3), dtype=np.uint8)
        px = np.array([[110, 200, 200]], dtype=np.float32)  # blue
        color, method = clf.classify(patch, px)
        # 应优雅回退到 HSV
        self.assertEqual(color, "blue")
        self.assertEqual(method, "ear_tag_color_rule")

    def test_no_session_triggers_hsv_directly(self):
        """_ort_session=None → 直接走 HSV 规则。"""
        from detection.models.classifier import EarTagClassifier
        clf = EarTagClassifier(use_cnn=False)
        clf._initialized = True  # 但 _ort_session=None
        clf._ort_available = False
        px = np.array([[25, 200, 200]], dtype=np.float32)
        color, method = clf.classify(None, px)
        self.assertEqual(color, "yellow")
        self.assertEqual(method, "ear_tag_color_rule")

    # ------------------------------------------------------------------
    # 9. 元数据验证边界
    # ------------------------------------------------------------------
    def test_validate_wrong_rank_falls_back(self):
        """输入 rank ≠ 4 → 验证失败。"""
        clf, sess, ok = self._classifier_with_fake(
            input_shape=[1, 3, 32],  # 3D 而非 4D
            run_outputs=[np.zeros((1, 6), dtype=np.float32)],
        )
        self.assertFalse(ok)
        # 验证失败后 session 应被置 None
        # (但 _classifier_with_fake 只返回 ok，不自动置 None)
        # 实际 _ensure_initialized 会置 None，此处只测 _validate_model_metadata 返回值

    def test_validate_wrong_channels_falls_back(self):
        """通道数 ≠ 3 → 验证失败。"""
        clf, sess, ok = self._classifier_with_fake(
            input_shape=[1, 1, 32, 32],  # 1 通道
            run_outputs=[np.zeros((1, 6), dtype=np.float32)],
        )
        self.assertFalse(ok)

    def test_validate_no_inputs_falls_back(self):
        """无输入 → 验证失败。"""
        from detection.models.classifier import EarTagClassifier
        clf = EarTagClassifier(use_cnn=False)
        clf._ort_available = True
        clf._initialized = True

        class NoInputSession:
            def get_inputs(slf):
                return []
            def get_outputs(slf):
                return [TestCNNInferenceWithFakeSession._make_fake_meta("y", [1, 6])]
            def run(slf, *a, **kw):
                return []

        clf._ort_session = NoInputSession()
        ok = clf._validate_model_metadata()
        self.assertFalse(ok)

    def test_validate_no_outputs_falls_back(self):
        """无输出 → 验证失败。"""
        from detection.models.classifier import EarTagClassifier
        clf = EarTagClassifier(use_cnn=False)
        clf._ort_available = True
        clf._initialized = True

        class NoOutputSession:
            def get_inputs(slf):
                return [TestCNNInferenceWithFakeSession._make_fake_meta("x", [1, 3, 32, 32])]
            def get_outputs(slf):
                return []
            def run(slf, *a, **kw):
                return []

        clf._ort_session = NoOutputSession()
        ok = clf._validate_model_metadata()
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
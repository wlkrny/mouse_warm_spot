"""
耳标颜色分类器 — 多层回退架构。

优先级:
  1. AI 视觉模型 (Kimi/Minimax, 由 MOUSE_COLOR_AI_PROVIDER 控制)
  2. ONNX CNN (由 MOUSE_COLOR_USE_CNN + MOUSE_COLOR_MODEL_PATH 控制)
  3. HSV 规则 — 始终可用的精确回退

任何上层失败（网络错误/模型缺失/解析异常）均向下回退，不崩溃。
注意：本模块不假定模型权重存在，所有模型相关路径均为可选配置。
"""

import os
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# 排错：MOUSE_COLOR_AI_DEBUG=1 启用 DEBUG 级别日志
if os.environ.get("MOUSE_COLOR_AI_DEBUG", "").strip() in ("1", "true", "yes", "on"):
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(h)

# 固定 6 类（与训练时顺序一致）
CATEGORIES = ["red", "yellow", "blue", "green", "white", "unknown"]

# 默认空间尺寸（模型元数据动态维度时使用）
DEFAULT_SPATIAL_SIZE = 32

# CNN 置信度阈值：低于此值回退到 HSV 规则
CNN_CONFIDENCE_THRESHOLD = 0.3


# ------------------------------------------------------------------
# 共享 HSV 规则分类（IdentityAssist 和 EarTagClassifier 共用）
# ------------------------------------------------------------------
def hsv_rule_classify(hsv_pixels):
    """HSV 像素投票 → 颜色，使用主阈值 + 次级回退两级门控。

    :param hsv_pixels: (N,3) HSV 像素数组，可为 None 或空
    :returns: (color: str, is_low_confidence: bool, best_ratio: float)
              color ∈ {"red","yellow","blue","green","white","unknown"}
              is_low_confidence: True 表示使用了放宽的次级阈值
              best_ratio: 主色占比（0.0 if unknown）
    """
    if hsv_pixels is None or len(hsv_pixels) == 0:
        return "unknown", True, 0.0

    h = hsv_pixels[:, 0]
    s = hsv_pixels[:, 1]
    v = hsv_pixels[:, 2]
    total_px = len(hsv_pixels)

    # --- Primary vote (tight thresholds) ---
    red_mask    = ((h <= 12) | (h >= 165)) & (s > 45)
    yellow_mask = (h >= 13) & (h <= 42) & (s > 45)
    green_mask  = (h >= 48) & (h <= 80) & (s > 50)
    blue_mask   = (h >= 90) & (h <= 135) & (s > 50)
    white_mask  = (s < 35) & (v > 170)

    counts = {
        "red":    int(np.sum(red_mask)),
        "yellow": int(np.sum(yellow_mask)),
        "blue":   int(np.sum(blue_mask)),
        "green":  int(np.sum(green_mask)),
        "white":  int(np.sum(white_mask)),
    }

    best_color = max(counts, key=counts.get)
    best_count = counts[best_color]
    if best_count > 0:
        best_ratio = best_count / max(1, total_px)
        # Dominance gate: min_pixels≥3, ratio≥0.35, margin≥0.15
        if best_count >= 3 and best_ratio >= 0.35:
            sorted_counts = sorted(counts.values(), reverse=True)
            second_best = sorted_counts[1] if len(sorted_counts) > 1 else 0
            second_ratio = second_best / max(1, total_px)
            if best_ratio - second_ratio >= 0.15:
                return best_color, False, best_ratio

    # --- Secondary fallback (relaxed thresholds) ---
    red_mask2    = ((h <= 12) | (h >= 165)) & (s > 35)
    yellow_mask2 = (h >= 10) & (h <= 45) & (s > 35)
    green_mask2  = (h >= 43) & (h <= 88) & (s > 35)
    blue_mask2   = (h >= 85) & (h <= 140) & (s > 35)
    white_mask2  = (s < 40) & (v > 160)

    counts2 = {
        "red":    int(np.sum(red_mask2)),
        "yellow": int(np.sum(yellow_mask2)),
        "blue":   int(np.sum(blue_mask2)),
        "green":  int(np.sum(green_mask2)),
        "white":  int(np.sum(white_mask2)),
    }

    best_color2 = max(counts2, key=counts2.get)
    best_count2 = counts2[best_color2]
    if best_count2 > 0:
        best_ratio2 = best_count2 / max(1, total_px)
        return best_color2, True, best_ratio2

    return "unknown", True, 0.0


class EarTagClassifier:
    """耳标颜色分类器，多层回退：AI 视觉 → ONNX CNN → HSV 规则。

    用法::
        clf = EarTagClassifier(color_model_path="path/to/model.onnx")
        color, method = clf.classify(patch_bgr, hsv_pixels)

    :param color_model_path: ONNX 模型文件路径，None 则自动发现或跳过 CNN
    :param use_cnn: 是否启用 CNN（None 时根据 MOUSE_COLOR_USE_CNN 环境变量和依赖可用性自动判定）
    :param vision_provider: VisionProvider 实例（None 时自动从 VisionProviderFactory 创建）
    """

    def __init__(self, color_model_path=None, use_cnn=None, vision_provider=None):
        self._color_model_path = color_model_path
        self._user_use_cnn = use_cnn
        self._ort_session = None
        self._ort_available = False
        self._initialized = False
        # 模型元数据缓存（由 _validate_model_metadata 填充）
        self._input_name = "input"
        self._model_h = DEFAULT_SPATIAL_SIZE
        self._model_w = DEFAULT_SPATIAL_SIZE
        # AI 视觉 provider（懒创建或注入）
        self._vision_provider = vision_provider
        self._vision_init_attempted = False

    # ------------------------------------------------------------------
    # 懒初始化：仅在首次调用 classify() 时触发
    # ------------------------------------------------------------------
    def _ensure_initialized(self):
        if self._initialized:
            return
        self._initialized = True

        # ---------- 0. 尝试初始化 AI 视觉 provider ----------
        self._init_vision_provider()

        # ---------- 1. 解析 use_cnn ----------
        effective_use_cnn = self._user_use_cnn
        if effective_use_cnn is None:
            env = os.environ.get("MOUSE_COLOR_USE_CNN", "").strip().lower()
            if env in ("1", "true", "yes", "on"):
                effective_use_cnn = True
            elif env in ("0", "false", "no", "off"):
                effective_use_cnn = False
            else:
                effective_use_cnn = True  # 默认尝试启用

        if not effective_use_cnn:
            logger.info("MOUSE_COLOR_USE_CNN=0, CNN disabled; will use HSV rules")
            return

        # ---------- 2. 尝试导入 onnxruntime ----------
        try:
            import onnxruntime  # noqa: F401 — 延迟导入，避免无依赖时崩溃
            self._ort_available = True
        except ImportError:
            logger.warning(
                "onnxruntime not installed; falling back to HSV rules. "
                "Install with: pip install -r requirements-ai.txt"
            )
            return

        # ---------- 3. 解析模型路径 ----------
        model_path = self._color_model_path
        if not model_path:
            model_path = os.environ.get("MOUSE_COLOR_MODEL_PATH", "").strip()
        if not model_path:
            model_path = self._auto_discover_model()

        if not model_path or not os.path.isfile(model_path):
            logger.info(
                f"ONNX model not found at {model_path!r}; falling back to HSV rules"
            )
            return

        # ---------- 4. 加载 ONNX session ----------
        try:
            import onnxruntime as ort
            self._ort_session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
            )
            logger.info(f"ONNX ear-tag classifier loaded: {model_path}")
        except Exception as exc:
            logger.warning(
                f"Failed to load ONNX model: {exc}; falling back to HSV rules"
            )
            self._ort_session = None
            return

        # ---------- 5. 验证模型元数据 ----------
        if not self._validate_model_metadata():
            self._ort_session = None

    # ------------------------------------------------------------------
    # Model auto-discovery
    # ------------------------------------------------------------------
    @staticmethod
    def _auto_discover_model():
        """自动查找 detection/models/ear_tag_classifier.onnx（相对于本文件）。"""
        try:
            this_dir = os.path.dirname(os.path.abspath(__file__))
            candidate = os.path.join(this_dir, "ear_tag_classifier.onnx")
            return candidate
        except Exception:
            return None

    # ------------------------------------------------------------------
    # AI 视觉 provider 初始化
    # ------------------------------------------------------------------
    def _init_vision_provider(self):
        """尝试从 VisionProviderFactory 创建 AI 视觉 provider。

        仅在 self._vision_provider 为 None 且未尝试过初始化时执行。
        构造错误（缺密钥等）不中断程序，仅记录日志并置 None。
        """
        if self._vision_provider is not None or self._vision_init_attempted:
            return
        self._vision_init_attempted = True

        try:
            from .vision_provider import VisionProviderFactory, VisionProviderError
            provider = VisionProviderFactory.create()
            if provider is not None:
                self._vision_provider = provider
                logger.info("AI vision provider created: %s", type(provider).__name__)
            else:
                logger.debug("MOUSE_COLOR_AI_PROVIDER=hsv or not set; vision provider skipped")
        except VisionProviderError as exc:
            logger.warning("AI vision provider init failed: %s; will use CNN/HSV", exc)
            self._vision_provider = None
        except ImportError as exc:
            logger.warning("vision_provider module not importable: %s; will use CNN/HSV", exc)
            self._vision_provider = None
        except Exception as exc:
            logger.warning("Unexpected vision provider init error: %s; will use CNN/HSV", exc)
            self._vision_provider = None

    # ------------------------------------------------------------------
    # AI 视觉分类
    # ------------------------------------------------------------------
    def _vision_classify(self, patch_bgr):
        """通过 AI 视觉 provider 分类。

        :returns: (color: str, method: str)
        :raises VisionProviderError: 网络/API 可诊断错误
        """
        if self._vision_provider is None:
            return "unknown", "ear_tag_color_rule"

        color, confidence, method = self._vision_provider.classify(patch_bgr)

        # 低置信度 → 返回 unknown，触发下层回退
        if confidence < CNN_CONFIDENCE_THRESHOLD:
            logger.debug(
                "Vision confidence %.3f below threshold %.2f; → fallback",
                confidence, CNN_CONFIDENCE_THRESHOLD,
            )
            return "unknown", method

        return color, method

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def is_available(self):
        """CNN 推理是否实际可用（已加载 session）。"""
        self._ensure_initialized()
        return self._ort_session is not None

    @property
    def is_vision_available(self):
        """AI 视觉 provider 是否可用。"""
        self._ensure_initialized()
        return self._vision_provider is not None

    def classify(self, patch_bgr, hsv_pixels, allow_vision: bool = True):
        """对耳标 patch 分类：AI 视觉 → CNN → HSV 规则。

        :param patch_bgr: (H,W,3) BGR 图像 patch，可为 None
        :param hsv_pixels: (N,3) HSV 像素数组（同一轮廓），可为 None 或空
        :param allow_vision: 是否允许调用 AI 视觉 provider（默认 True）。
                             设为 False 可防止因预算耗尽等原因意外触发网络调用。
        :returns: (color: str, method: str)
                  color  ∈ CATEGORIES
                  method ∈ {"ear_tag_color_cnn", "ear_tag_color_rule",
                            "ear_tag_color_kimi", "ear_tag_color_minimax"}
        """
        self._ensure_initialized()

        # --- Try AI Vision (highest priority) ---
        if (allow_vision
                and self._vision_provider is not None
                and patch_bgr is not None
                and patch_bgr.size > 0):
            try:
                color, method = self._vision_classify(patch_bgr)
                if color != "unknown":
                    return color, method
            except Exception as exc:
                logger.debug(f"Vision model exception: {exc}; → CNN/HSV fallback")

        # --- Try CNN ---
        if (self._ort_session is not None
                and patch_bgr is not None
                and patch_bgr.size > 0):
            try:
                color = self._cnn_infer(patch_bgr)
                if color != "unknown":
                    return color, "ear_tag_color_cnn"
            except Exception as exc:
                logger.debug(f"CNN inference exception: {exc}; → HSV fallback")

        # --- Fallback: HSV rules ---
        color = self._rule_classify(hsv_pixels)
        return color, "ear_tag_color_rule"

    def classify_segment_frames(self, frames: list, clip_id: str | None = None) -> dict:
        """Return segment-level VLM count/colors without invoking local fallback.

        Invalid, unavailable, or low-confidence responses are represented as an
        invalid result so IdentityAssist can preserve its complete local path.
        """
        self._ensure_initialized()
        fallback = {"mouse_count": 1, "colors": ["unknown"], "confidence": 0.0,
                    "thermometer_present": False, "method": "ear_tag_color_rule", "raw_response": None,
                    "parse_status": "unavailable"}
        if self._vision_provider is None:
            return fallback
        try:
            if clip_id is None:
                result = self._vision_provider.classify_segment_frames(frames)
            else:
                try:
                    result = self._vision_provider.classify_segment_frames(frames, clip_id=clip_id)
                except TypeError:
                    # Third-party providers with the pre-clip signature remain usable.
                    result = self._vision_provider.classify_segment_frames(frames)
            if not isinstance(result, dict):
                raise ValueError("non-dict segment response")
            count, colors, confidence = result.get("mouse_count"), result.get("colors"), result.get("confidence")
            thermometer_present = result.get("thermometer_present", False)
            if (isinstance(count, bool) or count not in (1, 2) or not isinstance(colors, list)
                    or len(colors) != count or any(c not in CATEGORIES for c in colors)
                    or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1
                    or not isinstance(thermometer_present, bool)):
                raise ValueError("invalid segment response")
            return result
        except (AttributeError, ValueError, TypeError):
            # Providers predating the new API retain the old one-color protocol.
            try:
                if clip_id is None:
                    color, confidence, method, raw = self._vision_provider.classify_frames(frames)
                else:
                    try:
                        color, confidence, method, raw = self._vision_provider.classify_frames(frames, clip_id=clip_id)
                    except TypeError:
                        color, confidence, method, raw = self._vision_provider.classify_frames(frames)
                return {"mouse_count": 1, "colors": [color], "confidence": confidence,
                        "thermometer_present": False, "method": method, "raw_response": raw,
                        "parse_status": "legacy_provider"}
            except Exception as exc:
                logger.debug("Vision segment classification failed: %s", exc)
                return fallback
        except Exception as exc:
            logger.debug("Vision segment classification failed: %s", exc)
            return fallback

    def classify_frames(self, frames: list, clip_id: str | None = None):
        """多帧上下文颜色分类（每个 segment 一次调用）。

        仅使用 AI 视觉 provider，不做 CNN/HSV 回退（回退由调用方在更高层处理）。

        :param frames: (H,W,3) uint8 BGR 图像列表，按时间顺序排列
        :returns: (color: str, method: str, confidence: float)
                  color  ∈ CATEGORIES
        :raises VisionProviderError: provider 配置/网络/API 错误
        """
        self._ensure_initialized()

        if self._vision_provider is None:
            return "unknown", "ear_tag_color_rule", 0.0

        if clip_id is None:
            color, confidence, method, _raw = self._vision_provider.classify_frames(frames)
        else:
            try:
                color, confidence, method, _raw = self._vision_provider.classify_frames(frames, clip_id=clip_id)
            except TypeError:
                color, confidence, method, _raw = self._vision_provider.classify_frames(frames)

        # 低置信度 → 视为 unknown
        if confidence < CNN_CONFIDENCE_THRESHOLD:
            logger.debug(
                "Vision context confidence %.3f below threshold %.2f; → fallback",
                confidence, CNN_CONFIDENCE_THRESHOLD,
            )
            return "unknown", method, confidence

        return color, method, confidence

    # ------------------------------------------------------------------
    # 模型元数据验证（不崩溃，失败返回 False → 调用方回退 HSV）
    # ------------------------------------------------------------------
    def _validate_model_metadata(self):
        """验证并缓存模型输入/输出元数据。

        Returns:
            True:  元数据合法，self._input_name / self._model_h /
                   self._model_w 已更新。
            False: 元数据不合法，调用方应将 self._ort_session 置 None。
        """
        try:
            inputs = self._ort_session.get_inputs()
            if not inputs:
                logger.warning("ONNX model has no inputs; falling back to HSV rules")
                return False

            inp = inputs[0]
            self._input_name = inp.name
            shape = list(inp.shape)  # e.g. ['batch', 3, 32, 32] or [1, 3, 32, 32]

            # --- 验证 rank (NCHW=4D) ---
            if len(shape) != 4:
                logger.warning(
                    f"Expected 4D input (NCHW), got {len(shape)}D shape={shape}; "
                    "falling back to HSV rules"
                )
                return False

            # --- 验证通道数 (dim 1) ---
            ch = shape[1]
            if isinstance(ch, int) and ch != 3:
                logger.warning(
                    f"Expected 3 input channels, got {ch}; falling back to HSV rules"
                )
                return False

            # --- 解析空间尺寸 (dim 2, dim 3) ---
            h_raw, w_raw = shape[2], shape[3]
            if (isinstance(h_raw, int) and isinstance(w_raw, int)
                    and h_raw > 0 and w_raw > 0):
                self._model_h, self._model_w = int(h_raw), int(w_raw)
            else:
                # 动态维度 → 回退到 32×32
                logger.info(
                    "Dynamic spatial dimensions (%s, %s); using %d×%d fallback",
                    h_raw, w_raw, DEFAULT_SPATIAL_SIZE, DEFAULT_SPATIAL_SIZE,
                )
                self._model_h = DEFAULT_SPATIAL_SIZE
                self._model_w = DEFAULT_SPATIAL_SIZE

            # --- 验证输出 ---
            outputs = self._ort_session.get_outputs()
            if not outputs:
                logger.warning("ONNX model has no outputs; falling back to HSV rules")
                return False

            logger.info(
                "ONNX model metadata OK: input=%r shape=%r h=%d w=%d",
                self._input_name, shape, self._model_h, self._model_w,
            )
            return True

        except Exception as exc:
            logger.warning(
                "Model metadata validation failed: %s; falling back to HSV rules", exc
            )
            return False

    # ------------------------------------------------------------------
    # CNN 推理
    # ------------------------------------------------------------------
    def _cnn_infer(self, patch_bgr):
        """运行 ONNX 推理，返回 CATEGORIES 之一。

        任何异常 / 输出不合法 → 返回 "unknown"（调用方回退 HSV）。
        """
        # 预处理
        tensor = self._preprocess(patch_bgr)

        # 推理（使用元数据中记录的输入名，不硬编码 "input"）
        outputs = self._ort_session.run(None, {self._input_name: tensor})

        # 安全解码输出
        result = self._safe_decode_output(outputs)
        if result is None:
            return "unknown"

        pred_idx, max_prob = result

        if max_prob < CNN_CONFIDENCE_THRESHOLD:
            return "unknown"

        return CATEGORIES[pred_idx] if pred_idx < len(CATEGORIES) else "unknown"

    # ------------------------------------------------------------------
    # 输出安全解码
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_decode_output(outputs):
        """安全解析 ONNX 模型第一输出。

        支持形状:
          - (1, 6) → squeeze 为 (6,)
          - (6,)   → 直接使用

        自动检测 logits vs 已归一化概率:
          - 若所有值 ∈ [0,1] 且之和 ≈ 1 → 视为概率，不 softmax
          - 否则视为 logits，施加 softmax

        :returns: (pred_idx: int, max_prob: float) 或 None（需回退 HSV）
        """
        if not outputs:
            logger.warning("ONNX model produced empty output; falling back to HSV")
            return None

        try:
            raw = outputs[0]
            arr = np.asarray(raw, dtype=np.float64).squeeze()
        except (ValueError, TypeError) as exc:
            logger.warning("Cannot convert ONNX output to array: %s", exc)
            return None

        # 形状检查
        if arr.ndim != 1:
            logger.warning(
                "Expected 1D output after squeeze, got %dD; falling back to HSV",
                arr.ndim,
            )
            return None

        n_classes = len(arr)
        if n_classes != len(CATEGORIES):
            logger.warning(
                "Expected %d classes, got %d; falling back to HSV",
                len(CATEGORIES), n_classes,
            )
            return None

        if n_classes == 0:
            logger.warning("Zero-class output; falling back to HSV")
            return None

        # NaN / Inf 检查
        if not np.isfinite(arr).all():
            logger.warning("Output contains NaN or Inf; falling back to HSV")
            return None

        # 判断是否已是合法概率分布
        all_in_01 = bool(np.all(arr >= 0.0) and np.all(arr <= 1.0))
        sum_close_1 = bool(np.isclose(arr.sum(), 1.0, atol=1e-3))
        is_prob = all_in_01 and sum_close_1

        if is_prob:
            probs = arr.astype(np.float64)
        else:
            # 视为 logits → softmax
            probs = EarTagClassifier._softmax(arr)

        pred_idx = int(np.argmax(probs))
        max_prob = float(probs[pred_idx])
        return pred_idx, max_prob

    # ------------------------------------------------------------------
    # 预处理：BGR patch → (1,3,H,W) float32
    # ------------------------------------------------------------------
    def _preprocess(self, patch_bgr):
        """预处理流水线:

        1. resize → (model_h, model_w)   — 来自模型元数据或默认 32×32
        2. BGR → RGB
        3. 归一化 [0, 255] → [0.0, 1.0] float32
        4. HWC → CHW
        5. 添加 batch 维度 → (1, 3, model_h, model_w)
        """
        # Step 1: resize 到模型期望尺寸
        target_size = (self._model_w, self._model_h)
        resized = cv2.resize(patch_bgr, target_size, interpolation=cv2.INTER_LINEAR)

        # Step 2: BGR → RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Step 3: normalize
        rgb_f = rgb.astype(np.float32) / 255.0

        # Step 4: HWC → CHW
        chw = np.transpose(rgb_f, (2, 0, 1))

        # Step 5: batch dim
        batch = np.expand_dims(chw, axis=0)  # (1, 3, H, W)
        return batch.astype(np.float32)

    @staticmethod
    def _softmax(x):
        """稳定 softmax（沿最后一维）。"""
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    # ------------------------------------------------------------------
    # HSV 规则分类（委托给共享的 hsv_rule_classify）
    # ------------------------------------------------------------------
    @staticmethod
    def _rule_classify(hsv_pixels):
        """HSV 像素投票 → 颜色。无有效像素时返回 'unknown'。

        委托给模块级 hsv_rule_classify() 以避免与
        IdentityAssist._classify_contour 的阈值逻辑重复。
        """
        color, _low_conf, _ratio = hsv_rule_classify(hsv_pixels)
        return color
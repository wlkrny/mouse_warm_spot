"""
AI 视觉颜色识别 Provider — 可配置的 Kimi/Minimax 视觉模型适配层。

架构:
   VisionProvider (ABC)
     ├── KimiVisionProvider   (OpenAI-compatible)
     ├── MinimaxVisionProvider (OpenAI-compatible)
     └── VisionProviderFactory.create() → 根据 MOUSE_COLOR_AI_PROVIDER 创建实例

环境变量（命名明确，不提交真实密钥，无默认 base/model 值；用户必须从供应商文档获取）:
  MOUSE_COLOR_AI_PROVIDER  — "kimi" | "minimax" | "openrouter" | "hsv" (默认 "hsv")
  MOUSE_COLOR_AI_TIMEOUT   — 请求超时秒数 (默认 30)
  MOUSE_COLOR_AI_MAX_REQUESTS_PER_SEGMENT — 每个 segment 最多云视觉请求数 (默认 3, 0=禁用)

  Kimi:
    KIMI_API_KEY           — API 密钥（必填）
    KIMI_API_BASE          — API base URL（必填，从 Moonshot 文档获取）
    KIMI_VISION_MODEL      — 视觉模型名（必填，须为支持视觉的模型）

  Minimax:
    MINIMAX_API_KEY        — API 密钥（必填）
    MINIMAX_API_BASE       — API base URL（必填，从 MiniMax 文档获取）
    MINIMAX_VISION_MODEL   — 视觉模型名（必填，须为支持视觉的模型）

  OpenRouter:
    OPENROUTER_API_KEY     — API 密钥（必填）
    OPENROUTER_API_BASE    — API base URL（必填）
    OPENROUTER_VISION_MODEL — 视觉模型名（必填，须确认模型支持 image input）

输入/输出契约:
  classify(patch_bgr: np.ndarray) -> (color: str, confidence: float, method: str)
    color ∈ {"red","yellow","blue","green","white","unknown"}
    method = "ear_tag_color_kimi" | "ear_tag_color_minimax" | "ear_tag_color_openrouter"
  异常时返回 ("unknown", 0.0, method) 且不崩溃。
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 排错：MOUSE_COLOR_AI_DEBUG=1 启用 DEBUG 级别日志
if os.environ.get("MOUSE_COLOR_AI_DEBUG", "").strip() in ("1", "true", "yes", "on"):
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(h)

CATEGORIES = ["red", "yellow", "blue", "green", "white", "unknown"]

# 超时默认值（秒）
DEFAULT_TIMEOUT = 30

# AI 视觉输出 token 上限默认值
DEFAULT_MAX_TOKENS = 512


def _read_max_tokens() -> int:
    """从环境变量 MOUSE_COLOR_AI_MAX_TOKENS 读取 AI 视觉输出 token 上限。

    要求严格正整数；非法/非正/未设置时回退到 DEFAULT_MAX_TOKENS (512)。
    日志记录最终采用的值，方便运营商追溯。
    """
    raw = os.environ.get("MOUSE_COLOR_AI_MAX_TOKENS", "").strip()
    if not raw:
        return DEFAULT_MAX_TOKENS
    try:
        val = int(raw)
    except ValueError:
        logger.warning(
            "MOUSE_COLOR_AI_MAX_TOKENS=%r 不是合法整数；回退默认 %d",
            raw, DEFAULT_MAX_TOKENS,
        )
        return DEFAULT_MAX_TOKENS
    if val <= 0:
        logger.warning(
            "MOUSE_COLOR_AI_MAX_TOKENS=%d 非正数；回退默认 %d",
            val, DEFAULT_MAX_TOKENS,
        )
        return DEFAULT_MAX_TOKENS
    logger.debug("MOUSE_COLOR_AI_MAX_TOKENS=%d (user override)", val)
    return val


# ------------------------------------------------------------------
# 辅助：BGR → JPEG base64 data URI
# ------------------------------------------------------------------
def _bgr_to_jpeg_data_uri(patch_bgr: np.ndarray, quality: int = 85) -> str:
    """将 BGR numpy 图像编码为 `data:image/jpeg;base64,…`。

    :param patch_bgr: (H,W,3) uint8 BGR 图像
    :param quality: JPEG 质量 (1-100)
    :returns: data URI 字符串
    """
    success, buf = cv2.imencode(".jpg", patch_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ValueError("JPEG encoding failed")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# ------------------------------------------------------------------
# 辅助：解析 AI 模型结构化 JSON 输出
# ------------------------------------------------------------------
def _parse_segment_json(raw_text: str) -> dict | None:
    """Parse only a final, complete segment JSON response.

    Unlike the legacy color parser, this intentionally accepts no prose/code
    fences and never consults provider reasoning fields.  A legacy
    ``{color, confidence}`` final JSON object is normalized to one mouse.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None
    try:
        data = json.loads(raw_text.strip())
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        confidence = float(data["confidence"])
    except (KeyError, TypeError, ValueError):
        return None
    if not 0.0 <= confidence <= 1.0:
        return None

    # Backward-compatible final JSON contract.  A missing thermometer field
    # predates the interference detector and is safely treated as false.
    thermometer_present = data.get("thermometer_present", False)
    if not isinstance(thermometer_present, bool):
        return None
    if "mouse_count" not in data and "colors" not in data:
        color = data.get("color")
        if not isinstance(color, str) or color.strip().lower() not in CATEGORIES:
            return None
        return {"mouse_count": 1, "colors": [color.strip().lower()],
                "confidence": confidence, "thermometer_present": thermometer_present,
                "parse_status": "legacy"}

    count = data.get("mouse_count")
    colors = data.get("colors")
    # bool is an int subclass but is not a valid count.
    if isinstance(count, bool) or not isinstance(count, int) or count not in (1, 2):
        return None
    if not isinstance(colors, list) or len(colors) != count:
        return None
    normalized = []
    for color in colors:
        if not isinstance(color, str) or color.strip().lower() not in CATEGORIES:
            return None
        normalized.append(color.strip().lower())
    return {"mouse_count": count, "colors": normalized,
            "confidence": confidence, "thermometer_present": thermometer_present,
            "parse_status": "ok"}


def _parse_color_json(raw_text: str) -> tuple[str, float] | None:
    """从模型返回文本中提取颜色分类 JSON。

    容错策略:
      1. 尝试直接 json.loads 整个文本
      2. 尝试提取第一个 {…} 块再解析
      3. 验证 color ∈ CATEGORIES, confidence ∈ [0,1]
      4. 失败返回 None

    :returns: (color, confidence) 或 None
    """
    if not raw_text or not isinstance(raw_text, str):
        return None

    candidates = [raw_text.strip()]

    # 尝试提取 JSON 块
    if not raw_text.strip().startswith("{"):
        import re
        m = re.search(r"\{[^{}]*\}", raw_text)
        if m:
            candidates.append(m.group(0))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue

        if not isinstance(data, dict):
            continue

        color = str(data.get("color", "")).strip().lower()
        if color not in CATEGORIES:
            continue

        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        if not (0.0 <= confidence <= 1.0):
            confidence = max(0.0, min(1.0, confidence))

        return color, confidence

    return None


# ------------------------------------------------------------------
# VisionProvider 抽象基类
# ------------------------------------------------------------------
class VisionProvider(ABC):
    """AI 视觉颜色识别 provider 抽象基类。"""

    @abstractmethod
    def classify(self, patch_bgr: np.ndarray) -> tuple[str, float, str]:
        """对耳标 patch 图像进行颜色分类。

        :param patch_bgr: (H,W,3) uint8 BGR 图像
        :returns: (color, confidence, method)
        :raises VisionProviderError: 可诊断的配置/网络/API 错误
        """
        ...

    @abstractmethod
    def classify_frames(self, frames: list[np.ndarray]) -> tuple[str, float, str, dict | None]:
        """多帧上下文颜色分类（每个 segment 一次调用，携带大范围上下文 crops）。

        :param frames: (H,W,3) uint8 BGR 图像列表，按时间顺序排列
        :returns: (color, confidence, method, raw_response_json_dict_or_None)
        :raises VisionProviderError: 可诊断的配置/网络/API 错误
        """
        ...

    def classify_segment_frames(self, frames: list[np.ndarray]) -> dict:
        """Backward-compatible segment API for count and ordered colors."""
        color, confidence, method, raw = self.classify_frames(frames)
        return {"mouse_count": 1, "colors": [color], "confidence": confidence,
                "method": method, "raw_response": raw,
                "parse_status": "legacy_adapter"}


class VisionProviderError(Exception):
    """AI 视觉 provider 可诊断错误。"""

    def __init__(self, message: str, provider: str = "", status_code: int | None = None):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


# ------------------------------------------------------------------
# OpenAI 兼容 Provider 基类（Kimi / Minimax 共用）
# ------------------------------------------------------------------
class _OpenAICompatVisionProvider(VisionProvider):
    """OpenAI-compatible Chat Completions API 通用实现。

    子类需提供:
      - _env_prefix: 环境变量前缀 ("KIMI" | "MINIMAX" | "OPENROUTER")
      - _method_name: identity_method 字段值
    """

    _env_prefix: str = ""
    _method_name: str = ""

    def __init__(self):
        self._api_key: str = ""
        self._api_base: str = ""
        self._model: str = ""
        self._timeout: float = DEFAULT_TIMEOUT
        self._max_tokens: int = DEFAULT_MAX_TOKENS
        self._configured: bool = False
        self._init_config()

    def _init_config(self):
        """从环境变量读取配置并验证。

        不再提供默认 base URL / 模型名：用户必须从供应商当前文档获取
        正确的端点和视觉模型名填入环境变量。
        """
        prefix = self._env_prefix

        self._api_key = os.environ.get(f"{prefix}_API_KEY", "").strip()
        self._api_base = os.environ.get(f"{prefix}_API_BASE", "").strip()
        self._model = os.environ.get(f"{prefix}_VISION_MODEL", "").strip()

        timeout_str = os.environ.get("MOUSE_COLOR_AI_TIMEOUT", "").strip()
        if timeout_str:
            try:
                self._timeout = float(timeout_str)
            except ValueError:
                logger.warning("Invalid MOUSE_COLOR_AI_TIMEOUT=%r; using default %ds", timeout_str, DEFAULT_TIMEOUT)
                self._timeout = DEFAULT_TIMEOUT

        # 验证必填项（不再提供默认值）
        missing = []
        if not self._api_key:
            missing.append(f"{prefix}_API_KEY")
        if not self._api_base:
            missing.append(f"{prefix}_API_BASE")
        if not self._model:
            missing.append(f"{prefix}_VISION_MODEL")

        if missing:
            raise VisionProviderError(
                f"缺少必填环境变量: {', '.join(missing)}。"
                f"请从 {self._method_name.replace('ear_tag_color_', '').upper()} 供应商文档获取正确的端点和视觉模型名，并设置为环境变量。",
                provider=self._method_name,
            )

        self._configured = True
        self._max_tokens = _read_max_tokens()
        logger.info(
            "%s configured: base=%s model=%s timeout=%.0fs max_tokens=%d",
            self._method_name, self._api_base, self._model, self._timeout, self._max_tokens,
        )

    def classify(self, patch_bgr: np.ndarray) -> tuple[str, float, str]:
        """对耳标 patch 调用视觉模型分类。

        :returns: (color, confidence, method)
        """
        if not self._configured:
            raise VisionProviderError(
                "Provider 未正确配置；缺少 API 密钥或端点。",
                provider=self._method_name,
            )

        if patch_bgr is None or patch_bgr.size == 0:
            return "unknown", 0.0, self._method_name

        # 编码图像
        try:
            data_uri = _bgr_to_jpeg_data_uri(patch_bgr)
        except Exception as exc:
            logger.warning("%s: image encoding failed: %s", self._method_name, exc)
            return "unknown", 0.0, self._method_name

        # 构造请求
        payload = self._build_payload(data_uri)
        headers = self._build_headers()
        endpoint = f"{self._api_base.rstrip('/')}/chat/completions"

        # 排错日志：每次云视觉请求
        logger.info(
            "%s -> POST %s model=%s patch_size=%dx%d bytes",
            self._method_name, endpoint, self._model,
            patch_bgr.shape[1], patch_bgr.shape[0],
        )

        # 发送请求
        resp_json = self._post_json(endpoint, headers, payload)

        # 排错：保存输入 patch 和原始返回 (MOUSE_COLOR_AI_SAVE_DEBUG=1)
        self._save_debug(patch_bgr, resp_json)

        # 解析响应
        return self._parse_response(resp_json)

    def classify_frames(self, frames: list[np.ndarray]) -> tuple[str, float, str, dict | None]:
        """多帧上下文颜色分类（每个 segment 一次调用）。

        :param frames: (H,W,3) uint8 BGR 图像列表，按时间顺序排列
        :returns: (color, confidence, method, raw_response_json_or_None)
        """
        if not self._configured:
            raise VisionProviderError(
                "Provider 未正确配置；缺少 API 密钥或端点。",
                provider=self._method_name,
            )

        if not frames:
            return "unknown", 0.0, self._method_name, None

        # 编码所有帧
        data_uris = []
        for i, frame in enumerate(frames):
            if frame is None or frame.size == 0:
                logger.warning("%s: frame %d empty, skipping", self._method_name, i)
                continue
            try:
                data_uris.append(_bgr_to_jpeg_data_uri(frame))
            except Exception as exc:
                logger.warning("%s: frame %d encoding failed: %s", self._method_name, i, exc)

        if not data_uris:
            logger.warning("%s: no valid frames to send", self._method_name)
            return "unknown", 0.0, self._method_name, None

        # 构造多帧请求
        payload = self._build_multi_payload(data_uris, len(frames))
        headers = self._build_headers()
        endpoint = f"{self._api_base.rstrip('/')}/chat/completions"

        # 排错日志
        logger.info(
            "%s -> POST %s model=%s frames=%d context_call",
            self._method_name, endpoint, self._model, len(data_uris),
        )

        # 发送请求
        resp_json = self._post_json(endpoint, headers, payload)

        # 排错：保存多帧 context 输入和 API 原始返回
        self._save_debug_multi(frames, resp_json)

        result = self.classify_segment_frames_from_response(resp_json)
        return result["colors"][0], result["confidence"], result["method"], resp_json

    def classify_segment_frames(self, frames: list[np.ndarray]) -> dict:
        """Classify one segment and return validated count/colors metadata."""
        if not self._configured:
            raise VisionProviderError("Provider 未正确配置；缺少 API 密钥或端点。", provider=self._method_name)
        if not frames:
            return {"mouse_count": 1, "colors": ["unknown"], "confidence": 0.0,
                    "method": self._method_name, "raw_response": None, "parse_status": "no_frames"}
        data_uris = []
        for frame in frames:
            if frame is not None and frame.size > 0:
                try:
                    data_uris.append(_bgr_to_jpeg_data_uri(frame))
                except Exception as exc:
                    logger.warning("%s: frame encoding failed: %s", self._method_name, exc)
        if not data_uris:
            return {"mouse_count": 1, "colors": ["unknown"], "confidence": 0.0,
                    "method": self._method_name, "raw_response": None, "parse_status": "no_valid_frames"}
        endpoint = f"{self._api_base.rstrip('/')}/chat/completions"
        logger.info("%s -> POST %s model=%s frames=%d segment_count_call", self._method_name, endpoint, self._model, len(data_uris))
        resp_json = self._post_json(endpoint, self._build_headers(), self._build_multi_payload(data_uris, len(frames)))
        self._save_debug_multi(frames, resp_json)
        return self.classify_segment_frames_from_response(resp_json)

    def classify_segment_frames_from_response(self, resp_json: dict) -> dict:
        """Decode final-content JSON only; reasoning is deliberately ignored."""
        try:
            choice = (resp_json.get("choices") or [])[0]
            content = (choice.get("message", {}).get("content") or "").strip()
            parsed = _parse_segment_json(content)
            if parsed is None:
                logger.warning("%s: invalid segment JSON final content", self._method_name)
                return {"mouse_count": 1, "colors": ["unknown"], "confidence": 0.0,
                        "method": self._method_name, "raw_response": resp_json, "parse_status": "invalid"}
            parsed.update({"method": self._method_name, "raw_response": resp_json})
            return parsed
        except Exception as exc:
            logger.warning("%s: segment response parsing exception: %s", self._method_name, exc)
            return {"mouse_count": 1, "colors": ["unknown"], "confidence": 0.0,
                    "method": self._method_name, "raw_response": resp_json, "parse_status": "invalid"}

    def _build_payload(self, data_uri: str) -> dict:
        """构造 OpenAI-compatible Chat Completions 请求体。"""
        return {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a precise color classifier for mouse ear tags. "
                        "Analyze the image and output ONLY a JSON object with no markdown formatting, no code fences, no extra text. "
                        "The JSON must have exactly two fields: "
                        '"color" (one of: "red", "yellow", "blue", "green", "white", "unknown") '
                        'and "confidence" (a float between 0.0 and 1.0). '
                        "Respond with just the raw JSON object, nothing else."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Classify the dominant color of the ear tag in this image crop.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_uri, "detail": "low"},
                        },
                    ],
                },
            ],
            "max_tokens": self._max_tokens,
            "temperature": 0.0,
        }

    def _build_multi_payload(self, data_uris: list[str], num_frames: int | None = None) -> dict:
        """构造多帧上下文 OpenAI-compatible Chat Completions 请求体。

        所有帧放在同一个 user content 数组中，按时间顺序排列。
        提示词明确说明这是同一 clip 的前/中/后上下文图。
        """
        n = num_frames if num_frames is not None else len(data_uris)
        # 将帧分成三组：前 1/3、中 1/3、后 1/3
        if n >= 9:
            group_desc = "the beginning (first 3), middle (next 3), and end (last 3)"
        elif n >= 3:
            g1 = (n + 2) // 3
            g2 = (n - g1 + 1) // 2
            group_desc = f"the beginning (first {g1}), middle (next {g2}), and end (last {n - g1 - g2})"
        else:
            group_desc = "the same short video clip"

        content_parts = [
            {
                "type": "text",
                "text": (
                    f"You are analyzing {n} context frames from the SAME video clip of a mouse experiment, "
                    f"sampled from {group_desc} in chronological order. "
                    "Determine whether ONE or TWO mice simultaneously occupy the warm-spot region during this same segment. "
                    "Do not invent a second mouse from reflections, afterimages, or mice replacing each other at different times. "
                    "Also determine whether any frame shows a handheld thermometer, temperature probe, or other non-mouse device obstructing or intruding into the warm-spot area. "
                    "Do not mistake the warm spot, reflections, or mouse ear tags for an instrument. "
                    "If uncertain choose 1 with low confidence and unknown color(s). Return each mouse's ear-tag color in order. "
                    "Output ONLY raw JSON, exactly: {\"mouse_count\": 1 or 2, \"colors\": [colors matching count], \"confidence\": 0.0 to 1.0, \"thermometer_present\": true or false}. "
                    "thermometer_present must be a JSON boolean. Each color must be red, yellow, blue, green, white, or unknown; no markdown or other text."
                ),
            }
        ]
        for uri in data_uris:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": uri, "detail": "low"},
            })

        return {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a precise segment-level mouse count and ear-tag color classifier. "
                        "Return only the requested final JSON; never put an answer in reasoning."
                    ),
                },
                {
                    "role": "user",
                    "content": content_parts,
                },
            ],
            "max_tokens": self._max_tokens,
            "temperature": 0.0,
        }

    def _build_headers(self) -> dict:
        """构造 HTTP 请求头。"""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

    def _post_json(self, endpoint: str, headers: dict, payload: dict) -> dict:
        """发送 POST 请求并返回解析后的 JSON。

        :raises VisionProviderError: 网络/API 错误
        """
        import requests

        try:
            resp = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=self._timeout,
            )
        except requests.exceptions.Timeout as exc:
            raise VisionProviderError(
                f"{self._method_name} API 请求超时 ({self._timeout}s): {exc}",
                provider=self._method_name,
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise VisionProviderError(
                f"{self._method_name} API 连接失败: {exc}",
                provider=self._method_name,
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise VisionProviderError(
                f"{self._method_name} API 请求异常: {exc}",
                provider=self._method_name,
            ) from exc

        if not resp.ok:
            body_preview = resp.text[:500] if resp.text else "(empty)"
            raise VisionProviderError(
                f"{self._method_name} API 返回 HTTP {resp.status_code}: {body_preview}",
                provider=self._method_name,
                status_code=resp.status_code,
            )

        try:
            return resp.json()
        except ValueError as exc:
            raise VisionProviderError(
                f"{self._method_name} API 返回非 JSON 响应: {resp.text[:300]}",
                provider=self._method_name,
            ) from exc

    def _parse_response(self, resp_json: dict) -> tuple[str, float, str]:
        """从 OpenAI-compatible 响应中提取颜色分类结果。

        当 finish_reason == "length" 且 content 无法解析为有效 JSON 时，
        发出可诊断警告（含当前 max_tokens 和调高建议），绝不回退解读
        reasoning/reasoning_details 中的自然语言文本。

        :returns: (color, confidence, method)
        """
        try:
            choices = resp_json.get("choices", [])
            if not choices:
                raise VisionProviderError(
                    f"{self._method_name} API 返回空 choices 列表",
                    provider=self._method_name,
                )

            choice0 = choices[0]
            finish_reason = choice0.get("finish_reason", "unknown")
            message = choice0.get("message", {})
            content = (message.get("content") or "").strip()

            # 截断诊断：finish_reason == "length" 且 content 空/无法解析 JSON
            # 常见于带 reasoning 的模型（MiniMax M3 等）在低 max_tokens 时被截断
            if finish_reason == "length":
                has_reasoning = bool(
                    message.get("reasoning") or message.get("reasoning_details")
                )
                parsed = _parse_color_json(content) if content else None
                if parsed is None:
                    logger.warning(
                        "%s: 模型输出被截断 (finish_reason=length, content=%r). "
                        "当前 MOUSE_COLOR_AI_MAX_TOKENS=%d 可能不足以让带 reasoning 的模型完成最终 JSON 输出。"
                        "建议提高 MOUSE_COLOR_AI_MAX_TOKENS（如 1024 或更高）。"
                        "%s reasoning 文本（截断前内容不可作为分类结果）。",
                        self._method_name,
                        content[:200] if content else "(empty)",
                        self._max_tokens,
                        "模型存在" if has_reasoning else "模型不存在",
                    )
                    return "unknown", 0.0, self._method_name

            if not content:
                raise VisionProviderError(
                    f"{self._method_name} API 返回空 content (finish_reason={finish_reason})",
                    provider=self._method_name,
                )

            result = _parse_color_json(content)
            if result is None:
                # 二次截断诊断：有 content 但解析失败也可能是 length 截断
                if finish_reason == "length":
                    logger.warning(
                        "%s: 模型输出被截断 (finish_reason=length)，content 无法解析为 JSON: %s. "
                        "当前 MOUSE_COLOR_AI_MAX_TOKENS=%d；建议调高。",
                        self._method_name,
                        content[:200],
                        self._max_tokens,
                    )
                else:
                    logger.warning(
                        "%s: failed to parse structured output from: %s",
                        self._method_name,
                        content[:200],
                    )
                return "unknown", 0.0, self._method_name

            color, confidence = result
            logger.info(
                "%s classify result: color=%s confidence=%.3f model_raw=%s",
                self._method_name, color, confidence, content[:120].replace("\n", " "),
            )
            return color, confidence, self._method_name

        except VisionProviderError:
            raise
        except Exception as exc:
            logger.warning("%s: response parsing exception: %s", self._method_name, exc)
            return "unknown", 0.0, self._method_name

    # ------------------------------------------------------------------
    # 排错：保存输入 patch 和 API 原始返回
    # ------------------------------------------------------------------
    _debug_counter: int = 0

    def _save_debug(self, patch_bgr: np.ndarray, resp_json: dict):
        """MOUSE_COLOR_AI_SAVE_DEBUG=1 时，保存 patch 图片和 response JSON 到 debug_vision_output/"""
        if os.environ.get("MOUSE_COLOR_AI_SAVE_DEBUG", "").strip() not in ("1", "true", "yes", "on"):
            return
        try:
            out_dir = Path.cwd() / "debug_vision_output"
            out_dir.mkdir(parents=True, exist_ok=True)
            idx = _OpenAICompatVisionProvider._debug_counter
            _OpenAICompatVisionProvider._debug_counter += 1
            # 保存 patch
            cv2.imwrite(str(out_dir / f"patch_{idx:03d}.jpg"), patch_bgr)
            # 保存 response
            with open(out_dir / f"response_{idx:03d}.json", "w", encoding="utf-8") as f:
                json.dump(resp_json, f, indent=2, ensure_ascii=False, default=str)
            logger.info("debug saved: patch_%03d.jpg + response_%03d.json", idx, idx)
        except Exception as exc:
            logger.debug("debug save failed: %s", exc)

    def _save_debug_multi(self, frames: list[np.ndarray], resp_json: dict):
        """MOUSE_COLOR_AI_SAVE_DEBUG=1 时，保存多帧 context crops 和 response JSON。

        注意：不保存 Authorization header/API key，不保存 base64 payload。
        日志记录 finish_reason 与 parse 状态，方便诊断截断问题。
        """
        if os.environ.get("MOUSE_COLOR_AI_SAVE_DEBUG", "").strip() not in ("1", "true", "yes", "on"):
            return
        try:
            out_dir = Path.cwd() / "debug_vision_output"
            out_dir.mkdir(parents=True, exist_ok=True)
            idx = _OpenAICompatVisionProvider._debug_counter
            _OpenAICompatVisionProvider._debug_counter += 1
            # 保存每个 context crop
            for i, crop in enumerate(frames):
                cv2.imwrite(str(out_dir / f"context_{idx:03d}_{i:02d}.jpg"), crop)
            # 保存 response
            with open(out_dir / f"context_response_{idx:03d}.json", "w", encoding="utf-8") as f:
                json.dump(resp_json, f, indent=2, ensure_ascii=False, default=str)
            # Manifest intentionally contains only safe metadata, never headers,
            # keys, request payloads, or base64 image data.
            choices = resp_json.get("choices", []) if isinstance(resp_json, dict) else []
            content = (choices[0].get("message", {}).get("content") or "").strip() if choices else ""
            parsed_segment = _parse_segment_json(content)
            manifest = {
                "num_context_crops": len(frames),
                "ai_mouse_count": parsed_segment.get("mouse_count") if parsed_segment else None,
                "ai_colors": parsed_segment.get("colors") if parsed_segment else None,
                "confidence": parsed_segment.get("confidence") if parsed_segment else None,
                "thermometer_present": parsed_segment.get("thermometer_present") if parsed_segment else None,
                "parse_status": parsed_segment.get("parse_status") if parsed_segment else "invalid",
            }
            with open(out_dir / f"context_manifest_{idx:03d}.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

            # 排错：记录 finish_reason 与解析状态（不含 key / base64）
            try:
                choices = resp_json.get("choices", [])
                finish_reason = choices[0].get("finish_reason", "?") if choices else "?"
                content = (choices[0].get("message", {}).get("content") or "").strip() if choices else ""
                parsed = _parse_color_json(content) if content else None
                if parsed:
                    logger.info(
                        "debug saved: context_%03d (x%d frames) + context_response_%03d.json "
                        "finish_reason=%s parse=OK color=%s",
                        idx, len(frames), idx, finish_reason, parsed[0],
                    )
                else:
                    logger.info(
                        "debug saved: context_%03d (x%d frames) + context_response_%03d.json "
                        "finish_reason=%s parse=FAIL content_len=%d",
                        idx, len(frames), idx, finish_reason, len(content),
                    )
            except Exception:
                logger.info(
                    "debug saved: context_%03d (x%d frames) + context_response_%03d.json",
                    idx, len(frames), idx,
                )
        except Exception as exc:
            logger.debug("debug save multi failed: %s", exc)


# ------------------------------------------------------------------
# Kimi Vision Provider
# ------------------------------------------------------------------
class GPTVisionProvider(_OpenAICompatVisionProvider):
    """GPT vision provider (the supported production VLM path for segment counting)."""

    _env_prefix = "GPT"
    _method_name = "ear_tag_color_gpt"


class KimiVisionProvider(_OpenAICompatVisionProvider):
    """Kimi (Moonshot) 视觉模型 provider。

    环境变量（全部必填，无默认值；请从 Moonshot 文档获取当前正确的端点与模型名）:
      KIMI_API_KEY       — API 密钥
      KIMI_API_BASE      — API base URL
      KIMI_VISION_MODEL  — 视觉模型名（须支持视觉）
    """

    _env_prefix = "KIMI"
    _method_name = "ear_tag_color_kimi"


# ------------------------------------------------------------------
# Minimax Vision Provider
# ------------------------------------------------------------------
class MinimaxVisionProvider(_OpenAICompatVisionProvider):
    """MiniMax 视觉模型 provider。

    环境变量（全部必填，无默认值；请从 MiniMax 文档获取当前正确的端点与模型名）:
      MINIMAX_API_KEY       — API 密钥
      MINIMAX_API_BASE      — API base URL
      MINIMAX_VISION_MODEL  — 视觉模型名（须支持视觉）
    """

    _env_prefix = "MINIMAX"
    _method_name = "ear_tag_color_minimax"


# ------------------------------------------------------------------
# OpenRouter Vision Provider
# ------------------------------------------------------------------
class OpenRouterVisionProvider(_OpenAICompatVisionProvider):
    """OpenRouter 视觉模型 provider（使用 OpenAI-compatible 格式）。

    环境变量（全部必填，无默认值；用户须确认所选模型具备 image input 能力）:
      OPENROUTER_API_KEY       — API 密钥
      OPENROUTER_API_BASE      — API base URL
      OPENROUTER_VISION_MODEL  — 视觉模型名（须支持视觉输入）
    """

    _env_prefix = "OPENROUTER"
    _method_name = "ear_tag_color_openrouter"


# ------------------------------------------------------------------
# Provider 工厂
# ------------------------------------------------------------------
class VisionProviderFactory:
    """根据环境变量创建 VisionProvider 实例。

    用法::

        provider = VisionProviderFactory.create()
        try:
            color, confidence, method = provider.classify(patch_bgr)
        except VisionProviderError as e:
            logger.warning("Vision classification failed: %s", e)

    环境变量:
      MOUSE_COLOR_AI_PROVIDER  — "kimi" | "minimax" | "openrouter" | "hsv" (默认 "hsv")
    """

    # provider 注册表
    _registry: dict[str, type[VisionProvider]] = {
        "gpt": GPTVisionProvider,
        "kimi": KimiVisionProvider,
        "minimax": MinimaxVisionProvider,
        "openrouter": OpenRouterVisionProvider,
    }

    @classmethod
    def create(cls) -> VisionProvider | None:
        """根据 MOUSE_COLOR_AI_PROVIDER 创建 provider 实例。

        :returns: VisionProvider 实例，若配置为 "hsv" 或无有效配置则返回 None
        :raises VisionProviderError: provider 配置错误（缺密钥等）
        """
        provider_name = os.environ.get("MOUSE_COLOR_AI_PROVIDER", "hsv").strip().lower()

        if provider_name == "hsv":
            logger.info("MOUSE_COLOR_AI_PROVIDER=hsv; using HSV rule classifier only")
            return None

        if provider_name not in cls._registry:
            supported = ", ".join(sorted(cls._registry.keys()))
            raise VisionProviderError(
                f"不支持的 MOUSE_COLOR_AI_PROVIDER={provider_name!r}。"
                f"支持的值: hsv, {supported}",
            )

        provider_cls = cls._registry[provider_name]
        logger.info("Creating AI vision provider: %s", provider_name)
        return provider_cls()

    @classmethod
    def register(cls, name: str, provider_cls: type[VisionProvider]):
        """注册自定义 provider（扩展点）。"""
        cls._registry[name.lower()] = provider_cls
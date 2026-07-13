# 小鼠暖点占据半自动标注系统

桌面 GUI 程序，用于分析小鼠暖点实验视频。系统自动检测暖点被遮挡的时间段，并估计小鼠数量；人工通过快捷键快速审核并标注小鼠身份，最后导出统计表。

## 安装与运行

```bash
# 1. 进入项目目录
cd mouse_warm_spot

# 2. 安装依赖
pip install -r requirements.txt

# 3. (可选) 安装 AI 推理依赖（ONNX CNN + AI 视觉模型 HTTP 客户端）
pip install -r requirements-ai.txt

# 4. (可选) 配置 AI 视觉识别（Kimi / MiniMax / OpenRouter）
# 将以下环境变量导出到进程环境中，或创建 .env 文件后 source：
#   MOUSE_COLOR_AI_PROVIDER=openrouter
#   OPENROUTER_API_KEY=<your-key>
#   OPENROUTER_API_BASE=https://openrouter.ai/api/v1
#   OPENROUTER_VISION_MODEL=minimax/minimax-m3
# 详见下方「可选 AI 视觉识别」章节

# 5. 启动（若使用 .env，需 source 后再启动）
set -a && source .env && set +a && python main.py

### 可选 AI 推理（耳标颜色 CNN 分类器）

系统默认使用 HSV 规则对耳标颜色进行分类，无需任何额外依赖。如需更准确的颜色识别，可启用基于 ONNX 的轻量 CNN 分类器：

```bash
# 安装可选 AI 推理依赖
pip install -r requirements-ai.txt

# 将训练好的 ONNX 模型放入 detection/models/ 目录
# 期望文件名: ear_tag_classifier.onnx
# 程序会自动发现该文件并启用 CNN 推理
```

**环境变量控制**（无需修改代码）：

| 环境变量 | 说明 | 默认值 |
|---|---|---|
| `MOUSE_COLOR_MODEL_PATH` | 自定义 ONNX 模型文件路径 | 自动发现 `detection/models/ear_tag_classifier.onnx` |
| `MOUSE_COLOR_USE_CNN` | 是否启用 CNN（`0`=禁用，`1`=启用） | 自动检测（依赖 + 文件均可用时启用） |

回退机制：

| 场景 | 行为 |
|---|---|
| `onnxruntime` 未安装 | 静默降级为 HSV 规则，不影响程序功能 |
| 模型文件不存在 | 静默降级为 HSV 规则 |
| CNN 推理失败（任何异常） | 该轮廓自动回退到 HSV 规则；其余轮廓正常处理 |
| `MOUSE_COLOR_USE_CNN=0` | 强制使用 HSV 规则，不尝试加载模型 |

识别结果中 `identity_method` 字段会标明实际使用的分类方式：
- `ear_tag_color_vlm|exact_match` — AI 视觉模型推理成功 (Kimi / Minimax / OpenRouter)
- `ear_tag_color_cnn|exact_match` — ONNX CNN 推理成功
- `ear_tag_color_rule|exact_match` — 使用 HSV 规则

### 可选 AI 视觉识别（GPT）

系统支持将 segment 的大范围上下文图像（多帧）发送到 AI 视觉大模型进行颜色识别，
每个 segment 仅调用一次，优先级高于 CNN 和 HSV 规则。

**工作原理**：
- 每个 CountSegment 选择最多 9 帧上下文图像：头三帧 (start, start+5, start+10)、中间三帧 (mid-5, mid, mid+5)、尾三帧 (end-10, end-5, end)，帧间隔为 5
- 每帧裁剪外圈大范围图像：以 ROI 核心中心为中心，搜索 ROI（Core × 2.8）× 1.5 的矩形区域
- 9 帧在一个请求中按时间顺序发送给 VLM，附带前/中/后上下文提示词
- 短 segment 帧数不足时会去重并安全夹取到 [start, end]

**成本控制**：
- 每个 segment 仅消耗 1 次 `MOUSE_COLOR_AI_MAX_REQUESTS_PER_SEGMENT` 预算
- 预算在真实 API 调用前预扣，网络/API/解析失败仍消耗该次额度
- 预算为 0 时完全禁用 AI 视觉（不产生网络调用）

**结果融合**：
- VLM 返回合法 `mouse_count`（仅 1/2）、首个已知颜色且置信度 ≥ 0.3 时，优先重确认 segment 计数，并同步 `estimated_mouse_count`、`mouse_count` 与返回的 `target_count`
- VLM 的有序颜色直接作为结果；单鼠不会被规则候选覆盖。双鼠第二色为 `unknown` 或两色重复时保守标记复核，绝不凭规则臆造颜色
- VLM 失败、unknown、低置信度、无效 JSON 或无有效 crop 时完整回退到原 CNN/HSV 和本地计数
- 已人工 confirmed 的 segment 不会被自动写回覆盖

**环境变量配置**（全部必填，无默认值；请从供应商文档获取当前正确的端点与模型名）：

| 环境变量 | 说明 | 默认值 |
|---|---|---|
| `MOUSE_COLOR_AI_PROVIDER` | 本功能使用的视觉识别 provider：`gpt`（或 `hsv` 禁用） | `hsv`（仅使用 HSV 规则） |
| `MOUSE_COLOR_AI_TIMEOUT` | API 请求超时秒数 | `30` |
| `MOUSE_COLOR_AI_MAX_TOKENS` | AI 视觉输出 token 上限（必须为正整数；对带 reasoning 的模型建议 ≥512） | `512` |
| `MOUSE_COLOR_AI_MAX_REQUESTS_PER_SEGMENT` | 每个 segment 最多云视觉请求数（`0`=禁用云视觉） | `3` |

### Kimi (Moonshot) 所需环境变量（全部必填）：

| 环境变量 | 说明 |
|---|---|
| `KIMI_API_KEY` | API 密钥 |
| `KIMI_API_BASE` | API 端点 URL（从 Moonshot 文档获取，不可猜测） |
| `KIMI_VISION_MODEL` | 视觉模型名（须为支持视觉的模型，不可猜测） |

**MiniMax** 所需环境变量（全部必填）：

| 环境变量 | 说明 |
|---|---|
| `MINIMAX_API_KEY` | API 密钥 |
| `MINIMAX_API_BASE` | API 端点 URL（从 MiniMax 文档获取，不可猜测） |
| `MINIMAX_VISION_MODEL` | 视觉模型名（须为支持视觉的模型，不可猜测） |

**OpenRouter** 所需环境变量（全部必填；用户须确认所选模型具备 image input 能力）：

| 环境变量 | 说明 |
|---|---|
| `OPENROUTER_API_KEY` | API 密钥 |
| `OPENROUTER_API_BASE` | API 端点 URL（从 OpenRouter 文档获取） |
| `OPENROUTER_VISION_MODEL` | 视觉模型名（如 `<vision-capable-model-id>`，须支持视觉输入） |

**配置方式**：

这些环境变量需要导出到进程环境，或由用户现有启动器注入。项目**不会**自动加载 `.env` 文件。

如果用户的启动器支持 `.env`（如 `python-dotenv`、shell 脚本 `source .env`、IDE 运行配置等），
则将以下变量写入 `.env` 作为参考方案：

```bash
# 使用 Kimi 视觉模型（示例，请替换为实际值）
MOUSE_COLOR_AI_PROVIDER=kimi
KIMI_API_KEY=<your-api-key>
KIMI_API_BASE=<provider-base-url>
KIMI_VISION_MODEL=<vision-capable-model-id>

# 或使用 MiniMax 视觉模型
# MOUSE_COLOR_AI_PROVIDER=minimax
# MINIMAX_API_KEY=<your-api-key>
# MINIMAX_API_BASE=<provider-base-url>
# MINIMAX_VISION_MODEL=<vision-capable-model-id>

# 每 segment 最多 3 次云请求（超出后仅 CNN/HSV）
MOUSE_COLOR_AI_MAX_REQUESTS_PER_SEGMENT=3
```

直接启动时，请用环境变量方式：
```bash
KIMI_API_KEY=<your-api-key> KIMI_API_BASE=<provider-base-url> KIMI_VISION_MODEL=<vision-capable-model-id> python main.py
```

**响应 JSON 契约**（模型需返回以下结构）：

```json
{
  "mouse_count": 1,
  "colors": ["red"],
  "confidence": 0.85,
  "thermometer_present": false
}
```

- `mouse_count` 只能为整数 `1` 或 `2`；`colors` 长度必须与其一致，且每项只能为 `red/yellow/blue/green/white/unknown`
- `confidence` 必须为 0.0~1.0，低于 0.3 自动回退。仅解析最终 `content` 中的完整 JSON，绝不从 reasoning 推断
- `thermometer_present` 必须为 JSON boolean；缺失时为兼容旧响应按 `false` 处理，存在但不是 boolean 时整条 VLM 结果无效并回退
- VLM 已接受且 `thermometer_present=true` 时，不采用其 count/colors 覆盖本地结果，并将 `identity_confidence=0.0`、`identity_needs_review=true`、`identity_conflict=false`，`identity_method` 加 `thermometer_detected`；提示人工复核。已 confirmed 事件仍不被写回覆盖
- 兼容旧最终 JSON `{ "color": "red", "confidence": 0.85 }`，按单鼠处理；新请求始终要求新契约

**回退行为**：

| 场景 | 行为 |
|---|---|
| `MOUSE_COLOR_AI_PROVIDER=hsv` 或未设置 | 不启用 AI 视觉，走 CNN/HSV |
| `MOUSE_COLOR_AI_MAX_REQUESTS_PER_SEGMENT=0` | 禁用云视觉，仅 CNN/HSV |
| API 密钥缺失 | 启动时记录警告，回退到 CNN/HSV |
| API Base URL 或模型名缺失 | 启动时报告具体缺失变量（可诊断错误） |
| VLM 上下文请求失败（网络/API/解析） | 消耗该次预算，回退到 CNN/HSV，不标 VLM |
| 模型返回无效 JSON | 解析失败，回退到 CNN/HSV |
| 模型输出被截断 (finish_reason=length, 无可解析 JSON) | 回退到 CNN/HSV，日志建议提高 `MOUSE_COLOR_AI_MAX_TOKENS`。注意：reasoning 中的自然语言文本不会被当作分类结果 |
| VLM 返回 unknown 或置信度 < 0.3 | 视为失败，消耗预算，回退到 CNN/HSV |
| Segment 无有效 context crop | 不消耗预算，直接走 CNN/HSV |

**GUI 范围选择**：点击“颜色识别”后选择“当前事件”“全部事件”或“取消”。未选事件时“当前事件”不可用，但仍可选择“全部事件”。全部事件仅处理非空片段，使用一个后台 worker 串行执行（不并发云请求）；进度显示当前 i/N、事件 ID 和内部进度，取消会停止后续事件并保留已完成结果。每个完成事件立即更新事件列表和标注面板。

**Debug 排错**：

设置 `MOUSE_COLOR_AI_SAVE_DEBUG=1` 后，每次 segment 的 VLM 上下文请求保存到 `debug_vision_output/`：
- 所有 context crops（文件名含排序编号和原视频帧索引）
- API 响应 JSON（成功或失败均保存）
- Manifest JSON（provider/model 不含 API key、选中帧索引、crop 算法/范围、解析结果或错误信息）

**provider 架构**：
- `VisionProvider` (ABC): 抽象接口 `classify(patch_bgr) → (color, confidence, method)` + `classify_frames(frames) → (color, confidence, method, raw_json)`
- `KimiVisionProvider` / `MinimaxVisionProvider` / `OpenRouterVisionProvider`: OpenAI-compatible 实现
- `VisionProviderFactory`: 工厂创建，支持通过 `register()` 扩展新 provider
- 位于 `detection/models/vision_provider.py`

## 技术栈

| 模块 | 技术 |
|---|---|
| 语言 | Python 3.9+ |
| GUI | PySide6 |
| 视频/图像 | OpenCV + NumPy |
| AI 推理（可选） | ONNX Runtime (onnxruntime) / AI 视觉模型 (Kimi/Minimax) |

## 核心设计

```
不自动识别小鼠身份。
不逐帧全视频高精度检测。

隔帧粗筛 → 触发回溯 → 局部逐帧精查 → 人工标注鼠标号
```

检测逻辑围绕 **暖点核心 ROI 是否被深色小鼠身体遮挡**，不做全画面运动检测。

---

## 项目结构

```
mouse_warm_spot/
├── main.py                      # 入口 + 暗色主题
├── requirements.txt             # PySide6 + OpenCV + NumPy
├── requirements-ai.txt          # 可选 AI 推理依赖 (onnxruntime + requests)
├── README.md
├── gui/
│   ├── main_window.py           # 主窗口：工具栏、菜单、信号连接、检测控制
│   ├── video_widget.py          # 视频显示 + ROI A/B/C 叠加 + 循环播放
│   ├── zoom_widget.py           # 暖点 ROI 放大窗口 (2x)
│   ├── metrics_panel.py         # 遮挡指标 + 小鼠计数 + 调试信息
│   ├── event_list_widget.py     # 事件列表 (CountSegment) + 颜色状态 + 右键菜单
│   ├── annotation_panel.py      # 标注面板：小鼠选择 + 数量确认
│   └── calibration_store.py     # 校准样本多帧管理器
├── detection/
│   ├── metrics.py               # 单帧遮挡检测指标 (occupancy)
│   ├── engine.py                # 全视频两层检测引擎 (OccupancyEpisode + CountSegment)
│   ├── counter.py               # 小鼠数量估计引擎 (MouseCounter)
│   ├── identity_assist.py       # 耳标颜色辅助检测（HSV 规则 + 可选 ONNX CNN）
│   └── models/                  # AI 推理模型
│       ├── __init__.py
│       ├── classifier.py        # EarTagClassifier: ONNX 包装器 + HSV 回退
│       └── vision_provider.py   # AI 视觉 provider (Kimi/MiniMax/OpenRouter)
├── export/
│   ├── __init__.py
│   └── csv_exporter.py          # CSV/Markdown 导出
└── tests/
    ├── __init__.py
    ├── test_color_classifier.py    # 颜色分类器单元测试
    ├── test_vision_provider.py     # AI 视觉 provider 单元测试
    └── test_thermometer.py         # 温度计检测单元测试
```

---

## 使用流程

### 1. 打开视频

菜单 `文件 → 打开视频` (Ctrl+O) 或点击工具栏 `打开视频`。

### 2. 绘制暖点核心 ROI

在视频上按住鼠标左键拖拽，圈定暖点圆片区域（椭圆）。

视频上会显示三个 ROI：
- **ROI A** (绿色实线): 暖点核心，用于判断是否被遮挡
- **ROI B** (黄色虚线): 缓冲 ROI (=ROI A × 1.8)，用于粗筛
- **ROI C** (蓝色虚线): 计数 ROI (=ROI A × 1.6)，用于小鼠数量估计

### 3. 标记参考帧（任意顺序）

校准按钮在工具栏中：

| 按钮 | 用途 | 统计方式 |
|---|---|---|
| `[0]` 标记0只 | 空场背景帧（必须至少一个） | 取最后一帧 |
| `[1]` 标记1只 | 单鼠面积校准 | 多帧取 P80 |
| `[2]` 标记2只 | 双鼠参考面积 | 多帧取中位数 |
| `[3]` 标记3只 | 三鼠参考面积 | 多帧取中位数 |
| `[4]` 标记4只 | 四鼠参考面积 | 多帧取中位数 |

- **左键点击** = 追加样本（不限次数）
- **右键点击** = 撤回/清空菜单
- 按钮显示当前样本数，如 `[2] 2只 x3/3`
- 不强制顺序：先点 [1] 再点 [0] 也能正常生效（[0] 出现后自动重测之前标记的帧）
- 无背景时按 `R` 使用 dark-only fallback 模式（置信度降低）

### 4. 自动检测全视频

点击 `自动检测全视频` → 后台运行两层检测：

- **Layer 1** — 隔帧粗筛 (ROI B) → 局部精查 (ROI A) → 状态机 → 占据大事件 (OccupancyEpisode)
- **Layer 2** — 在占据事件内逐帧估计小鼠数量 → 按数量变化切分 → 计数子片段 (CountSegment)

检测完成后事件列表自动填充，等待人工审核。

### 5. 人工审核

| 操作 | 快捷键 |
|---|---|
| 选择小鼠身份 (多选 toggle) | `1` `2` `3` `4` |
| 确认当前片段数量 | `Shift+1` `Shift+2` |
| 刷新当前帧计数 | `R` |
| 播放/暂停 | `Space` |
| 单帧前进/后退 | `→` `←` |
| 快进/快退 10 帧 | `J` `L` |
| 上一个/下一个事件 | `↑` `↓` |
| 保存并跳到下一个 | `Enter` |
| 标记误检 | `X` |
| 清空当前选择 | `C` |

标注面板操作：
- `[v] 确认` — 确认当前片段的小鼠标注
- `[x] 误检` — 标记当前片段为误检
- `保存并下一个` — 保存并自动跳到下一个待审核片段

### 6. 导出

菜单 `文件 → 导出Markdown统计表...` 生成报告，包含：
- 事件明细表（每段起止时间、小鼠编号、数量）
- 每只小鼠汇总（总占据时长、事件次数）
- 暖点汇总（总被占据时长、多鼠事件数）

---

## 检测算法概要

### 占据检测 (Layer 1)

1. 每隔 10 帧对 ROI B 采样，计算遮挡比例
2. 遮挡比例 ≥ 20% → 触发精细扫描
3. 对 ROI A 逐帧计算 5 项指标（暖色保留、深色比例、背景差异等）
4. 状态机判断事件边界：连续 4 帧 ≥ 20% 确认进入，连续 10 帧 ≤ 8% 确认离开
5. 过滤 < 0.8 秒的过短事件，合并间隔 < 0.8 秒的相邻事件

### 小鼠计数 (Layer 2)

保守计数策略，核心原则：**单个连通区最多判 2 只，绝不判 3/4**。

```
前景提取 (ROI C 内):
  strong_dark (V < 55) | (dark_candidate (V < 95) & 背景差异 > 25)
  → 形态学去噪 (开3x3 + 闭9x9)
  → 连通区分析
  → 碎片合并 (Union-Find 近邻聚类)
  → 过滤: 面积 < 50px、长宽比 > 5、与 ROI A 接触 < 20px 或 < 2%

计数:
  count_by_blob = 合并后接触 ROI A 的连通区数 (0~4)
  count_by_area = 总面积 / 单鼠参考面积 → 阈值映射 (< 1.7=1, 1.7-2.7=2, ...)

综合判定:
  blob=0               → 0只, 0.9
  blob=1, ratio<1.7    → 1只, 0.85
  blob=1, 1.7≤r<2.7    → 2只, 0.35 (低置信)
  blob=1, r≥2.7        → 2只, 0.25 (极低置信)
  blob=1, 有参考面积匹配 → 采用匹配结果 (偏差<35%)
  blob≥2               → 相信blob数, 结合面积

时间稳定性: 1→2需8帧, 2→3需10帧, 1→3需15帧+额外证据
```

---

## 快捷键总览

| 键 | 功能 |
|---|---|
| `1` `2` `3` `4` | 选择/取消小鼠身份 |
| `Shift+1` `Shift+2` | 确认片段小鼠数量 |
| `R` | 刷新当前帧计数 |
| `Space` | 播放/暂停 |
| `→` `←` | 单帧前进/后退 |
| `J` `L` | 快进/快退 10 帧 |
| `↑` `↓` | 上一个/下一个事件 |
| `Enter` | 保存并跳到下一个 |
| `X` | 标记误检 |
| `C` | 清空当前选择 |
| `S` | 保存当前标注 |

---

## 菜单

| 菜单 | 功能 |
|---|---|
| 文件 → 打开视频 (Ctrl+O) | 打开 mp4/avi/mov 等视频 |
| 文件 → 保存/加载 ROI | 保存/加载暖点 ROI 坐标 (JSON) |
| 文件 → 保存/加载背景帧 | 保存/加载空场背景图像 (PNG) |
| 文件 → 导出Markdown统计表... | 导出标注结果 |
| 文件 → 退出 (Ctrl+Q) | 退出程序 |
| 视图 → 暖点放大窗口 | 显示/隐藏放大窗口 |
| 视图 → 检测指标 | 显示/隐藏指标面板 |

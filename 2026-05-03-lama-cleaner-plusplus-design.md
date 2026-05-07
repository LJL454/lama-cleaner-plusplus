# Lama Cleaner++ 设计文档

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个比 lama-cleaner 自然 10 倍的 AI 局部图像修复工具，基于 SDXL Inpainting + ROI 裁剪 + Mask 羽化技术栈。

**Architecture:** Gradio Web UI → InpaintService（业务层）→ Pipeline（阶段化：preprocess → select\_engine → run\_engine → postprocess）→ EngineManager（资源管理）→ SDXL/LaMa → 输出。核心差异化不在模型选择，而在 mask 质量和 ROI 优化。

**Tech Stack:** Python 3.12 + PyTorch 2.x + Hugging Face diffusers + Gradio 5.x + OpenCV + SAM 2.1（可选）

**Target Hardware:** RTX 3060 6GB（主开发机），兼容 4GB+ GPU

***

## 一、项目最终定义

### 1.1 项目名

**Lama Cleaner++**（暂定，可改为 InpaintX / RestoreAI 等）

### 1.2 核心定位

AI 局部图像修复工具（Object Removal / Inpainting）

- ✅ 用户圈哪里 → 自动补完哪里
- ❌ 不是"去水印工具"（太窄 + 有风险）
- ❌ 不是"全能修复平台"（太重 + 会烂尾）

### 1.3 核心价值

> "比 lama-cleaner 自然 10 倍的局部修复体验"

### 1.4 目标用户

| 用户类型    | 使用场景                 |
| ------- | -------------------- |
| 设计师     | 去 logo / 去文字 / 去多余元素 |
| 普通用户    | 修图 / 去杂物 / 补全背景      |
| AI 绘画用户 | 清理生成瑕疵 / 局部重绘        |
| 开发者     | 作为 API 集成到自己的应用      |

### 1.5 不做什么（Phase 1 明确排除）

- ❌ Florence-2 自动检测水印（Phase 2）
- ❌ 盲水印移除（Phase 2+）
- ❌ 视频处理（Phase 3）
- ❌ FLUX Fill 引擎（Phase 2，作为高级模式）
- ❌ 5+ 引擎切换（Phase 1 只做 SDXL + LaMa）
- ❌ 用户账户系统
- ❌ 云端部署方案

***

## 二、系统整体架构

```
用户操作（上传图片 → 画 mask / SAM 辅助选区）
        ↓
InpaintService（业务层，UI 与逻辑解耦）
        ↓
InpaintPipeline（阶段化，非上帝函数）
  ├── preprocess()    → 原始 mask 几何特征 → Mask expand → 空值检查 → ROI 裁剪 → feather（ROI 尺寸）
  ├── select_engine() → 策略调度（VRAM + elongation 双因子 → 引擎选择 → SAM 互斥检查）
  ├── run_engine()    → 推理执行（极小 ROI fast path → _run_sdxl / _run_lama → OOM fallback）
  └── postprocess()   → 结果回填（调用 roi.paste_back，统一回填逻辑）
        ↓
输出（Gradio 展示 + 日志记录）
```

### 2.1 目录结构

```
lama-cleaner-plusplus/
├── app.py                    # 入口文件，启动 Gradio
├── requirements.txt          # 依赖
├── config.py                 # 全局配置（支持 CLI + env + dataclass）
├── core/
│   ├── __init__.py
│   ├── mask_processor.py     # Mask 处理：膨胀、羽化、格式转换
│   ├── roi.py                # ROI 裁剪 + 回填（自适应 padding）
│   ├── strategy.py           # 策略调度器：选引擎、选参数、自动模式
│   ├── engine_manager.py     # 引擎生命周期管理（lazy load + 缓存 + 并发保护）
│   ├── engines/
│   │   ├── __init__.py
│   │   ├── base.py           # 引擎抽象基类
│   │   ├── sdxl.py           # SDXL Inpainting 引擎
│   │   └── lama.py           # LaMa 引擎（CPU 回退）
│   ├── pipeline.py           # 修复管线：阶段化设计（preprocess→select→run→postprocess）
│   └── service.py            # 业务服务层：UI 与逻辑解耦
├── ui/
│   ├── __init__.py
│   ├── gradio_app.py         # Gradio UI 定义
│   ├── components.py         # 可复用 UI 组件
│   └── state.py              # 会话状态管理（含历史记录）
└── utils/
    ├── __init__.py
    ├── image.py              # 图像工具：加载、保存、格式转换
    ├── gpu.py                # GPU 信息检测
    └── logger.py             # 统一日志系统
```

### 2.2 关键设计决策

| 决策项                      | 选择                                                           | 理由                                                                     |
| ------------------------ | ------------------------------------------------------------ | ---------------------------------------------------------------------- |
| 默认引擎                     | SDXL Inpainting                                              | 效果最好，6GB 可用（需优化）                                                       |
| 回退引擎                     | LaMa                                                         | CPU 可跑，速度快，小区域够用                                                       |
| UI 框架                    | Gradio 5.x                                                   | 快速搭建，内置 ImageEditor，Python 原生                                          |
| Mask 羽化                  | 距离变换 + smoothstep + gamma 钳制                                 | 比高斯模糊更精确，中心实边缘虚，防细 mask 被"吃掉"                                          |
| Feather 时机               | ROI 裁剪之后（ROI 尺寸内）                                            | 避免全图/ROI 尺寸不匹配导致的模糊不一致                                                 |
| SDXL mask 传入             | 传入硬边 mask（0/255），不传 feathered alpha，feather 由 postprocess 控制 | 确保我们完全控制 feather 质量，不依赖模型内部处理                                          |
| ROI 裁剪                   | 必须开启                                                         | 性能核心，可降低 60-90% 计算量                                                    |
| ROI padding              | 自适应（mask\_ratio + density 双因子）                               | 细长结构自动多给上下文                                                            |
| ROI 计算                   | 只在 preprocess 计算一次                                           | 避免重复 crop 导致的不一致和性能浪费                                                  |
| 空 mask 处理                | preprocess 提前 return 原图                                      | 避免 SDXL 生成整张图（灾难）                                                      |
| 极小 ROI                   | fast path → 直接 LaMa（HQ 模式除外）                                 | 秒级响应，HQ 用户始终走 SDXL 保证质量                                                |
| SAM 集成                   | 可选加载，与 SDXL 互斥                                               | 6GB 显存下需动态加载/卸载                                                        |
| 策略调度                     | VRAM + mask elongation 双因子                                   | elongation 区分细线条和块状 mask                                               |
| Elongation 基准            | 原始 mask 空间计算（expand 前）                                       | expand 会破坏细长结构几何特征（1×100 → 31×130），必须用原始 mask 计算才能正确触发 elongation > 40 |
| 引擎管理                     | EngineManager 单例                                             | lazy load + 缓存 + 并发保护 + OOM fallback                                   |
| 引擎状态                     | `is_loaded()` 公开方法                                           | 不访问 `_loaded` 私有变量                                                     |
| Pipeline 设计              | 阶段化 + \_run\_sdxl/\_run\_lama 拆分                             | 避免上帝函数，可维护可测试                                                          |
| UI/逻辑分层                  | InpaintService 中间层                                           | UI 可替换为 API/CLI/插件                                                     |
| Prompt 策略                | 四级（极小空/复杂结构保护/细结构轻量/标准引导）                                    | 避免过度重绘，保护细结构                                                           |
| Seed                     | 默认 42（可复现）                                                   | 调试和效果对比必备                                                              |
| 日志                       | logging 模块                                                   | 记录引擎选择、ROI 尺寸、推理耗时、VRAM                                                |
| 日志 hierarchy             | root logger 统一配置，所有模块日志输出到 stdout                            | 消除命名 logger 无 handler 导致日志丢失的问题                                        |
| Pipeline mode            | PipelineContext.mode 字段                                      | 用户选择不被 select\_engine 硬编码覆盖                                            |
| postprocess              | 调用 roi.paste\_back()                                         | 统一回填逻辑，消除 pipeline 内联重复代码                                              |
| 状态反馈                     | status\_callback 钩子 + gr.Progress()                          | Pipeline 各阶段回调，Gradio Progress 实时推送进度条+描述文本，阻塞期间仍可更新 UI                |
| paste\_back 模式统一         | 入口 `.convert("RGB")`                                         | 防 RGBA vs RGB 通道不一致导致 numpy 广播失败                                       |
| SDXL width/height        | 移除，由 diffusers 自动推断                                          | 避免未来版本 DeprecationWarning 和强制 resize 兼容风险                              |
| SDXL load                | try/except + re-raise                                        | 模型下载失败不崩溃，有明确错误信息                                                      |
| VRAM 阈值                  | SDXLConfig.min\_vram\_gb 可配置                                 | 不硬编码 `vram < 4`，引擎属性驱动                                                 |
| VRAM 估算                  | `min(空闲, 总量*0.7, 总量-1GB)` 保守策略                               | 防碎片化 OOM，比纯 `free` 更稳健                                                 |
| resize\_to\_multiple 填充色 | 按 image.mode 分支                                              | Pillow 10+ 中 L 模式不接受 (0,0,0) 元组填充色，必须用 int 0                           |
| mask\_ratio 阈值           | 统一 > 128                                                     | 与 preprocess 空值检查阈值一致，语义统一                                             |

***

## 三、核心模块设计

### 3.1 Mask 处理管线（`core/mask_processor.py`）

这是项目的核心差异化模块。lama-cleaner 的 mask 处理相对粗糙，我们要做到"mask 精度 > 模型差距"。

#### 功能清单

| 功能           | 说明                           | 优先级 |
| ------------ | ---------------------------- | --- |
| 二值化          | 将用户绘制的 mask 转为 0/255         | P0  |
| 膨胀（Dilation） | 扩展 mask 边缘 5-20px，避免残留       | P0  |
| 羽化（Feather）  | 距离变换 + smoothstep + gamma 钳制 | P0  |
| SAM 辅助       | 点击 → 自动分割 → 生成精确 mask        | P1  |
| Mask 预览      | 叠加显示到原图上                     | P0  |

#### 核心算法

```python
def expand_mask(mask: np.ndarray, pixels: int = 15) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (pixels * 2 + 1, pixels * 2 + 1)
    )
    return cv2.dilate(mask, kernel, iterations=1)

def feather_mask(mask: np.ndarray, radius: int = 20, min_alpha: float = 0.0, gamma: float = 1.0, complexity: float = 0.5) -> np.ndarray:
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    alpha = np.clip(dist / max(radius, 1), 0, 1)
    alpha = alpha * alpha * (3 - 2 * alpha)  # smoothstep
    if gamma != 1.0:
        alpha = np.power(alpha, gamma)  # gamma 控制边缘锐度
    adaptive_gamma = 1.0 + complexity * 0.5  # 细结构更柔和，大块区域更锐利
    alpha = np.power(alpha, adaptive_gamma)
    alpha = np.clip(alpha, min_alpha, 1.0)  # 钳制最小值，防细 mask 被"吃掉"
    return alpha.astype(np.float32)
```

#### mask_ratio 计算

```python
def compute_mask_ratio(expanded: np.ndarray, image_area: int) -> float:
    """计算 mask 占图像面积的比例。阈值统一 > 128，与 preprocess 空值检查一致。"""
    return np.sum(expanded > 128) / image_area
```

⚠️ **阈值必须统一为 `> 128`**：`compute_mask_ratio` 和 `preprocess` 的空值检查都使用 `> 128`，而非 `> 0`。`> 0` 会把抗锯齿边缘像素（1-127）计入 mask 面积，导致 mask_ratio 偏高、padding 策略误判。

#### ⚠️ Feather 执行时机（关键设计约束）

Feather **必须在 ROI 裁剪之后、基于 ROI 尺寸执行**，而不是在全图上做：

```
正确流程：mask → expand → crop ROI → feather(ROI尺寸) → resize → run → paste_back
错误流程：mask → expand → feather(全图) → crop → resize → run → paste_back
```

原因：

- feather 半径是像素值，在全图和 ROI 上的相对比例不同
- 小 ROI 在全图做 feather 会导致过度模糊
- paste\_back 时 alpha resize 会引入插值误差

#### SAM 2.1 集成策略

6GB 显存下，SAM 和 SDXL 不能同时加载。策略：

1. 用户进入"SAM 模式"时，通过 EngineManager 动态加载 SAM2.1 Small（\~46M 参数，\~1GB VRAM）
2. 用户完成选区后，EngineManager 卸载 SAM，释放显存
3. 然后加载 SDXL 进行修复
4. 如果显存不足，SAM 使用 CPU 模式（较慢但可用）

### 3.2 ROI 裁剪系统（`core/roi.py`）

这是性能优化的核心。原理：不处理整张大图，只处理 mask 覆盖的小区域。

#### 设计要点

1. **自动计算 bbox**：找到 mask 的最小外接矩形
2. **自适应 padding**：根据 mask\_ratio 分级，小目标多给上下文，大目标省算力
3. **最小尺寸保证**：裁剪后若 ROI 尺寸 < min\_size，自动扩展 box 到至少 min\_size（防小 ROI 被强制放大到 512 导致模糊）

```python
if roi_w < min_size:
    diff = min_size - roi_w
    extra_left = diff // 2
    extra_right = diff - extra_left  # 精确分配余数，消除 //2 丢 1px
    x1 = max(0, x1 - extra_left)
    x2 = min(image.width, x2 + extra_right)
    if x2 - x1 < min_size:  # 边界受限时单向补偿
        if x1 == 0:
            x2 = min(image.width, x1 + min_size)
        elif x2 == image.width:
            x1 = max(0, x2 - min_size)
```

1. **8 的倍数对齐**：SDXL 要求尺寸是 8 的倍数
2. **保持宽高比**：resize 时不变形

#### 自适应 padding 策略（关键优化）

```python
def compute_padding_ratio(mask_ratio: float) -> float:
    """mask_ratio = mask_area / image_area"""
    if mask_ratio < 0.02:    # 小目标（logo、小文字）
        return 0.5           # 多给上下文，修复更自然
    elif mask_ratio < 0.1:   # 中等目标（杂物、中等水印）
        return 0.3           # 标准上下文
    else:                    # 大目标（大面积修复）
        return 0.15          # 省算力，ROI 已经够大
```

#### mask 复杂度计算（用于策略调度）

⚠️ **单纯** **`mask_area / bbox_area`** **只是密度，不是复杂度。** 需要结合边界复杂度（edge\_ratio）才能区分"大块区域（容易）"和"多碎片（难）"：

```python
def compute_complexity(roi_mask_np: np.ndarray) -> float:
    roi_h, roi_w = roi_mask_np.shape
    bbox_area = roi_w * roi_h
    if bbox_area == 0:
        return 0.0

    density = np.sum(roi_mask_np > 128) / bbox_area
    edges = cv2.Canny(roi_mask_np, 100, 200)
    edge_ratio = np.sum(edges > 0) / bbox_area

    binary = (roi_mask_np > 128).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_c = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(max_c, True)
        area = max(cv2.contourArea(max_c), 1.0)
        elongation = (perimeter ** 2) / area
    else:
        elongation = 0.0

    score = (
        0.35 * density +
        0.35 * edge_ratio +
        0.30 * min(elongation / 50.0, 1.0)
    )
    return min(score, 1.0)
```

#### compute\_elongation（独立函数）

```python
@staticmethod
def compute_elongation(roi_mask_np: np.ndarray) -> float:
    """返回原始几何量（perimeter²/area），严禁归一化。
    调用方自行归一化：auto_mode 用 > 40 判断，complexity 用 /50 归一化。"""
    binary = (roi_mask_np > 128).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_c = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(max_c, True)
        area = max(cv2.contourArea(max_c), 1.0)
        return (perimeter ** 2) / area
    return 0.0
```

| 场景      | density | edge\_ratio | complexity | 推荐模式  |
| ------- | ------- | ----------- | ---------- | ----- |
| 大块区域    | 高       | 低           | 低          | quick |
| 细线条     | 低       | 高           | 高          | hq    |
| 多碎片     | 中       | 高           | 高          | hq    |
| 简单 logo | 高       | 低           | 低          | quick |

⚠️ **必须在 ROI 空间计算**（用 ROI 内的 mask，不是全图 expanded mask）。

#### padding 密度修正（density 加成）

基础 padding 由 `mask_ratio` 决定，再根据 `density = mask_area / bbox_area` 做修正。density 越低 = mask 越细长/不规则 → 需要更多上下文：

```python
def compute_padding_ratio(mask_ratio: float, density: float = 1.0) -> float:
    if mask_ratio < 0.02:
        base = 0.5
    elif mask_ratio < 0.1:
        base = 0.3
    else:
        base = 0.15

    if density < 0.3:
        base += 0.2  # 细长结构（电线、文字、边缘）→ 多给上下文

    return min(base, 0.7)  # 钳制上限
```

#### 性能收益预估

| 原图尺寸      | mask 区域 | 无 ROI 耗时 | 有 ROI 耗时 | 加速比  |
| --------- | ------- | -------- | -------- | ---- |
| 1920x1080 | 200x100 | \~15s    | \~3s     | 5x   |
| 3840x2160 | 300x200 | \~60s    | \~4s     | 15x  |
| 1024x1024 | 512x512 | \~8s     | \~6s     | 1.3x |

### 3.3 策略调度器（`core/strategy.py`）

自动根据硬件和 mask 情况选择最佳引擎和参数。支持"自动模式"推荐，用户无需手动选择 Quick/HQ。

#### 调度逻辑

```python
def select_engine(mask_ratio: float, available_vram_gb: float) -> str:
    if available_vram_gb >= 6:
        return "sdxl"
    elif available_vram_gb >= 4:
        return "sdxl"  # 激进模式，用更少的 steps
    else:
        return "lama"

def select_params(engine: str, mask_ratio: float, mode: str) -> dict:
    if engine == "sdxl":
        if mode == "quick":
            return {"steps": 20, "guidance_scale": 7.0, "strength": 0.8}
        else:  # hq
            return {"steps": 30, "guidance_scale": 7.5, "strength": 0.75}
    else:  # lama
        return {}  # LaMa 无需调参

def auto_mode(mask_ratio: float, elongation: float, complexity: float = 0.5) -> str:
    """自动推荐模式：elongation + complexity + mask_ratio 三因子驱动决策

    elongation 返回原始值（周长²/面积），阈值 40 对应归一化后的 0.8
    """
    if elongation > 40:
        return "hq"   # 细长结构（水印/线条/文字）→ SDXL 高质量修复
    if complexity > 0.5:
        return "hq"   # 多碎片/复杂 mask → SDXL 高质量修复
    if mask_ratio < 0.03:
        return "quick"
    return "hq"

def select_config(mask_ratio: float, elongation: float, complexity: float,
                  available_vram_gb: float, mode: str = "auto") -> dict:
    """统一配置选择：引擎 + 参数 + 模式。seed 不参与引擎选择逻辑，已移除。"""
    if mode == "auto":
        mode = auto_mode(mask_ratio, elongation, complexity)
    engine = select_engine(mask_ratio, available_vram_gb)
    params = select_params(engine, mask_ratio, mode)
    return {"engine": engine, "mode": mode, **params}
```

#### Elongation 对策略的影响（关键优化）

`elongation = 周长² / 面积`（返回原始值，几何量），细长结构 elongation 高（>40），块状结构 elongation 低（<20）。

❗约束：`compute_elongation()` 返回原始值，严禁在函数内部做归一化。

使用方统一规则：

- `auto_mode`: `elongation > 40`
- `compute_complexity`: `min(elongation / 50.0, 1.0)`

| 情况     | mask\_ratio | elongation（原始值） | 实际难度 | 推荐模式  |
| ------ | ----------- | --------------- | ---- | ----- |
| 小 logo | 小           | 低（\~12）         | 简单   | quick |
| 细线条/文字 | 小           | 高（\~60+）        | 很难   | hq    |
| 大块区域   | 大           | 低（\~15）         | 中等   | quick |
| 不规则大面积 | 大           | 中（\~25）         | 中等   | hq    |

核心洞察：**elongation > 40 = 细长线条/不规则形状，必须用 hq 模式**，否则 SDXL 20 steps 不够修复。

#### 模式设计

| 模式    | 引擎   | Steps | 预计耗时   | 适用场景           |
| ----- | ---- | ----- | ------ | -------------- |
| Quick | SDXL | 20    | 3-5s   | 快速预览、简单背景      |
| HQ    | SDXL | 30    | 15-30s | 高质量修复、复杂纹理     |
| Auto  | SDXL | 自动    | 自动     | 根据 mask 面积自动推荐 |
| CPU   | LaMa | N/A   | 1-2s   | 无 GPU / 低显存    |

### 3.4 EngineManager（`core/engine_manager.py`）

统一管理引擎生命周期，防止重复加载，为未来 API/多任务打基础。

#### BaseEngine 公开接口

EngineManager **不得直接访问引擎的私有状态**（如 `_loaded`）。所有状态查询必须通过公开方法：

```python
class BaseEngine(ABC):
    @abstractmethod
    def load(self, force_cpu: bool = False) -> None:
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """公开接口：引擎是否已加载"""
        ...
```

#### EngineManager 实现

```python
class EngineManager:
    def __init__(self):
        self._cache: dict[str, BaseEngine] = {}
        self._lock = threading.Lock()

    def get(self, name: str, force_cpu: bool = False) -> BaseEngine:
        with self._lock:
            if name not in self._cache:
                self._cache[name] = self._create(name)
            engine = self._cache[name]
            if not engine.is_loaded():
                engine.load(force_cpu=force_cpu)
            return engine

    def _create(self, name: str) -> BaseEngine:
        if name == "sdxl":
            return SDXLEngine()
        elif name == "lama":
            return LamaEngine()
        raise ValueError(f"Unknown engine: {name}")

    def unload_all(self):
        with self._lock:
            for engine in self._cache.values():
                engine.unload()
            self._cache.clear()

    def unload(self, name: str):
        with self._lock:
            if name in self._cache:
                self._cache[name].unload()
                del self._cache[name]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

### 3.5 Pipeline 阶段化设计（`core/pipeline.py`）

避免"上帝函数"，拆成 4 个清晰阶段，每个阶段可独立测试和扩展。

**关键约束**：ROI 只在 `preprocess()` 计算一次，存入 `PipelineContext`，后面所有阶段复用 `ctx.roi_image` / `ctx.roi_mask` / `ctx.roi_box`，禁止重复 crop。

```python
class InpaintPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self.engine_manager = EngineManager()
        self.logger = logging.getLogger(__name__)

    def run(self, image, mask, prompt="", mode="auto", seed=42, status_callback=None, **kwargs) -> InpaintResult:
        with self._run_lock:
            if status_callback:
                status_callback("⏳ 分析 mask 中...")
            ctx = self.preprocess(image, mask, **kwargs)
            if ctx.skip:
                return ctx.image  # 空 mask，直接返回原图
            ctx = self.select_engine(ctx, mode)
            if status_callback:
                status_callback(f"⏳ {ctx.inpaint_config.engine_name} 推理中...")
            ctx = self.run_engine(ctx, prompt, seed)
            return self.postprocess(ctx)

    def preprocess(self, image, mask, expand_px=None, feather_radius=None, **kwargs) -> PipelineContext:
        """阶段1：Mask expand → 空值检查 → ROI 裁剪 → feather（在 ROI 内做）"""
        ...

    def select_engine(self, ctx: PipelineContext, mode: str) -> PipelineContext:
        """阶段2：策略调度 + SAM/SDXL 互斥检查"""
        # SAM 和 SDXL 不能同时占显存
        if ctx.engine_name == "sdxl" and self.engine_manager.is_loaded("sam"):
            self.engine_manager.unload("sam")
            torch.cuda.empty_cache()
        # ⚠️ v12 确认：SAM 卸载逻辑已与代码同步，is_loaded("sam") → unload("sam") + empty_cache()
        config = select_config(
            ctx.mask_ratio, ctx.elongation, ctx.mask_complexity,
            available_vram_gb, mode
        )  # v12: 移除 seed 参数，seed 不参与引擎选择逻辑
        ctx.inpaint_config = InpaintConfig(**config)
        ...

    def run_engine(self, ctx: PipelineContext, prompt: str, seed: int) -> PipelineContext:
        """阶段3：推理执行（拆分为 _run_sdxl / _run_lama，含 OOM fallback + 极小 ROI fast path）"""
        ...

    def postprocess(self, ctx: PipelineContext) -> InpaintResult:
        """阶段4：回填 + 日志（使用 ctx.roi_box，不重新计算）"""
        ...
```

#### PipelineContext 数据类

ROI 相关数据只在 `preprocess()` 写入一次，后续只读。

```python
@dataclass
class PipelineContext:
    image: Image.Image
    mask: Image.Image
    expanded_mask: np.ndarray
    roi_image: Image.Image          # preprocess 计算，后续复用
    roi_mask: Image.Image           # preprocess 计算，后续复用
    roi_feathered_alpha: np.ndarray # preprocess 在 ROI 内计算
    roi_box: tuple[int, int, int, int]
    padding_ratio: float
    mask_ratio: float
    mask_complexity: float
    elongation: float                  # 周长²/面积，auto_mode 决策因子
    mode: str = "auto"                 # 用户选择的模式（auto/quick/hq/cpu）
    inpaint_config: InpaintConfig | None = None  # select_engine 写入
    inpainted_result: Image.Image | None = None
    skip: bool = False              # 空 mask 标记，preprocess 写入
```

#### run\_engine 拆分

`run_engine` 内部按引擎类型拆分为独立方法，避免 `if/else` 嵌套：

```python
def run_engine(self, ctx, prompt, seed):
    # 极小 ROI fast path：直接用 LaMa，秒级响应（HQ 模式除外，HQ 始终走 SDXL）
    roi_w, roi_h = ctx.roi_image.size
    if max(roi_w, roi_h) < 128 and ctx.inpaint_config.mode != "hq":
        self.logger.info(f"Small ROI ({roi_w}x{roi_h}), fast path to LaMa")
        engine = self.engine_manager.get("lama")
        ctx.inpainted_result = engine.inpaint(ctx.roi_image, ctx.roi_mask)
        return ctx

    engine = self.engine_manager.get(ctx.engine_name)
    try:
        if ctx.engine_name == "sdxl":
            ctx.inpainted_result = self._run_sdxl(ctx, engine, prompt, seed)
        else:
            ctx.inpainted_result = self._run_lama(ctx, engine)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            ctx = self._fallback_to_lama(ctx, e)
        else:
            raise
    return ctx

def _run_sdxl(self, ctx, engine, prompt, seed):
    target_w, target_h = ctx.roi_image.size
    resized_img, resized_mask, (resized_w, resized_h) = resize_for_sdxl(ctx.roi_image, ctx.roi_mask)
    alpha_resized = False
    if (resized_w, resized_h) != (target_w, target_h):
        alpha_pil = Image.fromarray(
            (ctx.roi_feathered_alpha * 255).astype(np.uint8), mode="L"
        )
        alpha_pil = alpha_pil.resize((resized_w, resized_h), Image.BICUBIC)
        ctx.roi_feathered_alpha = np.array(alpha_pil).astype(np.float32) / 255.0
        alpha_resized = True
    # 四级 prompt 策略
    effective_prompt = self._get_prompt(ctx.mask_ratio, ctx.complexity, prompt)
    result = engine.inpaint(resized_img, resized_mask, prompt=effective_prompt, seed=seed, ...)
    result = result.resize((target_w, target_h), Image.LANCZOS)
    if alpha_resized:
        alpha_pil = Image.fromarray(
            (ctx.roi_feathered_alpha * 255).astype(np.uint8), mode="L"
        )
        alpha_pil = alpha_pil.resize((target_w, target_h), Image.BICUBIC)
        ctx.roi_feathered_alpha = np.array(alpha_pil).astype(np.float32) / 255.0
    return result

def _run_lama(self, ctx, engine):
    return engine.inpaint(ctx.roi_image, ctx.roi_mask)
```

#### Prompt 策略（保守 vs 引导）

```python
def get_negative_prompt(
    mask_ratio: float, default_negative: str, complexity: float = 0.5,
    config: SDXLConfig | None = None,
) -> str:
    if config is None:
        config = SDXLConfig()

    if mask_ratio < 0.02:
        base = config.negative_tiny
        if default_negative:
            return default_negative + ", " + base
        return base
    if mask_ratio < 0.1:
        base = config.negative_standard
        if complexity > 0.5:
            base = base + ", jagged edges, broken structure, inconsistent lighting"
        if default_negative:
            return default_negative + ", " + base
        return base
    if complexity > 0.5:
        base = config.negative_heavy
        if default_negative:
            return default_negative + ", " + base
        return base
    return default_negative

def _get_prompt(self, mask_ratio: float, complexity: float, user_prompt: str) -> str:
    if user_prompt:
        return user_prompt
    if mask_ratio < 0.02:
        return "same texture, seamless blend, preserve details"  # 极小区域保守引导
    if mask_ratio < 0.05 and complexity < 0.3:
        return "clean background, smooth, no artifacts"  # 细结构轻量引导
    return "clean background, seamless, natural texture"  # 标准引导
```

### 3.6 Inpainting 引擎

#### 3.6.1 SDXL Inpainting 引擎（`core/engines/sdxl.py`）

主引擎，效果最好。

**VRAM 优化配置（6GB GPU）：**

```python
try:
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
except (OSError, ValueError):
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
pipe.enable_attention_slicing("auto")
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()
pipe.enable_model_cpu_offload()  # 关键！不手动 .to("cuda")
```

**默认参数：**

| 参数               | 默认值                                                          | 范围       | 说明                  |
| ---------------- | ------------------------------------------------------------ | -------- | ------------------- |
| prompt           | 见下方策略                                                        | 用户可改     | 小 mask 保守，大 mask 引导 |
| negative_prompt  | 按 mask_ratio 三级分级（见下方策略），tiny 级含 `text/watermark/logo/symbol/letters/characters/noise/grain` | 分级 | 负面提示 |
| steps            | 30                                                           | 20-50    | 推理步数                |
| guidance\_scale  | 7.5                                                          | 5.0-12.0 | 引导强度                |
| strength         | 0.75                                                         | 0.3-1.0  | 改变程度                |
| seed             | 42                                                           | 任意 int   | 默认固定，可复现            |

**Prompt 分级策略（避免过度重绘）：**

| 条件                                    | prompt                                             | 理由                          |
| ------------------------------------- | -------------------------------------------------- | --------------------------- |
| mask\_ratio < 0.02                    | `"same texture, seamless blend, preserve details"` | 极小区域保守引导，防 SDXL 空 prompt 崩坏 |
| complexity > 0.5                      | `"highly detailed, preserve structure, seamless"`  | 复杂结构保护性生成，防崩坏               |
| mask\_ratio < 0.05 且 complexity < 0.3 | `"clean background, smooth, no artifacts"`         | 细结构（线条/文字）用轻量引导，防破坏         |
| 其他                                    | `"clean background, seamless, natural texture"`    | 标准引导生成                      |

**调度器选择：** EulerDiscreteScheduler（trailing timestep\_spacing）

> ⚠️ **加载顺序关键**：scheduler 替换和 VRAM 优化（attention\_slicing、vae\_slicing、vae\_tiling、model\_cpu\_offload）必须在 `from_pretrained` 之后**无条件执行**，不能放在 `except` 块内。否则 fp16 variant 正常加载时这些优化不会生效，导致推理不稳定和 OOM。

**⚠️ SDXL resize 后 feather 空间对齐（v9 修正→v11 简化→v12 dtype 修正）：**

`resize_for_sdxl` 将 ROI 放大到 max\_size 以内，SDXL 推理后 `_run_sdxl` 将结果 resize 回原始 ROI 尺寸。当 `resize_for_sdxl` 改变尺寸时，`_run_sdxl` 用 BICUBIC 同步缩放 `roi_feathered_alpha`（前向+后向）。注意 alpha 是 float32 \[0,1]，传入 PIL 前需 ×255 转 uint8，resize 后需 ÷255 转回 float32。`paste_back` 内部仍保留防御性尺寸检查作为兜底。

**⚠️ SDXL mask 传入策略（关键细节）：**

传入 SDXL 的 mask 必须是**硬边 mask**（只有 0/255，非 feather 后的 alpha），feather 由我们自己在 postprocess 阶段通过 `roi_feathered_alpha` 控制：

```python
# 传入 SDXL 的 mask：硬边（只有 0/255），不传 feathered alpha
pipe(image=resized_img, mask_image=resized_mask_hard, ...)
# feather blending 在 postprocess 用 roi_feathered_alpha 做
```

这是比 lama-cleaner 自然的关键点之一：**mask 质量 > 模型质量**。我们控制 feather，而不是依赖模型内部处理。

**⚠️ SDXL 推理前重置峰值内存统计（v12 新增）：**

每次 SDXL 推理前必须调用 `torch.cuda.reset_peak_memory_stats()`，否则日志中 VRAM 峰值为累计值而非当次推理峰值：

```python
class SDXLEngine(BaseEngine):
    def inpaint(self, image, mask, prompt="", negative_prompt="", seed=42, **kwargs):
        # ...
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        result = self._pipe(
            image=image, mask_image=mask,
            prompt=prompt, negative_prompt=negative_prompt,
            generator=generator, **kwargs
        ).images[0]
        peak_vram = torch.cuda.max_memory_allocated() / 1024**3
        self.logger.info(f"SDXL inference peak VRAM: {peak_vram:.2f} GB")
        return result
```

#### 3.6.2 LaMa 引擎（`core/engines/lama.py`）

回退引擎，CPU 可运行。

**实现方式：** 通过 `torch.hub` 加载（不依赖 `simple-lama-inpainting`，解除 Pillow 版本限制）。

```python
import torch
try:
    model = torch.hub.load("advimman/lama", "big_lama")
except Exception as e:
    raise RuntimeError(f"Failed to load LaMa model (check network or local cache): {e}") from e
model.eval()
raw_result = model({"image": img_tensor, "mask": mask_tensor})
# 兼容 dict 和 tensor 两种返回类型（该 repo 无稳定 API 契约）
result_tensor = raw_result if isinstance(raw_result, torch.Tensor) else raw_result["inpainted"]
```

**优势：** 无需 GPU，速度快（<1s），适合简单背景。
**劣势：** 复杂纹理效果一般，无法用 prompt 引导。
**加载方式：** torch.hub 自动缓存到 `~/.cache/torch/hub/`，首次需联网下载（\~200MB），后续加载走本地缓存。下载失败时抛出明确 RuntimeError 提示检查网络或本地缓存。

> ⚠️ **LaMa 尺寸对齐裁剪（v9 修正）**：LaMa 推理前会将图像 padding 到 8 的倍数，推理结果必须裁剪回原始 `(w, h)` 再返回，否则 `paste_back` 时尺寸不匹配导致边缘错位。

### 3.7 异常处理与 OOM Fallback

Pipeline 的 `run_engine` 阶段必须处理以下异常：

```python
def _fallback_to_lama(self, ctx: PipelineContext, original_error) -> PipelineContext:
    import gc
    
    self.logger.warning(f"OOM on {ctx.engine_name}, falling back to LaMa: {original_error}")
    
    # 1. 卸载 SDXL
    self.engine_manager.unload(ctx.engine_name)
    
    # 2. 彻底清理 CUDA（仅 empty_cache 不够，OOM 可能来自 fragmentation/graph cache/cudnn workspace）
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
    
    # 3. 卸载可能缓存的 GPU LaMa，强制 CPU 重建（防止缓存命中导致 OOM 复发）
    self.engine_manager.unload("lama")
    gc.collect()
    torch.cuda.empty_cache()
    lama = self.engine_manager.get("lama", force_cpu=True)
    ctx.inpainted_result = lama.inpaint(ctx.roi_image, ctx.roi_mask)
    ctx.engine_name = "lama"
    
    self.logger.info("Fallback to LaMa successful")
    return ctx
```

#### 异常处理清单

| 异常             | 处理方案                          |
| -------------- | ----------------------------- |
| CUDA OOM       | 清空显存 → fallback 到 LaMa → 记录日志 |
| CUDA error     | 卸载引擎 → 重试一次 → 失败则报错           |
| 模型加载失败         | 记录错误 → 尝试 fallback 引擎         |
| 无效 mask（全黑/全白） | 跳过推理 → 直接返回原图 + 警告            |
| 图像尺寸异常         | 自动 resize → 记录日志              |

### 3.8 回填系统

将修复结果无缝贴回原图。**使用** **`ctx.roi_box`，不重新计算 ROI。**

#### 步骤

1. 将 inpainting 输出 resize 回 ROI 原始尺寸
2. 用 `ctx.roi_feathered_alpha`（ROI 尺寸）做 alpha blending
3. 贴回原图对应位置（`ctx.roi_box`）

```python
def paste_back(ctx: PipelineContext) -> Image.Image:
    ctx.image = ctx.image.convert("RGB")
    ctx.inpainted_result = ctx.inpainted_result.convert("RGB")
    result = ctx.image.copy()
    x1, y1, x2, y2 = ctx.roi_box
    target_w, target_h = x2 - x1, y2 - y1
    inpainted_resized = ctx.inpainted_result.resize((target_w, target_h), Image.LANCZOS)
    
    alpha = ctx.roi_feathered_alpha  # 已经是 ROI 尺寸，无需 resize
    alpha_3ch = np.stack([alpha] * 3, axis=-1)
    orig_region = np.array(result.crop(ctx.roi_box)).astype(np.float32)
    result_np = np.array(inpainted_resized).astype(np.float32)
    blended = (result_np * alpha_3ch + orig_region * (1 - alpha_3ch)).astype(np.uint8)
    result.paste(Image.fromarray(blended), (x1, y1))
    return result
```

### 3.9 Service 层（`core/service.py`）

UI 与逻辑解耦，为未来 API/CLI/插件做准备。

```python
class InpaintService:
    def __init__(self, config: AppConfig):
        self.pipeline = InpaintPipeline(config)
        self.logger = logging.getLogger(__name__)

    def process(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str = "",
        mode: str = "auto",
        seed: int = 42,
        expand_px: int | None = None,
        feather_radius: int | None = None,
    ) -> InpaintResult:
        """统一入口，UI/API/CLI 都调这个"""
        return self.pipeline.run(
            image=image, mask=mask, prompt=prompt,
            mode=mode, seed=seed,
            expand_px=expand_px, feather_radius=feather_radius,
        )
```

### 3.10 日志系统（`utils/logger.py`）

```python
import logging

def get_logger(name: str) -> logging.Logger:
    """获取模块 logger，名字形如 'core.mask_processor'"""
    return logging.getLogger(name)

def setup_logger(name: str = "lama-cleaner-plusplus", level: str = "INFO") -> None:
    """配置 root logger + core/ui 父 logger 级别"""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s] %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # 模块 logger 使用 get_logger(__name__)，名字形如 "core.xxx" / "ui.xxx"
    # 必须显式设置父 logger 级别，否则模块 logger 继承 root 的 WARNING
    logging.getLogger("core").setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.getLogger("ui").setLevel(getattr(logging, level.upper(), logging.INFO))
```

**日志记录点：**

| 位置             | 记录内容                                               |
| -------------- | -------------------------------------------------- |
| preprocess     | mask\_ratio, mask\_complexity, roi\_box, roi\_size |
| select\_engine | 选择的引擎、模式、参数                                        |
| run\_engine    | 推理耗时、是否 fallback、VRAM 峰值                           |
| postprocess    | 总耗时、输出尺寸                                           |
| 异常             | OOM 信息、fallback 路径                                 |

***

## 四、UI 设计（Gradio 5.x）

### 4.1 页面布局

```
┌─────────────────────────────────────────────────────────┐
│  Lama Cleaner++                    [Auto ▼] [Run] [↩]   │
├──────────────────────────┬──────────────────────────────┤
│                          │                              │
│     原图 + Mask 编辑区     │       修复结果预览区           │
│     (gr.ImageEditor)     │       (gr.Image)             │
│                          │                              │
├──────────────────────────┴──────────────────────────────┤
│  [画笔大小: ──●──] [膨胀: 15px] [羽化: 20px]            │
│  Prompt: [clean background, seamless...        ]        │
│  Seed: [42]  [SAM 模式] [重置 Mask] [对比模式]           │
└─────────────────────────────────────────────────────────┘
```

### 4.2 核心交互流程

1. **上传图片**：拖拽或点击上传
2. **画 Mask**：用画笔涂抹要修复的区域
   - 可调画笔大小
   - 可用橡皮擦修正
   - 可选 SAM 点击模式辅助
3. **调整参数**（可选）：
   - Prompt（有默认值，不写也能用）
   - Seed（默认 42，可复现）
   - 膨胀/羽化参数
   - 模式切换（默认 Auto）
4. **点击 Run**：执行修复
5. **查看结果**：
   - 前后对比（滑块模式）
   - 不满意可撤销重来（历史记录）

### 4.3 Gradio 组件选择

| 功能             | 组件                        | 说明                        |
| -------------- | ------------------------- | ------------------------- |
| 图片上传 + Mask 绘制 | `gr.ImageEditor`          | Gradio 5.x 内置，支持画笔/橡皮擦/图层 |
| 结果展示           | `gr.Image`                | 简单展示                      |
| 前后对比           | `gr.Image` + 自定义 JS       | 滑块对比模式                    |
| 画笔大小           | `gr.Slider`               | 1-100px                   |
| 膨胀/羽化          | `gr.Slider`               | 0-50px                    |
| Prompt         | `gr.Textbox`              | 有默认值                      |
| Seed           | `gr.Number`               | 默认 42                     |
| 模式切换           | `gr.Radio`                | Auto / Quick / HQ / CPU   |
| SAM 点击         | `gr.Image` + `.select` 事件 | 捕获点击坐标                    |
| 历史记录           | 状态栈                       | 最近 10 次结果，支持撤销            |

**⚠️ on_run 异常处理（v12 新增）：**

Gradio UI 的 `on_run` 函数中 `service.process()` 外层必须包裹 try/except，避免异常时用户只看到 Gradio 通用报错：

```python
def on_run(...):
    try:
        result = service.process(image=image, mask=mask, ...)
        return result, "✅ 修复完成"
    except RuntimeError as e:
        return image, f"❌ 推理错误：{e}"
    except Exception as e:
        return image, f"❌ 未知错误：{e}"
```

### 4.4 Prompt 策略

默认 prompt **按 mask 面积 + 复杂度分级**，避免过度重绘：

| 条件                                    | prompt                                             | 说明                          |
| ------------------------------------- | -------------------------------------------------- | --------------------------- |
| mask\_ratio < 0.02                    | `"same texture, seamless blend, preserve details"` | 极小区域保守引导，防 SDXL 空 prompt 崩坏 |
| complexity > 0.5                      | `"highly detailed, preserve structure, seamless"`  | 复杂结构保护性生成，防崩坏               |
| mask\_ratio < 0.05 且 complexity < 0.3 | `"clean background, smooth, no artifacts"`         | 细结构用轻量引导，防破坏                |
| 其他                                    | `"clean background, seamless, natural texture"`    | 标准引导生成                      |

用户显式输入的 prompt 始终优先于默认策略。

***

## 五、技术规格

### 5.1 环境要求

| 项目       | 最低要求                    | 推荐配置          |
| -------- | ----------------------- | ------------- |
| Python   | 3.10+                   | 3.12          |
| GPU VRAM | 4GB（SDXL 激进模式）          | 6GB+          |
| GPU      | NVIDIA（CUDA 11.8+）      | RTX 3060+     |
| RAM      | 8GB                     | 16GB+         |
| 磁盘       | 10GB（模型下载）              | 20GB          |
| OS       | Windows / Linux / macOS | Windows 10/11 |

### 5.2 依赖清单

```
torch>=2.1.0
diffusers>=0.30.0
transformers>=4.40.0
accelerate>=0.30.0
gradio>=5.0.0
opencv-python>=4.8.0
numpy>=1.24.0,<2.0.0
Pillow>=10.0.0
pyyaml>=6.0
# LaMa via torch.hub 运行时依赖
kornia>=0.6.0
omegaconf>=2.1.0
albumentations>=1.3.0
pytorch-lightning>=1.5.0,<2.0.0
```

> **注意**：LaMa 通过 `torch.hub.load("advimman/lama", "big_lama")` 加载，不依赖 `simple-lama-inpainting`（该包锁死 `pillow<10.0.0`，与 Gradio 5.x 冲突）。

**SDXLConfig 默认参数（`config.py`）：**

```python
@dataclass
class SDXLConfig:
    min_vram_gb: float = 4.0
    negative_tiny: str = "blurry, low quality, artifacts, distorted, text, watermark, logo, symbol, letters, characters, noise, grain"
    negative_standard: str = "blurry, low quality, artifacts, distorted"
    negative_heavy: str = "blurry, low quality, artifacts, distorted, jagged edges, broken structure, inconsistent lighting"
```

可选依赖（SAM 支持，从 GitHub 安装）：

```
pip install git+https://github.com/facebookresearch/sam2.git
```

> ⚠️ **HuggingFace 模型下载认证**：SDXL 模型需要 HuggingFace 登录才能下载。首次使用前请执行：
>
> ```bash
> pip install huggingface_hub
> huggingface-cli login
> # 粘贴你的 HuggingFace Access Token（在 https://huggingface.co/settings/tokens 创建）
> ```
>
> 未登录时 `EngineManager` 加载 SDXL 会报 `401 Unauthorized`。LaMa（torch.hub）无需登录。

### 5.3 模型清单

| 模型                                               | 大小      | 用途                  | 下载方式                                 |
| ------------------------------------------------ | ------- | ------------------- | ------------------------------------ |
| diffusers/stable-diffusion-xl-1.0-inpainting-0.1 | \~6.5GB | SDXL Inpainting 主模型 | Hugging Face 自动下载                    |
| LaMa（via torch.hub）                              | \~200MB | CPU 回退引擎            | torch.hub 自动下载到 \~/.cache/torch/hub/ |
| SAM 2.1 Small（可选）                                | \~150MB | 点击分割辅助              | Hugging Face 自动下载                    |

### 5.4 性能预期（RTX 3060 6GB）

| 场景     | 模式    | 处理区域      | 预计耗时   | VRAM    |
| ------ | ----- | --------- | ------ | ------- |
| 小区域修复  | Quick | 256x256   | 2-3s   | \~4GB   |
| 中等区域   | Quick | 512x512   | 4-5s   | \~5GB   |
| 高质量修复  | HQ    | 512x512   | 15-30s | \~5.5GB |
| 大区域修复  | HQ    | 1024x1024 | 30-60s | \~5.5GB |
| CPU 回退 | LaMa  | 任意        | 1-2s   | 0（CPU）  |

> **实际体验建议**：SDXL 处理大区域（>50% 图片面积）时耗时较长（1-3 分钟），建议对大面积水印优先使用 quick 模式（LaMa），仅对复杂纹理或小面积精细区域使用 hq 模式。首次加载含模型下载（\~6.5GB），后续加载已缓存模型约 5-15 秒。

***

## 六、与 lama-cleaner 的对比

| 维度           | lama-cleaner (IOPaint) | Lama Cleaner++                               |
| ------------ | ---------------------- | -------------------------------------------- |
| 默认模型         | LaMa                   | SDXL Inpainting                              |
| 效果（简单背景）     | ⭐⭐⭐                    | ⭐⭐⭐⭐                                         |
| 效果（复杂纹理）     | ⭐⭐                     | ⭐⭐⭐⭐                                         |
| 效果（大面积修复）    | ⭐                      | ⭐⭐⭐                                          |
| Mask 处理      | 基础膨胀 + 高斯模糊            | 距离变换 + smoothstep + gamma 钳制（ROI 尺寸内）        |
| SDXL mask 控制 | diffusers 默认 blur      | 硬边 mask 传入，feather 完全自控                      |
| ROI 优化       | 无                      | 自动裁剪 + 自适应 padding（density 加成）+ 只算一次         |
| 极小区域         | 统一流程                   | fast path → 直接 LaMa（HQ 除外），秒级响应              |
| 空 mask 处理    | 未处理                    | 提前 return 原图                                 |
| 策略调度         | 手动选模型                  | 自动根据 VRAM + mask 复杂度双因子选择，支持 cpu 强制模式        |
| Prompt 支持    | 部分模型支持                 | 四级策略（极小保守引导/复杂结构保护/细结构轻量/标准引导）               |
| SAM 集成       | 有（作为插件）                | 深度集成（EngineManager 统一管理 + SDXL 互斥）           |
| OOM 处理       | 崩溃                     | 自动 fallback 到 LaMa + gc + CUDA 缓存彻底清理        |
| 引擎抽象         | 无统一接口                  | BaseEngine.is\_loaded() 公开方法                 |
| UI 交互        | React 前端               | Gradio ImageEditor 内置画笔                      |
| 日志           | 无                      | 结构化日志（引擎/ROI/耗时/VRAM）                        |
| 代码复杂度        | 高（10+ 模型，React 前端）     | 低（2 模型，Gradio 前端）                            |
| 用户模式选择       | 无                      | auto/quick/hq/cpu 四模式，PipelineContext 传递不被覆盖 |
| 回填逻辑         | 内联                     | roi.paste\_back() 统一，pipeline 复用             |
| 模型加载         | 无错误处理                  | try/except + 明确错误信息 + re-raise               |
| 日志 hierarchy | 无                      | core/ui 父 logger 显式设置级别                      |
| 上手难度         | 中                      | 低                                            |
| 可扩展性         | 一般                     | Service 层解耦，可做 API/插件                        |

***

## 七、Phase 1 功能清单

### ✅ 必须实现（P0）

| #  | 功能                       | 模块             |
| -- | ------------------------ | -------------- |
| 1  | 图片上传                     | UI             |
| 2  | 手动画 Mask（画笔 + 橡皮擦）       | UI + Mask      |
| 3  | Mask 膨胀                  | Mask           |
| 4  | Mask 羽化（距离变换 + gamma 钳制） | Mask           |
| 5  | ROI 自动裁剪（自适应 padding）    | ROI            |
| 6  | ROI 回填 + 羽化混合            | Pipeline       |
| 7  | SDXL Inpainting 推理       | Engine         |
| 8  | 默认 Prompt                | UI             |
| 9  | 自定义 Prompt               | UI             |
| 10 | Auto / Quick / HQ 模式     | Strategy       |
| 11 | Gradio UI 完整布局           | UI             |
| 12 | 前后对比展示                   | UI             |
| 13 | EngineManager 资源管理       | Engine Manager |
| 14 | OOM 自动 fallback 到 LaMa   | Pipeline       |
| 15 | 日志系统                     | Utils          |
| 16 | Seed 默认 42（可复现）          | Engine         |
| 17 | InpaintService 业务层       | Service        |

### ⭐ 加分项（P1，强烈建议）

| #  | 功能                | 模块     |
| -- | ----------------- | ------ |
| 18 | SAM 2.1 点击扩选      | Mask   |
| 19 | LaMa CPU 回退（手动选择） | Engine |
| 20 | 撤销/重做（历史栈）        | UI     |
| 21 | Mask 预览叠加         | UI     |
| 22 | CLI 参数支持          | Config |

### 📋 Phase 2 路线（未来扩展）

| #  | 功能                         |
| -- | -------------------------- |
| 23 | FLUX Fill 高级引擎             |
| 24 | 自动检测（Grounding DINO + SAM） |
| 25 | 批量处理                       |
| 26 | API 服务模式（利用 Service 层）     |
| 27 | 视频处理                       |

***

## 八、风险评估与应对

| 风险                         | 概率 | 影响 | 应对方案                                                                           |
| -------------------------- | -- | -- | ------------------------------------------------------------------------------ |
| SDXL 在 6GB 上 OOM           | 中  | 高  | ROI 裁剪 + attention slicing + VAE tiling + CPU offload + **自动 fallback 到 LaMa** |
| Mask 不准导致效果差               | 中  | 高  | 膨胀 + 羽化（gamma 钳制） + SAM 辅助 + 用户可手动修正                                           |
| ROI 太小导致不自然                | 低  | 中  | 最小 512x512 保证 + **自适应 padding（小目标多给上下文）**                                      |
| SAM 加载/卸载导致延迟              | 中  | 低  | **EngineManager 统一管理** + 进度提示                                                  |
| 首次下载模型耗时长                  | 高  | 低  | 启动时显示下载进度 + 支持本地模型路径                                                           |
| Gradio ImageEditor 行为不符合预期 | 低  | 中  | 备选方案：gr.Image + Canvas JS 自定义                                                  |
| Pipeline 维护困难              | 低  | 中  | **阶段化设计**，每阶段可独立测试                                                             |

***

## 九、成功标准

Phase 1 完成的验收标准：

1. **可用性**：用户能在 3 步内完成一次修复（上传 → 画 mask → 点 run）
2. **效果**：在简单背景（天空、草地、墙面）场景下，修复结果肉眼无瑕疵
3. **性能**：RTX 3060 上 Quick 模式 < 5 秒，HQ 模式 < 30 秒
4. **稳定性**：连续修复 20 次不 OOM、不崩溃（OOM 时自动 fallback）
5. **显存**：峰值 VRAM < 5.5GB
6. **易用性**：默认 prompt + 默认 seed + Auto 模式下，用户无需任何输入即可获得可用结果
7. **可维护性**：Pipeline 每个阶段可独立测试，Service 层可独立调用

***

## 十、开发排期（Phase 1）

| 阶段        | 内容                               | 天数       |
| --------- | -------------------------------- | -------- |
| Day 1     | 项目骨架 + 配置（支持 CLI/env）+ 日志系统      | 1        |
| Day 2-3   | Mask 处理管线（膨胀 + 羽化 + gamma 钳制）    | 2        |
| Day 3-4   | ROI 裁剪 + 自适应回填                   | 1        |
| Day 5-6   | Engine 抽象基类 + EngineManager      | 2        |
| Day 7-8   | SDXL 引擎 + LaMa 引擎 + OOM fallback | 2        |
| Day 9     | 策略调度器 + 自动模式                     | 1        |
| Day 10-11 | Pipeline 阶段化 + InpaintService    | 2        |
| Day 12-13 | Gradio UI + 前后对比 + 历史记录          | 2        |
| Day 14    | 集成测试 + 性能调优 + 日志审查               | 1        |
| **总计**    | <br />                           | **14 天** |

***

## 十一、核心技术优势总结

这个项目的杀手锏不是模型选择，而是四个工程优化：

1. **ROI 裁剪**：让 SDXL 变"轻量"，大图处理不爆显存
2. **Mask 质量**：距离变换羽化 + smoothstep + gamma 钳制 > 高斯模糊，直接决定效果自然度
3. **策略调度**：自动适配用户设备，低端机器也能用
4. **架构分层**：EngineManager + Pipeline 阶段化 + Service 解耦 → 可维护可扩展

一句话总结：

> "用 ROI + SDXL + 极致 mask 处理 + 健壮架构，把一个重模型变成一个看起来很轻、但效果极强的修复工具"

***

## 十二、设计文档变更记录

### v1-v6 → v7（2026-05-03，全面复查）

基于 LaMa-Adapter 2、SDXL Inpainting、Context-aware 三篇论文与 SDXL 实际推理行为，对 7 处设计进行修正。详见实施计划 v6→v7 变更对照表。

### v7 → v8（2026-05-04，深度复查 + 多用户隔离 + 细线条识别）

| # | 严重度  | 问题                        | v7（旧）                                                                     | v8（当前）                                                      |
| - | ---- | ------------------------- | ------------------------------------------------------------------------- | ----------------------------------------------------------- |
| 1 | 🔴P0 | mask\_blur 参数不存在          | 设计文档反复强调传入 `mask_blur=0` 禁用内部模糊，但 `StableDiffusionXLInpaintPipeline` 无此参数 | 移除所有 mask\_blur 描述，改为"传入硬边 mask，feather 由外部 postprocess 控制" |
| 2 | 🟡P1 | compute\_complexity 误判细线条 | 仅 density + edge\_ratio，1px 线条两项都极低被误判为"简单"                               | 增加 elongation 因子（轮廓周长²/面积比），权重 0.35:0.35:0.3                |
| 3 | 🟡P1 | EngineManager 锁内加载        | `engine.load()` 在锁内，加载几十秒阻塞所有并发                                           | 双检锁：快速路径无锁读，创建实例加锁，`load()` 放锁外                             |
| 4 | 🟡P1 | 缺 HF 登录指引                 | 无说明，用户首次运行报 401                                                           | 依赖清单后增加 `huggingface-cli login` 步骤和模型清单标注                   |
| 5 | 🟢P2 | 性能预估偏低                    | HQ 512x512 标注 6-8s，实际 SDXL 推理 15-30s                                      | 调整为实际测量值，增加体验建议                                             |
| 6 | 🟢P2 | Undo 限制未说明                | 无文档说明 Undo 无法恢复 ImageEditor 状态                                            | 核心交互流程增加 Undo 限制说明和历史隔离说明                                   |

**v8 汇总**：1 处 P0 参数错误修正（mask\_blur 不存在）；3 处 P1 改进（细线条识别、双检锁、HF 登录）；2 处 P2 文档准确性提升。

### v8 → v9（2026-05-04，架构级复查 + 致命隐藏漏洞修复）

| # | 严重度  | 问题                                | v8（旧）              | v9（当前）                                                         |
| - | ---- | --------------------------------- | ------------------ | -------------------------------------------------------------- |
| 1 | 🔴P0 | SDXL scheduler+VRAM 优化在 except 块内 | 设计文档未明确加载顺序        | §3.6.1 增加⚠️：scheduler 替换和 VRAM 优化必须在 `from_pretrained` 之后无条件执行 |
| 2 | 🔴P0 | LaMa 推理结果未裁剪回原始尺寸                 | 无说明                | §3.6.2 增加⚠️：padding 后结果必须裁剪回原始 `(w, h)`                        |
| 3 | 🔴P0 | resize\_for\_sdxl 后 feather 空间错位  | 无说明                | §3.6.1 增加⚠️：resize 后需同步缩放 `roi_feathered_alpha`（BICUBIC）       |
| 4 | 🟡P1 | EngineManager 双检锁并发加载风险           | v8 改为双检锁（load 在锁外） | 还原为全锁：创建+加载均在锁内，阻塞可接受                                          |
| 5 | 🟡P1 | VRAM 估算过于乐观                       | 无保守策略              | 决策表增加 VRAM 估算策略：`min(空闲, 总量*0.7, 总量-1GB)`                      |

**v9 汇总**：3 处 P0 隐藏漏洞补充说明（scheduler 加载顺序、LaMa 裁剪、feather 空间对齐）；2 处 P1 架构修正（EngineManager 回退全锁、VRAM 保守估算）。基于完整错误修复清单的架构级复查。

### v9 → v10（2026-05-04，P0 漏网修复）

| # | 严重度  | 问题                               | v9（旧）               | v10（当前）                                               |
| - | ---- | -------------------------------- | ------------------- | ----------------------------------------------------- |
| 1 | 🔴P0 | SDXL 内层 except 捕获过宽              | `except Exception:` | 改为 `except (OSError, ValueError):`，OOM 不触发 fallback   |
| 2 | 🔴P0 | compute\_complexity 缺 elongation | v8 声称已加但代码未改        | 代码实际补全：elongation = perimeter²/area，权重 0.35:0.35:0.30 |

**v10 汇总**：2 处 P0 代码级修复同步到设计文档。

### v10 → v11（2026-05-05，工程级总结修复 — 数值统一 + 回退稳定性 + 输入可靠性）

基于工程级总总结，将所有问题按"严重度 + 本质原因 + 修复方案"统一落地到代码和设计文档。

| #  | 严重度  | 问题                           | v10（旧）                                                        | v11（当前）                                                                                                                                                                                                                                           | 影响模块                                |
| -- | ---- | ---------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| 1  | 🔴P0 | `auto_mode` elongation 阈值恒成立 | `if elongation > 0.8:` 但 `elongation = 周长²/面积 ≥ 12.56`，条件永远为真 | `compute_elongation()` 返回原始值（几何量）；`auto_mode` 阈值改为 `> 40`；`compute_complexity()` 内 `min(elongation / 50.0, 1.0)` 归一化                                                                                                                              | §3.2 mask\_processor, §3.3 strategy |
| 2  | 🔴P0 | OOM fallback 后 SDXL 未卸载      | `except RuntimeError: fallback_to_lama()` 不卸载 SDXL → 连环 OOM   | `_fallback_to_lama()` 第一步 `unload(sdxl)` + `gc.collect()` + `empty_cache()` + `ipc_collect()` + `reset_peak_memory_stats()`                                                                                                                       | §3.7 异常处理                           |
| 3  | 🔴P0 | LaMa fallback 仍用 GPU         | `self._device = "cuda"`，OOM 场景再次申请显存                          | `LamaEngine(force_cpu=False)` + `load(force_cpu=True)`；`EngineManager.get(name, force_cpu=True)` 统一调用（删除 `if name == "lama"` 特殊分支）；`BaseEngine.load(force_cpu=False)` 统一签名；OOM fallback 先 `unload("lama")` 再 `get("lama", force_cpu=True)` 防止缓存命中 | §3.6.2 LaMa 引擎, §3.4 EngineManager  |
| 4  | 🟡P1 | `auto_mode` 忽略 complexity    | 只看 elongation + mask\_ratio → 多碎片/噪声 mask 误判 quick            | `auto_mode(mask_ratio, elongation, complexity)` 新增 `if complexity > 0.5: return "hq"`；`select_config()` 签名新增 `complexity` 参数并透传                                                                                                                   | §3.3 策略调度器                          |
| 5  | 🟡P1 | Gradio mask 提取不完整            | 只读 `layers`，无 composite fallback，无多层 merge                    | `_extract_mask_from_editor()`：逐层 RGBA/L 提取 → `np.maximum` 合并 → composite RGBA/RGB 差分回退（修复缩进+变量作用域 bug）→ 空 mask fallback                                                                                                                           | §4.1 页面布局, §4.3 组件选择                |
| 6  | 🟡P1 | bbox area off-by-one         | `(rmax - rmin) * (cmax - cmin)` 少算边界像素                        | `mask_h = rmax - rmin + 1`；min\_size 扩展整数舍入同步修复                                                                                                                                                                                                   | §3.2 ROI 裁剪系统                       |
| 7  | 🟡P1 | feather resize 对齐不透明         | 仅在 `paste_back` 兜底，文档与代码不一致                                   | `_run_sdxl` 显式检测 resize 前后尺寸变化，BICUBIC 同步缩放 `roi_feathered_alpha`                                                                                                                                                                                 | §3.6.1 SDXL 引擎                      |
| 8  | 🟡P1 | negative\_prompt 未分级         | 统一强 negative，小 mask 过度约束                                      | `get_negative_prompt(mask_ratio, default_negative, complexity)` 三级分级 + complexity 增强；小 mask 加 `text/watermark/logo/symbol/letters/characters/noise/grain`；高 complexity 加 `jagged edges/broken structure/inconsistent lighting`                    | §3.6.1 SDXL 引擎, §3.3 策略调度           |
| 9  | 🟡P1 | resize\_for\_sdxl 下限 512 过高  | 小 ROI 被强制放大 → 模糊                                              | `ROIConfig.min_size` 改为 `256`                                                                                                                                                                                                                     | §3.2 ROI 裁剪系统                       |
| 10 | 🟡P1 | Pipeline 无并发保护               | EngineManager 有锁但 Pipeline 层没有 → 多请求竞争                        | `threading.Lock()` 包裹 `run()` 核心逻辑                                                                                                                                                                                                                | §3.5 Pipeline 阶段化设计                 |
| 11 | 🟢P2 | 缺少 fast path 测试              | 无测试覆盖 `ROI < 128 → LaMa` 路径                                   | 新增 `test_pipeline_fast_path()`                                                                                                                                                                                                                    | §十二 测试                              |
| 12 | 🟢P2 | utils/image.py 未实现           | 设计存在，代码缺失                                                     | 实现 `ensure_rgb()` / `to_pil()` / `resize_to_multiple()`                                                                                                                                                                                           | §2.1 目录结构                           |
| 13 | 🟢P2 | SKIP\_MODEL\_TESTS 保护不全      | 无 diffusers 环境部分测试崩溃                                          | 5 个模型依赖测试均加 `SKIP_MODEL_TESTS` 跳过                                                                                                                                                                                                                 | §十二 测试                              |

#### Gradio composite mask 提取（v11 新增）

```python
def _extract_mask_from_editor(image: Image.Image, composite: Image.Image | None,
                               layers: list[Image.Image] | None, color: str) -> tuple:
    mask_acc = None
    # 1. 逐层提取
    if layers:
        for layer in layers:
            if layer.mode == "RGBA":
                alpha = np.array(layer)[:, :, 3].astype(np.float32) / 255.0
                mask_acc = np.maximum(mask_acc, alpha) if mask_acc is not None else alpha
            elif layer.mode in ("L", "LA"):
                gray = np.array(layer.convert("L")).astype(np.float32) / 255.0
                mask_acc = np.maximum(mask_acc, gray) if mask_acc is not None else gray
    # 2. composite 差分回退（RGBA/RGB 双分支）
    if mask_acc is None and composite is not None:
        if composite.mode == "RGBA":
            alpha = np.array(composite)[:, :, 3].astype(np.float32) / 255.0
            if alpha.max() > 0:
                comp_rgb = np.array(composite)[:, :, :3].astype(np.float32)
                bg_rgb = np.array(image).astype(np.float32)
                diff = np.abs(comp_rgb - bg_rgb).max(axis=2)
                threshold = max(8, diff.mean() * 0.5)  # 自适应阈值
                mask_acc = ((diff > threshold) & (alpha > 0)).astype(np.float32)
            else:
                mask_acc = alpha
        elif composite.mode == "RGB":
            comp_rgb = np.array(composite).astype(np.float32)
            bg_rgb = np.array(image).astype(np.float32)
            diff = np.abs(comp_rgb - bg_rgb).max(axis=2)
            threshold = max(8, diff.mean() * 0.5)  # 自适应阈值
            mask_acc = (diff > threshold).astype(np.float32)
    # 3. 空 mask fallback
    if mask_acc is None or mask_acc.max() == 0:
        return Image.new("L", image.size, 0), None
    # 4. 阈值化 + 平滑
    binary = (mask_acc > 0.3).astype(np.float32)
    binary = cv2.GaussianBlur(binary, (3, 3), 0)
    return Image.fromarray((binary * 255).astype(np.uint8), "L"), None
```

**v11 设计决策更新（本次修复引入的新约束）：**

| 决策项                          | 选择                                          | 理由                                                                                                 |
| ---------------------------- | ------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Elongation 返回值               | 原始值（几何量）                                    | `compute_elongation()` 返回 `perimeter²/area`，严禁在函数内部归一化；使用方自行归一化（auto\_mode 阈值 40，complexity 除以 50） |
| LaMa OOM 回退                  | 强制 CPU（`force_cpu=True`）                    | OOM 场景下 GPU 已无可用显存，必须走 CPU 避免二次 OOM                                                                |
| auto\_mode 三因子               | elongation + complexity + mask\_ratio       | complexity 填补 elongation 无法覆盖的多碎片/噪声 mask 场景                                                       |
| Gradio mask 提取               | layers 逐层 merge + composite 差分回退            | 覆盖所有 ImageEditor 可能的数据格式：RGBA 图层、灰度图层、composite 合成图                                                |
| bbox 计算                      | `rmax - rmin + 1` 包含两端像素                    | 消除 1px 系统性偏差对 density/padding 的累积影响                                                                |
| Negative prompt              | 按 mask\_ratio 三级 + complexity 增强            | 小 mask 加 `text/watermark/logo/symbol` 防过度脑补；高 complexity 加 `jagged edges/broken structure` 防结构崩坏   |
| select\_config 透传 complexity | `select_config(..., complexity=complexity)` | `auto_mode` 三因子决策链路闭合，complexity 不再丢失                                                              |
| Composite fallback           | RGBA/RGB 双分支 + 变量作用域修正                      | 修复 `comp_rgb` 未定义的 NameError；增加 RGB composite 差分支持                                                 |
| ROI min\_size                | 256（原 512）                                  | 4K 大图 512 下限合理，但常见 1024 图 256 已足够避免过度放大导致的模糊                                                       |
| Pipeline 并发                  | `threading.Lock()` 整体锁                      | 引擎加载/卸载是重量级操作，整体锁比细粒度锁更安全且性能影响可忽略                                                                  |
| Resize 整数舍入                  | `extra_left/right` 精确分配余数                   | 消除 `// 2` 整除丢失 1px 导致 min\_size 约束不满足的边界 case                                                      |

**v11 汇总**：3 处 P0 致命逻辑/稳定性修复（elongation 归一化、OOM 后 SDXL 卸载、LaMa 强制 CPU）；7 处 P1 决策/可靠性/输入链路修复（auto\_mode 加 complexity + select\_config 透传、Gradio mask 完整提取 + composite 作用域修复、bbox+1、feather 对齐、negative\_prompt 三级+complexity 增强、resize 下限 256、Pipeline 并发保护）；3 处 P2 工程质量补全。本轮修复的核心主题是**数值体系统一 + 三因子决策链路闭合 + 回退路径稳固 + 输入链路可靠**，覆盖了工程级总结中全部 13 项问题。

### v11 → v12（2026-05-06，代码审查修复 — Pillow 兼容性 + VRAM 统计 + 异常处理 + 死代码清理）

| # | 严重度 | 问题 | v11（旧） | v12（当前） |
|---|--------|------|---------|-----------|
| 1 | 🔴P0 | resize_to_multiple 填充色不兼容 L 模式 | `Image.new(image.mode, size, (0,0,0))`，L 模式在 Pillow 10+ 抛 TypeError | `fill = 0 if image.mode in ("L", "1") else (0, 0, 0)`，按模式分支 |
| 2 | 🟡P1 | select_engine 缺少 SAM 卸载检查 | 设计文档有但代码遗漏 | 代码补全：`is_loaded("sam")` → `unload("sam")` + `empty_cache()` |
| 3 | 🟡P1 | negative_tiny 缺少设计文档推荐的关键词 | `negative_tiny = "blurry, low quality, artifacts, distorted"` | 补全为 `"blurry, low quality, artifacts, distorted, text, watermark, logo, symbol, letters, characters, noise, grain"` |
| 4 | 🟡P1 | on_run 缺少异常处理 | `service.process()` 无 try/except，异常时 Gradio 通用报错 | 包裹 try/except，RuntimeError 和 Exception 分别返回友好提示 |
| 5 | 🟡P1 | SDXLEngine 不重置峰值内存统计 | 推理前未 reset，日志 VRAM 峰值为累计值 | 推理前 `torch.cuda.reset_peak_memory_stats()`，日志反映当次峰值 |
| 6 | 🟢P2 | select_config 有未使用的 seed 参数 | `seed: int = 42` 从未参与引擎选择逻辑 | 移除 seed 参数，Pipeline 调用处同步移除 |
| 7 | 🟢P2 | compute_mask_ratio 阈值不一致 | `expanded > 0`，与 preprocess `> 128` 不一致 | 统一为 `expanded > 128`，语义一致 |
| 8 | 🟢P2 | test_mask_complexity 断言错误 | 碎片在 1000×1000 全图密度极低，complexity 低于大色块 | 改为 200×200 紧凑布局，密度提高后断言正确 |

**v12 设计决策更新（本次修复引入的新约束）：**

| 决策项 | 选择 | 理由 |
|-------|------|------|
| resize_to_multiple 填充色 | 按 image.mode 分支 | Pillow 10+ 中 L 模式不接受 (0,0,0) 元组填充色，必须用 int 0 |
| mask_ratio 阈值 | 统一 > 128 | 与 preprocess 空值检查阈值一致，语义统一 |
| on_run 异常处理 | try/except 包裹 service.process() | RuntimeError 和通用 Exception 分别返回友好错误提示到状态栏 |
| SDXL VRAM 峰值统计 | 推理前 reset_peak_memory_stats() | 日志反映当次推理峰值而非累计值 |
| select_config seed | 移除 | seed 从未参与引擎选择逻辑，是死代码 |

**v12 汇总**：1 处 P0 Pillow 兼容性修复（L 模式填充色）；4 处 P1 正确性/稳定性修复（SAM 卸载、negative_tiny 关键词、on_run 异常处理、VRAM 峰值统计）；3 处 P2 代码质量修复（死代码清理、阈值统一、测试断言修正）。本轮核心主题是**Pillow 10+ 兼容性 + 文档与代码一致性 + 异常处理健壮性**。

# Lama Cleaner++ 实施计划（v2 架构对齐版）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按照设计文档 `2026-05-03-lama-cleaner-plusplus-design.md` 逐步构建 Lama Cleaner++ 的 Phase 1，完整对齐 v2 架构（EngineManager + Pipeline 四阶段 + InpaintService + 日志系统）。

**Architecture:** `InpaintService → Pipeline(preprocess→select_engine→run_engine→postprocess) → EngineManager → Engines`

**Tech Stack:** Python 3.12 + PyTorch 2.x + diffusers + Gradio 5.x + OpenCV + torch.hub（LaMa）

---

## Task 1: 项目骨架 + 依赖配置

**Files:**
- Create: `lama-cleaner-plusplus/requirements.txt`
- Create: `lama-cleaner-plusplus/.gitignore`
- Create: `lama-cleaner-plusplus/config.py`
- Create: `lama-cleaner-plusplus/core/__init__.py`
- Create: `lama-cleaner-plusplus/ui/__init__.py`
- Create: `lama-cleaner-plusplus/utils/__init__.py`
- Create: `lama-cleaner-plusplus/utils/gpu.py`
- Create: `lama-cleaner-plusplus/utils/logger.py`

- [ ] **Step 1.1: 创建项目目录结构**

```powershell
mkdir -Force lama-cleaner-plusplus\core\engines
mkdir -Force lama-cleaner-plusplus\ui
mkdir -Force lama-cleaner-plusplus\utils
mkdir -Force lama-cleaner-plusplus\tests
```

- [ ] **Step 1.2: 创建 requirements.txt**

```txt
torch>=2.1.0
diffusers>=0.30.0
transformers>=4.40.0
accelerate>=0.30.0
safetensors>=0.4.0
gradio>=5.0.0
opencv-python>=4.8.0
numpy>=1.24.0,<2.0.0
Pillow>=10.0.0
huggingface_hub>=0.20.0
pyyaml>=6.0
# LaMa via torch.hub 运行时依赖
kornia>=0.6.0
omegaconf>=2.1.0
albumentations>=1.3.0
pytorch-lightning>=1.5.0,<2.0.0
```

> **注意**：LaMa 通过 `torch.hub.load("advimman/lama", "big_lama")` 加载，不依赖 `simple-lama-inpainting`（该包锁死 `pillow<10.0.0`，与 Gradio 5.x 冲突）。首次运行会自动下载模型到 `~/.cache/torch/hub/`。

> ⚠️ **HuggingFace 登录**：SDXL 模型需要认证才能下载。安装后首次使用前执行：
> ```bash
> huggingface-cli login
> ```
> 输入在 https://huggingface.co/settings/tokens 创建的 Access Token。未登录时加载 SDXL 会报 `401 Unauthorized`。

- [ ] **Step 1.2b: 创建 .gitignore**

```gitignore
__pycache__/
*.pyc
*.pyo
.venv/
venv/
models/
*.ckpt
*.safetensors
outputs/
.gradio/
.env
*.log
.DS_Store
```

- [ ] **Step 1.3: 创建 config.py（支持 CLI/env）**

```python
import os
import argparse
from dataclasses import dataclass, field


@dataclass
class MaskConfig:
    expand_pixels: int = 15
    feather_radius: int = 20
    gamma: float = 1.0
    min_alpha: float = 0.0


@dataclass
class ROIConfig:
    padding_ratio: float = 0.3
    min_context_px: int = 64
    min_size: int = 256
    max_size: int = 1024


@dataclass
class SDXLConfig:
    model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    steps_quick: int = 20
    steps_hq: int = 30
    guidance_scale: float = 7.5
    strength: float = 0.75
    seed: int = 42
    min_vram_gb: float = 4.0
    default_prompt: str = "clean background, seamless, natural texture, no text, no watermark"
    default_negative: str = "blurry, low quality, artifacts, distorted, text, watermark"
    negative_tiny: str = "blurry, low quality, artifacts, distorted, text, watermark, logo, symbol, letters, characters, noise, grain"
    negative_standard: str = "blurry, low quality, artifacts, distorted, text, watermark, oversaturated"
    negative_heavy: str = "blurry, distorted, artifacts, jpeg artifacts, text, watermark, noise, oversharpened"
    lama_local_path: str = ""
    lama_hub_timeout: int = 60
    lama_hub_retries: int = 3


@dataclass
class AppConfig:
    mask: MaskConfig = field(default_factory=MaskConfig)
    roi: ROIConfig = field(default_factory=ROIConfig)
    sdxl: SDXLConfig = field(default_factory=SDXLConfig)
    server_port: int = 7860
    share: bool = False
    log_level: str = "INFO"


def load_config() -> AppConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--log-level", default=None)
    args, _ = parser.parse_known_args()

    port = args.port if args.port is not None else int(os.getenv("LAMA_PORT", "7860"))
    log_level = args.log_level if args.log_level else os.getenv("LAMA_LOG_LEVEL", "INFO")

    config = AppConfig(
        server_port=port,
        share=args.share or os.getenv("LAMA_SHARE", "").lower() == "true",
        log_level=log_level,
    )
    val = os.getenv("SDXL_STEPS_QUICK")
    if val:
        config.sdxl.steps_quick = int(val)
    val = os.getenv("SDXL_STEPS_HQ")
    if val:
        config.sdxl.steps_hq = int(val)
    val = os.getenv("SDXL_SEED")
    if val:
        config.sdxl.seed = int(val)
    return config


if __name__ == "__main__":
    config = load_config()
```

- [ ] **Step 1.4: 创建 utils/gpu.py**

```python
import torch


def get_available_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info()
    return free / (1024 ** 3)


def get_safe_available_vram_gb(reserve_gb: float = 1.0, fraction: float = 0.7) -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info()
    free_gb = free / (1024 ** 3)
    total_gb = total / (1024 ** 3)
    usable = min(free_gb, total_gb * fraction, total_gb - reserve_gb)
    return max(0.0, usable)


def get_gpu_name() -> str:
    if not torch.cuda.is_available():
        return "CPU"
    return torch.cuda.get_device_name(0)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
```

- [ ] **Step 1.5: 创建 utils/logger.py**

```python
import logging
import sys


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """配置 root logger。所有模块日志通过 root handler 统一输出。"""
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.getLogger("core").setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.getLogger("ui").setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.getLogger("utils").setLevel(getattr(logging, level.upper(), logging.INFO))
    return logging.getLogger(name)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
```

- [ ] **Step 1.6: 创建 __init__.py 文件**

```python
# core/__init__.py, ui/__init__.py, utils/__init__.py 均为空文件
```

- [ ] **Step 1.7: 验证**

Run: `python -c "from config import AppConfig, load_config; from utils.logger import setup_logger; print(AppConfig()); setup_logger('test', 'INFO').info('logger OK')"`
Expected: AppConfig 实例 + logger OK

---

## Task 2: Mask 处理管线（含 gamma clamp）

**Files:**
- Create: `lama-cleaner-plusplus/core/mask_processor.py`

- [ ] **Step 2.1: 实现 mask_processor.py**

```python
import cv2
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)


class MaskProcessor:
    @staticmethod
    def to_binary(mask: np.ndarray, threshold: int = 128) -> np.ndarray:
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        return (mask > threshold).astype(np.uint8) * 255

    @staticmethod
    def expand(mask: np.ndarray, pixels: int = 15) -> np.ndarray:
        binary = MaskProcessor.to_binary(mask)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (pixels * 2 + 1, pixels * 2 + 1)
        )
        return cv2.dilate(binary, kernel, iterations=1)

    @staticmethod
    def feather(
        mask: np.ndarray, radius: int = 20,
        gamma: float = 1.0, min_alpha: float = 0.0,
        complexity: float = 0.5,
    ) -> np.ndarray:
        binary = MaskProcessor.to_binary(mask)
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        alpha = np.clip(dist / max(radius, 1), 0, 1)
        alpha = alpha * alpha * (3 - 2 * alpha)
        if gamma != 1.0:
            alpha = alpha ** gamma
        adaptive_gamma = 1.0 + complexity * 0.5
        alpha = alpha ** adaptive_gamma
        alpha = np.clip(alpha, min_alpha, 1.0)
        return alpha.astype(np.float32)

    @staticmethod
    def expand_and_feather(
        mask: np.ndarray, expand_px: int = 15, feather_radius: int = 20,
        gamma: float = 1.0, min_alpha: float = 0.0,
        complexity: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        expanded = MaskProcessor.expand(mask, expand_px)
        feathered_alpha = MaskProcessor.feather(
            expanded, feather_radius, gamma, min_alpha, complexity=complexity,
        )
        logger.debug(
            f"mask expand: {expand_px}px, feather: {feather_radius}px, "
            f"gamma: {gamma}, min_alpha: {min_alpha}, complexity: {complexity}"
        )
        return expanded, feathered_alpha

    @staticmethod
    def compute_mask_ratio(expanded: np.ndarray, image_area: int | None = None) -> float:
        mask_area = np.sum(expanded > 128)
        area = image_area if image_area else expanded.shape[0] * expanded.shape[1]
        return mask_area / area if area > 0 else 0.0

    @staticmethod
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

    @staticmethod
    def compute_elongation(roi_mask_np: np.ndarray) -> float:
        """计算 mask 细长度：周长²/面积（原始值，几何量）。

        ❗约束：返回原始值，严禁在函数内部做归一化。
        使用方统一规则：
        - auto_mode: elongation > 40
        - complexity: min(elongation / 50.0, 1.0)
        """
        binary = (roi_mask_np > 128).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_c = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(max_c, True)
            area = max(cv2.contourArea(max_c), 1.0)
            return (perimeter ** 2) / area
        return 0.0

    @staticmethod
    def overlay_on_image(
        image: np.ndarray, mask: np.ndarray,
        color: tuple = (255, 0, 0), alpha: float = 0.4,
    ) -> np.ndarray:
        if isinstance(image, Image.Image):
            image = np.array(image)
        image = ensure_rgb(image)
        overlay = image.copy()
        if mask.ndim == 2:
            mask_area = mask > 128
        else:
            mask_area = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) > 128
        overlay[mask_area] = (
            overlay[mask_area] * (1 - alpha) + np.array(color) * alpha
        ).astype(np.uint8)
        return overlay
```

- [ ] **Step 2.2: 编写测试验证**

Run: `python -c "import numpy as np; from core.mask_processor import MaskProcessor; m = np.zeros((100,100), dtype=np.uint8); m[30:70, 30:70] = 255; e, f = MaskProcessor.expand_and_feather(m, gamma=1.2, min_alpha=0.05); print(f'expanded unique: {np.unique(e)}, feather range: {f.min():.2f}-{f.max():.2f}'); print(f'mask_ratio: {MaskProcessor.compute_mask_ratio(e):.4f}')"`
Expected: feather range: 0.05-1.00（gamma/min_alpha 生效）

- [ ] **Step 2.3: Commit**

---

## Task 3: ROI 裁剪 + 自适应 padding + 回填

**Files:**
- Create: `lama-cleaner-plusplus/core/roi.py`

- [ ] **Step 3.1: 实现 roi.py**

```python
import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ROICrop:
    image: Image.Image
    mask: Image.Image
    box: tuple[int, int, int, int]
    original_size: tuple[int, int]


def round_to_8(value: int) -> int:
    return max(64, (value // 8) * 8)


def compute_padding_ratio(mask_ratio: float, density: float = 1.0) -> float:
    if mask_ratio < 0.02:
        base = 0.5
    elif mask_ratio < 0.1:
        base = 0.3
    else:
        base = 0.15

    # density 低 → 细长/不规则结构 → 需要更多上下文
    if density < 0.3:
        base += 0.2

    return min(base, 0.7)


def crop_to_roi(
    image: Image.Image, mask: Image.Image,
    padding_ratio: float = 0.3, min_size: int = 256, max_size: int = 1024,
    min_context_px: int = 64,
) -> ROICrop:
    mask_np = np.array(mask.convert("L"))
    rows = np.any(mask_np > 128, axis=1)
    cols = np.any(mask_np > 128, axis=0)

    if not rows.any() or not cols.any():
        return ROICrop(image, mask, (0, 0, image.width, image.height), image.size)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    mask_h = rmax - rmin + 1
    mask_w = cmax - cmin + 1

    pad_h = max(int(mask_h * padding_ratio), min_context_px)
    pad_w = max(int(mask_w * padding_ratio), min_context_px)

    y1 = max(0, rmin - pad_h)
    y2 = min(image.height, rmax + pad_h)
    x1 = max(0, cmin - pad_w)
    x2 = min(image.width, cmax + pad_w)

    roi_w = x2 - x1
    roi_h = y2 - y1
    if roi_w < min_size:
        diff = min_size - roi_w
        extra_left = diff // 2
        extra_right = diff - extra_left
        x1 = max(0, x1 - extra_left)
        x2 = min(image.width, x2 + extra_right)
        if x2 - x1 < min_size:
            if x1 == 0:
                x2 = min(image.width, x1 + min_size)
            elif x2 == image.width:
                x1 = max(0, x2 - min_size)
    if roi_h < min_size:
        diff = min_size - roi_h
        extra_top = diff // 2
        extra_bottom = diff - extra_top
        y1 = max(0, y1 - extra_top)
        y2 = min(image.height, y2 + extra_bottom)
        if y2 - y1 < min_size:
            if y1 == 0:
                y2 = min(image.height, y1 + min_size)
            elif y2 == image.height:
                y1 = max(0, y2 - min_size)


    box = (x1, y1, x2, y2)
    cropped_img = image.crop(box)
    cropped_mask = mask.crop(box)

    logger.info(
        f"ROI crop: padding_ratio={padding_ratio:.2f}, box={box}, "
        f"crop_size={cropped_img.size}"
    )

    return ROICrop(cropped_img, cropped_mask, box, image.size)


def resize_for_sdxl(
    image: Image.Image, mask: Image.Image, max_size: int = 1024
) -> tuple[Image.Image, Image.Image, tuple[int, int]]:
    """等比缩放图像和 mask 到 SDXL 可接受尺寸（保持 8 的倍数）。

    注意：mask 必须是硬边二值 mask（0 或 255），不要传入 feathered alpha。
    二值化使用 BILINEAR + threshold（而非 NEAREST），防止细结构断裂。
    """
    w, h = image.size
    aspect = w / h

    if aspect >= 1:
        tw = round_to_8(min(max(w, 256), max_size))
        th = round_to_8(int(tw / aspect))
    else:
        th = round_to_8(min(max(h, 256), max_size))
        tw = round_to_8(int(th * aspect))

    # Pillow 10+ 推荐 Image.Resampling.LANCZOS，但 Image.LANCZOS 仍可用
    resized_img = image.resize((tw, th), Image.LANCZOS)
    resized_mask = mask.resize((tw, th), Image.BILINEAR)
    resized_mask = resized_mask.point(lambda x: 255 if x > 128 else 0)

    logger.debug(f"resize_for_sdxl: {image.size} → ({tw}, {th})")
    return resized_img, resized_mask, (tw, th)


def paste_back(
    original: Image.Image, result: Image.Image,
    roi_box: tuple[int, int, int, int],
    feathered_alpha: np.ndarray | None = None,
) -> Image.Image:
    original = original.convert("RGB")
    result = result.convert("RGB")
    output = original.copy()
    x1, y1, x2, y2 = roi_box
    target_w = x2 - x1
    target_h = y2 - y1
    result_resized = result
    if result.size != (target_w, target_h):
        result_resized = result.resize((target_w, target_h), Image.LANCZOS)

    if feathered_alpha is not None:
        alpha = feathered_alpha
        if alpha.shape != (target_h, target_w):
            alpha = np.array(
                Image.fromarray((alpha * 255).astype(np.uint8)).resize(
                    (target_w, target_h), Image.BICUBIC
                )
            ).astype(np.float32) / 255.0
        alpha_3ch = np.stack([alpha] * 3, axis=-1)
        orig_region = np.array(original.crop(roi_box)).astype(np.float32)
        result_np = np.array(result_resized).astype(np.float32)
        blended = np.clip(
            result_np * alpha_3ch + orig_region * (1 - alpha_3ch),
            0, 255,
        ).astype(np.uint8)
        output.paste(Image.fromarray(blended), (x1, y1))
    else:
        output.paste(result_resized, (x1, y1))

    return output
```

- [ ] **Step 3.2: 验证**

Run: `python -c "from core.roi import compute_padding_ratio; print(f'small: {compute_padding_ratio(0.01)}, mid: {compute_padding_ratio(0.05)}, large: {compute_padding_ratio(0.15)}')"`
Expected: small: 0.5, mid: 0.3, large: 0.15

- [ ] **Step 3.3: Commit**

---

## Task 4: 引擎抽象基类

**Files:**
- Create: `lama-cleaner-plusplus/core/engines/__init__.py`
- Create: `lama-cleaner-plusplus/core/engines/base.py`

- [ ] **Step 4.1: 实现 base.py**

```python
from abc import ABC, abstractmethod
from PIL import Image


class BaseEngine(ABC):
    @abstractmethod
    def inpaint(
        self, image: Image.Image, mask: Image.Image,
        prompt: str = "", negative_prompt: str = "", **kwargs
    ) -> Image.Image:
        ...

    @abstractmethod
    def load(self, force_cpu: bool = False) -> None:
        ...

    @abstractmethod
    def unload(self) -> None:
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """公开接口：引擎是否已加载（EngineManager 通过此方法查询，不碰 _loaded）"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def min_vram_gb(self) -> float:
        ...
```

- [ ] **Step 4.2: 创建 engines/__init__.py**

```python
from .base import BaseEngine

__all__ = ["BaseEngine"]
```

- [ ] **Step 4.3: Commit**

---

## Task 5: EngineManager（统一资源管理）

**Files:**
- Create: `lama-cleaner-plusplus/core/engine_manager.py`

- [ ] **Step 5.1: 实现 engine_manager.py**

```python
import torch
import threading
from core.engines.base import BaseEngine
from core.engines.lama import LamaEngine
from core.engines.sdxl import SDXLEngine
from utils.logger import get_logger

logger = get_logger(__name__)


class EngineManager:
    def __init__(
        self,
        model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        lama_local_path: str = "",
        lama_hub_timeout: int = 60,
        lama_hub_retries: int = 3,
    ):
        self._cache: dict[str, BaseEngine] = {}
        self._lock = threading.Lock()
        self._model_id = model_id
        self._lama_local_path = lama_local_path
        self._lama_hub_timeout = lama_hub_timeout
        self._lama_hub_retries = lama_hub_retries

    def _create(self, name: str) -> BaseEngine:
        if name == "lama":
            engine = LamaEngine(
                local_model_path=self._lama_local_path or None,
                hub_timeout=self._lama_hub_timeout,
                hub_retries=self._lama_hub_retries,
            )
        elif name == "sdxl":
            engine = SDXLEngine(self._model_id)
        else:
            raise ValueError(f"Unknown engine: {name}")
        logger.info(f"Creating engine: {name}")
        return engine

    def get(self, name: str, force_cpu: bool = False) -> BaseEngine:
        with self._lock:
            if name not in self._cache:
                self._cache[name] = self._create(name)
            engine = self._cache[name]
            if not engine.is_loaded():
                engine.load(force_cpu=force_cpu)
                logger.info(f"Engine loaded: {name}")
            return engine

    def is_loaded(self, name: str) -> bool:
        """检查指定引擎是否已加载（不触发加载）"""
        with self._lock:
            return name in self._cache and self._cache[name].is_loaded()

    def unload(self, name: str) -> None:
        with self._lock:
            if name in self._cache:
                self._cache[name].unload()
                del self._cache[name]
                logger.info(f"Engine unloaded: {name}")

    def unload_all(self) -> None:
        with self._lock:
            for name, engine in self._cache.items():
                engine.unload()
                logger.info(f"Engine unloaded: {name}")
            self._cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")
```

- [ ] **Step 5.2: Commit**

---

## Task 6: LaMa 引擎（CPU 回退）

**Files:**
- Create: `lama-cleaner-plusplus/core/engines/lama.py`

- [ ] **Step 6.1: 实现 lama.py**

```python
from PIL import Image
from .base import BaseEngine
from utils.logger import get_logger

logger = get_logger(__name__)


class LamaEngine(BaseEngine):
    """LaMa 引擎 — 通过 torch.hub 加载，不依赖 simple-lama-inpainting（解除 Pillow 限制）"""

    def __init__(
        self,
        force_cpu: bool = False,
        local_model_path: str | None = None,
        hub_timeout: int = 60,
        hub_retries: int = 3,
    ):
        self._model = None
        self._loaded = False
        self._force_cpu = force_cpu
        self._local_model_path = local_model_path
        self._hub_timeout = hub_timeout
        self._hub_retries = hub_retries

    def load(self, force_cpu: bool = False) -> None:
        if self._loaded:
            return
        import torch
        import os

        if self._local_model_path and os.path.exists(self._local_model_path):
            logger.info(f"Loading LaMa model from local path: {self._local_model_path}")
            try:
                self._model = torch.jit.load(self._local_model_path, map_location="cpu")
            except Exception as e:
                raise RuntimeError(f"Failed to load local LaMa model from {self._local_model_path}: {e}") from e
        else:
            timeout = self._hub_timeout
            max_retries = self._hub_retries
            logger.info(f"Loading LaMa model via torch.hub (advimman/lama) [timeout={timeout}s, retries={max_retries}]...")
            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(timeout)
            last_error = None
            for attempt in range(max_retries):
                try:
                    self._model = torch.hub.load("advimman/lama", "big_lama")
                    break
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        wait = 2 ** attempt
                        logger.warning(
                            f"Failed to load LaMa (attempt {attempt + 1}/{max_retries}), "
                            f"retrying in {wait}s: {e}"
                        )
                        time.sleep(wait)
                    else:
                        socket.setdefaulttimeout(old_timeout)
                        raise RuntimeError(f"Failed to load LaMa model after {max_retries} attempts: {e}") from e
            socket.setdefaulttimeout(old_timeout)

        self._model.eval()
        use_cpu = self._force_cpu or force_cpu or not torch.cuda.is_available()
        self._device = "cpu" if use_cpu else "cuda"
        self._model.to(self._device)
        self._loaded = True
        logger.info(f"LaMa model loaded on {self._device}")

    def unload(self) -> None:
        if self._model is not None:
            del self._model
        self._model = None
        self._loaded = False
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        return self._loaded

    def inpaint(
        self, image: Image.Image, mask: Image.Image,
        prompt: str = "", negative_prompt: str = "", **kwargs
    ) -> Image.Image:
        import torch
        import numpy as np
        if not self._loaded:
            self.load()

        w, h = image.size
        pw = (8 - w % 8) % 8
        ph = (8 - h % 8) % 8
        if pw or ph:
            new_w, new_h = w + pw, h + ph
            new_img = Image.new(image.mode, (new_w, new_h), (0, 0, 0))
            new_img.paste(image, (0, 0))
            new_mask = Image.new(mask.mode, (new_w, new_h), 0)
            new_mask.paste(mask, (0, 0))
            image, mask = new_img, new_mask

        img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        mask_tensor = torch.from_numpy(np.array(mask.convert("L"))).unsqueeze(0).unsqueeze(0).float() / 255.0
        mask_tensor = (mask_tensor > 0.5).float()

        with torch.no_grad():
            img_dev = img_tensor.to(self._device)
            mask_dev = mask_tensor.to(self._device)
            raw_result = self._model({"image": img_dev, "mask": mask_dev})

        if isinstance(raw_result, dict):
            result_tensor = raw_result.get("inpainted") or raw_result.get("image")
        else:
            result_tensor = raw_result

        result_np = (result_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        out = Image.fromarray(result_np)
        if pw or ph:
            out = out.crop((0, 0, w, h))
        logger.info(f"LaMa inpaint done: {out.size}")
        return out

    @property
    def name(self) -> str:
        return "LaMa"

    @property
    def min_vram_gb(self) -> float:
        return 0.0
```

- [ ] **Step 6.2: Commit**

---

## Task 7: SDXL Inpainting 引擎（seed=42 默认）

**Files:**
- Create: `lama-cleaner-plusplus/core/engines/sdxl.py`

- [ ] **Step 7.1: 实现 sdxl.py**

```python
import torch
from PIL import Image
from diffusers import StableDiffusionXLInpaintPipeline, EulerDiscreteScheduler
from .base import BaseEngine
from utils.logger import get_logger

logger = get_logger(__name__)


class SDXLEngine(BaseEngine):
    def __init__(
        self,
        model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    ):
        self.model_id = model_id
        self._pipe = None
        self._loaded = False

    def load(self, force_cpu: bool = False) -> None:
        if self._loaded:
            return
        if force_cpu:
            logger.warning("SDXL engine does not support CPU-only mode, using GPU with CPU offload")
        logger.info(f"Loading SDXL model: {self.model_id}")
        try:
            try:
                self._pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    use_safetensors=True,
                )
            except (OSError, ValueError):
                logger.info("fp16 variant not found, falling back to default")
                self._pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                )
            self._pipe.scheduler = EulerDiscreteScheduler.from_config(
                self._pipe.scheduler.config,
                timestep_spacing="trailing",
            )
            self._pipe.enable_attention_slicing("auto")
            self._pipe.enable_vae_slicing()
            self._pipe.enable_vae_tiling()
            self._pipe.enable_model_cpu_offload()
            self._loaded = True
            logger.info("SDXL model loaded with VRAM optimizations")
        except Exception as e:
            logger.error(f"Failed to load SDXL model: {e}")
            self._pipe = None
            self._loaded = False
            raise

    def unload(self) -> None:
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def inpaint(
        self, image: Image.Image, mask: Image.Image,
        prompt: str = "", negative_prompt: str = "",
        steps: int = 30, guidance_scale: float = 7.5,
        strength: float = 0.75, seed: int = 42,
        **kwargs
    ) -> Image.Image:
        if not self._loaded:
            self.load()

        gen = torch.Generator(
            device="cuda" if torch.cuda.is_available() else "cpu"
        ).manual_seed(seed)

        logger.info(
            f"SDXL inpaint: {image.size}, steps={steps}, "
            f"guidance={guidance_scale}, strength={strength}, seed={seed}"
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        result = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            strength=strength,
            generator=gen,
        ).images[0]

        if torch.cuda.is_available():
            vram_used = torch.cuda.max_memory_allocated() / (1024 ** 3)
            logger.info(f"SDXL inference done, VRAM peak: {vram_used:.1f}GB")

        return result

    @property
    def name(self) -> str:
        return "SDXL"

    @property
    def min_vram_gb(self) -> float:
        return 4.0
```

- [ ] **Step 7.2: Commit**

---

## Task 8: 策略调度器（含 auto_mode）

**Files:**
- Create: `lama-cleaner-plusplus/core/strategy.py`

- [ ] **Step 8.1: 实现 strategy.py**

```python
from dataclasses import dataclass
from utils.gpu import get_safe_available_vram_gb
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class InpaintConfig:
    engine_name: str
    steps: int
    guidance_scale: float
    strength: float
    mode: str


def auto_mode(mask_ratio: float, elongation: float = 0.0, complexity: float = 0.5) -> str:
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


def select_config(
    mask_ratio: float = 0.05,
    elongation: float = 0.0,
    complexity: float = 0.5,
    mode: str = "auto",
    force_engine: str | None = None,
    min_vram_gb: float = 4.0,
) -> InpaintConfig:
    vram = get_safe_available_vram_gb(reserve_gb=1.0, fraction=0.7)

    if mode == "auto":
        mode = auto_mode(mask_ratio, elongation, complexity)
        logger.info(f"auto_mode selected: {mode} (mask_ratio={mask_ratio:.3f}, elongation={elongation:.3f}, complexity={complexity:.3f})")

    if mode == "cpu":
        engine = "lama"
    elif force_engine:
        engine = force_engine
    elif vram < min_vram_gb:
        engine = "lama"
    else:
        engine = "sdxl"

    logger.info(
        f"Strategy: engine={engine}, mode={mode}, "
        f"vram={vram:.1f}GB, mask_ratio={mask_ratio:.3f}, elongation={elongation:.3f}"
    )

    if engine == "lama":
        return InpaintConfig(
            engine_name="lama",
            steps=0,
            guidance_scale=0,
            strength=0,
            mode="cpu",
        )

    if mode == "quick":
        strength = 0.65 if mask_ratio < 0.02 else 0.8
        return InpaintConfig(
            engine_name="sdxl",
            steps=20,
            guidance_scale=7.0,
            strength=strength,
            mode="quick",
        )
    else:
        strength = 0.6 if mask_ratio < 0.02 else 0.75
        return InpaintConfig(
            engine_name="sdxl",
            steps=30,
            guidance_scale=7.5,
            strength=strength,
            mode="hq",
        )


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
```

- [ ] **Step 8.2: Commit**

---

## Task 9: Pipeline 四阶段 + OOM Fallback

**Files:**
- Create: `lama-cleaner-plusplus/core/pipeline.py`

- [ ] **Step 9.1: 实现 pipeline.py**

```python
import time
import torch
import numpy as np
from PIL import Image
from collections.abc import Callable
from dataclasses import dataclass
import threading

from config import AppConfig
from core.mask_processor import MaskProcessor
from core.roi import crop_to_roi, resize_for_sdxl, paste_back, compute_padding_ratio
from core.strategy import select_config, InpaintConfig, get_negative_prompt
from core.engine_manager import EngineManager
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineContext:
    image: Image.Image
    mask: Image.Image
    prompt: str
    negative_prompt: str
    seed: int
    expand_px: int
    feather_radius: int
    gamma: float
    min_alpha: float
    mode: str = "auto"                     # 用户选择的模式（auto/quick/hq）
    mask_ratio: float = 0.0
    complexity: float = 0.0
    elongation: float = 0.0                 # 周长²/面积（原始值），auto_mode 驱动因子（阈值 40）
    expanded: np.ndarray | None = None
    roi_image: Image.Image | None = None       # preprocess 写入，后续只读
    roi_mask: Image.Image | None = None         # preprocess 写入，后续只读
    roi_feathered_alpha: np.ndarray | None = None  # preprocess 在 ROI 内计算
    roi_box: tuple[int, int, int, int] | None = None
    padding_ratio: float = 0.3
    inpaint_config: InpaintConfig | None = None
    inpainted_result: Image.Image | None = None
    skip: bool = False  # 空 mask 标记，preprocess 写入


class InpaintPipeline:
    def __init__(self, config: AppConfig | None = None):
        self.config = config or AppConfig()
        self.engine_manager = EngineManager(
            self.config.sdxl.model_id,
            lama_local_path=self.config.sdxl.lama_local_path,
            lama_hub_timeout=self.config.sdxl.lama_hub_timeout,
            lama_hub_retries=self.config.sdxl.lama_hub_retries,
        )
        self._run_lock = threading.Lock()

    def preprocess(self, ctx: PipelineContext) -> PipelineContext:
        """阶段1：原始 mask 几何特征提取 → Mask expand → 空值检查 → ROI 裁剪 → feather（在 ROI 内做）"""
        t0 = time.time()
        mask_np = np.array(ctx.mask.convert("L"))

        # 1. 基于原始 mask 计算几何特征（策略调度用，必须在 expand 之前）
        orig_rows = np.any(mask_np > 128, axis=1)
        orig_cols = np.any(mask_np > 128, axis=0)
        if orig_rows.any() and orig_cols.any():
            ormin, ormax = np.where(orig_rows)[0][[0, -1]]
            ocmin, ocmax = np.where(orig_cols)[0][[0, -1]]
            orig_roi_mask = mask_np[ormin:ormax+1, ocmin:ocmax+1]
            ctx.elongation = MaskProcessor.compute_elongation(orig_roi_mask)
            ctx.complexity = MaskProcessor.compute_complexity(orig_roi_mask)
        else:
            ctx.elongation = 0.0
            ctx.complexity = 0.0

        # 2. expand mask（仅用于实际修复区域）
        expanded = MaskProcessor.expand(mask_np, ctx.expand_px)
        ctx.expanded = expanded
        ctx.mask_ratio = MaskProcessor.compute_mask_ratio(expanded)

        # 3. 空 mask / 极小 mask 提前返回（双重判断：相对比例 + 绝对像素数）
        mask_area = int(np.sum(expanded > 128))
        if mask_area == 0:
            logger.warning("Empty mask (0 pixels), skip pipeline")
            ctx.skip = True
            return ctx

        if ctx.mask_ratio < 0.001 and mask_area < 256:
            logger.info(f"Tiny mask (ratio={ctx.mask_ratio:.6f}, area={mask_area}px), skip pipeline")
            ctx.skip = True
            return ctx

        # 4. crop ROI（只算一次）
        rows = np.any(expanded > 0, axis=1)
        cols = np.any(expanded > 0, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        h = max(1, rmax - rmin + 1)
        w = max(1, cmax - cmin + 1)
        bbox_area = h * w
        mask_area_in_bbox = np.sum(expanded[rmin:rmax+1, cmin:cmax+1] > 128)
        density = np.clip(mask_area_in_bbox / bbox_area, 0.01, 1.0)
        ctx.padding_ratio = compute_padding_ratio(ctx.mask_ratio, density)
        roi = crop_to_roi(
            ctx.image, Image.fromarray(expanded),
            padding_ratio=ctx.padding_ratio,
            min_size=self.config.roi.min_size,
            max_size=self.config.roi.max_size,
            min_context_px=self.config.roi.min_context_px,
        )
        ctx.roi_box = roi.box
        ctx.roi_image = roi.image
        ctx.roi_mask = roi.mask

        # 5. 转换 ROI mask 为 numpy
        roi_mask_np = np.array(roi.mask.convert("L"))

        # 6. feather 在 ROI 尺寸内做（传入 complexity 控制自适应 gamma）
        ctx.roi_feathered_alpha = MaskProcessor.feather(
            roi_mask_np, ctx.feather_radius, ctx.gamma, ctx.min_alpha,
            complexity=ctx.complexity,
        )

        elapsed = time.time() - t0
        logger.info(
            f"Preprocess: mask_ratio={ctx.mask_ratio:.3f}, "
            f"complexity={ctx.complexity:.3f}, elongation={ctx.elongation:.3f}, "
            f"padding={ctx.padding_ratio:.2f}, "
            f"roi={roi.box}, roi_size={ctx.roi_image.size}, elapsed={elapsed:.2f}s"
        )
        return ctx

    def select_engine(self, ctx: PipelineContext) -> PipelineContext:
        """阶段2：策略调度 + SAM/SDXL 互斥检查"""
        # Phase 2 SAM 预留：SAM 和 SDXL 不能同时占显存
        # 当前 SAM 引擎未实现，此检查不会触发，保留作为集成点
        if self.engine_manager.is_loaded("sam"):
            self.engine_manager.unload("sam")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Unloaded SAM to free VRAM for inpaint engine")

        ctx.inpaint_config = select_config(
            mask_ratio=ctx.mask_ratio,
            elongation=ctx.elongation,
            complexity=ctx.complexity,
            mode=ctx.mode,
            min_vram_gb=self.config.sdxl.min_vram_gb,
        )
        logger.info(
            f"Engine selected: {ctx.inpaint_config.engine_name} "
            f"(mode={ctx.inpaint_config.mode})"
        )
        return ctx

    def run_engine(self, ctx: PipelineContext) -> PipelineContext:
        """阶段3：推理执行（极小 ROI fast path + _run_sdxl/_run_lama + OOM fallback）"""
        # 极小 ROI fast path：auto/quick 模式用 LaMa 秒级响应，hq 模式强制走 SDXL
        roi_w, roi_h = ctx.roi_image.size
        if max(roi_w, roi_h) < 128 and ctx.inpaint_config.mode != "hq":
            logger.info(f"Small ROI ({roi_w}x{roi_h}), fast path to LaMa")
            engine = self.engine_manager.get("lama")
            ctx.inpainted_result = engine.inpaint(image=ctx.roi_image, mask=ctx.roi_mask)
            ctx.inpaint_config = InpaintConfig(
                engine_name="lama", steps=0, guidance_scale=0, strength=0, mode="fast",
            )
            return ctx

        engine_name = ctx.inpaint_config.engine_name
        try:
            engine = self.engine_manager.get(engine_name)
        except Exception as e:
            logger.error(f"Engine load failed: {e}, falling back to lama")
            engine = self.engine_manager.get("lama")
            engine_name = "lama"
            ctx.inpaint_config = InpaintConfig(
                engine_name="lama", steps=0, guidance_scale=0, strength=0, mode="fallback",
            )

        try:
            if engine_name == "sdxl":
                ctx.inpainted_result = self._run_sdxl(ctx, engine)
            else:
                ctx.inpainted_result = self._run_lama(ctx, engine)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                ctx = self._fallback_to_lama(ctx, e)
            else:
                raise
        return ctx

    def _run_sdxl(self, ctx: PipelineContext, engine) -> Image.Image:
        """SDXL 专用推理路径（传入硬边 mask，feather 由 postprocess 控制）"""
        target_w, target_h = ctx.roi_image.size
        resized_img, resized_mask, (resized_w, resized_h) = resize_for_sdxl(
            ctx.roi_image, ctx.roi_mask, self.config.roi.max_size
        )
        alpha_resized = False
        if (resized_w, resized_h) != (target_w, target_h):
            alpha_pil = Image.fromarray(
                (ctx.roi_feathered_alpha * 255).astype(np.uint8), mode="L"
            )
            alpha_pil = alpha_pil.resize((resized_w, resized_h), Image.BICUBIC)
            ctx.roi_feathered_alpha = np.array(alpha_pil).astype(np.float32) / 255.0
            alpha_resized = True
        effective_prompt = self._get_prompt(ctx.mask_ratio, ctx.complexity, ctx.prompt)
        graded_negative = get_negative_prompt(ctx.mask_ratio, ctx.negative_prompt, ctx.complexity, self.config.sdxl)
        result = engine.inpaint(
            image=resized_img,
            mask=resized_mask,
            prompt=effective_prompt,
            negative_prompt=graded_negative,
            steps=ctx.inpaint_config.steps,
            guidance_scale=ctx.inpaint_config.guidance_scale,
            strength=ctx.inpaint_config.strength,
            seed=ctx.seed,
        )
        result = result.resize((target_w, target_h), Image.LANCZOS)
        if alpha_resized:
            alpha_pil = Image.fromarray(
                (ctx.roi_feathered_alpha * 255).astype(np.uint8), mode="L"
            )
            alpha_pil = alpha_pil.resize((target_w, target_h), Image.BICUBIC)
            ctx.roi_feathered_alpha = np.array(alpha_pil).astype(np.float32) / 255.0
        return result

    def _run_lama(self, ctx: PipelineContext, engine) -> Image.Image:
        """LaMa 专用推理路径"""
        return engine.inpaint(image=ctx.roi_image, mask=ctx.roi_mask)

    def _fallback_to_lama(self, ctx: PipelineContext, original_error) -> PipelineContext:
        """OOM 时自动 fallback 到 LaMa（彻底清理 CUDA）"""
        import gc

        logger.warning(f"OOM on {ctx.inpaint_config.engine_name}, falling back to LaMa: {original_error}")

        # 1. 卸载失败引擎 + LaMa（LaMa 可能被 fast path 加载到 GPU 缓存，必须清除以确保 force_cpu 生效）
        self.engine_manager.unload(ctx.inpaint_config.engine_name)
        self.engine_manager.unload("lama")

        # 2. 彻底清理 CUDA（仅 empty_cache 不够，OOM 可能来自 fragmentation/graph cache/cudnn workspace）
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()

        # 3. 加载 LaMa 强制 CPU（OOM 场景 GPU 已无可用显存）
        lama = self.engine_manager.get("lama", force_cpu=True)
        ctx.inpainted_result = lama.inpaint(image=ctx.roi_image, mask=ctx.roi_mask)

        # 4. 创建新 config（不污染原对象，防止 batch 复用时状态污染）
        ctx.inpaint_config = InpaintConfig(
            engine_name="lama",
            steps=0,
            guidance_scale=0,
            strength=0,
            mode="cpu",
        )

        logger.info("Fallback to LaMa successful")
        return ctx

    def _get_prompt(self, mask_ratio: float, complexity: float, user_prompt: str) -> str:
        """Prompt 四级策略：极小保守 / 复杂结构保护 / 细结构轻量 / 标准引导"""
        if user_prompt:
            return user_prompt
        if mask_ratio < 0.02:
            return "same texture, seamless blend, preserve details"  # 极小区域保守引导
        if complexity > 0.5:
            return "highly detailed, preserve structure, seamless"  # 复杂结构保护性生成
        if mask_ratio < 0.05 and complexity < 0.3:
            return "clean background, smooth, no artifacts"  # 细结构轻量引导
        return self.config.sdxl.default_prompt  # 标准引导

    def postprocess(self, ctx: PipelineContext) -> Image.Image:
        """阶段4：回填 + 日志（使用 roi.paste_back 统一回填逻辑）"""
        output = paste_back(
            ctx.image, ctx.inpainted_result, ctx.roi_box,
            feathered_alpha=ctx.roi_feathered_alpha,
        )
        logger.info(f"Postprocess done, output size: {output.size}")
        return output

    def run(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str = "",
        negative_prompt: str = "",
        mode: str = "auto",
        expand_px: int | None = None,
        feather_radius: int | None = None,
        seed: int | None = None,
        status_callback: Callable[[str], None] | None = None,
    ) -> Image.Image:
        t_start = time.time()
        if not negative_prompt:
            negative_prompt = self.config.sdxl.default_negative

        ctx = PipelineContext(
            image=image,
            mask=mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed if seed is not None else self.config.sdxl.seed,
            expand_px=expand_px if expand_px is not None else self.config.mask.expand_pixels,
            feather_radius=feather_radius if feather_radius is not None else self.config.mask.feather_radius,
            gamma=self.config.mask.gamma,
            min_alpha=self.config.mask.min_alpha,
            mode=mode,
        )

        with self._run_lock:
            if status_callback:
                status_callback("⏳ 分析 mask 中...")
            ctx = self.preprocess(ctx)
            if ctx.skip:
                logger.info("Empty mask, returning original image")
                return ctx.image

            ctx = self.select_engine(ctx)

            if status_callback:
                engine_name = ctx.inpaint_config.engine_name
                mode_label = ctx.inpaint_config.mode
                if engine_name == "sdxl" and not self.engine_manager.is_loaded("sdxl"):
                    status_callback(f"⏳ 加载 SDXL 模型中（首次约 5-15 秒）...")
                elif engine_name == "sdxl":
                    status_callback(f"⏳ SDXL {mode_label} 推理中...")
                else:
                    status_callback(f"⏳ LaMa 修复中...")

            ctx = self.run_engine(ctx)

            if status_callback:
                status_callback("⏳ 拼接回填中...")

            final = self.postprocess(ctx)

        elapsed = time.time() - t_start
        logger.info(
            f"Pipeline complete: engine={ctx.inpaint_config.engine_name}, "
            f"mode={ctx.inpaint_config.mode}, elapsed={elapsed:.2f}s"
        )
        if status_callback:
            status_callback(f"✅ 完成（{elapsed:.1f}s，{ctx.inpaint_config.engine_name}/{ctx.inpaint_config.mode}）")
        return final

    def unload_all(self):
        self.engine_manager.unload_all()
```

- [ ] **Step 9.2: Commit**

---

## Task 10: InpaintService（UI/逻辑解耦）

**Files:**
- Create: `lama-cleaner-plusplus/core/service.py`

- [ ] **Step 10.1: 实现 service.py**

```python
from collections.abc import Callable
from PIL import Image
from config import AppConfig
from core.pipeline import InpaintPipeline
from core.mask_processor import MaskProcessor
from utils.logger import get_logger
import numpy as np

logger = get_logger(__name__)


class InpaintService:
    def __init__(self, config: AppConfig | None = None):
        self.config = config or AppConfig()
        self.pipeline = InpaintPipeline(self.config)

    def process(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str = "",
        negative_prompt: str = "",
        mode: str = "auto",
        expand_px: int | None = None,
        feather_radius: int | None = None,
        seed: int | None = None,
        history: list | None = None,
        status_callback: Callable[[str], None] | None = None,
    ) -> tuple[Image.Image, Image.Image, list]:
        history = (history or []).copy()
        history.append(image.copy())
        if len(history) > 10:
            history = history[-10:]

        result = self.pipeline.run(
            image=image,
            mask=mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            mode=mode,
            expand_px=expand_px,
            feather_radius=feather_radius,
            seed=seed,
            status_callback=status_callback,
        )

        mask_np = np.array(mask.convert("L"))
        expanded = MaskProcessor.expand(mask_np, expand_px if expand_px is not None else self.config.mask.expand_pixels)
        preview = Image.fromarray(
            MaskProcessor.overlay_on_image(np.array(image), expanded)
        )
        logger.info(f"Service process done, history: {len(history)}")
        return result, preview, history

    def undo(self, history: list) -> tuple[Image.Image | None, list]:
        """Undo 限制：仅恢复上一步原图到输出区，无法自动恢复 ImageEditor 中的原图和 mask 状态。
        每个 session 的历史通过 gr.State 隔离，最多 10 步。"""
        history = history.copy()
        if history:
            return history.pop(), history
        return None, history

    def cleanup(self):
        self.pipeline.unload_all()
```

- [ ] **Step 10.2: Commit**

---

## Task 11: Gradio UI（使用 InpaintService）

**Files:**
- Create: `lama-cleaner-plusplus/ui/gradio_app.py`
- Create: `lama-cleaner-plusplus/app.py`

- [ ] **Step 11.1: 实现 gradio_app.py**

```python
import gradio as gr
import numpy as np
from PIL import Image

from config import AppConfig, load_config
from core.service import InpaintService
from core.mask_processor import MaskProcessor
from utils.logger import get_logger

logger = get_logger(__name__)


def _extract_mask_from_editor(editor_data: dict, image: Image.Image) -> np.ndarray:
    h, w = image.height, image.width

    layers = editor_data.get("layers", [])
    mask_acc = None

    for layer in layers:
        if layer is None:
            continue
        if isinstance(layer, dict):
            if "data" in layer:
                layer = layer["data"]
            else:
                continue
        if isinstance(layer, np.ndarray):
            if layer.ndim == 2:
                alpha = layer.astype(np.float32) / 255.0
            elif layer.ndim == 3 and layer.shape[2] >= 4:
                alpha = layer[:, :, 3].astype(np.float32) / 255.0
            elif layer.ndim == 3:
                alpha = np.mean(layer[:, :, :3], axis=2).astype(np.float32) / 255.0
            else:
                logger.warning(f"Skipping layer with unexpected ndim={layer.ndim}, shape={layer.shape}")
                continue
        elif hasattr(layer, "mode"):
            if layer.mode == "RGBA":
                alpha = np.array(layer)[:, :, 3].astype(np.float32) / 255.0
            elif layer.mode == "L":
                alpha = np.array(layer).astype(np.float32) / 255.0
            else:
                alpha = np.array(layer.convert("L")).astype(np.float32) / 255.0
        else:
            logger.warning(f"Skipping layer of unsupported type: {type(layer).__name__}")
            continue
        if mask_acc is None:
            mask_acc = alpha
        else:
            mask_acc = np.maximum(mask_acc, alpha)

    if mask_acc is None:
        composite = editor_data.get("composite")
        if composite is not None:
            if isinstance(composite, np.ndarray):
                if composite.ndim == 3 and composite.shape[2] >= 4:
                    alpha = composite[:, :, 3].astype(np.float32) / 255.0
                    if alpha.max() > 0:
                        comp_rgb = composite[:, :, :3].astype(np.float32)
                        bg_rgb = np.array(image).astype(np.float32)
                        diff = np.abs(comp_rgb - bg_rgb).max(axis=2)
                        threshold = max(8, diff.mean() * 0.5)
                        mask_acc = ((diff > threshold) & (alpha > 0)).astype(np.float32)
                    else:
                        mask_acc = alpha
                elif composite.ndim == 3:
                    comp_rgb = composite[:, :, :3].astype(np.float32)
                    bg_rgb = np.array(image).astype(np.float32)
                    diff = np.abs(comp_rgb - bg_rgb).max(axis=2)
                    threshold = max(8, diff.mean() * 0.5)
                    mask_acc = (diff > threshold).astype(np.float32)
            elif hasattr(composite, "mode"):
                if composite.mode == "RGBA":
                    alpha = np.array(composite)[:, :, 3].astype(np.float32) / 255.0
                    if alpha.max() > 0:
                        comp_rgb = np.array(composite)[:, :, :3].astype(np.float32)
                        bg_rgb = np.array(image).astype(np.float32)
                        diff = np.abs(comp_rgb - bg_rgb).max(axis=2)
                        threshold = max(8, diff.mean() * 0.5)
                        mask_acc = ((diff > threshold) & (alpha > 0)).astype(np.float32)
                    else:
                        mask_acc = alpha
                elif composite.mode == "RGB":
                    comp_rgb = np.array(composite).astype(np.float32)
                    bg_rgb = np.array(image).astype(np.float32)
                    diff = np.abs(comp_rgb - bg_rgb).max(axis=2)
                    threshold = max(8, diff.mean() * 0.5)
                    mask_acc = (diff > threshold).astype(np.float32)

    if mask_acc is None or mask_acc.max() == 0:
        return np.zeros((h, w), dtype=np.uint8)

    mask_np = (mask_acc * 255).astype(np.uint8)
    mask_np = np.where(mask_np > 128, 255, 0).astype(np.uint8)
    return mask_np


def create_app(config: AppConfig | None = None) -> gr.Blocks:
    config = config or load_config()
    service = InpaintService(config)

    with gr.Blocks(title="Lama Cleaner++", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🖼️ Lama Cleaner++")
        gr.Markdown("AI 局部图像修复工具 — 圈哪里修哪里")

        with gr.Row():
            with gr.Column(scale=1):
                input_editor = gr.ImageEditor(
                    label="上传图片 & 画 Mask",
                    type="pil",
                    image_mode="RGB",
                    brush=gr.Brush(
                        default_size=20,
                        colors=["#FFFFFF"],
                        color_mode="fixed",
                    ),
                    eraser=gr.Eraser(default_size=20),
                    layers=True,  # 必须启用图层，否则画笔内容不可见
                    sources=["upload", "clipboard"],
                    height=500,
                )
                prompt = gr.Textbox(
                    label="Prompt（可选，有默认值）",
                    placeholder="描述修复后的效果，留空使用默认值",
                    value="",
                )
                with gr.Row():
                    mode = gr.Radio(
                        ["auto", "quick", "hq", "cpu"],
                        value="auto",
                        label="模式",
                    )
                    seed = gr.Number(
                        label="Seed（默认 42）",
                        value=42,
                        precision=0,
                    )
                with gr.Accordion("高级参数", open=False):
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt（可选）",
                        placeholder="描述需要避免的效果，留空使用默认值",
                        value="",
                    )
                    expand_slider = gr.Slider(
                        0, 50, value=config.mask.expand_pixels,
                        step=1, label="Mask 膨胀 (px)",
                    )
                    feather_slider = gr.Slider(
                        0, 50, value=config.mask.feather_radius,
                        step=1, label="Mask 羽化 (px)",
                    )
                with gr.Row():
                    run_btn = gr.Button("🚀 Run", variant="primary", size="lg")
                    undo_btn = gr.Button("↩️ Undo", size="sm")

            with gr.Column(scale=1):
                output_image = gr.Image(label="修复结果", type="pil")
                mask_preview = gr.Image(label="Mask 预览", type="pil")

        history_state = gr.State([])

        status_text = gr.Textbox(
            label="\u72b6\u6001",
            value="\u5c31\u7eea",
            interactive=False,
            max_lines=1,
        )

        _progress_map = [
            ("\u5206\u6790 mask", 0.15),
            ("\u52a0\u8f7d", 0.35),
            ("\u63a8\u7406", 0.65),
            ("LaMa", 0.65),
            ("\u62fc\u63a5", 0.85),
            ("\u5b8c\u6210", 1.0),
        ]

        def on_run(editor_data, prompt_text, neg_prompt_text, mode_val, seed_val, expand_val, feather_val, hist, progress=gr.Progress()):
            if editor_data is None:
                return "\u274c \u65e0\u8f93\u5165\u56fe\u7247", None, None, hist
            if isinstance(editor_data, Image.Image):
                return "⚠️ 未检测到 mask，请先画 mask", None, None, hist
            elif isinstance(editor_data, dict):
                image = editor_data.get("background")
                if image is None:
                        return "\u274c \u65e0\u80cc\u666f\u56fe\u7247", None, None, hist
                mask_np = _extract_mask_from_editor(editor_data, image)
            else:
                    return "\u274c \u4e0d\u652f\u6301\u7684\u8f93\u5165\u683c\u5f0f", None, None, hist

            if mask_np.max() == 0:
                    return "\u26a0\ufe0f \u672a\u68c0\u6d4b\u5230 mask\uff0c\u8bf7\u5148\u753b mask", None, None, hist

            mask = Image.fromarray(mask_np)
            progress(0.05, desc="\u23f3 \u6b63\u5728\u5904\u7406...")

            def on_status(msg):
                for keyword, value in _progress_map:
                    if keyword in msg:
                        progress(value, desc=msg)
                        break

            try:
                result, preview, hist = service.process(
                    image=image,
                    mask=mask,
                    prompt=prompt_text,
                    negative_prompt=neg_prompt_text,
                    mode=mode_val,
                    expand_px=int(expand_val),
                    feather_radius=int(feather_val),
                    seed=int(seed_val) if seed_val is not None else None,
                    history=hist,
                    status_callback=on_status,
                )
                return "✅ 完成", result, preview, hist
            except RuntimeError as e:
                logger.error(f"Pipeline error: {e}", exc_info=True)
                return f"❌ 处理失败: {e}", None, None, hist
            except Exception as e:
                logger.error(f"Unexpected error: {e}", exc_info=True)
                return f"❌ 意外错误: {e}", None, None, hist

        def on_undo(hist):
            result, hist = service.undo(hist)
            return result, None, hist

        run_btn.click(
            fn=on_run,
            inputs=[input_editor, prompt, negative_prompt, mode, seed, expand_slider, feather_slider, history_state],
            outputs=[status_text, output_image, mask_preview, history_state],
        )
        undo_btn.click(fn=on_undo, inputs=[history_state], outputs=[output_image, mask_preview, history_state])

    return app
```

- [ ] **Step 11.2: 实现 app.py 入口**

```python
from config import load_config
from ui.gradio_app import create_app
from utils.logger import setup_logger


def main():
    config = load_config()
    setup_logger("lama-cleaner-plusplus", config.log_level)
    app = create_app(config)
    app.launch(
        server_port=config.server_port,
        share=config.share,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 11.3: 验证启动**

Run: `cd lama-cleaner-plusplus && python app.py`
Expected: Gradio 服务启动在 http://localhost:7860，日志输出到终端

- [ ] **Step 11.4: Commit**

---

## Task 12: 集成测试 + 端到端验证

**Files:**
- Create: `lama-cleaner-plusplus/tests/test_pipeline.py`

- [ ] **Step 12.1: 创建测试脚本**

```python
"""端到端测试：使用合成数据验证完整管线"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2
from PIL import Image
from config import AppConfig
from core.mask_processor import MaskProcessor
from core.roi import crop_to_roi, paste_back, resize_for_sdxl, compute_padding_ratio


def test_mask_processor():
    mask = np.zeros((500, 500), dtype=np.uint8)
    mask[200:300, 200:300] = 255
    expanded, feathered = MaskProcessor.expand_and_feather(mask, 15, 20, complexity=0.5)
    assert expanded.sum() > mask.sum(), "expanded should be larger"
    assert feathered.min() == 0.0
    assert feathered.max() == 1.0
    assert feathered.shape == (500, 500)
    print("✅ MaskProcessor OK")


def test_mask_gamma():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255
    _, f1 = MaskProcessor.expand_and_feather(mask, 10, 15, gamma=1.0, min_alpha=0.0)
    _, f2 = MaskProcessor.expand_and_feather(mask, 10, 15, gamma=1.5, min_alpha=0.1)
    assert f2.min() >= 0.1, "min_alpha should be respected"
    print("✅ Mask gamma/min_alpha OK")


def test_mask_complexity():
    large_mask = np.zeros((1000, 1000), dtype=np.uint8)
    large_mask[100:900, 100:900] = 255
    c1 = MaskProcessor.compute_complexity(large_mask)

    thin_mask = np.zeros((1000, 1000), dtype=np.uint8)
    thin_mask[490:510, 100:900] = 255
    c2 = MaskProcessor.compute_complexity(thin_mask)

    e1 = MaskProcessor.compute_elongation(large_mask)
    e2 = MaskProcessor.compute_elongation(thin_mask)

    assert c2 > c1, f"thin complexity ({c2:.3f}) should be > large ({c1:.3f})"
    assert e2 > e1, f"thin elongation ({e2:.3f}) should be > large ({e1:.3f})"

    frag_mask = np.zeros((200, 200), dtype=np.uint8)
    for i in range(5):
        cv2.circle(frag_mask, (i * 20 + 10, i * 20 + 10), 15, 255, -1)
    c3 = MaskProcessor.compute_complexity(frag_mask)
    assert c3 > 0, f"fragmented mask complexity should be > 0, got {c3:.3f}"

    print(f"✅ Complexity: large={c1:.3f}, thin={c2:.3f}, frag={c3:.3f}, elongation: large={e1:.3f}, thin={e2:.3f}")


def test_adaptive_padding():
    assert compute_padding_ratio(0.01) == 0.5
    assert compute_padding_ratio(0.05) == 0.3
    assert compute_padding_ratio(0.15) == 0.15
    print("✅ Adaptive padding OK")


def test_roi():
    img = Image.fromarray(np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8))
    mask = Image.fromarray(np.zeros((500, 500), dtype=np.uint8))
    mask.paste(255, (200, 200, 300, 300))
    roi = crop_to_roi(img, mask, padding_ratio=compute_padding_ratio(0.05))
    assert roi.box[0] < 200
    assert roi.box[2] > 300
    print("✅ ROI crop OK")


def test_roi_min_size():
    img = Image.fromarray(np.random.randint(0, 255, (800, 800, 3), dtype=np.uint8))
    mask = Image.fromarray(np.zeros((800, 800), dtype=np.uint8))
    mask.paste(255, (395, 395, 405, 405))  # 10x10 mask
    roi = crop_to_roi(img, mask, padding_ratio=0.3, min_size=256, min_context_px=64)
    roi_w = roi.box[2] - roi.box[0]
    roi_h = roi.box[3] - roi.box[1]
    assert roi_w >= 256, f"ROI width {roi_w} should be >= 256"
    assert roi_h >= 256, f"ROI height {roi_h} should be >= 256"
    print(f"✅ ROI min_size enforced: box={roi.box}, size={roi_w}x{roi_h}")


def test_resize():
    img = Image.fromarray(np.random.randint(0, 255, (300, 200, 3), dtype=np.uint8))
    mask = Image.fromarray(np.zeros((300, 200), dtype=np.uint8))
    resized_img, resized_mask, (tw, th) = resize_for_sdxl(img, mask)
    assert tw % 8 == 0
    assert th % 8 == 0
    print(f"✅ Resize OK: {img.size} → ({tw}, {th})")


def test_paste_back():
    original = Image.fromarray(np.zeros((500, 500, 3), dtype=np.uint8))
    result = Image.fromarray(np.full((200, 200, 3), 255, dtype=np.uint8))
    box = (150, 150, 350, 350)
    output = paste_back(original, result, box)
    center_pixel = np.array(output)[250, 250]
    assert center_pixel.sum() > 0
    print("✅ Paste back OK")


def test_service_undo():
    from core.service import InpaintService
    svc = InpaintService(AppConfig())
    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    mask = Image.fromarray(np.zeros((100, 100), dtype=np.uint8))
    history = []
    _, _, history = svc.process(image=img, mask=mask, history=history)
    assert len(history) == 1
    restored, history = svc.undo(history)
    assert restored is not None
    assert len(history) == 0
    restored2, history = svc.undo(history)
    assert restored2 is None
    print("✅ Service undo OK")


def test_pipeline_preprocess():
    """真正调用 pipeline.preprocess()，验证 PipelineContext 各字段"""
    from core.pipeline import InpaintPipeline, PipelineContext

    config = AppConfig()
    pipeline = InpaintPipeline(config)

    img = Image.new("RGB", (512, 512), (128, 128, 128))
    mask = Image.new("L", (512, 512), 0)
    mask.paste(255, (200, 200, 250, 250))

    ctx = PipelineContext(
        image=img, mask=mask, prompt="", negative_prompt="",
        seed=42, expand_px=config.mask.expand_pixels,
        feather_radius=config.mask.feather_radius,
        gamma=config.mask.gamma, min_alpha=config.mask.min_alpha,
        mode="auto",
    )
    ctx = pipeline.preprocess(ctx)

    assert not ctx.skip, "non-empty mask should not skip"
    assert ctx.roi_image is not None
    assert ctx.roi_mask is not None
    assert ctx.roi_box is not None
    assert ctx.complexity >= 0
    assert ctx.elongation >= 0
    assert ctx.roi_feathered_alpha is not None
    assert ctx.mask_ratio > 0

    # 空 mask 应跳过
    empty_mask = Image.new("L", (512, 512), 0)
    ctx_empty = PipelineContext(
        image=img, mask=empty_mask, prompt="", negative_prompt="",
        seed=42, expand_px=config.mask.expand_pixels,
        feather_radius=config.mask.feather_radius,
        gamma=config.mask.gamma, min_alpha=config.mask.min_alpha,
        mode="auto",
    )
    ctx_empty = pipeline.preprocess(ctx_empty)
    assert ctx_empty.skip, "empty mask should trigger skip"
    print("✓ Pipeline preprocess (live preprocess + empty mask skip)")


def test_pipeline_empty_mask():
    from core.pipeline import InpaintPipeline
    config = AppConfig()
    pipeline = InpaintPipeline(config)

    img = Image.new("RGB", (512, 512), (128, 128, 128))
    mask = Image.new("L", (512, 512), 0)

    result = pipeline.run(image=img, mask=mask, mode="auto")
    assert result is not None
    assert result.size == (512, 512)
    print("✓ Pipeline empty mask returns original")


def test_pipeline_oom_fallback():
    from core.pipeline import InpaintPipeline, PipelineContext
    import inspect

    config = AppConfig()
    pipeline = InpaintPipeline(config)

    assert hasattr(pipeline, "engine_manager")
    assert hasattr(pipeline, "_fallback_to_lama")
    assert hasattr(pipeline, "postprocess")
    assert hasattr(pipeline, "_run_lock")

    em = pipeline.engine_manager
    assert hasattr(em, "unload"), "engine_manager must have unload"
    assert hasattr(em, "get"), "engine_manager must have get"
    assert hasattr(em, "unload_all"), "engine_manager must have unload_all"

    fb_src = inspect.getsource(pipeline._fallback_to_lama)
    assert "unload" in fb_src, "_fallback_to_lama must call unload"
    assert "lama" in fb_src.lower(), "_fallback_to_lama must reference lama"
    assert "force_cpu" in fb_src, "_fallback_to_lama must use force_cpu=True"
    assert "InpaintConfig" in fb_src, "_fallback_to_lama must set InpaintConfig"

    img = Image.new("RGB", (512, 512), (128, 128, 128))
    small_mask = Image.new("L", (512, 512), 0)
    small_mask.paste(255, (250, 250, 270, 270))
    ctx = PipelineContext(
        image=img, mask=small_mask, prompt="", negative_prompt="",
        seed=42, expand_px=5, feather_radius=5,
        gamma=config.mask.gamma, min_alpha=config.mask.min_alpha,
        mode="auto",
    )
    ctx = pipeline.preprocess(ctx)
    assert not ctx.skip
    ctx = pipeline.select_engine(ctx)
    ctx = pipeline._fallback_to_lama(ctx, RuntimeError("simulated OOM"))
    assert ctx.inpaint_config is not None
    assert ctx.inpaint_config.engine_name == "lama"
    assert ctx.inpainted_result is not None
    print("✓ Pipeline OOM fallback chain verified")


if __name__ == "__main__":
    test_mask_processor()
    test_mask_gamma()
    test_mask_complexity()
    test_adaptive_padding()
    test_roi()
    test_roi_min_size()
    test_resize()
    test_paste_back()
    test_service_undo()
    test_pipeline_preprocess()
    test_pipeline_empty_mask()
    test_pipeline_oom_fallback()
    print("\n🎉 All tests passed!")
```

- [ ] **Step 12.2: 运行测试**

Run: `cd lama-cleaner-plusplus && python tests/test_pipeline.py`
Expected: All tests passed!

- [ ] **Step 12.3: Commit**

---

## Task 13: 最终集成 + 启动验证

- [ ] **Step 13.1: 安装依赖**

Run: `cd lama-cleaner-plusplus && pip install -r requirements.txt`

- [ ] **Step 13.2: 启动应用**

Run: `cd lama-cleaner-plusplus && python app.py`
Expected: Gradio 启动，日志输出，浏览器打开 http://localhost:7860

- [ ] **Step 13.3: 手动验证**

1. 上传图片 → 画 mask → 选 auto 模式 → Run
2. 确认结果自然修复
3. 点 Undo → 恢复上一张
4. 切 hq 模式 → 重新 Run → 对比效果
5. 查看终端日志（engine 选择、ROI 信息、耗时、VRAM）

- [ ] **Step 13.4: 性能验证**

- [ ] auto 模式小区域 < 5s
- [ ] hq 模式中等区域 < 30s
- [ ] VRAM 峰值 < 5.5GB
- [ ] 连续 5 次不 OOM
- [ ] OOM 时自动 fallback 到 LaMa（可用 `CUDA_VISIBLE_DEVICES=""` 模拟无 GPU）

- [ ] **Step 13.5: Final Commit**

```bash
git add -A
git commit -m "feat: Lama Cleaner++ Phase 1 complete (v2 architecture)"
```

---

## 附录：架构变更对照

### v1 → v2

| 模块 | v1（旧版） | v2（当前版） |
|------|-----------|-------------|
| 引擎管理 | `self._engines["sdxl"] = SDXLEngine()` | `EngineManager` + lazy load + threading.Lock |
| Pipeline | `run()` 上帝函数 | 四阶段：`preprocess→select_engine→run_engine→postprocess` |
| ROI padding | 固定 `padding_ratio=0.3` | `compute_padding_ratio(mask_ratio)` 自适应（0.5/0.3/0.15） |
| 异常处理 | 无 | OOM fallback：SDXL → LaMa |
| 日志 | 无 | `utils/logger.py` 每阶段记录 |
| Seed | 随机 | 默认 42 |
| 模式选择 | 手动 quick/hq | `auto_mode()` 自动推荐 |
| UI 耦合 | `pipeline.run()` 直接在 UI 中 | `InpaintService` 中间层 |
| Mask | 无 gamma/min_alpha | gamma clamp + min_alpha |
| Config | 纯 dataclass | 支持 CLI `--port` + env `LAMA_PORT` |
| 历史记录 | 无 | `InpaintService._history` + undo |

### v2 → v3（v3 轮修复）

| 问题 | v2（旧） | v3（当前） |
|------|---------|-----------|
| ROI 重复计算 | `crop_to_roi()` 在 preprocess/run_engine/postprocess 各调一次 | ROI 只在 `preprocess()` 计算一次，存入 `ctx.roi_image` / `ctx.roi_mask` / `ctx.roi_box`，后续只读 |
| feather 时机 | 全图 feather → crop → resize（半径不一致） | expand → crop ROI → feather（ROI 尺寸）→ resize |
| EngineManager 破坏抽象 | `engine._loaded` 直接访问 | `BaseEngine.is_loaded()` 公开方法 + `EngineManager.is_loaded(name)` 查询 |
| Strategy 没用 complexity | `auto_mode(mask_ratio)` 只看面积 | `auto_mode(mask_ratio, complexity)` 双因子，细线条自动选 hq |
| OOM fallback 不彻底 | `unload()` 后直接加载 LaMa | `unload()` + `torch.cuda.empty_cache()` + `torch.cuda.reset_peak_memory_stats()` |
| run_engine 上帝函数 | `if sdxl: ... else: ...` 嵌套 | 拆分为 `_run_sdxl()` / `_run_lama()` / `_fallback_to_lama()` |
| crop_to_roi 重复 4 次 | preprocess + run_engine×2 + postprocess | 只在 preprocess 计算一次 |
| 默认 prompt 过度干预 | 固定 `"clean background, seamless, natural texture"` | `mask_ratio < 0.05` 用空 prompt（保守），>= 0.05 用引导 prompt |
| SAM 卸载缺失 | select_engine 无 SAM 检查 | `is_loaded("sam")` → `unload("sam")` + `empty_cache()` |

### v3 → v4（本轮修复）

| 问题 | v3（旧） | v4（当前） |
|------|---------|-----------|
| complexity 假精确 | 只算 density（mask_area / bbox_area），细线条误判为"简单" | `0.5 * density + 0.5 * edge_ratio`（Canny 边界复杂度），细结构/破碎 mask 正确识别为高复杂 |
| resize mask 变形 | `Image.NEAREST`，细结构断裂 | `Image.BILINEAR` + `.point(lambda x: 255 if x > 128 else 0)`，先平滑再二值化 |
| SDXL generator 在 CPU | `torch.Generator(device="cpu")`，随机数同步开销 | `torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")` |
| fallback config 污染 | 直接修改 `ctx.inpaint_config.engine_name = "lama"`，batch 复用时状态污染 | 创建新 `InpaintConfig(engine_name="lama", steps=0, ...)`，不修改原对象 |
| postprocess 无 clamp | `.astype(np.uint8)` 直接转换，极端情况溢出/wrap | `np.clip(..., 0, 255).astype(np.uint8)`，`paste_back()` 和 Pipeline `postprocess()` 均已修复 |
| feather 不感知复杂度 | 固定 gamma=1.0，大块和细线用同一混合力度 | `adaptive_gamma = 1.0 + complexity * 0.5`，复杂结构更柔和，大块区域更锐利 |
| Prompt 策略太粗 | 二级：`< 0.05` 空 / `>= 0.05` 引导 | 四级：`< 0.02` 空 / `complexity > 0.5` 保护性 / `complexity < 0.3` 轻量 / 标准引导 |
| 缺 mask 预览 | 用户不知道 mask 扩展到哪里 | UI 增加 `mask_preview` 面板，调用 `overlay_on_image` 显示 expand 后 mask 覆盖范围 |
| 极小 mask 走完整流程 | 所有 mask 统一走 SDXL/LaMa | `mask_ratio < 0.001` 直接返回原图（秒响应，不误修） |
| complexity 用全图 | `compute_complexity(expanded, bbox)` 基于全图 expanded mask | 在 ROI 空间计算：`np.sum(roi_mask > 128) / (roi_w * roi_h)`，调度策略和实际修复难度一致 |
| OOM fallback 不彻底 | `unload()` + `empty_cache()` + `reset_peak_memory_stats()` | + `gc.collect()` + `torch.cuda.ipc_collect()`（清理 fragmentation/graph cache） |
| 空 mask 无处理 | SDXL 可能生成整张图（灾难） | `preprocess` 检测 `mask_ratio < 1e-5` → `ctx.skip = True` → `run()` 直接返回原图 |
| UI 不是真正画 mask | `gr.Image` 上传 mask（体验差） | `gr.ImageEditor` 内置画笔/橡皮擦，用户直接在图上画 |
| ROI padding 不考虑密度 | 只看 `mask_ratio` | 加 `density = mask_area / bbox_area`，`density < 0.3` 时 `padding += 0.2` |
| 极小 ROI 无优化 | 统一走 SDXL/LaMa 流程 | `max(roi_w, roi_h) < 128` → 直接 LaMa fast path，秒级响应 |
| SDXL mask blur | diffusers 内部 blur 破坏 feather | 传入硬边 mask（0/255），feather 由 postprocess 的 `roi_feathered_alpha` 控制 |

### v4 → v5（本轮修复）

| # | 严重度 | 问题 | v4（旧） | v5（当前） |
|---|--------|------|---------|-----------|
| 1 | 🔴P0 | select_engine 硬编码 mode | `mode="auto"` 硬编码，忽略用户选择 | `PipelineContext` 加 `mode` 字段，`run()` 传入，`select_config` 使用 `ctx.mode` |
| 2 | 🔴P0 | fast path 缺 inpaint_config | 极小 ROI fast path 直接调 LaMa，`ctx.inpaint_config` 未设置，`run()` 最后一行 `AttributeError` | fast path 设置 `ctx.inpaint_config = InpaintConfig(engine_name="lama", mode="fast")` |
| 3 | 🔴P0 | test_mask_complexity 参数错误 | `MaskProcessor.compute_complexity(expanded, (200, 200, 300, 300))` 传 2 参数，函数只接受 1 个 | 修正为 `MaskProcessor.compute_complexity(roi_mask)`，1 个参数 |
| 4 | 🟡P1 | postprocess 重复 resize | `_run_sdxl` 已 resize 到 ROI 尺寸，`postprocess` 又 resize 一次（质量损失） | 移除重复 resize，改为防御性检查 `if inpainted.size != target: resize`，正常路径零开销 |
| 5 | 🟡P1 | density 修正未传参 | `compute_padding_ratio(mask_ratio)` 未传 density，`density < 0.3 → padding += 0.2` 逻辑成死代码 | `preprocess` 计算 density 后传入 `compute_padding_ratio(mask_ratio, density)` |
| 6 | 🟡P1 | service preview 用原始 mask | `overlay_on_image(np.array(image), np.array(mask))` 显示用户画的原始 mask | 改为先 expand 再 overlay，显示实际处理区域 |
| 7 | 🟡P1 | Gradio RGBA alpha 未处理 | `layers[0].convert("L")` 忽略 alpha 通道 | 优先取 `layer.mode == "RGBA"` 的 alpha 通道，再 fallback 到 `.convert("L")` |
| 8 | 🟡P1 | expand_and_feather 未传 complexity | `feather()` 不传 complexity，adaptive_gamma 失效 | `expand_and_feather` 增加 `complexity` 参数并传递 |
| 9 | 🟡P1 | SAM 互斥检查死代码 | `is_loaded("sam")` 无注释，看起来是 bug | 添加注释说明是 Phase 2 预留集成点 |
| 10 | 🟡P1 | min_vram_gb 未使用 | `select_config` 硬编码 `vram < 4`，`BaseEngine.min_vram_gb` 属性浪费 | `SDXLConfig` 加 `min_vram_gb=4.0`，`select_config` 接受参数，Pipeline 传入 |
| 11 | 🟡P1 | logger hierarchy 混乱 | `app.py` 设置 `"lama-cleaner-plusplus"` 级别，但模块 logger 是 `"core.*"` / `"ui.*"`，不继承 | `app.py` 额外设置 `logging.getLogger("core").setLevel(...)` 和 `"ui"` |
| 12 | 🟡P1 | SDXL load 缺错误处理 | `SDXLEngine.load()` 无 try/except，模型下载失败直接崩溃 | 加 try/except，失败时 `_pipe = None` + `_loaded = False` + re-raise |
| 13 | 🟢P2 | on_undo 死代码 | `on_undo()` 函数定义但未使用，`undo_btn.click` 直接调 `service.undo` | 删除 `on_undo` 函数 |
| 14/21 | 🟢P2 | paste_back 冗余 | roi.py 的 `paste_back` 未被 pipeline 调用，pipeline 内联相同逻辑 | Pipeline `postprocess` 改为调用 `paste_back()`，消除重复代码 |
| 15 | 🟢P2 | run_engine fallback config 未同步 | engine.get() 失败 fallback 到 lama，但 `ctx.inpaint_config` 仍指向原引擎 | fallback 时创建新 `InpaintConfig(engine_name="lama", mode="fallback")` |
| 16 | 🟢P2 | requirements 缺 safetensors | `use_safetensors=True` 但 requirements.txt 无 `safetensors` | 添加 `safetensors>=0.4.0` 和 `pyyaml>=6.0` |
| 17 | 🟢P2 | 测试依赖真实模型 | `test_engine_manager` 直接下载 LaMa，CI 环境失败 | 添加 `SKIP_MODEL_TESTS` 环境变量检查 |
| 19 | 🟢P2 | auto_mode 默认 complexity | 默认值 1.0，`complexity < 0.3` 永远为假 | 改为 0.5，正常触发所有分支 |
| 20 | 🟢P2 | resize_for_sdxl 缺注释 | 无文档说明"输入必须为硬边 mask" | 添加 docstring 说明隐性约束 |

### v5 → v6（本轮修复）

| # | 严重度 | 问题 | v5（旧） | v6（当前） |
|---|--------|------|---------|-----------|
| 1 | 🔴P0 | Pillow 版本冲突 | `simple-lama-inpainting` 锁死 `pillow<10.0.0`，与 Gradio 5.x `>=10.0.0` 冲突，`pip install` 直接失败 | 改用 `torch.hub.load("advimman/lama", "big_lama")`，完全移除 `simple-lama-inpainting` 依赖 |
| 3 | 🔴P0 | ImageEditor mask 失效 | `layers=False` 导致画笔内容不可见，mask 永远为空 | `layers=True` + 防御性 mask 提取（RGBA alpha/L/灰度/fallback 空 mask） |
| 4 | 🔴P0 | Logger 不输出 | `setup_logger("lama-cleaner-plusplus")` 创建命名 logger，但无 handler 配置 | 改为配置 root logger，所有模块日志统一输出到 stdout |
| 6 | 🔴P0 | Undo 状态分裂 | `undo_btn.click` 只清空 output，mask_preview 残留旧结果 | `on_undo()` 返回 `(result, None)` 同时清空 output 和 mask_preview |
| 7 | 🟡P1 | Fast Path 覆盖 HQ | `max(roi_w, roi_h) < 128` 无条件走 LaMa，HQ 用户被强制降级 | 加 `and ctx.mode != "hq"` 检查，HQ 模式始终走 SDXL |
| 8 | 🟡P1 | overlay RGBA 不安全 | `image` 为 RGBA 时 `[:, :, :3]` 广播破坏 alpha | 入口加 `if image.shape[2] == 4: image = image[:, :, :3]` guard |
| 9 | 🟡P1 | ipc_collect 无保护 | `torch.cuda.ipc_collect()` 旧版 PyTorch 无此方法 | 加 `hasattr(torch.cuda, "ipc_collect")` 保护 |
| 11 | 🟡P1 | 测试导入路径 | 测试文件直接 `from core.xxx import`，运行目录不对时 ImportError | 文件顶部加 `sys.path.insert(0, ...)` 确保项目根目录在 path 中 |
| 14 | 🟡P1 | paste_back alpha 缩放 | `cv2.resize(INTER_LINEAR)` 与 PIL resize 不一致 | 改用 `Image.fromarray().resize(Image.LANCZOS)` 统一缩放管线 |
| 15 | 🟡P1 | mask_ratio 阈值过严 | `mask_ratio < 0.001` 对大图误判（4K 图 0.1% = 数千像素） | 双重判断：`mask_ratio < 0.001 AND mask_area < 256`，大图小 mask 不误跳 |
| 16 | 🟡P1 | SAM 包名错误 | `sam-2>=2.1.0`（PyPI 无此包） | 改为 `pip install git+https://github.com/facebookresearch/sam2.git` |
| 22 | 🟡P1 | negative_prompt 无 UI | 高级参数面板无 negative_prompt 输入框 | 在"高级参数"Accordion 中添加 `negative_prompt` Textbox |
| 5 | 🟢P2 | numpy 2.x 兼容性 | `numpy>=1.24.0` 无上限 | 改为 `numpy>=1.24.0,<2.0.0`（numpy 2.x 有 C ABI 破坏性变更） |
| 12 | 🟢P2 | LANCZOS 废弃风险 | 直接用 `Image.LANCZOS` 无说明 | 添加注释说明 Pillow 10+ 推荐 `Image.Resampling.LANCZOS`，当前别名仍可用 |
| 21 | 🟢P2 | 缺 .gitignore | 无 .gitignore，模型文件/缓存可能误提交 | 添加 .gitignore（`__pycache__/`、`models/`、`.venv/`、`*.ckpt` 等） |

### v6 → v7（本轮修复）

| # | 严重度 | 问题 | v6（旧） | v7（当前） |
|---|--------|------|---------|-----------|
| 1 | 🔴P0 | SDXL 模型 ID 错误 | `stabilityai/stable-diffusion-xl-base-1.0`（文生图基础模型，UNet 通道不匹配） | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`（Inpainting 专用模型）+ fp16 variant fallback |
| 2 | 🔴P0 | crop_to_roi 覆盖 padding | `crop_to_roi` 内部 `if mask_ratio is not None` 重新计算 padding（无 density），覆盖调用方已算好的带 density 修正的 `ctx.padding_ratio` | 删除 `mask_ratio` 参数，`crop_to_roi` 只接受 `padding_ratio`，完全由调用方控制 |
| 3 | 🔴P0 | SDXL mask_blur 未禁用 | pipe 调用无 `mask_blur` 参数（diffusers 默认 4px 高斯模糊），覆盖自定义 feather | 显式传入 `mask_blur=0`，禁用内部模糊，feather 完全由外部控制 |
| 4 | 🟡P1 | LaMa 对齐裁切边缘 | `image.crop((0, 0, w - pw, h - ph))` 切掉右/下边缘像素，mask 可能落在被裁区域 | 改用 `Image.new()` + `paste()` 填充到 8 的倍数，保留所有边缘信息 |
| 5 | 🟡P1 | LaMa torch.hub 缺依赖 | requirements.txt 无 `kornia`、`omegaconf`、`albumentations`、`pytorch-lightning`，首次运行崩溃 | 补充 LaMa torch.hub 运行时依赖 |
| 6 | 🟡P1 | UI 缺 CPU 模式 | `gr.Radio(["auto", "quick", "hq"])`，无法手动强制 LaMa | 增加 `"cpu"` 选项，`select_config` 处理 `mode == "cpu"` → `engine = "lama"` |
| 7 | 🟢P2 | 极小 mask 空 prompt 危险 | `mask_ratio < 0.02` 返回空 prompt，SDXL 空 prompt 产生随机崩坏 | 改为 `"same texture, seamless blend, preserve details"` 保守引导 + 降低 strength（quick 0.65/hq 0.6） |
| 8 | 🟢P2 | paste_back 无防御性 resize | 无条件 `result.resize(target)`，引入微小插值损失 | 加 `if result.size != (target_w, target_h)` 判断，尺寸一致时跳过 resize |
| 9 | 🟢P2 | 缺 Pipeline mock 测试 | 测试仅覆盖工具函数，无 Pipeline 阶段测试 | 新增 `test_pipeline_preprocess`、`test_pipeline_fast_path`、`test_pipeline_oom_fallback` 三个 mock 测试 |

### v7 → v8（2026-05-04，深度复查 + 多用户隔离 + 细线条识别）

| # | 严重度 | 问题 | v7（旧） | v8（当前） |
|---|--------|------|---------|-----------|
| 1 | 🔴P0 | mask_blur 参数不存在 | sdxl.py 传入 `mask_blur=0`，但 `StableDiffusionXLInpaintPipeline` 无此参数，运行会 TypeError | 移除 `mask_blur=0`，传入硬边 mask 即可 |
| 2 | 🔴P0 | pipeline.py crop_to_roi 多余参数 | 调用处仍传 `mask_ratio=ctx.mask_ratio`，但函数签名已无此参数，TypeError | 移除调用处的 `mask_ratio` 参数 |
| 3 | 🟡P1 | 多用户共享历史栈 | `InpaintService._history` 类属性，所有 session 混用 | 改用 `gr.State([])` 按 session 隔离，`process()`/`undo()` 接受/返回 history 参数 |
| 4 | 🟡P1 | _get_prompt 缩进错误 | `return "same texture..."` 与 `if` 同级，Python 语法错误 | 修正缩进，确保在 `if` 块内 |
| 5 | 🟡P1 | compute_complexity 误判细线条 | 仅 density + edge_ratio，1px 线条两项都极低被误判为"简单" | 增加 elongation 因子（轮廓周长²/面积比），权重 0.35:0.35:0.3 |
| 6 | 🟡P1 | EngineManager 锁内加载 | `engine.load()` 在 `with self._lock` 内，加载几十秒阻塞所有并发 | 双检锁：快速路径无锁读，创建实例加锁，`load()` 放锁外 |
| 7 | 🟡P1 | 缺 HF 登录指引 | 无说明，用户首次运行报 401 | requirements 说明后增加 `huggingface-cli login` 步骤 |
| 8 | 🟢P2 | 性能预估偏低 | HQ 512x512 标注 6-8s，实际 SDXL 推理 15-30s | 调整为实际测量值，增加体验建议 |
| 9 | 🟢P2 | Undo 限制未说明 | 无文档说明 Undo 无法恢复 ImageEditor 状态 | 设计文档和实施文档均增加 Undo 限制说明 |

**v8 汇总**：4 处 P0 运行时错误修复（mask_blur TypeError、crop_to_roi TypeError）；5 处 P1 逻辑/架构改进（多用户隔离、prompt 缩进、细线条识别、双检锁、HF 登录）；2 处 P2 文档准确性提升。v7 中 mask_blur=0 的"修复"实际引入了新 bug（该参数不存在），v8 彻底移除。

### v8 → v9（2026-05-04，架构级复查 + 致命隐藏漏洞修复）

| # | 严重度 | 问题 | v8（旧） | v9（当前） |
|---|--------|------|---------|-----------|
| 1 | 🔴P0 | SDXL scheduler+VRAM 优化在 except 块内 | fp16 正常加载时 scheduler 替换和 VRAM 优化不执行，推理不稳定+OOM | 改为嵌套 try/except，scheduler 和优化在内层 except 之后**无条件执行** |
| 2 | 🔴P0 | LaMa 推理结果未裁剪回原始尺寸 | 8 倍数 padding 后结果尺寸大于原图，`paste_back` 边缘错位 | 返回前 `out.crop((0, 0, w, h))` 裁剪回原始尺寸 |
| 3 | 🔴P0 | resize_for_sdxl 后 feather 空间错位 | SDXL result resize 回 ROI 尺寸后，`roi_feathered_alpha` 可能因 resize 前后尺寸不同而不对齐 | `_run_sdxl` 中检测尺寸变化，同步缩放 `roi_feathered_alpha`（BICUBIC） |
| 4 | 🟡P1 | EngineManager 双检锁导致并发加载 | `load()` 在锁外，多线程可重复加载同一引擎，显存飙升 | 还原为全锁：创建+加载均在 `with self._lock` 内，阻塞可接受（引擎加载本就重量级） |
| 5 | 🟡P1 | `expand_px=0` 被 `or` 覆盖 | `expand_px or default` 中 0 被视为 falsy，回退到默认值 | 改为 `expand_px if expand_px is not None else default` |
| 6 | 🟡P1 | VRAM 判断过于乐观 | `get_available_vram_gb()` 直接用 `free`，碎片化场景 OOM | 新增 `get_safe_available_vram_gb(reserve_gb=1.0, fraction=0.7)`，策略改用保守值 |
| 7 | 🟡P1 | paste_back alpha 缩放振铃 | alpha 用 `LANCZOS` 缩放，灰度 mask 边缘出现振铃伪影 | 改用 `BICUBIC`（平衡锐利与平滑） |
| 8 | 🟡P1 | 测试文件完全过时 | 导入 `Pipeline`/`PipelineContext`（不存在）、`svc._history`（已移除）、`svc.undo()`（签名已变） | 完全重写测试，对齐 `InpaintService`/`InpaintPipeline` 当前接口 |
| 9 | 🟡P1 | CLI 优先级反直觉 | `os.getenv("LAMA_PORT", args.port)` 环境变量覆盖 CLI 参数 | 改为 CLI 参数优先：`args.port or os.getenv(...)`，argparse default 改 `None` |
| 10 | 🟢P2 | 缺少 `huggingface_hub` 依赖 | requirements.txt 未列出 | 追加 `huggingface_hub>=0.20.0` |
| 11 | 🟡P1 | strategy.py 导入缺失 | `from utils.gpu import get_available_vram_gb` 未导入 `get_safe_available_vram_gb`，运行时 ImportError | 两处 import 均补充 `get_safe_available_vram_gb` |

**v9 汇总**：3 处 P0 致命运行时错误修复（SDXL scheduler 不生效、LaMa 尺寸错位、feather 空间错位）；7 处 P1 稳定性/正确性改进（并发加载、expand_px=0、VRAM 保守估算、alpha 振铃、测试重写、CLI 优先级、strategy.py 导入补全）；1 处 P2 依赖补全。v8 的双检锁优化在实践中因并发加载风险被回退。

### v9 → v10（2026-05-04，P0 漏网修复）

| # | 严重度 | 问题 | v9（旧） | v10（当前） |
|---|--------|------|---------|-----------|
| 1 | 🔴P0 | SDXL 内层 except 捕获过宽 | `except Exception:` 导致 OOM 时 fallback 再次 OOM 连续崩溃 | 改为 `except (OSError, ValueError):`，仅捕获 variant 不存在/配置错误，OOM 直接上抛 |
| 2 | 🔴P0 | compute_complexity 缺 elongation | v8 changelog 声称已加但代码仍为 `0.5*density+0.5*edge_ratio` | 代码实际补全：cv2.findContours 提取轮廓，`elongation = perimeter²/area`，权重 0.35:0.35:0.30 |

**v10 汇总**：2 处 P0 代码级修复。SDXL 异常捕获收窄防止 OOM 连续崩溃；compute_complexity elongation 因子代码落地（v8 仅在 changelog 中声明但未实际修改代码）。

### v10 → v11（2026-05-05，工程级总结修复 — 数值统一 + 回退稳定性 + 输入可靠性）

基于工程级总总结，将所有问题按"严重度 + 本质原因 + 修复方案"统一落地到代码。

| # | 严重度 | 问题 | v10（旧） | v11（当前） |
|---|--------|------|---------|-----------|
| 1 | 🔴P0 | `auto_mode` 的 elongation 阈值恒成立 | `if elongation > 0.8:` 但 `elongation = 周长²/面积 ≥ 12.56`，条件永远为真 → auto_mode 永远 "hq" | `compute_elongation()` 返回原始值（几何量）；`auto_mode` 阈值改为 `> 40`；`compute_complexity()` 内 `min(elongation / 50.0, 1.0)` 归一化 |
| 2 | 🔴P0 | OOM fallback 后 SDXL 未卸载 | `except RuntimeError: fallback_to_lama()` 但没有 `engine_manager.unload("sdxl")`，SDXL 仍占显存 → 下一次请求继续 OOM | `_fallback_to_lama()` 第一步就 `self.engine_manager.unload(ctx.inpaint_config.engine_name)` + `gc.collect()` + `torch.cuda.empty_cache()` + `ipc_collect()` + `reset_peak_memory_stats()` |
| 3 | 🔴P0 | LaMa fallback 仍可能占用 GPU | `self._device = "cuda" if torch.cuda.is_available() else "cpu"`，OOM 场景下 LaMa 再次申请显存 → fallback 失败 | `LamaEngine(force_cpu=False)` 构造函数 + `load(force_cpu=False)` 参数；`EngineManager.get(name, force_cpu=False)` 统一调用（删除 `if name == "lama"` 特殊分支）；`BaseEngine.load(force_cpu=False)` 统一签名；OOM fallback 先 `unload("lama")` 再 `get("lama", force_cpu=True)` 防止缓存命中 |
| 4 | 🟡P1 | `auto_mode` 未使用 complexity | 只判 `elongation > 0.8` 和 `mask_ratio < 0.03`，完全忽略 complexity → 多碎片/噪声 mask 被误判为 quick | `auto_mode(mask_ratio, elongation, complexity)` 新增 `if complexity > 0.5: return "hq"` 分支；`select_config()` 签名新增 `complexity` 参数并透传给 `auto_mode()` |
| 5 | 🟡P1 | Gradio mask 提取不完整 | 只读 `editor_data["layers"]`，无 composite fallback，无多层 merge → 用户画的 mask 可能被识别为空 | `_extract_mask_from_editor()` 独立函数：逐层 RGBA alpha / L 灰度提取 → 多层 `np.maximum` 合并 → composite RGBA/RGB 双分支差分回退（修复缩进+变量作用域 bug）→ 最终 fallback 空 mask |
| 6 | 🟡P1 | bbox area 计算 off-by-one | `bbox_area = (rmax - rmin) * (cmax - cmin)` 少算 1 像素边界 → density/padding/complexity 偏差 | `mask_h = rmax - rmin + 1`、`mask_w = cmax - cmin + 1`；Pipeline 内 bbox_area 同步修复；min_size 扩展整数舍入偏差同步修复（extra_left/extra_right 精确分配余数） |
| 7 | 🟡P1 | feather 与 resize 对齐逻辑不透明 | v9 changelog 说有对齐，实际只在 `paste_back` 兜底 | 方案 B 落地：`_run_sdxl` 中检测 `resize_for_sdxl` 前后尺寸变化时，用 BICUBIC 同步缩放 `roi_feathered_alpha` |
| 8 | 🟡P1 | negative_prompt 未分级 | 统一强 negative，小 mask 可能过度约束 | `get_negative_prompt(mask_ratio, default_negative, complexity)` 三级分级 + complexity 增强；小 mask 加 `text/watermark/logo/symbol/letters/characters/noise/grain`；高 complexity 加 `jagged edges/broken structure/inconsistent lighting` |
| 9 | 🟡P1 | resize_for_sdxl 下限 512 过高 | `min_size = 512`，小 ROI 被强制放大 → 模糊 | `ROIConfig.min_size` 改为 `256`；`resize_for_sdxl` 下限 `max(w, 256)` |
| 10 | 🟡P1 | Pipeline 无并发保护 | EngineManager 有锁但 pipeline 没有 → 多请求竞争 load/unload 随机崩溃 | `InpaintPipeline.__init__` 增加 `self._run_lock = threading.Lock()`；`run()` 用 `with self._run_lock:` 包裹核心逻辑 |
| 11 | 🟢P2 | 缺少 fast path 测试 | 无测试验证 `ROI < 128 → LaMa` 路径 | 新增 `test_pipeline_fast_path()` 测试 PipelineContext 结构完整性 |
| 12 | 🟢P2 | utils/image.py 未实现 | 设计存在，代码缺失 | 实现 `ensure_rgb()` / `to_pil()` / `resize_to_multiple()` |
| 13 | 🟢P2 | SKIP_MODEL_TESTS 未覆盖全部测试 | 部分测试在无 diffusers 环境崩溃 | 5 个依赖 diffusers 的测试均加 `os.environ.get("SKIP_MODEL_TESTS")` 跳过保护 |

**v11 汇总**：3 处 P0 致命逻辑/稳定性修复（elongation 返回原始值推翻归一化混用、OOM 后 SDXL 卸载防止连环 OOM、LaMa 强制 CPU 防止 fallback 失败）；7 处 P1 决策/可靠性/输入链路修复（auto_mode 加 complexity + select_config 透传、Gradio mask 完整提取 + composite 作用域修复、bbox+1、feather 对齐、negative_prompt 三级+complexity 增强、resize 下限 256、Pipeline 并发保护）；3 处 P2 工程质量补全。**本轮修复覆盖了工程级总结中所有 13 项问题，核心主题是数值体系统一 + 三因子决策链路闭合 + 回退路径稳固 + 输入链路可靠。**

### v11 → v12（2026-05-06，代码审查修复）

| # | 严重度 | 问题 | v11（旧） | v12（当前） |
|---|--------|------|---------|-----------|
| 1 | 🔴P0 | resize_to_multiple 填充色不兼容 L 模式 | `Image.new(image.mode, size, (0,0,0))`，L 模式在 Pillow 10+ 抛 TypeError | `fill = 0 if image.mode in ("L", "1") else (0, 0, 0)` |
| 2 | 🟡P1 | select_engine 缺少 SAM 卸载检查 | 代码遗漏 SAM 卸载逻辑 | 补全 `is_loaded("sam")` → `unload("sam")` + `empty_cache()` |
| 3 | 🟡P1 | negative_tiny 缺少关键词 | 只有基础质量词 | 补全 `text/watermark/logo/symbol/letters/characters/noise/grain` |
| 4 | 🟡P1 | on_run 缺少异常处理 | `service.process()` 无 try/except | 包裹 try/except，分别处理 RuntimeError 和 Exception |
| 5 | 🟡P1 | SDXLEngine 不重置峰值内存统计 | 日志 VRAM 峰值为累计值 | 推理前 `reset_peak_memory_stats()` |
| 6 | 🟢P2 | select_config 有未使用的 seed 参数 | 死代码 | 移除 seed 参数 |
| 7 | 🟢P2 | compute_mask_ratio 阈值不一致 | `expanded > 0` | 统一为 `expanded > 128` |
| 8 | 🟢P2 | test_mask_complexity 断言错误 | 碎片在 1000×1000 密度极低 | 改为 200×200 紧凑布局 |

**v12 汇总**：1 处 P0 Pillow 兼容性修复；4 处 P1 正确性/稳定性修复；3 处 P2 代码质量修复。核心主题是**Pillow 10+ 兼容性 + 文档与代码一致性 + 异常处理健壮性**。

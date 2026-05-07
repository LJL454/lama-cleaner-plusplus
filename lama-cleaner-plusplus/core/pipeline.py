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
    mode: str = "auto"
    mask_ratio: float = 0.0
    complexity: float = 0.0
    elongation: float = 0.0
    expanded: np.ndarray | None = None
    roi_image: Image.Image | None = None
    roi_mask: Image.Image | None = None
    roi_feathered_alpha: np.ndarray | None = None
    roi_box: tuple[int, int, int, int] | None = None
    padding_ratio: float = 0.3
    inpaint_config: InpaintConfig | None = None
    inpainted_result: Image.Image | None = None
    skip: bool = False


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
        t0 = time.time()
        mask_np = np.array(ctx.mask.convert("L"))

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

        expanded = MaskProcessor.expand(mask_np, ctx.expand_px)
        ctx.expanded = expanded
        ctx.mask_ratio = MaskProcessor.compute_mask_ratio(expanded)

        mask_area = int(np.sum(expanded > 128))
        if mask_area == 0:
            logger.warning("Empty mask (0 pixels), skip pipeline")
            ctx.skip = True
            return ctx

        if ctx.mask_ratio < 0.001 and mask_area < 256:
            logger.info(f"Tiny mask (ratio={ctx.mask_ratio:.6f}, area={mask_area}px), skip pipeline")
            ctx.skip = True
            return ctx

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

        roi_mask_np = np.array(roi.mask.convert("L"))

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
        return engine.inpaint(image=ctx.roi_image, mask=ctx.roi_mask)

    def _fallback_to_lama(self, ctx: PipelineContext, original_error) -> PipelineContext:
        import gc

        logger.warning(f"OOM on {ctx.inpaint_config.engine_name}, falling back to LaMa: {original_error}")

        self.engine_manager.unload(ctx.inpaint_config.engine_name)
        self.engine_manager.unload("lama")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()

        lama = self.engine_manager.get("lama", force_cpu=True)
        ctx.inpainted_result = lama.inpaint(image=ctx.roi_image, mask=ctx.roi_mask)

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
        if user_prompt:
            return user_prompt
        if mask_ratio < 0.02:
            return "same texture, seamless blend, preserve details, no new objects"
        if complexity > 0.5:
            return "preserve structure, seamless continuation, no new objects"
        if mask_ratio < 0.05 and complexity < 0.3:
            return "clean smooth background, seamless, no new objects"
        return self.config.sdxl.default_prompt

    def postprocess(self, ctx: PipelineContext) -> Image.Image:
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

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
        self._esrgan = None

    def _get_esrgan(self, release_inpaint: bool = True):
        if release_inpaint:
            self.pipeline.unload_all()
        if self._esrgan is None:
            from core.engines.realesrgan import RealESRGANEngine
            self._esrgan = RealESRGANEngine()
        if not self._esrgan.is_loaded():
            self._esrgan.load()
        return self._esrgan

    def _release_esrgan(self):
        if self._esrgan is not None:
            self._esrgan.unload()
            self._esrgan = None

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
        upscale: str = "none",
    ) -> tuple[Image.Image, Image.Image, list]:
        history = (history or []).copy()
        history.append(image.copy())
        if len(history) > 10:
            history = history[-10:]

        if upscale == "only":
            if status_callback:
                status_callback("upscaling...")
            result = self._get_esrgan(release_inpaint=True).upscale(image)
            mask_np = np.array(mask.convert("L"))
            expanded = MaskProcessor.expand(mask_np, expand_px if expand_px is not None else self.config.mask.expand_pixels)
            preview = Image.fromarray(
                MaskProcessor.overlay_on_image(np.array(image), expanded)
            )
            logger.info(f"Service upscale-only done, {image.size} -> {result.size}")
            return result, preview, history

        if upscale == "input":
            if status_callback:
                status_callback("upscaling input...")
            image = self._get_esrgan(release_inpaint=True).upscale(image)
            mask = mask.resize(image.size, Image.NEAREST)
            self._release_esrgan()

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

        if upscale == "output":
            if status_callback:
                status_callback("upscaling output...")
            self.pipeline.unload_all()
            result = self._get_esrgan(release_inpaint=False).upscale(result)
            self._release_esrgan()

        mask_np = np.array(mask.convert("L"))
        expanded = MaskProcessor.expand(mask_np, expand_px if expand_px is not None else self.config.mask.expand_pixels)
        preview = Image.fromarray(
            MaskProcessor.overlay_on_image(np.array(image), expanded)
        )
        logger.info(f"Service process done, history: {len(history)}, upscale: {upscale}")
        return result, preview, history

    def undo(self, history: list) -> tuple[Image.Image | None, list]:
        history = history.copy()
        if history:
            return history.pop(), history
        return None, history

    def cleanup(self):
        self.pipeline.unload_all()
        self._release_esrgan()

import torch
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionXLInpaintPipeline, EulerDiscreteScheduler
from .base import BaseEngine
from utils.logger import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_SDXL_MODEL = str(_PROJECT_ROOT / "models" / "sdxl-inpainting")


class SDXLEngine(BaseEngine):
    def __init__(
        self,
        model_id: str = _DEFAULT_SDXL_MODEL,
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

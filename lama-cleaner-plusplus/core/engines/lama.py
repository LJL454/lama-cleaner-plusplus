import os
import time
import socket
from PIL import Image
from .base import BaseEngine
from utils.logger import get_logger

logger = get_logger(__name__)


class LamaEngine(BaseEngine):
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

    def _load_from_state_dict(self, path: str):
        import torch
        from modelscope.models.cv.image_inpainting.base import BaseInpaintingTrainingModule

        logger.info(f"Loading LaMa model from state dict: {path}")
        training_module = BaseInpaintingTrainingModule(predict_only=True)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        gen_keys = {k: v for k, v in ckpt.items() if k.startswith("generator.")}
        training_module.load_state_dict(gen_keys, strict=False)
        training_module.eval()
        self._model = training_module.generator
        del ckpt, gen_keys

    def _load_from_torchscript(self, path: str):
        import torch
        import io

        logger.info(f"Loading LaMa model from TorchScript: {path}")
        with open(path, "rb") as f:
            buf = io.BytesIO(f.read())
        self._model = torch.jit.load(buf, map_location="cpu")

    def _load_from_hub(self):
        import torch

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
                    raise RuntimeError(
                        f"Failed to load LaMa model after {max_retries} attempts. "
                        f"Check network or set local_model_path in config. "
                        f"Last error: {e}"
                    ) from e
        socket.setdefaulttimeout(old_timeout)

    def load(self, force_cpu: bool = False) -> None:
        if self._loaded:
            return
        import torch

        if self._local_model_path and os.path.exists(self._local_model_path):
            try:
                self._load_from_state_dict(self._local_model_path)
            except Exception:
                logger.info("State dict loading failed, trying TorchScript format...")
                try:
                    self._load_from_torchscript(self._local_model_path)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load local LaMa model from {self._local_model_path}: {e}"
                    ) from e
        else:
            self._load_from_hub()

        self._model.eval()
        use_cpu = self._force_cpu or force_cpu or not torch.cuda.is_available()
        self._device = "cpu" if use_cpu else "cuda"
        self._model.to(self._device)
        self._loaded = True
        logger.info(f"LaMa model loaded on {self._device}")
        if self._device == "cuda":
            vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            logger.info(f"LaMa VRAM usage: {vram_mb:.1f} MB on {torch.cuda.get_device_name(0)}")

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
            masked_img = img_dev * (1 - mask_dev)
            inp = torch.cat([masked_img, mask_dev], dim=1)
            raw_result = self._model(inp)

        if isinstance(raw_result, dict):
            result_tensor = raw_result.get("predicted_image") or raw_result.get("inpainted") or raw_result.get("image")
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

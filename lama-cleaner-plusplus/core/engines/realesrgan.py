import os
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
from .base import BaseEngine
from utils.logger import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_REALESRGAN_PATH = str(_PROJECT_ROOT / "models" / "realesrgan" / "RealESRGAN_x4plus.pth")


def _gaussian_window(h: int, w: int, sigma: float = 0.25) -> "np.ndarray":
    import numpy as np
    y = np.linspace(0, 1, h)
    x = np.linspace(0, 1, w)
    gy = np.exp(-((y - 0.5) ** 2) / (2 * sigma ** 2))
    gx = np.exp(-((x - 0.5) ** 2) / (2 * sigma ** 2))
    window = gy[:, None] * gx[None, :]
    window = np.stack([window] * 3, axis=-1)
    return window.astype(np.float32)


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.conv_body(self.body(fea))
        fea = fea + trunk
        fea = self.lrelu(self.conv_up1(self.up1(fea)))
        fea = self.lrelu(self.conv_up2(self.up2(fea)))
        fea = self.lrelu(self.conv_hr(fea))
        out = self.conv_last(fea)
        return out


class RealESRGANEngine(BaseEngine):
    def __init__(self, model_path: str = _DEFAULT_REALESRGAN_PATH):
        self._model_path = model_path
        self._model = None
        self._loaded = False

    def load(self, force_cpu: bool = False) -> None:
        if self._loaded:
            return

        logger.info(f"Loading RealESRGAN model: {self._model_path}")
        self._model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4)

        ckpt = torch.load(self._model_path, map_location="cpu", weights_only=False)
        if "params_ema" in ckpt:
            self._model.load_state_dict(ckpt["params_ema"], strict=True)
        elif "params" in ckpt:
            self._model.load_state_dict(ckpt["params"], strict=True)
        else:
            self._model.load_state_dict(ckpt, strict=True)
        del ckpt

        use_cpu = force_cpu or not torch.cuda.is_available()
        self._device = "cpu" if use_cpu else "cuda"
        self._model.to(self._device)
        self._model.eval()
        self._loaded = True
        logger.info(f"RealESRGAN model loaded on {self._device}")

    def unload(self) -> None:
        if self._model is not None:
            del self._model
        self._model = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        return self._loaded

    def upscale(self, image: Image.Image, tile_size: int = 512) -> Image.Image:
        if not self._loaded:
            self.load()

        import numpy as np

        w, h = image.size
        need_tile = (h * w > tile_size * tile_size) and self._device == "cuda"

        if not need_tile:
            return self._upscale_whole(image)

        logger.info(f"RealESRGAN tiling mode: {image.size}, tile={tile_size}")
        scale = self._model.scale
        overlap = tile_size // 4

        out_w, out_h = w * scale, h * scale
        result_np = np.zeros((out_h, out_w, 3), dtype=np.float32)
        weight_np = np.zeros((out_h, out_w, 3), dtype=np.float32)

        window = _gaussian_window(tile_size * scale, tile_size * scale)

        step = tile_size - overlap
        y = 0
        while y < h:
            x = 0
            while x < w:
                tile_h = min(tile_size, h - y)
                tile_w = min(tile_size, w - x)
                crop = image.crop((x, y, x + tile_w, y + tile_h))

                if tile_w < tile_size or tile_h < tile_size:
                    padded = Image.new(image.mode, (tile_size, tile_size), (0, 0, 0))
                    padded.paste(crop, (0, 0))
                    tile_result = self._upscale_tensor(padded)
                    tile_result = tile_result[:tile_h * scale, :tile_w * scale]
                else:
                    tile_result = self._upscale_tensor(crop)

                win = window[:tile_h * scale, :tile_w * scale]

                oy, ox = y * scale, x * scale
                th, tw = tile_result.shape[:2]
                result_np[oy:oy + th, ox:ox + tw] += tile_result * win
                weight_np[oy:oy + th, ox:ox + tw] += win

                x += step
            y += step

        result_np = np.clip(result_np / np.maximum(weight_np, 1e-8), 0, 255).astype(np.uint8)
        result = Image.fromarray(result_np)
        logger.info(f"RealESRGAN upscale (tiled): {image.size} -> {result.size}")
        return result

    def _upscale_whole(self, image: Image.Image) -> Image.Image:
        import numpy as np

        img_np = np.array(image).astype(np.float32) / 255.0
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(self._device)

        h, w = img_tensor.shape[2], img_tensor.shape[3]
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        if pad_h or pad_w:
            img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode="reflect")

        with torch.no_grad():
            output = self._model(img_tensor)

        output = output[:, :, :h * 4, :w * 4]
        out_np = output[0].permute(1, 2, 0).cpu().numpy()
        out_np = (out_np * 255.0).clip(0, 255).astype(np.uint8)
        result = Image.fromarray(out_np)
        logger.info(f"RealESRGAN upscale: {image.size} -> {result.size}")
        return result

    def _upscale_tensor(self, image: Image.Image) -> "np.ndarray":
        import numpy as np

        img_np = np.array(image).astype(np.float32) / 255.0
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(self._device)

        h, w = img_tensor.shape[2], img_tensor.shape[3]
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        if pad_h or pad_w:
            img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode="reflect")

        with torch.no_grad():
            output = self._model(img_tensor)

        output = output[:, :, :h * 4, :w * 4]
        out_np = output[0].permute(1, 2, 0).cpu().numpy() * 255.0
        torch.cuda.empty_cache()
        return out_np

    def inpaint(self, image: Image.Image, mask: Image.Image, **kwargs) -> Image.Image:
        return self.upscale(image)

    @property
    def name(self) -> str:
        return "RealESRGAN"

    @property
    def min_vram_gb(self) -> float:
        return 1.0

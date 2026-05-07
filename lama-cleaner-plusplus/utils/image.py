import numpy as np
from PIL import Image


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, :3]
    return image


def to_pil(image: np.ndarray) -> Image.Image:
    if image.ndim == 2:
        return Image.fromarray(image, mode="L")
    return Image.fromarray(image, mode="RGB")


def resize_to_multiple(image: Image.Image, multiple: int = 8) -> tuple[Image.Image, int, int]:
    w, h = image.size
    pw = (multiple - w % multiple) % multiple
    ph = (multiple - h % multiple) % multiple
    if pw or ph:
        fill = 0 if image.mode in ("L", "1") else (0, 0, 0)
        new_img = Image.new(image.mode, (w + pw, h + ph), fill)
        new_img.paste(image, (0, 0))
        return new_img, pw, ph
    return image, 0, 0

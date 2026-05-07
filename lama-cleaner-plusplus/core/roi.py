import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass
from utils.logger import get_logger
from utils.image import resize_to_multiple

logger = get_logger(__name__)


@dataclass
class ROICrop:
    image: Image.Image
    mask: Image.Image
    box: tuple[int, int, int, int]
    original_size: tuple[int, int]


def compute_padding_ratio(mask_ratio: float, density: float = 1.0) -> float:
    if mask_ratio < 0.02:
        base = 0.5
    elif mask_ratio < 0.1:
        base = 0.3
    else:
        base = 0.15

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
        extra_left = (min_size - roi_w) // 2
        extra_right = (min_size - roi_w) - extra_left
        x1 = max(0, x1 - extra_left)
        x2 = min(image.width, x2 + extra_right)
        if x2 - x1 < min_size:
            if x1 == 0:
                x2 = min(image.width, x1 + min_size)
            elif x2 == image.width:
                x1 = max(0, x2 - min_size)
    if roi_h < min_size:
        extra_top = (min_size - roi_h) // 2
        extra_bottom = (min_size - roi_h) - extra_top
        y1 = max(0, y1 - extra_top)
        y2 = min(image.height, y2 + extra_bottom)
        if y2 - y1 < min_size:
            if y1 == 0:
                y2 = min(image.height, y1 + min_size)
            elif y2 == image.height:
                y1 = max(0, y2 - min_size)

    actual_w = x2 - x1
    actual_h = y2 - y1
    if actual_w < min_size or actual_h < min_size:
        logger.debug(f"ROI hit image boundary: target={min_size}px, actual={actual_w}x{actual_h}")

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
    w, h = image.size
    aspect = w / h

    if aspect >= 1:
        tw = min(max(w, 256), max_size)
        th = max(int(tw / aspect), 1)
    else:
        th = min(max(h, 256), max_size)
        tw = max(int(th * aspect), 1)

    resized_img = image.resize((tw, th), Image.LANCZOS)
    resized_mask = mask.resize((tw, th), Image.BILINEAR)
    resized_mask = resized_mask.point(lambda x: 255 if x > 128 else 0)

    resized_img, pw, ph = resize_to_multiple(resized_img, multiple=8)
    if pw or ph:
        resized_mask, _, _ = resize_to_multiple(resized_mask, multiple=8)

    final_w, final_h = resized_img.size
    logger.debug(f"resize_for_sdxl: {image.size} → ({final_w}, {final_h})")
    return resized_img, resized_mask, (final_w, final_h)


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

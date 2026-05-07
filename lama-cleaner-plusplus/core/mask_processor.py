import cv2
import numpy as np
from PIL import Image
from utils.logger import get_logger
from utils.image import ensure_rgb

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
            raw_elongation = (perimeter ** 2) / area
        else:
            raw_elongation = 0.0

        elongation = min(raw_elongation / 50.0, 1.0)

        score = (
            0.35 * density +
            0.35 * edge_ratio +
            0.30 * elongation
        )
        return min(score, 1.0)

    @staticmethod
    def compute_elongation(roi_mask_np: np.ndarray) -> float:
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

from dataclasses import dataclass
from config import SDXLConfig
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
    if mask_ratio < 0.15 and complexity < 0.5:
        return "remove"
    if elongation > 40:
        return "hq"
    if complexity > 0.5:
        return "hq"
    if mask_ratio < 0.03:
        return "remove"
    if mask_ratio < 0.10:
        return "remove"
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

    if mode == "remove":
        engine = "lama"
    elif mode == "cpu":
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
            mode=mode,
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

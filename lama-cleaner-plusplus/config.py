import os
import argparse
from dataclasses import dataclass, field
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
    model_id: str = str(_PROJECT_ROOT / "models" / "sdxl-inpainting")
    steps_quick: int = 20
    steps_hq: int = 30
    guidance_scale: float = 7.5
    strength: float = 0.75
    seed: int = 42
    min_vram_gb: float = 4.0
    default_prompt: str = "empty area, clean smooth background, seamless continuation of surrounding texture"
    default_negative: str = "blurry, low quality, artifacts, distorted, text, watermark, new objects, people, faces, animals"
    negative_tiny: str = "blurry, low quality, artifacts, distorted, text, watermark, logo, symbol, letters, characters, noise, grain"
    negative_standard: str = "blurry, low quality, artifacts, distorted, text, watermark, oversaturated"
    negative_heavy: str = "blurry, distorted, artifacts, jpeg artifacts, text, watermark, noise, oversharpened"
    lama_local_path: str = str(_PROJECT_ROOT / "models" / "lama" / "big_lama.pt")
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
    print(config)

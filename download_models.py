import sys
import os
import urllib.request
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent
_MODELS_DIR = _PROJECT_ROOT / "models"

MODELS = {
    "lama": {
        "save_path": _MODELS_DIR / "lama" / "big_lama.pt",
        "description": "LaMa inpainting model (~391 MB)",
        "required": True,
    },
    "sdxl": {
        "save_path": _MODELS_DIR / "sdxl-inpainting" / "model_index.json",
        "description": "SDXL inpainting model (~5 GB, fp16 only)",
        "required": False,
    },
    "realesrgan": {
        "save_path": _MODELS_DIR / "realesrgan" / "RealESRGAN_x4plus.pth",
        "description": "RealESRGAN x4 super-resolution model (~64 MB)",
        "required": False,
    },
}


def _progress_hook(count, block_size, total_size):
    downloaded = count * block_size
    mb = downloaded / (1024 * 1024)
    total_mb = total_size / (1024 * 1024) if total_size > 0 else 0
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        print(f"\r  Progress: {mb:.1f}/{total_mb:.1f} MB ({pct}%)", end="", flush=True)
    else:
        print(f"\r  Progress: {mb:.1f} MB", end="", flush=True)


def _download_lama() -> bool:
    save_path = MODELS["lama"]["save_path"]
    if save_path.exists():
        print(f"  Already exists: {save_path}")
        return True

    url = "https://www.modelscope.cn/models/iic/cv_fft_inpainting_lama/resolve/master/pytorch_model.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading to: {save_path}")
    print()

    tmp = save_path.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_progress_hook)
        tmp.rename(save_path)
        print(f"\n  Download complete!")
        return True
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        print(f"\n  Download failed: {e}")
        print(f"  Please download manually:")
        print(f"    URL: {url}")
        print(f"    Save to: {save_path}")
        return False


def _download_sdxl() -> bool:
    save_path = MODELS["sdxl"]["save_path"]
    if save_path.exists():
        print(f"  Already exists: {save_path.parent}")
        return True

    local_dir = str(save_path.parent)
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("  modelscope not installed, installing...")
        os.system(f"{sys.executable} -m pip install modelscope")
        from modelscope import snapshot_download

    print(f"  Downloading to: {local_dir}")
    print(f"  (fp16 only, skipping FP32 weights)")
    print()

    try:
        snapshot_download(
            model_id="AI-ModelScope/stable-diffusion-xl-1.0-inpainting-0.1",
            local_dir=local_dir,
            ignore_patterns=["*.safetensors"],
            allow_patterns=[
                "*.fp16.safetensors",
                "*.json",
                "*.txt",
                "merges.txt",
                "vocab.json",
                "special_tokens_map.json",
                "tokenizer_config.json",
            ],
        )
        print(f"\n  Download complete!")
        return True
    except Exception as e:
        print(f"\n  Download failed: {e}")
        print(f"  Please download manually from:")
        print(f"    https://www.modelscope.cn/models/AI-ModelScope/stable-diffusion-xl-1.0-inpainting-0.1")
        return False


def _download_realesrgan() -> bool:
    save_path = MODELS["realesrgan"]["save_path"]
    if save_path.exists():
        print(f"  Already exists: {save_path}")
        return True

    url = "https://www.modelscope.cn/models/muse/RealESRGAN_x4plus/resolve/master/RealESRGAN_x4plus.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading to: {save_path}")
    print()

    tmp = save_path.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_progress_hook)
        tmp.rename(save_path)
        print(f"\n  Download complete!")
        return True
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        print(f"\n  Download failed: {e}")
        print(f"  Please download manually:")
        print(f"    URL: {url}")
        print(f"    Save to: {save_path}")
        return False


def download_model(name: str) -> bool:
    if name not in MODELS:
        print(f"Unknown model: {name}")
        print(f"Available: {', '.join(MODELS.keys())}")
        return False

    info = MODELS[name]
    print(f"\n{'='*50}")
    print(f"  {name}: {info['description']}")
    print(f"{'='*50}")

    if name == "lama":
        return _download_lama()
    elif name == "sdxl":
        return _download_sdxl()
    elif name == "realesrgan":
        return _download_realesrgan()
    return False


def check_models():
    print("\nModel Status:")
    print("-" * 50)
    for name, info in MODELS.items():
        exists = info["save_path"].exists()
        tag = "required" if info.get("required") else "optional"
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {name} ({tag}): {info['description']}")
    print()


def interactive_download():
    check_models()

    print("Which models to download?")
    print("  1. LaMa (required, ~391 MB)")
    print("  2. SDXL (optional, ~5 GB fp16)")
    print("  3. RealESRGAN (optional, ~64 MB)")
    print("  4. All models")
    print("  0. Exit")
    print()

    choice = input("Enter choice [0-4]: ").strip()

    if choice == "1":
        download_model("lama")
    elif choice == "2":
        download_model("sdxl")
    elif choice == "3":
        download_model("realesrgan")
    elif choice == "4":
        for name in MODELS:
            download_model(name)
    elif choice == "0":
        print("Bye!")
    else:
        print("Invalid choice.")

    print("\nFinal status:")
    check_models()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download models for Lama Cleaner++")
    parser.add_argument("model", nargs="?", default=None,
                        help="Model to download: lama, sdxl, realesrgan, all")
    parser.add_argument("--check", action="store_true", help="Only check model status")
    args = parser.parse_args()

    if args.check:
        check_models()
    elif args.model == "all":
        for name in MODELS:
            download_model(name)
    elif args.model in MODELS:
        download_model(args.model)
    else:
        interactive_download()

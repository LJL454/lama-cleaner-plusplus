import os
from pathlib import Path


def remove_fp32_weights(models_dir: str | None = None, dry_run: bool = False):
    if models_dir is None:
        models_dir = str(Path(__file__).resolve().parent / "models")

    models_path = Path(models_dir)
    if not models_path.exists():
        print(f"Models directory not found: {models_path}")
        return

    removed_size = 0
    removed_files = 0

    for fp32_file in models_path.rglob("*.safetensors"):
        name = fp32_file.name
        if ".fp16." in name:
            continue

        stem = name.replace(".safetensors", "")
        fp16_name = f"{stem}.fp16.safetensors"
        fp16_file = fp32_file.parent / fp16_name

        if fp16_file.exists():
            size_mb = fp32_file.stat().st_size / (1024 * 1024)
            action = "WOULD DELETE" if dry_run else "DELETING"
            print(f"  {action}: {fp32_file.relative_to(models_path)} ({size_mb:.1f} MB)")
            if not dry_run:
                fp32_file.unlink()
            removed_size += size_mb
            removed_files += 1
        else:
            print(f"  SKIP (no fp16 pair): {fp32_file.relative_to(models_path)}")

    verb = "Would free" if dry_run else "Freed"
    print(f"\n{verb} {removed_size:.1f} MB ({removed_files} files)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Remove FP32 model weights when fp16 variants exist")
    parser.add_argument("--models-dir", default=None, help="Path to models directory")
    parser.add_argument("--dry-run", action="store_true", help="Preview without deleting")
    args = parser.parse_args()

    if args.dry_run:
        print("=== DRY RUN (no files will be deleted) ===\n")
    remove_fp32_weights(args.models_dir, dry_run=args.dry_run)

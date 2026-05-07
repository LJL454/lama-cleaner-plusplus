import torch


def get_available_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info()
    return free / (1024 ** 3)


def get_safe_available_vram_gb(reserve_gb: float = 1.0, fraction: float = 0.7) -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info()
    free_gb = free / (1024 ** 3)
    total_gb = total / (1024 ** 3)
    usable = min(free_gb, total_gb * fraction, total_gb - reserve_gb)
    return max(0.0, usable)


def get_gpu_name() -> str:
    if not torch.cuda.is_available():
        return "CPU"
    return torch.cuda.get_device_name(0)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

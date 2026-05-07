import torch
import threading
from pathlib import Path
from core.engines.base import BaseEngine
from core.engines.lama import LamaEngine
from utils.logger import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_SDXL_MODEL = str(_PROJECT_ROOT / "models" / "sdxl-inpainting")


class EngineManager:
    def __init__(
        self,
        model_id: str = _DEFAULT_SDXL_MODEL,
        lama_local_path: str = "",
        lama_hub_timeout: int = 60,
        lama_hub_retries: int = 3,
    ):
        self._cache: dict[str, BaseEngine] = {}
        self._lock = threading.Lock()
        self._model_id = model_id
        self._lama_local_path = lama_local_path
        self._lama_hub_timeout = lama_hub_timeout
        self._lama_hub_retries = lama_hub_retries

    def _create(self, name: str) -> BaseEngine:
        if name == "lama":
            engine = LamaEngine(
                local_model_path=self._lama_local_path or None,
                hub_timeout=self._lama_hub_timeout,
                hub_retries=self._lama_hub_retries,
            )
        elif name == "sdxl":
            try:
                from core.engines.sdxl import SDXLEngine
            except ImportError as e:
                raise ImportError(
                    "SDXL engine requires the 'diffusers' package. "
                    "Install it with: pip install diffusers transformers accelerate"
                ) from e
            engine = SDXLEngine(self._model_id)
        else:
            raise ValueError(f"Unknown engine: {name}")
        logger.info(f"Creating engine: {name}")
        return engine

    def get(self, name: str, force_cpu: bool = False) -> BaseEngine:
        with self._lock:
            if name not in self._cache:
                self._cache[name] = self._create(name)
            engine = self._cache[name]
            if not engine.is_loaded():
                engine.load(force_cpu=force_cpu)
                logger.info(f"Engine loaded: {name}")
            return engine

    def is_loaded(self, name: str) -> bool:
        with self._lock:
            return name in self._cache and self._cache[name].is_loaded()

    def unload(self, name: str) -> None:
        with self._lock:
            if name in self._cache:
                self._cache[name].unload()
                del self._cache[name]
                logger.info(f"Engine unloaded: {name}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared after unload")

    def unload_all(self) -> None:
        with self._lock:
            for name, engine in self._cache.items():
                engine.unload()
                logger.info(f"Engine unloaded: {name}")
            self._cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")

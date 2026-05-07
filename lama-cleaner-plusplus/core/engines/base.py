from abc import ABC, abstractmethod
from PIL import Image


class BaseEngine(ABC):
    @abstractmethod
    def inpaint(
        self, image: Image.Image, mask: Image.Image,
        prompt: str = "", negative_prompt: str = "", **kwargs
    ) -> Image.Image:
        ...

    @abstractmethod
    def load(self, force_cpu: bool = False) -> None:
        ...

    @abstractmethod
    def unload(self) -> None:
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def min_vram_gb(self) -> float:
        ...

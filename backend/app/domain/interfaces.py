from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator

from app.domain.models import CompareInput, GenerationOutput


class InferenceRuntime(ABC):
    @abstractmethod
    def runtime_name(self) -> str:
        pass

    @abstractmethod
    def start_loading_async(self) -> None:
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        pass

    @abstractmethod
    def get_loading_status(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def error_detail(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def comparison_debug(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def generate_base(self, data: CompareInput) -> GenerationOutput:
        pass

    @abstractmethod
    def generate_lora(self, data: CompareInput) -> GenerationOutput:
        pass

    @abstractmethod
    def stream_base_chunks(self, data: CompareInput) -> Iterator[str]:
        pass

    @abstractmethod
    def stream_lora_chunks(self, data: CompareInput) -> Iterator[str]:
        pass

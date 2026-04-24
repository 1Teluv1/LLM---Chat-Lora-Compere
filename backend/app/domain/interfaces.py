from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator

from app.domain.models import CompareInput, GenerationOutput


class InferenceRuntime(ABC):
    @abstractmethod
    def runtime_name(self) -> str:
        pass

    @abstractmethod
    def start_loading_async(self, data: CompareInput | None = None) -> None:
        """백그라운드 로드 시작. llama_cpp는 data.llama_load로 설정이 바뀌면 재로드합니다."""
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

    @abstractmethod
    def prompt_token_info(self, data: CompareInput) -> dict[str, Any]:
        """실제 생성 호출과 동일하게 렌더된 프롬프트의 문자 수·토큰 수(가능한 경우). 미지원이면 빈 dict."""
        pass

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


RuntimeType = Literal["llama_cpp", "transformers"]
LoraStrategyType = Literal["adapter", "merged", "auto"]
DeviceHintType = Literal["auto", "cpu", "cuda"]


@dataclass(slots=True)
class GenerationOptions:
    seed: int
    top_k: int
    top_p: float
    temperature: float
    max_tokens: int


@dataclass(slots=True)
class LlamaLoadOverrides:
    """UI/API에서 넘긴 llama.cpp 로드 옵션. 필드가 None이면 서버 환경 변수 기본값을 씁니다."""

    n_ctx: int | None = None
    n_threads: int | None = None
    n_gpu_layers: int | None = None
    n_batch: int | None = None
    use_mmap: bool | None = None
    use_mlock: bool | None = None


@dataclass(slots=True)
class CompareInput:
    prompt: str
    system_prompt: str | None = None
    enable_thinking: bool = False
    runtime: RuntimeType = "llama_cpp"
    base_model_id: str | None = None
    lora_id: str | None = None
    lora_strategy: LoraStrategyType = "auto"
    device_hint: DeviceHintType = "auto"
    llama_load: LlamaLoadOverrides | None = None
    options: GenerationOptions = field(
        default_factory=lambda: GenerationOptions(
            seed=42,
            top_k=40,
            top_p=0.9,
            temperature=0.7,
            max_tokens=512,
        )
    )


@dataclass(slots=True)
class GenerationOutput:
    text: str
    duration_ms: int


@dataclass(slots=True)
class CompareOutput:
    base: GenerationOutput
    lora: GenerationOutput
    params: dict[str, Any]
    debug: dict[str, Any] | None = None

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
class CompareInput:
    prompt: str
    system_prompt: str | None = None
    enable_thinking: bool = False
    runtime: RuntimeType = "llama_cpp"
    base_model_id: str | None = None
    lora_id: str | None = None
    lora_strategy: LoraStrategyType = "auto"
    device_hint: DeviceHintType = "auto"
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

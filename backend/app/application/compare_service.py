from __future__ import annotations

from typing import Any

from app.domain.interfaces import InferenceRuntime
from app.domain.models import CompareInput, CompareOutput
from app.infrastructure.llama_cpp.runtime import LlamaCppRuntime
from app.infrastructure.resolver import ModelResolver
from app.infrastructure.transformers.runtime import TransformersRuntime


class RuntimeRegistry:
    def __init__(self) -> None:
        resolver = ModelResolver()
        self._runtimes: dict[str, InferenceRuntime] = {
            "llama_cpp": LlamaCppRuntime(resolver),
            "transformers": TransformersRuntime(),
        }

    def get(self, name: str) -> InferenceRuntime:
        return self._runtimes.get(name, self._runtimes["llama_cpp"])


class CompareService:
    def __init__(self) -> None:
        self._registry = RuntimeRegistry()

    def select_runtime(self, runtime_name: str) -> InferenceRuntime:
        runtime = self._registry.get(runtime_name)
        runtime.start_loading_async()
        return runtime

    def compare_once(self, data: CompareInput) -> CompareOutput:
        runtime = self.select_runtime(data.runtime)
        if not runtime.is_ready():
            raise RuntimeError(str(runtime.error_detail()))
        base = runtime.generate_base(data)
        lora = runtime.generate_lora(data)
        params: dict[str, Any] = {
            "prompt": data.prompt,
            "system_prompt": data.system_prompt,
            "enable_thinking": data.enable_thinking,
            "seed": data.options.seed,
            "top_k": data.options.top_k,
            "top_p": data.options.top_p,
            "temperature": data.options.temperature,
            "max_tokens": data.options.max_tokens,
            "runtime": data.runtime,
            "base_model_id": data.base_model_id,
            "lora_id": data.lora_id,
            "lora_strategy": data.lora_strategy,
            "device_hint": data.device_hint,
        }
        return CompareOutput(base=base, lora=lora, params=params, debug=runtime.comparison_debug())

    def loading_status(self, runtime_name: str) -> dict[str, Any]:
        runtime = self.select_runtime(runtime_name)
        return runtime.get_loading_status()

    def error_detail(self, runtime_name: str) -> dict[str, Any]:
        runtime = self.select_runtime(runtime_name)
        return runtime.error_detail()

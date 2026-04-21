from __future__ import annotations

import logging
import os
import threading
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

from app.config.settings import env_flag, optional_positive_int
from app.domain.interfaces import InferenceRuntime
from app.domain.models import CompareInput, GenerationOutput
from app.infrastructure.resolver import ModelResolver

logger = logging.getLogger(__name__)

try:
    from llama_cpp import Llama
except Exception:  # pragma: no cover
    Llama = None  # type: ignore[assignment]
try:
    import llama_cpp as _llama_cpp_module  # type: ignore
    LLAMA_SUPPORTS_GPU = bool(_llama_cpp_module.llama_supports_gpu_offload())
except Exception:  # pragma: no cover
    LLAMA_SUPPORTS_GPU = False

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


class LlamaCppRuntime(InferenceRuntime):
    def __init__(self, resolver: ModelResolver) -> None:
        self._resolver = resolver
        self._base_llm = None
        self._lora_llm = None
        self._stage = "idle"
        self._message = ""
        self._err = ""
        self._comparison_mode = "lora_adapter"
        self._resolved_base = ""
        self._resolved_compare = ""
        self._active_loras: list[dict[str, Any]] | None = None
        self._load_started = False
        self._lock = threading.Lock()

    def runtime_name(self) -> str:
        return "llama_cpp"

    def _set_stage(self, stage: str, message: str = "") -> None:
        with self._lock:
            self._stage = stage
            self._message = message

    def start_loading_async(self) -> None:
        with self._lock:
            if self._load_started:
                return
            self._load_started = True
        threading.Thread(target=self._load_models, daemon=True, name="llama-runtime-loader").start()

    def _load_models(self) -> None:
        self._set_stage("resolving", "llama.cpp 모델 경로 확인 중")
        if Llama is None:
            self._err = "llama_cpp 모듈 import 실패"
            self._set_stage("error", self._err)
            return
        resolved = self._resolver.resolve_for_llama()
        self._resolved_base = resolved["base_model_path"]
        self._resolved_compare = resolved["compare_model_path"]
        self._comparison_mode = resolved["comparison_mode"]
        if not self._resolved_base or not Path(self._resolved_base).is_file():
            self._err = f"베이스 모델 파일 없음: {self._resolved_base}"
            self._set_stage("error", self._err)
            return
        if not self._resolved_compare or not Path(self._resolved_compare).is_file():
            self._err = f"비교 모델 파일 없음: {self._resolved_compare}"
            self._set_stage("error", self._err)
            return

        n_ctx = int(os.getenv("LLAMA_N_CTX", "4096"))
        n_threads = int(os.getenv("LLAMA_N_THREADS", "8"))
        n_gpu_layers = int(os.getenv("LLAMA_N_GPU_LAYERS", "0"))
        use_mmap = env_flag("LLAMA_USE_MMAP", True)
        use_mlock = env_flag("LLAMA_USE_MLOCK", False)
        verbose = env_flag("LLAMA_VERBOSE", True)
        kw = {
            "n_ctx": n_ctx,
            "n_threads": n_threads,
            "n_gpu_layers": n_gpu_layers,
            "use_mmap": use_mmap,
            "use_mlock": use_mlock,
            "verbose": verbose,
        }
        n_batch = optional_positive_int("LLAMA_N_BATCH")
        if n_batch:
            kw["n_batch"] = n_batch
        try:
            self._set_stage("loading_base", "베이스 모델 로딩 중")
            self._base_llm = Llama(model_path=self._resolved_base, **kw)
            self._set_stage("loading_lora", "비교 모델 로딩 중")
            if self._comparison_mode == "merged_gguf":
                self._lora_llm = Llama(model_path=self._resolved_compare, **kw)
                self._active_loras = None
            else:
                self._lora_llm = Llama(model_path=self._resolved_base, **kw)
                self._lora_llm.load_lora("adapter", self._resolved_compare)
                loaded = list(self._lora_llm.list_loras() or [])
                if loaded:
                    scale = float(os.getenv("LLAMA_LORA_SCALE", "1.0"))
                    adapter_obj = self._lora_llm._model._lora_registry[loaded[0]]
                    self._lora_llm._ctx.apply_loras([(adapter_obj, scale)])
                    self._active_loras = [{"name": loaded[0], "scale": scale}]
            self._set_stage("ready", "모델 로드 완료")
        except Exception as exc:
            self._err = f"{type(exc).__name__}: {exc}"
            self._base_llm = None
            self._lora_llm = None
            logger.warning("llama runtime load failed", exc_info=True)
            self._set_stage("error", self._err)

    def is_ready(self) -> bool:
        return self._base_llm is not None and self._lora_llm is not None

    def _resolve_device(self) -> str:
        if torch is not None and bool(torch.cuda.is_available()):
            return "cuda"
        if int(os.getenv("LLAMA_N_GPU_LAYERS", "0")) > 0:
            return "cuda"
        return "cpu"

    def _process_stats(self) -> dict[str, Any]:
        if psutil is None:
            return {"error": "psutil unavailable"}
        try:
            process = psutil.Process()
            return {
                "rss_bytes": int(process.memory_info().rss),
                "vms_bytes": int(process.memory_info().vms),
                "private_bytes": int(getattr(process.memory_info(), "private", 0) or 0),
                "num_threads": int(process.num_threads()),
                "cpu_percent": float(process.cpu_percent(interval=None)),
            }
        except Exception as exc:  # pragma: no cover
            return {"error": str(exc)}

    def _system_ram_stats(self) -> dict[str, Any]:
        if psutil is None:
            return {}
        try:
            vm = psutil.virtual_memory()
            return {
                "system_ram_total_bytes": int(vm.total),
                "system_ram_available_bytes": int(vm.available),
                "system_ram_percent": float(vm.percent),
            }
        except Exception:  # pragma: no cover
            return {}

    def _gpu_stats(self) -> dict[str, Any]:
        gpu = {
            "available": False,
            "name": None,
            "vram_total_bytes": None,
            "vram_used_bytes": None,
            "vram_free_bytes": None,
            "utilization_percent": None,
            "error": None,
        }
        if torch is None:
            return gpu
        try:
            if not bool(torch.cuda.is_available()):
                return gpu
            index = 0
            free_mem, total_mem = torch.cuda.mem_get_info(index)
            used_mem = int(total_mem - free_mem)
            gpu.update(
                {
                    "available": True,
                    "name": str(torch.cuda.get_device_name(index)),
                    "vram_total_bytes": int(total_mem),
                    "vram_used_bytes": used_mem,
                    "vram_free_bytes": int(free_mem),
                }
            )
            return gpu
        except Exception as exc:  # pragma: no cover
            gpu["error"] = str(exc)
            return gpu

    def get_loading_status(self) -> dict[str, Any]:
        process_info = self._process_stats()
        process_info.update(self._system_ram_stats())
        device = self._resolve_device()
        gpu = self._gpu_stats()
        return {
            "stage": self._stage,
            "message": self._message,
            "ready": self.is_ready(),
            "runtime_name": self.runtime_name(),
            "device": device,
            "comparison_mode": self._comparison_mode,
            "model_identifiers": {
                "base": self._resolved_base or None,
                "lora": self._resolved_compare or None,
            },
            "model_loaded": {
                "base": self._base_llm is not None,
                "lora": self._lora_llm is not None,
                "overall": self.is_ready(),
            },
            "capabilities": {
                "gpu_runtime_available": bool(LLAMA_SUPPORTS_GPU),
                "gpu_metrics_available": bool(gpu.get("available")),
            },
            "base_file": {"path": self._resolved_base or None, "size_bytes": None},
            "lora_file": {"path": self._resolved_compare or None, "size_bytes": None},
            "process": process_info,
            "gpu": gpu,
            "llama_cpp": {
                "version": None,
                "supports_gpu_offload": bool(LLAMA_SUPPORTS_GPU),
                "n_gpu_layers_requested": int(os.getenv("LLAMA_N_GPU_LAYERS", "0")),
                "n_gpu_layers_effective": getattr(self, "_effective_n_gpu_layers", None),
                "gpu_forced_off": getattr(self, "_gpu_forced_off", False),
            },
            "error_reason": self._err or None,
        }

    def error_detail(self) -> dict[str, Any]:
        return {
            "ready": self.is_ready(),
            "summary": self._err or ("정상" if self.is_ready() else "로딩 중"),
            "loading_status": self.get_loading_status(),
            "comparison_mode": self._comparison_mode,
            "resolved_base_model_path": self._resolved_base or None,
            "resolved_lora_adapter_path": self._resolved_compare or None,
        }

    def comparison_debug(self) -> dict[str, Any]:
        loaded = []
        if self._lora_llm is not None:
            try:
                loaded = list(self._lora_llm.list_loras() or [])
            except Exception:
                loaded = []
        return {
            "runtime_effective": "llama_cpp",
            "comparison_mode": self._comparison_mode,
            "resolved_base_model_path": self._resolved_base or None,
            "resolved_compare_model_path": self._resolved_compare or None,
            "loaded_loras": loaded,
            "ready": self.is_ready(),
        }

    def _generate(self, llm: Any, data: CompareInput, active_loras: list[dict[str, Any]] | None = None) -> GenerationOutput:
        started = perf_counter()
        kwargs: dict[str, Any] = {
            "prompt": data.prompt,
            "max_tokens": data.options.max_tokens,
            "temperature": data.options.temperature,
            "top_k": data.options.top_k,
            "top_p": data.options.top_p,
            "seed": data.options.seed,
            "echo": False,
        }
        if active_loras is not None:
            kwargs["active_loras"] = active_loras
        out = llm(**kwargs)
        text = out["choices"][0]["text"]
        return GenerationOutput(text=text, duration_ms=int((perf_counter() - started) * 1000))

    def _stream(self, llm: Any, data: CompareInput, active_loras: list[dict[str, Any]] | None = None) -> Iterator[str]:
        kwargs: dict[str, Any] = {
            "prompt": data.prompt,
            "max_tokens": data.options.max_tokens,
            "temperature": data.options.temperature,
            "top_k": data.options.top_k,
            "top_p": data.options.top_p,
            "seed": data.options.seed,
            "echo": False,
            "stream": True,
        }
        if active_loras is not None:
            kwargs["active_loras"] = active_loras
        for part in llm(**kwargs):
            piece = part["choices"][0].get("text") or ""
            if piece:
                yield piece

    def generate_base(self, data: CompareInput) -> GenerationOutput:
        if self._base_llm is None:
            return GenerationOutput(text=f"[fallback-base] {self._err}\n\n{data.prompt}", duration_ms=0)
        return self._generate(self._base_llm, data)

    def generate_lora(self, data: CompareInput) -> GenerationOutput:
        if self._lora_llm is None:
            return GenerationOutput(text=f"[fallback-lora] {self._err}\n\n{data.prompt}", duration_ms=0)
        return self._generate(self._lora_llm, data, active_loras=self._active_loras)

    def stream_base_chunks(self, data: CompareInput) -> Iterator[str]:
        if self._base_llm is None:
            raise RuntimeError("베이스 모델이 로드되지 않았습니다.")
        yield from self._stream(self._base_llm, data)

    def stream_lora_chunks(self, data: CompareInput) -> Iterator[str]:
        if self._lora_llm is None:
            raise RuntimeError("LoRA 모델이 로드되지 않았습니다.")
        yield from self._stream(self._lora_llm, data, active_loras=self._active_loras)

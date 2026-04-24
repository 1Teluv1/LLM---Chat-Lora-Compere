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
from app.domain.models import CompareInput, GenerationOutput, LlamaLoadOverrides
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
        self._lock = threading.Lock()
        self._loader_lock = threading.Lock()
        self._loader_thread: threading.Thread | None = None
        self._loaded_kw: dict[str, Any] | None = None
        self._desired_kw: dict[str, Any] | None = None
        self._desired_model_sel: dict[str, Any] | None = None
        self._loaded_model_sel: dict[str, Any] | None = None
        self._gpu_forced_off = False
        self._effective_n_gpu_layers: int | None = None

    def runtime_name(self) -> str:
        return "llama_cpp"

    def _set_stage(self, stage: str, message: str = "") -> None:
        with self._lock:
            self._stage = stage
            self._message = message

    def _resolve_llama_kw(self, overrides: LlamaLoadOverrides | None) -> dict[str, Any]:
        o = overrides
        if o is None or o.n_ctx is None or o.n_ctx == 0:
            n_ctx = int(os.getenv("LLAMA_N_CTX", "8192"))
        else:
            n_ctx = int(o.n_ctx)
        n_ctx = max(512, min(262144, n_ctx))
        n_threads = int(os.getenv("LLAMA_N_THREADS", "8")) if o is None or o.n_threads is None else o.n_threads
        n_threads = max(1, min(256, n_threads))
        n_gpu_layers_req = (
            int(os.getenv("LLAMA_N_GPU_LAYERS", "0")) if o is None or o.n_gpu_layers is None else o.n_gpu_layers
        )
        if n_gpu_layers_req != 0 and not LLAMA_SUPPORTS_GPU:
            self._gpu_forced_off = True
            n_gpu_layers = 0
        else:
            self._gpu_forced_off = False
            n_gpu_layers = n_gpu_layers_req
        self._effective_n_gpu_layers = n_gpu_layers
        use_mmap = env_flag("LLAMA_USE_MMAP", True) if o is None or o.use_mmap is None else o.use_mmap
        use_mlock = env_flag("LLAMA_USE_MLOCK", False) if o is None or o.use_mlock is None else o.use_mlock
        verbose = env_flag("LLAMA_VERBOSE", False)
        n_batch: int | None = optional_positive_int("LLAMA_N_BATCH")
        if o is not None and o.n_batch is not None:
            n_batch = max(32, min(65536, o.n_batch))
        return {
            "n_ctx": n_ctx,
            "n_threads": n_threads,
            "n_gpu_layers": n_gpu_layers,
            "use_mmap": use_mmap,
            "use_mlock": use_mlock,
            "verbose": verbose,
            "n_batch": n_batch,
        }

    @staticmethod
    def _llama_ctor_kw(load_kw: dict[str, Any]) -> dict[str, Any]:
        ctor: dict[str, Any] = {k: v for k, v in load_kw.items() if k != "n_batch"}
        nb = load_kw.get("n_batch")
        if nb is not None:
            ctor["n_batch"] = nb
        return ctor

    def start_loading_async(self, data: CompareInput | None = None) -> None:
        overrides = data.llama_load if data else None
        kw = self._resolve_llama_kw(overrides)
        model_sel = {
            "base_model_id": data.base_model_id if data else None,
            "lora_id": data.lora_id if data else None,
            "lora_strategy": data.lora_strategy if data else "auto",
        }
        with self._loader_lock:
            self._desired_kw = kw
            self._desired_model_sel = model_sel
            if self._loader_thread is None or not self._loader_thread.is_alive():
                self._loader_thread = threading.Thread(
                    target=self._loader_main,
                    daemon=True,
                    name="llama-runtime-loader",
                )
                self._loader_thread.start()

    def _loader_main(self) -> None:
        while True:
            with self._loader_lock:
                target = dict(self._desired_kw) if self._desired_kw else self._resolve_llama_kw(None)
                target_model_sel = dict(self._desired_model_sel) if self._desired_model_sel else {
                    "base_model_id": None,
                    "lora_id": None,
                    "lora_strategy": "auto",
                }
            try:
                self._set_stage("resolving", "llama.cpp 로드 설정 적용 중…")
                self._load_models_impl(target, target_model_sel)
                with self._loader_lock:
                    self._loaded_kw = dict(target)
                    self._loaded_model_sel = dict(target_model_sel)
                    pending = dict(self._desired_kw) if self._desired_kw else {}
                    pending_model_sel = (
                        dict(self._desired_model_sel)
                        if self._desired_model_sel
                        else {"base_model_id": None, "lora_id": None, "lora_strategy": "auto"}
                    )
                    if pending == self._loaded_kw and pending_model_sel == self._loaded_model_sel:
                        self._set_stage("ready", "모델 로드 완료")
                        return
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                self._err = err
                self._base_llm = None
                self._lora_llm = None
                logger.warning("llama runtime load failed", exc_info=True)
                self._set_stage("error", err)
                return

    def _load_models_impl(self, load_kw: dict[str, Any], model_sel: dict[str, Any]) -> None:
        if Llama is None:
            self._err = "llama_cpp 모듈 import 실패"
            self._set_stage("error", self._err)
            raise RuntimeError(self._err)
        self._base_llm = None
        self._lora_llm = None
        resolved = self._resolver.resolve_for_llama(
            strategy=str(model_sel.get("lora_strategy") or "auto"),
            base_model_id=model_sel.get("base_model_id"),
            lora_id=model_sel.get("lora_id"),
        )
        self._resolved_base = resolved["base_model_path"]
        self._resolved_compare = resolved["compare_model_path"]
        self._comparison_mode = resolved["comparison_mode"]
        if not self._resolved_base or not Path(self._resolved_base).is_file():
            self._err = f"베이스 모델 파일 없음: {self._resolved_base}"
            self._set_stage("error", self._err)
            raise RuntimeError(self._err)
        if not self._resolved_compare or not Path(self._resolved_compare).is_file():
            self._err = f"비교 모델 파일 없음: {self._resolved_compare}"
            self._set_stage("error", self._err)
            raise RuntimeError(self._err)

        ctor = self._llama_ctor_kw(load_kw)
        try:
            ctx = ctor.get("n_ctx", "?")
            self._set_stage("loading_base", f"베이스 모델 로딩 중 (n_ctx={ctx})")
            self._base_llm = Llama(model_path=self._resolved_base, **ctor)
            self._set_stage("loading_lora", "비교 모델 로딩 중")
            if self._comparison_mode == "merged_gguf":
                self._lora_llm = Llama(model_path=self._resolved_compare, **ctor)
                self._active_loras = None
            else:
                self._lora_llm = Llama(model_path=self._resolved_base, **ctor)
                self._lora_llm.load_lora("adapter", self._resolved_compare)
                loaded = list(self._lora_llm.list_loras() or [])
                if loaded:
                    scale = float(os.getenv("LLAMA_LORA_SCALE", "1.0"))
                    adapter_obj = self._lora_llm._model._lora_registry[loaded[0]]
                    self._lora_llm._ctx.apply_loras([(adapter_obj, scale)])
                    self._active_loras = [{"name": loaded[0], "scale": scale}]
            self._err = ""
        except Exception:
            self._base_llm = None
            self._lora_llm = None
            raise

    def is_ready(self) -> bool:
        return self._base_llm is not None and self._lora_llm is not None

    def _resolve_device(self) -> str:
        ng = int(os.getenv("LLAMA_N_GPU_LAYERS", "0"))
        if self._loaded_kw is not None:
            ng = int(self._loaded_kw.get("n_gpu_layers", 0))
        elif self._desired_kw is not None:
            ng = int(self._desired_kw.get("n_gpu_layers", 0))
        if torch is not None and bool(torch.cuda.is_available()):
            return "cuda"
        if ng != 0:
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
                "version": getattr(_llama_cpp_module, "__version__", None) if Llama else None,
                "supports_gpu_offload": bool(LLAMA_SUPPORTS_GPU),
                "n_gpu_layers_requested": (
                    int(self._loaded_kw["n_gpu_layers"])
                    if self._loaded_kw is not None
                    else int(os.getenv("LLAMA_N_GPU_LAYERS", "0"))
                ),
                "n_gpu_layers_effective": getattr(self, "_effective_n_gpu_layers", None),
                "gpu_forced_off": getattr(self, "_gpu_forced_off", False),
                "load_kw_effective": dict(self._loaded_kw) if self._loaded_kw else None,
                "load_kw_pending": (
                    dict(self._desired_kw)
                    if self._desired_kw and self._loaded_kw != self._desired_kw
                    else None
                ),
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

    def _build_prompt(self, data: CompareInput) -> str:
        system_prompt = (data.system_prompt or "").strip()
        if not data.enable_thinking:
            think_off = (
                "Do not output chain-of-thought or <think> blocks. "
                "Return only the final answer concisely."
            )
            system_prompt = f"{system_prompt}\n\n{think_off}".strip() if system_prompt else think_off
        if not system_prompt:
            return data.prompt
        return f"System:\n{system_prompt}\n\nUser:\n{data.prompt}\n\nAssistant:\n"

    def prompt_token_info(self, data: CompareInput) -> dict[str, Any]:
        """GGUF에 포함된 llama.cpp 어휘로 `tokenize` (Base·LoRA 동일 베이스 가정)."""
        if self._base_llm is None:
            return {}
        prompt = self._build_prompt(data)
        llm = self._base_llm
        blob = prompt.encode("utf-8")
        token_param_sets: tuple[dict[str, Any], ...] = (
            {"add_bos": True, "special": True},
            {"add_bos": True, "special": False},
            {"add_bos": True},
            {},
        )
        ids: list[int] | None = None
        used_params = ""
        for kw in token_param_sets:
            try:
                ids = llm.tokenize(blob, **kw)
                used_params = ",".join(f"{k}={v}" for k, v in kw.items()) if kw else "positional"
                break
            except TypeError:
                continue
        if ids is None:
            return {
                "rendered_prompt_chars": len(prompt),
                "tokenizer_backend": "llama_cpp",
                "error": "llama_cpp.tokenize 호환 시그니처를 찾지 못했습니다.",
            }
        return {
            "rendered_prompt_chars": len(prompt),
            "rendered_prompt_tokens": len(ids),
            "tokenizer_backend": "llama_cpp",
            "tokenize_params": used_params,
            "note": "로드된 GGUF의 어휘로 계산. 생성 시 llama.cpp가 추가하는 특수 토큰과 1~2 토큰 차이 날 수 있음.",
        }

    def _generate(self, llm: Any, data: CompareInput, active_loras: list[dict[str, Any]] | None = None) -> GenerationOutput:
        started = perf_counter()
        prompt = self._build_prompt(data)
        kwargs: dict[str, Any] = {
            "prompt": prompt,
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
        prompt = self._build_prompt(data)
        kwargs: dict[str, Any] = {
            "prompt": prompt,
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

from __future__ import annotations

import os
import re
import threading
from time import perf_counter
from typing import Any, Iterator

from app.domain.interfaces import InferenceRuntime
from app.domain.models import CompareInput, GenerationOutput

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


class TransformersRuntime(InferenceRuntime):
    def __init__(self) -> None:
        self._stage = "idle"
        self._message = ""
        self._err = ""
        self._base_pipe = None
        self._lora_pipe = None
        self._load_started = False
        self._base_model_id = os.getenv("HF_BASE_MODEL_ID", "gpt2")
        self._lora_model_id = os.getenv("HF_LORA_MODEL_ID", "")

    def runtime_name(self) -> str:
        return "transformers"

    def _set_stage(self, stage: str, message: str = "") -> None:
        self._stage = stage
        self._message = message

    def start_loading_async(self) -> None:
        if self._load_started:
            return
        self._load_started = True
        threading.Thread(target=self._load_models, daemon=True, name="transformers-runtime-loader").start()

    def _load_models(self) -> None:
        self._set_stage("resolving", "transformers 모델 준비 중")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except Exception as exc:  # pragma: no cover
            self._err = f"transformers 런타임 import 실패: {exc}"
            self._set_stage("error", self._err)
            return
        model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device_index = 0 if device == "cuda" else -1

        base_id = self._base_model_id
        lora_id = self._lora_model_id or base_id
        try:
            self._set_stage("loading_base", f"base={base_id} 로딩")
            base_tokenizer = AutoTokenizer.from_pretrained(base_id)
            base_model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=model_dtype)
            self._base_pipe = pipeline(
                "text-generation",
                model=base_model,
                tokenizer=base_tokenizer,
                device=device_index,
            )

            self._set_stage("loading_lora", f"compare={lora_id} 로딩")
            lora_tokenizer = AutoTokenizer.from_pretrained(lora_id)
            lora_model = AutoModelForCausalLM.from_pretrained(lora_id, torch_dtype=model_dtype)
            self._lora_pipe = pipeline(
                "text-generation",
                model=lora_model,
                tokenizer=lora_tokenizer,
                device=device_index,
            )
            self._set_stage("ready", f"모델 로드 완료 ({device})")
        except Exception as exc:
            self._err = f"{type(exc).__name__}: {exc}"
            self._set_stage("error", self._err)
            self._base_pipe = None
            self._lora_pipe = None

    def is_ready(self) -> bool:
        return self._base_pipe is not None and self._lora_pipe is not None

    def _resolve_device(self) -> str:
        if torch is not None and bool(torch.cuda.is_available()):
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
            gpu.update(
                {
                    "available": True,
                    "name": str(torch.cuda.get_device_name(index)),
                    "vram_total_bytes": int(total_mem),
                    "vram_used_bytes": int(total_mem - free_mem),
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
            "comparison_mode": "hf_pair",
            "model_identifiers": {"base": self._base_model_id, "lora": self._lora_model_id or self._base_model_id},
            "model_loaded": {
                "base": self._base_pipe is not None,
                "lora": self._lora_pipe is not None,
                "overall": self.is_ready(),
            },
            "capabilities": {
                "gpu_runtime_available": bool(torch is not None and torch.cuda.is_available()),
                "gpu_metrics_available": bool(gpu.get("available")),
            },
            "base_file": {"path": self._base_model_id, "size_bytes": None},
            "lora_file": {"path": self._lora_model_id or self._base_model_id, "size_bytes": None},
            "process": process_info,
            "gpu": gpu,
            "error_reason": self._err or None,
        }

    def error_detail(self) -> dict[str, Any]:
        return {
            "ready": self.is_ready(),
            "summary": self._err or ("정상" if self.is_ready() else "로딩 중"),
            "loading_status": self.get_loading_status(),
            "comparison_mode": "hf_pair",
            "resolved_base_model_path": self._base_model_id,
            "resolved_lora_adapter_path": self._lora_model_id or self._base_model_id,
        }

    def comparison_debug(self) -> dict[str, Any]:
        return {
            "runtime_effective": "transformers",
            "comparison_mode": "hf_pair",
            "resolved_base_model_path": self._base_model_id,
            "resolved_compare_model_path": self._lora_model_id or self._base_model_id,
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

    def _strip_think_block(self, text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _generate(self, pipe: Any, data: CompareInput) -> GenerationOutput:
        started = perf_counter()
        prompt = self._build_prompt(data)
        outputs = pipe(
            prompt,
            do_sample=True,
            max_new_tokens=data.options.max_tokens,
            temperature=data.options.temperature,
            top_k=data.options.top_k,
            top_p=data.options.top_p,
            num_return_sequences=1,
        )
        text = outputs[0]["generated_text"]
        if text.startswith(prompt):
            text = text[len(prompt) :]
        if not data.enable_thinking:
            text = self._strip_think_block(text)
        return GenerationOutput(text=text, duration_ms=int((perf_counter() - started) * 1000))

    def generate_base(self, data: CompareInput) -> GenerationOutput:
        if self._base_pipe is None:
            return GenerationOutput(text=f"[fallback-base] {self._err}\n\n{data.prompt}", duration_ms=0)
        return self._generate(self._base_pipe, data)

    def generate_lora(self, data: CompareInput) -> GenerationOutput:
        if self._lora_pipe is None:
            return GenerationOutput(text=f"[fallback-lora] {self._err}\n\n{data.prompt}", duration_ms=0)
        return self._generate(self._lora_pipe, data)

    def stream_base_chunks(self, data: CompareInput) -> Iterator[str]:
        yield self.generate_base(data).text

    def stream_lora_chunks(self, data: CompareInput) -> Iterator[str]:
        yield self.generate_lora(data).text

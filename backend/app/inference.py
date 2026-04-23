import logging
import os
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

logger = logging.getLogger(__name__)

try:
    from llama_cpp import Llama
    LLAMA_IMPORT_ERROR = ""
except Exception:  # pragma: no cover
    Llama = None  # type: ignore[assignment]
    LLAMA_IMPORT_ERROR = str(__import__("sys").exc_info()[1])

try:
    import llama_cpp as _llama_cpp_module
    LLAMA_SUPPORTS_GPU = bool(_llama_cpp_module.llama_supports_gpu_offload())
except Exception:
    LLAMA_SUPPORTS_GPU = False

try:
    import psutil  # type: ignore
    PSUTIL_OK = True
except Exception:
    psutil = None  # type: ignore
    PSUTIL_OK = False

_PREFERRED_BASE_GGUF = "Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _discover_base_gguf() -> Path | None:
    """`<repo>/artifacts/base` 아래 GGUF를 찾는다 (`download_assets` 기본 출력과 동일)."""
    base_dir = _repo_root() / "artifacts" / "base"
    if not base_dir.is_dir():
        return None
    ggufs = sorted(base_dir.glob("*.gguf"))
    if not ggufs:
        ggufs = sorted({p for p in base_dir.rglob("*.gguf") if p.is_file()})
    if not ggufs:
        return None
    preferred = base_dir / _PREFERRED_BASE_GGUF
    if preferred.is_file():
        return preferred
    for p in ggufs:
        if p.name == _PREFERRED_BASE_GGUF:
            return p
    return ggufs[0]


def _discover_peft_safetensors() -> Path | None:
    """PEFT `adapter_model.safetensors` 경로 (llama.cpp가 직접 읽지 못함)."""
    lora_dir = _repo_root() / "artifacts" / "lora"
    if not lora_dir.is_dir():
        return None
    found = sorted(lora_dir.rglob("adapter_model.safetensors"))
    return found[0] if found else None


def _discover_lora_adapter_gguf() -> Path | None:
    """`<repo>/artifacts/lora`에서 llama.cpp용 LoRA GGUF (`adapter_model.gguf`)를 찾는다.

    PEFT는 `adapter_model.safetensors`이므로, 같은 디렉터리에 변환된 `adapter_model.gguf`가
    있으면 그것을 사용한다. safetensors만 있는 경우는 None.
    """
    lora_dir = _repo_root() / "artifacts" / "lora"
    if not lora_dir.is_dir():
        return None
    candidates = sorted(lora_dir.rglob("adapter_model.gguf"))
    if candidates:
        return candidates[0]
    for st in sorted(lora_dir.rglob("adapter_model.safetensors")):
        gguf = st.parent / "adapter_model.gguf"
        if gguf.is_file():
            return gguf
    return None


def _discover_merged_gguf() -> Path | None:
    merged_dir = _repo_root() / "artifacts" / "merged"
    if not merged_dir.is_dir():
        return None
    ggufs = sorted(merged_dir.rglob("*.gguf"))
    return ggufs[0] if ggufs else None


def _lora_safetensors_only_hint() -> str:
    st = _discover_peft_safetensors()
    if not st:
        return ""
    return (
        f" PEFT 어댑터가 확인됨: {st} — llama.cpp는 GGUF LoRA만 로드합니다. "
        "`pip install -r backend/requirements-lora-convert.txt` 후 `npm run convert:lora` 로 변환하거나, "
        "LORA_AUTO_CONVERT_GGUF=1(기본)일 때 서버 기동 시 자동 변환을 시도합니다."
    )


def _lora_auto_convert_enabled() -> bool:
    v = os.getenv("LORA_AUTO_CONVERT_GGUF", "1").strip().lower()
    return v not in ("0", "false", "no")


def _try_peft_autoconvert_to_gguf() -> bool:
    st = _discover_peft_safetensors()
    if st is None:
        return False
    out = st.parent / "adapter_model.gguf"
    if out.is_file() and out.stat().st_mtime >= st.stat().st_mtime:
        return True
    try:
        from app.lora_gguf_convert import (
            LoraGgufConvertError,
            convert_peft_adapter,
            infer_base_dir_for_lora,
        )
    except ImportError as exc:
        logger.warning("LoRA 자동 변환 스킵(import): %s", exc)
        return False
    try:
        convert_peft_adapter(
            st.parent,
            base_dir=infer_base_dir_for_lora(st.parent),
            outfile=out,
        )
        return True
    except LoraGgufConvertError as exc:
        logger.warning("LoRA GGUF 자동 변환 실패: %s", exc)
        return False
    except Exception as exc:
        logger.warning("LoRA GGUF 자동 변환 중 오류: %s", exc)
        return False


def _env_flag(name: str, default: bool = True) -> bool:
    v = os.getenv(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _env_use_mmap() -> bool:
    """베이스 GGUF mmap 여부. LoRA 적용 인스턴스는 llama-cpp-python이 mmap을 끈다."""
    return _env_flag("LLAMA_USE_MMAP", True)


def _env_use_mlock() -> bool:
    """모델 페이지를 RAM에 고정(가능한 OS에서). Windows는 권한·메모리 부족 시 실패할 수 있음."""
    return _env_flag("LLAMA_USE_MLOCK", False)


def _env_optional_positive_int(name: str) -> int | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        v = int(raw)
    except ValueError:
        return None
    return v if v > 0 else None


def _resolve_model_path() -> str:
    found = _discover_base_gguf()
    if found:
        return str(found.resolve())
    raw = os.getenv("BASE_MODEL_PATH", "").strip()
    if raw:
        p = Path(raw)
        if p.is_file():
            return str(p.resolve())
    return ""


def _resolve_lora_path() -> str:
    found = _discover_lora_adapter_gguf()
    if found:
        return str(found.resolve())
    raw = os.getenv("LORA_ADAPTER_PATH", "").strip()
    if raw:
        p = Path(raw)
        if p.is_file():
            if p.suffix.lower() in (".safetensors", ".bin"):
                return ""
            return str(p.resolve())
    return ""


def _resolve_merged_model_path() -> str:
    found = _discover_merged_gguf()
    if found:
        return str(found.resolve())
    raw = os.getenv("MERGED_MODEL_GGUF", "").strip()
    if raw:
        p = Path(raw)
        if p.is_file():
            return str(p.resolve())
    return ""


@dataclass
class InferenceOutput:
    text: str
    duration_ms: int


def _capture_llama_cpp_stderr(
    *,
    model_path: str,
    lora_path: str | None,
    n_ctx: int,
    n_threads: int,
    n_gpu_layers: int,
    use_mmap: bool = True,
    use_mlock: bool = False,
    max_bytes: int = 786_432,
) -> str:
    """로드 실패 후 llama.cpp가 stderr(FD 2)에 쓰는 로그를 수집한다. verbose=True로 한 번 더 시도."""
    if Llama is None:
        return ""
    read_fd, write_fd = os.pipe()
    old_stderr = os.dup(2)
    captured = bytearray()
    write_fd_open = True
    try:
        os.dup2(write_fd, 2)
        os.close(write_fd)
        write_fd_open = False
        try:
            kw: dict = dict(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                use_mmap=use_mmap,
                use_mlock=use_mlock,
                verbose=True,
            )
            if lora_path:
                Llama(**kw, lora_path=lora_path)
            else:
                Llama(**kw)
        except (ValueError, RuntimeError, OSError):
            pass
    finally:
        if write_fd_open:
            try:
                os.close(write_fd)
            except OSError:
                pass
        try:
            os.dup2(old_stderr, 2)
        finally:
            os.close(old_stderr)
        try:
            while len(captured) < max_bytes:
                chunk = os.read(read_fd, 16384)
                if not chunk:
                    break
                captured.extend(chunk)
        finally:
            os.close(read_fd)
    return captured.decode("utf-8", errors="replace").strip()


_LOAD_FAILURE_HINT = (
    "디스크에 파일이 있어도 llama.cpp가 모델을 메모리에 올리지 못하면 위 메시지가 납니다. "
    "흔한 원인: RAM/스왑 부족(특히 대형 MoE·긴 n_ctx), GGUF와 llama-cpp-python 빌드 불일치, "
    "다운로드 불완전·손상. 대응: LLAMA_N_CTX 축소, GPU 사용은 CUDA로 빌드된 llama-cpp-python 필수(`npm run install:llama-cuda`), "
    "LLAMA_N_GPU_LAYERS, RAM 적재는 LLAMA_USE_MMAP=0."
)


class InferenceService:
    STAGES = (
        "idle",
        "resolving",
        "converting_peft",
        "loading_base",
        "base_loaded",
        "loading_lora",
        "ready",
        "error",
    )

    def __init__(self) -> None:
        self._fallback_reason = ""
        self._base_llm = None
        self._lora_llm = None
        self._resolved_model_path = ""
        self._resolved_lora_path = ""
        self._comparison_mode = "lora_adapter"
        self._load_traceback = ""
        self._llama_cpp_stderr = ""
        self._load_failed_stage = ""
        self._active_loras: list[dict[str, Any]] | None = None
        self._lock = threading.Lock()
        self._stage = "idle"
        self._stage_message = ""
        self._loading_started_at = 0.0
        self._stage_started_at = 0.0
        self._stage_durations_ms: dict[str, int] = {}
        self._load_started = False

    def _set_stage(self, stage: str, message: str = "") -> None:
        now = perf_counter()
        with self._lock:
            prev = self._stage
            prev_started = self._stage_started_at
            if prev and prev not in ("idle", stage) and prev_started:
                self._stage_durations_ms[prev] = int((now - prev_started) * 1000)
            self._stage = stage
            self._stage_message = message
            self._stage_started_at = now
        logger.info("[stage] %s — %s", stage, message)

    def start_loading_async(self) -> None:
        """백그라운드 스레드에서 모델 로딩을 시작(이미 시작됐으면 no-op)."""
        with self._lock:
            if self._load_started:
                return
            self._load_started = True
            self._loading_started_at = perf_counter()
        threading.Thread(target=self._load_models, daemon=True, name="lora-compare-loader").start()

    def _mark_error(self, reason: str) -> None:
        self._fallback_reason = reason
        self._set_stage("error", reason)

    def _load_models(self) -> None:
        self._set_stage("resolving", "GGUF 경로 확인 중…")
        if Llama is None:
            import_error_suffix = (
                f" (import error: {LLAMA_IMPORT_ERROR})" if LLAMA_IMPORT_ERROR else ""
            )
            self._mark_error(
                "llama_cpp 모듈을 불러오지 못해 fallback 모드로 동작합니다." + import_error_suffix
            )
            return

        model_path = _resolve_model_path()
        self._resolved_model_path = model_path
        lora_path = _resolve_lora_path()
        self._resolved_lora_path = lora_path
        merged_model_path = _resolve_merged_model_path()
        use_merged_gguf = bool(merged_model_path)
        if use_merged_gguf:
            self._resolved_lora_path = merged_model_path
            self._comparison_mode = "merged_gguf"
        else:
            self._comparison_mode = "lora_adapter"
        if (
            not use_merged_gguf
            and not lora_path
            and _lora_auto_convert_enabled()
            and _discover_peft_safetensors() is not None
        ):
            self._set_stage(
                "converting_peft",
                "PEFT adapter_model.safetensors → adapter_model.gguf 변환 중… (최초 1회)",
            )
            if _try_peft_autoconvert_to_gguf():
                lora_path = _resolve_lora_path()
                self._resolved_lora_path = lora_path

        if not model_path:
            self._mark_error(
                "베이스 GGUF를 찾지 못했습니다. `artifacts/base`에 .gguf를 두거나 BASE_MODEL_PATH를 설정하세요."
            )
            return
        if not Path(model_path).exists():
            self._mark_error(f"베이스 GGUF 파일을 찾을 수 없습니다: {model_path}")
            return
        if not use_merged_gguf and not lora_path:
            env_lora = os.getenv("LORA_ADAPTER_PATH", "").strip()
            env_peft = ""
            if env_lora and Path(env_lora).suffix.lower() in (".safetensors", ".bin"):
                env_peft = (
                    f" LORA_ADAPTER_PATH가 PEFT 가중치(.safetensors/.bin)를 가리킵니다: {env_lora}. "
                    "llama.cpp에는 GGUF 어댑터 경로가 필요합니다."
                )
            self._mark_error(
                "LoRA 어댑터 GGUF(adapter_model.gguf)를 찾지 못했습니다. "
                "`artifacts/lora`에 변환된 GGUF를 두거나 LORA_ADAPTER_PATH에 .gguf 경로를 설정하세요."
                " 변환 시 base HF 경로가 필요할 수 있으므로 "
                "`npm run convert:lora -- --base <HF_BASE_DIR>`를 권장합니다."
                + (env_peft or _lora_safetensors_only_hint())
            )
            return
        compare_model_path = merged_model_path if use_merged_gguf else lora_path
        if not Path(compare_model_path).exists():
            if use_merged_gguf:
                self._mark_error(f"병합 GGUF 파일을 찾을 수 없습니다: {compare_model_path}")
            else:
                self._mark_error(f"LoRA 어댑터 파일을 찾을 수 없습니다: {compare_model_path}")
            return

        n_ctx = int(os.getenv("LLAMA_N_CTX", "262144"))
        n_threads = int(os.getenv("LLAMA_N_THREADS", "8"))
        n_gpu_layers_req = int(os.getenv("LLAMA_N_GPU_LAYERS", "0"))
        # llama-cpp-python이 CPU 전용 빌드면 n_gpu_layers를 강제로 0으로 덮어씀(hang 방지).
        if n_gpu_layers_req != 0 and not LLAMA_SUPPORTS_GPU:
            logger.warning(
                "LLAMA_N_GPU_LAYERS=%s 요청되었으나 llama-cpp-python이 CPU 전용 빌드입니다. "
                "0으로 강제합니다. GPU 사용 원하면 `npm run install:llama-cuda` 재설치.",
                n_gpu_layers_req,
            )
            self._gpu_forced_off = True
            n_gpu_layers = 0
        else:
            self._gpu_forced_off = False
            n_gpu_layers = n_gpu_layers_req
        # 기본값을 verbose=True로(진행 파악 위해). 명시적으로 꺼진 경우만 False.
        verbose_env = os.getenv("LLAMA_VERBOSE", "").strip().lower()
        verbose = verbose_env not in ("0", "false", "no", "off")
        use_mmap = _env_use_mmap()
        use_mlock = _env_use_mlock()

        self._effective_n_gpu_layers = n_gpu_layers

        llama_kw: dict = dict(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            use_mmap=use_mmap,
            use_mlock=use_mlock,
            verbose=verbose,
        )
        n_batch = _env_optional_positive_int("LLAMA_N_BATCH")
        if n_batch is not None:
            llama_kw["n_batch"] = n_batch

        base_size_mb = int(Path(model_path).stat().st_size / (1024 * 1024))
        self._set_stage(
            "loading_base",
            f"베이스 GGUF 로드 중… ({base_size_mb} MB, n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers})",
        )
        try:
            self._base_llm = Llama(**llama_kw)
        except Exception as exc:
            err = str(exc)
            self._load_failed_stage = "base_gguf"
            self._load_traceback = traceback.format_exc()
            self._base_llm = None
            self._lora_llm = None
            self._mark_error(f"[베이스 GGUF] {err} — {_LOAD_FAILURE_HINT}")
            logger.warning("Llama 베이스 로드 실패: %s", err, exc_info=True)
            return

        self._set_stage("base_loaded", "베이스 로드 완료. LoRA 인스턴스 로드 준비 중…")

        compare_size_mb = int(Path(compare_model_path).stat().st_size / (1024 * 1024))
        if use_merged_gguf:
            self._set_stage(
                "loading_lora",
                f"비교용 merged.gguf 로드 중… ({compare_size_mb} MB)",
            )
            try:
                merged_kw = dict(llama_kw)
                merged_kw["model_path"] = compare_model_path
                self._lora_llm = Llama(**merged_kw)
                self._active_loras = None
            except Exception as exc:
                err = str(exc)
                self._load_failed_stage = "merged_gguf"
                self._load_traceback = traceback.format_exc()
                self._lora_llm = None
                self._base_llm = None
                self._mark_error(f"[merged GGUF] {err} — {_LOAD_FAILURE_HINT}")
                logger.warning("merged GGUF 로드 실패: %s", err, exc_info=True)
                return
            self._set_stage("ready", f"모델 로드 완료. 비교 모드=merged_gguf")
            return

        self._set_stage(
            "loading_lora",
            f"LoRA 적용 Llama 인스턴스 로드 중… (어댑터 {compare_size_mb} MB + 베이스 {base_size_mb} MB)",
        )
        try:
            self._lora_llm = Llama(**llama_kw)
            self._lora_llm.load_lora("adapter", compare_model_path)
            loaded = list(self._lora_llm.list_loras() or [])
            if not loaded:
                raise RuntimeError(
                    "load_lora() 호출 후에도 list_loras()가 비어 있습니다. "
                    "LoRA 어댑터 GGUF가 베이스 모델 아키텍처와 호환되지 않을 수 있습니다."
                )
            adapter_obj = self._lora_llm._model._lora_registry[loaded[0]]
            lora_scale = float(os.getenv("LLAMA_LORA_SCALE", "1.0"))
            self._lora_llm._ctx.apply_loras([(adapter_obj, lora_scale)])
            self._lora_scale = lora_scale
            # llama-cpp-python 0.3.36: eval() 호출 시 active_loras=None이면 clear_loras()를 수행함.
            # 따라서 생성 호출마다 active_loras를 전달해 어댑터를 유지해야 한다.
            self._active_loras = [{"name": loaded[0], "scale": lora_scale}]
            logger.info("LoRA 어댑터 로드+적용 완료: %s (scale=%s)", loaded, lora_scale)
        except Exception as exc:
            err = str(exc)
            self._load_failed_stage = "lora_gguf"
            self._load_traceback = traceback.format_exc()
            self._lora_llm = None
            self._base_llm = None
            self._mark_error(f"[LoRA GGUF] {err} — {_LOAD_FAILURE_HINT}")
            logger.warning("Llama LoRA 로드 실패: %s", err, exc_info=True)
            return
        self._set_stage("ready", f"모델 로드 완료. LoRA 어댑터: {loaded}")

    @property
    def is_ready(self) -> bool:
        return self._base_llm is not None and self._lora_llm is not None

    def get_loading_status(self) -> dict[str, Any]:
        """현재 로딩 진행 스냅샷(UI 표시용)."""
        with self._lock:
            stage = self._stage
            message = self._stage_message
            total_ms = (
                int((perf_counter() - self._loading_started_at) * 1000)
                if self._loading_started_at
                else 0
            )
            stage_ms = (
                int((perf_counter() - self._stage_started_at) * 1000)
                if self._stage_started_at
                else 0
            )
            stage_history = dict(self._stage_durations_ms)
        base_p = Path(self._resolved_model_path) if self._resolved_model_path else None
        lora_p = Path(self._resolved_lora_path) if self._resolved_lora_path else None

        process_info: dict[str, Any] = {}
        if PSUTIL_OK:
            try:
                p = psutil.Process()
                mem = p.memory_info()
                process_info = {
                    "rss_bytes": int(mem.rss),
                    "vms_bytes": int(mem.vms),
                    "num_threads": p.num_threads(),
                    "cpu_percent": p.cpu_percent(interval=None),
                }
                try:
                    process_info["private_bytes"] = int(getattr(mem, "private", 0))
                except Exception:
                    pass
                try:
                    vm = psutil.virtual_memory()
                    process_info["system_ram_total_bytes"] = int(vm.total)
                    process_info["system_ram_available_bytes"] = int(vm.available)
                    process_info["system_ram_percent"] = float(vm.percent)
                except Exception:
                    pass
            except Exception as exc:
                process_info = {"error": str(exc)}

        return {
            "stage": stage,
            "message": message,
            "ready": self.is_ready,
            "elapsed_total_ms": total_ms,
            "elapsed_stage_ms": stage_ms,
            "stage_durations_ms": stage_history,
            "base_file": {
                "path": self._resolved_model_path or None,
                "size_bytes": base_p.stat().st_size if base_p and base_p.is_file() else None,
            },
            "lora_file": {
                "path": self._resolved_lora_path or None,
                "size_bytes": lora_p.stat().st_size if lora_p and lora_p.is_file() else None,
            },
            "comparison_mode": self._comparison_mode,
            "process": process_info,
            "llama_cpp": {
                "version": getattr(_llama_cpp_module, "__version__", "unknown") if Llama else None,
                "supports_gpu_offload": LLAMA_SUPPORTS_GPU,
                "n_gpu_layers_requested": int(os.getenv("LLAMA_N_GPU_LAYERS", "0")),
                "n_gpu_layers_effective": getattr(self, "_effective_n_gpu_layers", None),
                "gpu_forced_off": getattr(self, "_gpu_forced_off", False),
            },
            "error_reason": self._fallback_reason if stage == "error" else None,
        }

    def error_detail(self) -> dict[str, Any]:
        """클라이언트에 노출할 진단 정보(파일 존재·크기·설정·스택)."""
        status = self.get_loading_status()
        mp = self._resolved_model_path
        lp = self._resolved_lora_path
        base_p = Path(mp) if mp else None
        lora_p = Path(lp) if lp else None
        detail: dict[str, Any] = {
            "ready": self.is_ready,
            "summary": self._fallback_reason or ("정상" if self.is_ready else f"로딩 중: {status['stage']}"),
            "loading_status": status,
            "load_failed_stage": self._load_failed_stage or None,
            "resolved_base_model_path": mp,
            "resolved_lora_adapter_path": lp,
            "comparison_mode": self._comparison_mode,
            "base_file_exists": base_p.is_file() if base_p else False,
            "lora_file_exists": lora_p.is_file() if lora_p else False,
            "llama_n_ctx": int(os.getenv("LLAMA_N_CTX", "262144")),
            "llama_n_gpu_layers": int(os.getenv("LLAMA_N_GPU_LAYERS", "0")),
            "llama_n_threads": int(os.getenv("LLAMA_N_THREADS", "8")),
            "llama_use_mmap": _env_use_mmap(),
            "llama_use_mlock": _env_use_mlock(),
            "llama_n_batch": _env_optional_positive_int("LLAMA_N_BATCH"),
        }
        if base_p and base_p.is_file():
            detail["base_file_size_bytes"] = base_p.stat().st_size
        if lora_p and lora_p.is_file():
            detail["lora_file_size_bytes"] = lora_p.stat().st_size
        if self._load_traceback:
            detail["python_traceback"] = self._load_traceback
        if self._llama_cpp_stderr:
            detail["llama_cpp_stderr"] = self._llama_cpp_stderr
        try:
            import llama_cpp as _lc  # type: ignore

            detail["llama_cpp_python_version"] = getattr(_lc, "__version__", "unknown")
        except Exception:
            pass
        return detail

    def comparison_debug(self) -> dict[str, Any]:
        loaded_loras: list[str] = []
        if self._lora_llm is not None:
            try:
                loaded_loras = list(self._lora_llm.list_loras() or [])
            except Exception:
                loaded_loras = []
        return {
            "comparison_mode": self._comparison_mode,
            "resolved_base_model_path": self._resolved_model_path or None,
            "resolved_compare_model_path": self._resolved_lora_path or None,
            "lora_scale": getattr(self, "_lora_scale", None),
            "loaded_loras": loaded_loras,
            "ready": self.is_ready,
            "load_failed_stage": self._load_failed_stage or None,
        }

    def _generate(self, llm: "Llama", prompt: str, **kwargs) -> InferenceOutput:
        started_at = perf_counter()
        call_kwargs: dict[str, Any] = dict(
            prompt=prompt,
            max_tokens=kwargs["max_tokens"],
            temperature=kwargs["temperature"],
            top_k=kwargs["top_k"],
            top_p=kwargs["top_p"],
            seed=kwargs["seed"],
            echo=False,
        )
        if kwargs.get("active_loras") is not None:
            call_kwargs["active_loras"] = kwargs["active_loras"]
        output = llm(**call_kwargs)
        text = output["choices"][0]["text"]
        elapsed_ms = int((perf_counter() - started_at) * 1000)
        return InferenceOutput(text=text, duration_ms=elapsed_ms)

    def iter_completion_chunks(self, llm: "Llama", prompt: str, **kwargs) -> Iterator[str]:
        """스트리밍 생성: llama.cpp가 내보내는 텍스트 조각(토큰 디코딩 단위)을 순서대로 반환."""
        call_kwargs: dict[str, Any] = dict(
            prompt=prompt,
            max_tokens=kwargs["max_tokens"],
            temperature=kwargs["temperature"],
            top_k=kwargs["top_k"],
            top_p=kwargs["top_p"],
            seed=kwargs["seed"],
            echo=False,
            stream=True,
        )
        if kwargs.get("active_loras") is not None:
            call_kwargs["active_loras"] = kwargs["active_loras"]
        stream = llm(**call_kwargs)
        for part in stream:
            piece = part["choices"][0].get("text") or ""
            if piece:
                yield piece

    def stream_base_chunks(self, prompt: str, **kwargs) -> Iterator[str]:
        if self._base_llm is None:
            raise RuntimeError("베이스 모델이 로드되지 않았습니다.")
        yield from self.iter_completion_chunks(self._base_llm, prompt, **kwargs)

    def stream_lora_chunks(self, prompt: str, **kwargs) -> Iterator[str]:
        if self._lora_llm is None:
            raise RuntimeError("LoRA 모델이 로드되지 않았습니다.")
        yield from self.iter_completion_chunks(
            self._lora_llm,
            prompt,
            active_loras=self._active_loras,
            **kwargs,
        )

    def generate_base(self, prompt: str, **kwargs) -> InferenceOutput:
        if self._base_llm is None:
            return InferenceOutput(
                text=f"[fallback-base] {self._fallback_reason}\n\n{prompt}",
                duration_ms=0,
            )
        return self._generate(self._base_llm, prompt, **kwargs)

    def generate_lora(self, prompt: str, **kwargs) -> InferenceOutput:
        if self._lora_llm is None:
            return InferenceOutput(
                text=f"[fallback-lora] {self._fallback_reason}\n\n{prompt}",
                duration_ms=0,
            )
        return self._generate(
            self._lora_llm,
            prompt,
            active_loras=self._active_loras,
            **kwargs,
        )


_inference_singleton: InferenceService | None = None


def get_inference_service() -> InferenceService:
    global _inference_singleton
    if _inference_singleton is None:
        _inference_singleton = InferenceService()
    return _inference_singleton


def get_inference_service_if_loaded() -> InferenceService | None:
    return _inference_singleton


def peek_inference_environment() -> dict[str, Any]:
    """Llama 인스턴스를 만들지 않고 경로·환경만 진단한다(즉시 응답)."""
    mp = _resolve_model_path()
    lp = _resolve_lora_path()
    merged = _resolve_merged_model_path()
    use_merged = bool(merged)
    compare_path = merged if use_merged else lp
    base_p = Path(mp) if mp else None
    lora_p = Path(compare_path) if compare_path else None
    st = _discover_peft_safetensors()
    detail: dict[str, Any] = {
        "peek_only": True,
        "llama_cpp_import_ok": Llama is not None,
        "llama_cpp_import_error": LLAMA_IMPORT_ERROR or None,
        "resolved_base_model_path": mp or None,
        "resolved_lora_adapter_path": compare_path or None,
        "comparison_mode": "merged_gguf" if use_merged else "lora_adapter",
        "base_file_exists": base_p.is_file() if base_p else False,
        "lora_file_exists": lora_p.is_file() if lora_p else False,
        "llama_n_ctx": int(os.getenv("LLAMA_N_CTX", "262144")),
        "llama_n_gpu_layers": int(os.getenv("LLAMA_N_GPU_LAYERS", "0")),
        "llama_n_threads": int(os.getenv("LLAMA_N_THREADS", "8")),
        "llama_use_mmap": _env_use_mmap(),
        "llama_use_mlock": _env_use_mlock(),
        "llama_n_batch": _env_optional_positive_int("LLAMA_N_BATCH"),
        "lora_auto_convert_env": os.getenv("LORA_AUTO_CONVERT_GGUF", "1"),
        "windows_gpu_notes": (
            "정책: LLAMA_USE_MMAP=0이면 가중치를 RAM에 읽어 온 뒤, n_gpu_layers>0인 레이어는 VRAM에서 연산합니다. "
            "Windows pip 기본 llama-cpp-python은 CPU 빌드인 경우가 많아 GPU가 안 쓰입니다. "
            "루트에서 npm run install:llama-cuda (CUDA Toolkit+VS C++ 필요) 또는 CUDA 프리빌트 휠. 참고: "
            "https://github.com/abetlen/llama-cpp-python/issues/2079"
        ),
    }
    if base_p and base_p.is_file():
        detail["base_file_size_bytes"] = base_p.stat().st_size
    if lora_p and lora_p.is_file():
        detail["lora_file_size_bytes"] = lora_p.stat().st_size
    if not use_merged and not lp and st:
        detail["peft_safetensors_found"] = str(st)
        detail["hint"] = (
            "PEFT adapter_model.safetensors만 있고 GGUF가 없습니다. "
            "첫 요청 시 LORA_AUTO_CONVERT_GGUF=1이면 변환을 시도합니다. "
            "실패하면 `npm run convert:lora -- --base <HF_BASE_DIR>`로 명시 실행하세요."
        )
    if Llama is None:
        detail["summary"] = "llama_cpp 모듈을 불러올 수 없습니다."
    elif not mp:
        detail["summary"] = "베이스 GGUF 경로를 찾지 못했습니다."
    elif not base_p or not base_p.is_file():
        detail["summary"] = f"베이스 GGUF 파일이 없습니다: {mp}"
    elif not compare_path:
        detail["summary"] = (
            "MERGED_MODEL_GGUF 또는 LoRA 어댑터 GGUF를 찾지 못했습니다."
            if not use_merged
            else "MERGED_MODEL_GGUF를 찾지 못했습니다."
        )
    elif not lora_p or not lora_p.is_file():
        detail["summary"] = f"LoRA GGUF 파일이 없습니다: {lp}"
    else:
        detail["summary"] = (
            "경로·파일은 확인됨. 실제 로드는 첫 추론 요청 시 진행되며 수 분 걸릴 수 있습니다."
        )
    return detail

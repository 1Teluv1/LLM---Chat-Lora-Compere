import json
import logging
import os
import queue
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.application.compare_service import CompareService
from app.config.settings import env_flag, optional_positive_int
from app.download_assets import download_artifact
from app.domain.models import CompareInput, GenerationOptions, LlamaLoadOverrides
from app.schemas import (
    ArtifactDownloadRequest,
    ArtifactDownloadResponse,
    CompareRequest,
    CompareResponse,
    GenerationResult,
)

logger = logging.getLogger(__name__)
load_dotenv()

app = FastAPI(title="LoRA Compare Backend", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
compare_service = CompareService()


def _to_domain(req: CompareRequest) -> CompareInput:
    llama: LlamaLoadOverrides | None = None
    if req.llama_load is not None:
        lc = req.llama_load
        llama = LlamaLoadOverrides(
            n_ctx=lc.n_ctx,
            n_threads=lc.n_threads,
            n_gpu_layers=lc.n_gpu_layers,
            n_batch=lc.n_batch,
            use_mmap=lc.use_mmap,
            use_mlock=lc.use_mlock,
        )
    return CompareInput(
        prompt=req.prompt,
        system_prompt=req.system_prompt,
        enable_thinking=req.enable_thinking,
        runtime=req.runtime,
        base_model_id=req.base_model_id,
        lora_id=req.lora_id,
        lora_strategy=req.lora_strategy,  # type: ignore[arg-type]
        device_hint=req.device_hint,  # type: ignore[arg-type]
        llama_load=llama,
        options=GenerationOptions(
            seed=req.seed,
            top_k=req.top_k,
            top_p=req.top_p,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        ),
    )


@app.on_event("startup")
def _preload_models() -> None:
    runtime = os.getenv("INFERENCE_RUNTIME", "llama_cpp")
    compare_service.get_runtime(runtime).start_loading_async(None)


@app.get("/runtime/llama-defaults")
def llama_load_defaults() -> dict[str, Any]:
    """UI 초기값용: 현재 프로세스 환경 변수에서 llama.cpp 로드 기본을 읽습니다."""
    return {
        "n_ctx": int(os.getenv("LLAMA_N_CTX", "8192")),
        "n_threads": int(os.getenv("LLAMA_N_THREADS", "8")),
        "n_gpu_layers": int(os.getenv("LLAMA_N_GPU_LAYERS", "0")),
        "n_batch": optional_positive_int("LLAMA_N_BATCH"),
        "use_mmap": env_flag("LLAMA_USE_MMAP", True),
        "use_mlock": env_flag("LLAMA_USE_MLOCK", False),
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/debug/runtime")
def debug_runtime() -> dict[str, str]:
    runtime = os.getenv("INFERENCE_RUNTIME", "llama_cpp")
    return {"python": sys.executable, "runtime_default": runtime}


@app.get("/debug/inference")
def debug_inference(runtime: str = "llama_cpp") -> dict:
    return compare_service.error_detail(runtime)


@app.get("/runtime/status")
def runtime_status(runtime: str = "llama_cpp") -> dict:
    return compare_service.loading_status(runtime)


@app.post("/artifacts/download", response_model=ArtifactDownloadResponse)
def artifacts_download(request: ArtifactDownloadRequest) -> ArtifactDownloadResponse:
    try:
        payload = download_artifact(
            target_type=request.target_type,
            repo_id=request.repo_id,
            filename=request.filename,
            allow_patterns=request.allow_patterns,
            output_subdir=request.output_subdir,
            repo_type=request.repo_type,
        )
        return ArtifactDownloadResponse(**payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        message = str(exc)
        lower = message.lower()
        if "401" in lower or "403" in lower or "token" in lower:
            raise HTTPException(
                status_code=401,
                detail="Hugging Face 인증 실패: HF_TOKEN/HUGGING_FACE_HUB_TOKEN을 확인하세요.",
            ) from exc
        if "404" in lower or "not found" in lower:
            raise HTTPException(status_code=404, detail="repo_id 또는 파일명을 찾을 수 없습니다.") from exc
        if "429" in lower or "rate limit" in lower:
            raise HTTPException(status_code=429, detail="Hugging Face 요청 제한에 걸렸습니다. 잠시 후 재시도하세요.") from exc
        raise HTTPException(
            status_code=502,
            detail=f"Hugging Face 다운로드 실패: {message}",
        ) from exc


def _phase_inference_summary(
    phase: str,
    duration_ms: int,
    text: str,
    stream_chunks: int,
    t_start: float,
    t_first: float | None,
    max_tokens: int,
) -> dict[str, Any]:
    out_chars = len(text)
    lines = text.count("\n") + 1 if out_chars else 0
    ttft_ms = None
    if t_first is not None:
        ttft_ms = int((t_first - t_start) * 1000)
    return {
        "phase": phase,
        "duration_ms": duration_ms,
        "output_chars": out_chars,
        "output_lines": lines,
        "stream_chunks": stream_chunks,
        "time_to_first_chunk_ms": ttft_ms,
        "max_tokens_requested": max_tokens,
    }


def _run_inference_log(
    request: CompareRequest,
    base_summary: dict[str, Any],
    lora_summary: dict[str, Any],
    mode: str,
    prompt_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "runtime": request.runtime,
        "mode": mode,
        "base": base_summary,
        "lora": lora_summary,
        "run_total_ms": int(base_summary["duration_ms"] + lora_summary["duration_ms"]),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if prompt_info:
        row["prompt"] = prompt_info
    return row


def _inference_log_sync(
    request: CompareRequest,
    base_text: str,
    lora_text: str,
    base_ms: int,
    lora_ms: int,
    prompt_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base_s = _phase_inference_summary("base", base_ms, base_text, 1, 0.0, None, request.max_tokens)
    lora_s = _phase_inference_summary("lora", lora_ms, lora_text, 1, 0.0, None, request.max_tokens)
    return _run_inference_log(request, base_s, lora_s, "sync", prompt_info)


@app.post("/compare", response_model=CompareResponse)
def compare(request: CompareRequest) -> CompareResponse:
    data = _to_domain(request)
    try:
        out = compare_service.compare_once(data)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except RuntimeError as exc:
        runtime = compare_service.get_runtime(data.runtime)
        raise HTTPException(status_code=503, detail=runtime.error_detail()) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": str(exc), "error_type": type(exc).__name__, "traceback": traceback.format_exc()},
        ) from exc
    runtime = compare_service.get_runtime(data.runtime)
    pi = runtime.prompt_token_info(data)
    inf = _inference_log_sync(
        request,
        out.base.text,
        out.lora.text,
        out.base.duration_ms,
        out.lora.duration_ms,
        pi if pi else None,
    )
    pt = (pi or {}).get("rendered_prompt_tokens")
    logger.info(
        "[inference] Base·LoRA 비교(동기) 완료 — 총 %dms | Base %dms, %d자 | LoRA %dms, %d자 | max_tokens=%d | 프롬프트 토큰=%s",
        inf["run_total_ms"],
        out.base.duration_ms,
        len(out.base.text),
        out.lora.duration_ms,
        len(out.lora.text),
        request.max_tokens,
        pt if pt is not None else "—",
    )
    return CompareResponse(
        base=GenerationResult(text=out.base.text, duration_ms=out.base.duration_ms),
        lora=GenerationResult(text=out.lora.text, duration_ms=out.lora.duration_ms),
        params=request,
        debug=out.debug,
        inference_log=inf,
    )


def _sse_event(obj: dict) -> bytes:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")


_SSE_QUEUE_DONE = object()


@app.post("/compare/stream")
def compare_stream(request: CompareRequest):
    data = _to_domain(request)
    runtime = compare_service.get_runtime(data.runtime)

    def sse_blob_producer() -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=256)

        def worker() -> None:
            try:
                max_tok = int(request.max_tokens)
                prompt_info = runtime.prompt_token_info(data)
                q.put(_sse_event({"type": "phase", "phase": "base", "event": "start"}))
                t0 = time.perf_counter()
                base_parts: list[str] = []
                base_chunk_count = 0
                t_base_first: float | None = None
                for chunk in runtime.stream_base_chunks(data):
                    if t_base_first is None:
                        t_base_first = time.perf_counter()
                    base_chunk_count += 1
                    base_parts.append(chunk)
                    q.put(_sse_event({"type": "delta", "phase": "base", "text": chunk}))
                base_text = "".join(base_parts)
                base_ms = int((time.perf_counter() - t0) * 1000)
                base_inf = _phase_inference_summary(
                    "base", base_ms, base_text, base_chunk_count, t0, t_base_first, max_tok
                )
                logger.info(
                    "[inference] Base LLM 응답 완료 — %dms, 출력 %d자, 줄 %d, TTFT %s, 스트림 청크 %d, max_tokens=%d",
                    base_ms,
                    base_inf["output_chars"],
                    base_inf["output_lines"],
                    f"{base_inf['time_to_first_chunk_ms']}ms" if base_inf["time_to_first_chunk_ms"] is not None else "—",
                    base_chunk_count,
                    max_tok,
                )
                q.put(
                    _sse_event(
                        {
                            "type": "phase",
                            "phase": "base",
                            "event": "end",
                            "duration_ms": base_ms,
                            "text": base_text,
                            "inference": base_inf,
                        }
                    )
                )

                q.put(_sse_event({"type": "phase", "phase": "lora", "event": "start"}))
                t1 = time.perf_counter()
                lora_parts: list[str] = []
                lora_chunk_count = 0
                t_lora_first: float | None = None
                for chunk in runtime.stream_lora_chunks(data):
                    if t_lora_first is None:
                        t_lora_first = time.perf_counter()
                    lora_chunk_count += 1
                    lora_parts.append(chunk)
                    q.put(_sse_event({"type": "delta", "phase": "lora", "text": chunk}))
                lora_text = "".join(lora_parts)
                lora_ms = int((time.perf_counter() - t1) * 1000)
                lora_inf = _phase_inference_summary(
                    "lora", lora_ms, lora_text, lora_chunk_count, t1, t_lora_first, max_tok
                )
                logger.info(
                    "[inference] LoRA LLM 응답 완료 — %dms, 출력 %d자, 줄 %d, TTFT %s, 스트림 청크 %d, max_tokens=%d",
                    lora_ms,
                    lora_inf["output_chars"],
                    lora_inf["output_lines"],
                    f"{lora_inf['time_to_first_chunk_ms']}ms" if lora_inf["time_to_first_chunk_ms"] is not None else "—",
                    lora_chunk_count,
                    max_tok,
                )
                q.put(
                    _sse_event(
                        {
                            "type": "phase",
                            "phase": "lora",
                            "event": "end",
                            "duration_ms": lora_ms,
                            "text": lora_text,
                            "inference": lora_inf,
                        }
                    )
                )

                inference_log = _run_inference_log(
                    request, base_inf, lora_inf, "stream", prompt_info if prompt_info else None
                )
                pt = (prompt_info or {}).get("rendered_prompt_tokens")
                logger.info(
                    "[inference] Base·LoRA 비교 스트리밍 전체 완료 — 총 %dms (Base %dms + LoRA %dms) | 프롬프트 토큰=%s | %s",
                    inference_log["run_total_ms"],
                    base_ms,
                    lora_ms,
                    pt if pt is not None else "—",
                    inference_log["finished_at_utc"],
                )
                q.put(
                    _sse_event(
                        {
                            "type": "done",
                            "base": {"text": base_text, "duration_ms": base_ms},
                            "lora": {"text": lora_text, "duration_ms": lora_ms},
                            "params": request.model_dump(),
                            "debug": runtime.comparison_debug(),
                            "inference_log": inference_log,
                        }
                    )
                )
            except Exception as exc:
                q.put(
                    _sse_event(
                        {
                            "type": "error",
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                            "traceback": traceback.format_exc(),
                        }
                    )
                )
            finally:
                q.put(_SSE_QUEUE_DONE)

        threading.Thread(target=worker, daemon=True).start()
        return q

    def generate():
        yield _sse_event({"type": "meta", "event": "connected", "message": "SSE 연결됨."})
        runtime.start_loading_async(data)
        poll_interval = float(os.getenv("LOADING_POLL_SEC", "1.0") or "1.0")
        last_stage = ""
        while True:
            status = runtime.get_loading_status()
            stage = status["stage"]
            event_name = "stage" if stage != last_stage else "loading_progress"
            yield _sse_event({"type": "meta", "event": event_name, "stage": stage, "message": status.get("message"), "status": status})
            last_stage = stage
            if stage == "error":
                yield _sse_event({"type": "error", "error": status.get("error_reason") or "모델 로드 실패", "error_type": "InferenceLoadError", "detail": runtime.error_detail()})
                return
            if stage == "ready" and runtime.is_ready():
                break
            time.sleep(poll_interval)

        yield _sse_event({"type": "meta", "event": "ready", "message": "로드 완료. Base 스트리밍 시작.", "status": runtime.get_loading_status()})
        q = sse_blob_producer()
        keepalive_sec = float(os.getenv("SSE_KEEPALIVE_SEC", "12") or "12")
        while True:
            try:
                item = q.get(timeout=keepalive_sec)
            except queue.Empty:
                yield b": keepalive\n\n"
                continue
            if item is _SSE_QUEUE_DONE:
                break
            yield item

    return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})

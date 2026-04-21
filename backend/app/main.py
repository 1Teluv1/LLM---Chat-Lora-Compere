import json
import logging
import os
import queue
import sys
import threading
import time
import traceback

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.application.compare_service import CompareService
from app.domain.models import CompareInput, GenerationOptions
from app.schemas import CompareRequest, CompareResponse, GenerationResult

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
    return CompareInput(
        prompt=req.prompt,
        runtime=req.runtime,
        base_model_id=req.base_model_id,
        lora_id=req.lora_id,
        lora_strategy=req.lora_strategy,  # type: ignore[arg-type]
        device_hint=req.device_hint,  # type: ignore[arg-type]
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
    compare_service.select_runtime(runtime).start_loading_async()


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


@app.post("/compare", response_model=CompareResponse)
def compare(request: CompareRequest) -> CompareResponse:
    data = _to_domain(request)
    runtime = compare_service.select_runtime(data.runtime)
    if not runtime.is_ready():
        raise HTTPException(status_code=503, detail=runtime.error_detail())
    try:
        out = compare_service.compare_once(data)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": str(exc), "error_type": type(exc).__name__, "traceback": traceback.format_exc()},
        ) from exc
    return CompareResponse(
        base=GenerationResult(text=out.base.text, duration_ms=out.base.duration_ms),
        lora=GenerationResult(text=out.lora.text, duration_ms=out.lora.duration_ms),
        params=request,
        debug=out.debug,
    )


def _sse_event(obj: dict) -> bytes:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")


_SSE_QUEUE_DONE = object()


@app.post("/compare/stream")
def compare_stream(request: CompareRequest):
    data = _to_domain(request)
    runtime = compare_service.select_runtime(data.runtime)

    def sse_blob_producer() -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=256)

        def worker() -> None:
            try:
                q.put(_sse_event({"type": "phase", "phase": "base", "event": "start"}))
                t0 = time.perf_counter()
                base_parts: list[str] = []
                for chunk in runtime.stream_base_chunks(data):
                    base_parts.append(chunk)
                    q.put(_sse_event({"type": "delta", "phase": "base", "text": chunk}))
                base_ms = int((time.perf_counter() - t0) * 1000)
                q.put(_sse_event({"type": "phase", "phase": "base", "event": "end", "duration_ms": base_ms, "text": "".join(base_parts)}))

                q.put(_sse_event({"type": "phase", "phase": "lora", "event": "start"}))
                t1 = time.perf_counter()
                lora_parts: list[str] = []
                for chunk in runtime.stream_lora_chunks(data):
                    lora_parts.append(chunk)
                    q.put(_sse_event({"type": "delta", "phase": "lora", "text": chunk}))
                lora_ms = int((time.perf_counter() - t1) * 1000)
                q.put(_sse_event({"type": "phase", "phase": "lora", "event": "end", "duration_ms": lora_ms, "text": "".join(lora_parts)}))

                q.put(
                    _sse_event(
                        {
                            "type": "done",
                            "base": {"text": "".join(base_parts), "duration_ms": base_ms},
                            "lora": {"text": "".join(lora_parts), "duration_ms": lora_ms},
                            "params": request.model_dump(),
                            "debug": runtime.comparison_debug(),
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
        runtime.start_loading_async()
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

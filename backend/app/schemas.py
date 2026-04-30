from typing import Any
from typing import Literal

from pydantic import BaseModel, Field


class LlamaLoadConfig(BaseModel):
    """llama.cpp 모델 로드 시 적용(n_ctx 등). 값이 있으면 .env 기본값을 덮어씁니다."""

    # 0 또는 null: 서버 LLAMA_N_CTX(.env) 사용. 지정 시 512~262144로 클램프.
    n_ctx: int | None = Field(default=None, ge=0, le=262144)
    n_threads: int | None = Field(default=None, ge=1, le=256)
    n_gpu_layers: int | None = Field(default=None, ge=-1, le=65536)
    n_batch: int | None = Field(default=None, ge=32, le=65536)
    use_mmap: bool | None = None
    use_mlock: bool | None = None


class CompareRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    system_prompt: str | None = Field(default=None)
    enable_thinking: bool = Field(default=False)
    run_mode: Literal["both", "base_only", "lora_only"] = Field(
        default="both",
        description="both: Base·LoRA 순차 실행, base_only: 베이스만, lora_only: LoRA(어댑터) 경로만",
    )
    seed: int = Field(default=42)
    top_k: int = Field(default=40, ge=1)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    temperature: float = Field(default=0.7, ge=0.0)
    max_tokens: int = Field(default=512, ge=1, le=262144)
    runtime: str = Field(default="llama_cpp")
    base_model_id: str | None = Field(default=None)
    lora_id: str | None = Field(default=None)
    lora_strategy: str = Field(default="auto")
    device_hint: str = Field(default="auto")
    llama_load: LlamaLoadConfig | None = Field(
        default=None,
        description="llama_cpp 런타임 전용 로드 옵션. 바꾸면 모델을 다시 로드합니다.",
    )


class GenerationResult(BaseModel):
    text: str
    duration_ms: int


class CompareResponse(BaseModel):
    base: GenerationResult
    lora: GenerationResult
    params: CompareRequest
    debug: dict[str, Any] | None = None
    inference_log: dict[str, Any] | None = None


class ArtifactDownloadRequest(BaseModel):
    repo_id: str = Field(..., min_length=1, description="Hugging Face repo id")
    target_type: Literal["base", "lora"] = Field(
        ..., description="다운로드 대상 유형(base gguf / lora peft)"
    )
    filename: str | None = Field(
        default=None,
        description="단일 파일 다운로드 시 파일명. base에서는 필수 권장",
    )
    allow_patterns: list[str] | None = Field(
        default=None,
        description="snapshot_download 필터 패턴(예: adapter_*.*)",
    )
    output_subdir: str | None = Field(
        default=None,
        description="artifacts/base|lora 하위 저장 폴더",
    )
    repo_type: Literal["model", "dataset", "space"] = Field(default="model")


class ArtifactDownloadResponse(BaseModel):
    success: bool = True
    target_type: Literal["base", "lora"]
    repo_id: str
    resolved_path: str = Field(description="저장된 상대 경로")
    detected_files: dict[str, str | None]
    warnings: list[str] = Field(default_factory=list)

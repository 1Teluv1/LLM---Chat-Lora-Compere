from typing import Any
from typing import Literal

from pydantic import BaseModel, Field


class CompareRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    system_prompt: str | None = Field(default=None)
    enable_thinking: bool = Field(default=False)
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


class GenerationResult(BaseModel):
    text: str
    duration_ms: int


class CompareResponse(BaseModel):
    base: GenerationResult
    lora: GenerationResult
    params: CompareRequest
    debug: dict[str, Any] | None = None


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

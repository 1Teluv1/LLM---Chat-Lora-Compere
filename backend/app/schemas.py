from pydantic import BaseModel, Field
from typing import Any


class CompareRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    seed: int = Field(default=42)
    top_k: int = Field(default=40, ge=1)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    temperature: float = Field(default=0.7, ge=0.0)
    max_tokens: int = Field(default=512, ge=1, le=4096)
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

from __future__ import annotations

import os
from pathlib import Path

from app.config.settings import repo_root

_PREFERRED_BASE_GGUF = "Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"


class ModelResolver:
    def _discover_base_gguf(self) -> Path | None:
        base_dir = repo_root() / "artifacts" / "base"
        if not base_dir.is_dir():
            return None
        ggufs = sorted(base_dir.rglob("*.gguf"))
        if not ggufs:
            return None
        preferred = base_dir / _PREFERRED_BASE_GGUF
        if preferred.is_file():
            return preferred
        return ggufs[0]

    def _discover_lora_gguf(self) -> Path | None:
        lora_dir = repo_root() / "artifacts" / "lora"
        if not lora_dir.is_dir():
            return None
        candidates = sorted(lora_dir.rglob("adapter_model.gguf"))
        return candidates[0] if candidates else None

    def _discover_merged_gguf(self) -> Path | None:
        merged_dir = repo_root() / "artifacts" / "merged"
        if not merged_dir.is_dir():
            return None
        ggufs = sorted(merged_dir.rglob("*.gguf"))
        return ggufs[0] if ggufs else None

    def resolve_for_llama(self, strategy: str = "auto") -> dict[str, str]:
        base = self._discover_base_gguf()
        base_raw = os.getenv("BASE_MODEL_PATH", "").strip()
        base_path = str(base.resolve()) if base else (str(Path(base_raw).resolve()) if base_raw else "")

        merged = self._discover_merged_gguf()
        lora = self._discover_lora_gguf()
        merged_raw = os.getenv("MERGED_MODEL_GGUF", "").strip()
        lora_raw = os.getenv("LORA_ADAPTER_PATH", "").strip()
        merged_path = str(merged.resolve()) if merged else (str(Path(merged_raw).resolve()) if merged_raw else "")
        lora_path = str(lora.resolve()) if lora else (str(Path(lora_raw).resolve()) if lora_raw else "")

        if strategy == "merged" and merged_path:
            compare_mode = "merged_gguf"
            compare_path = merged_path
        elif strategy == "adapter" and lora_path:
            compare_mode = "lora_adapter"
            compare_path = lora_path
        elif merged_path:
            compare_mode = "merged_gguf"
            compare_path = merged_path
        else:
            compare_mode = "lora_adapter"
            compare_path = lora_path
        return {
            "base_model_path": base_path,
            "compare_model_path": compare_path,
            "comparison_mode": compare_mode,
        }

from __future__ import annotations

import logging
import os
from pathlib import Path

from app.config.settings import repo_root

_PREFERRED_BASE_GGUF = "Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"
logger = logging.getLogger(__name__)


class ModelResolver:
    @staticmethod
    def _resolve_user_path(raw: str | None) -> Path | None:
        if not raw:
            return None
        s = raw.strip()
        if not s:
            return None
        p = Path(s)
        if not p.is_absolute():
            p = (repo_root() / p).resolve()
        else:
            p = p.resolve()
        return p if p.exists() else None

    @staticmethod
    def _auto_convert_lora_dir_if_needed(lora_dir: Path) -> Path | None:
        gguf = lora_dir / "adapter_model.gguf"
        if gguf.is_file():
            return gguf
        st = lora_dir / "adapter_model.safetensors"
        if not st.is_file():
            return None
        try:
            from app.lora_gguf_convert import convert_peft_adapter, infer_base_dir_for_lora

            out = convert_peft_adapter(
                lora_dir,
                base_dir=infer_base_dir_for_lora(lora_dir),
                outfile=gguf,
            )
            return out if out.is_file() else None
        except Exception as exc:
            logger.warning("LoRA GGUF 자동 변환 실패 (%s): %s", lora_dir, exc)
            return None

    def _resolve_base_from_id(self, base_model_id: str | None) -> Path | None:
        p = self._resolve_user_path(base_model_id)
        if p is None:
            return None
        if p.is_file() and p.suffix.lower() == ".gguf" and p.name.lower() != "adapter_model.gguf":
            return p
        if p.is_dir():
            ggufs = sorted(x for x in p.rglob("*.gguf") if x.name.lower() != "adapter_model.gguf")
            return ggufs[0] if ggufs else None
        return None

    def _resolve_lora_from_id(self, lora_id: str | None, strategy: str) -> tuple[Path | None, str]:
        p = self._resolve_user_path(lora_id)
        if p is None:
            return (None, "lora_adapter")
        if p.is_file() and p.suffix.lower() == ".gguf":
            if p.name.lower() == "adapter_model.gguf":
                return (p, "lora_adapter")
            return (p, "merged_gguf")
        if not p.is_dir():
            return (None, "lora_adapter")

        adapter_gguf = p / "adapter_model.gguf"
        merged_candidates = sorted(x for x in p.rglob("*.gguf") if x.name.lower() != "adapter_model.gguf")
        merged_gguf = merged_candidates[0] if merged_candidates else None

        if strategy == "merged":
            return (merged_gguf, "merged_gguf")
        if strategy == "adapter":
            return (self._auto_convert_lora_dir_if_needed(p), "lora_adapter")
        if merged_gguf is not None:
            return (merged_gguf, "merged_gguf")
        if adapter_gguf.is_file():
            return (adapter_gguf, "lora_adapter")
        return (self._auto_convert_lora_dir_if_needed(p), "lora_adapter")

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
        artifacts_dir = repo_root() / "artifacts"
        if not artifacts_dir.is_dir():
            return None
        candidates = sorted(artifacts_dir.rglob("adapter_model.gguf"))
        return candidates[0] if candidates else None

    def _discover_merged_gguf(self) -> Path | None:
        merged_dir = repo_root() / "artifacts" / "merged"
        if not merged_dir.is_dir():
            return None
        ggufs = sorted(merged_dir.rglob("*.gguf"))
        return ggufs[0] if ggufs else None

    def resolve_for_llama(
        self,
        strategy: str = "auto",
        *,
        base_model_id: str | None = None,
        lora_id: str | None = None,
    ) -> dict[str, str]:
        base = self._resolve_base_from_id(base_model_id) or self._discover_base_gguf()
        base_raw = os.getenv("BASE_MODEL_PATH", "").strip()
        base_path = str(base.resolve()) if base else (str(Path(base_raw).resolve()) if base_raw else "")

        selected_compare, selected_mode = self._resolve_lora_from_id(lora_id, strategy)
        merged = self._discover_merged_gguf()
        lora = self._discover_lora_gguf()
        merged_raw = os.getenv("MERGED_MODEL_GGUF", "").strip()
        lora_raw = os.getenv("LORA_ADAPTER_PATH", "").strip()
        merged_path = str(merged.resolve()) if merged else (str(Path(merged_raw).resolve()) if merged_raw else "")
        lora_path = str(lora.resolve()) if lora else (str(Path(lora_raw).resolve()) if lora_raw else "")

        if selected_compare is not None:
            compare_mode = selected_mode
            compare_path = str(selected_compare.resolve())
            return {
                "base_model_path": base_path,
                "compare_model_path": compare_path,
                "comparison_mode": compare_mode,
            }

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

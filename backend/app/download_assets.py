import argparse
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, snapshot_download

DEFAULT_BASE_REPO_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
DEFAULT_BASE_FILENAME = "Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"
DEFAULT_LORA_REPO_ID = "Teluv/LLM-FineTuning"


def _load_repo_dotenv() -> None:
    """프로젝트 루트의 .env를 로드한다. `python backend/app/...` 직접 실행 시에도 동작하게 한다."""
    repo_root = Path(__file__).resolve().parents[2]
    env_path = repo_root / ".env"
    if env_path.is_file():
        load_dotenv(env_path)


def require_hf_token() -> str:
    _load_repo_dotenv()
    token = (
        os.getenv("HF_TOKEN", "").strip()
        or os.getenv("HUGGING_FACE_HUB_TOKEN", "").strip()
    )
    if not token:
        raise RuntimeError(
            "HF_TOKEN(또는 HUGGING_FACE_HUB_TOKEN)이 비어 있습니다. "
            "셸에 export 하거나 프로젝트 루트 .env에 넣은 뒤 다시 실행하세요."
        )
    return token


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _safe_output_dir(target_type: Literal["base", "lora"], output_subdir: str | None) -> Path:
    root = (_repo_root() / "artifacts" / target_type).resolve()
    if not output_subdir:
        return root
    candidate = (root / output_subdir).resolve()
    if not str(candidate).startswith(str(root)):
        raise ValueError("output_subdir는 artifacts 하위 경로만 허용됩니다.")
    return candidate


def _to_repo_relative(path: Path) -> str:
    return path.resolve().relative_to(_repo_root()).as_posix()


def _detect_downloaded_files(target_type: Literal["base", "lora"], saved_path: Path) -> dict[str, str | None]:
    if target_type == "base":
        base_file = saved_path if saved_path.is_file() else None
        return {
            "base_gguf": _to_repo_relative(base_file) if base_file else None,
            "adapter_model_safetensors": None,
            "adapter_config_json": None,
            "adapter_model_gguf": None,
        }

    root = saved_path if saved_path.is_dir() else saved_path.parent
    safetensors = sorted(root.rglob("adapter_model.safetensors"))
    config = sorted(root.rglob("adapter_config.json"))
    adapter_gguf = sorted(root.rglob("adapter_model.gguf"))
    return {
        "base_gguf": None,
        "adapter_model_safetensors": _to_repo_relative(safetensors[0]) if safetensors else None,
        "adapter_config_json": _to_repo_relative(config[0]) if config else None,
        "adapter_model_gguf": _to_repo_relative(adapter_gguf[0]) if adapter_gguf else None,
    }


def download_base_model(
    output_dir: Path,
    token: str,
    repo_id: str = DEFAULT_BASE_REPO_ID,
    filename: str = DEFAULT_BASE_FILENAME,
    repo_type: str = "model",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_file = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        token=token,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
    )
    return Path(model_file)


def download_lora_repo(
    output_dir: Path,
    token: str,
    repo_id: str = DEFAULT_LORA_REPO_ID,
    allow_patterns: list[str] | None = None,
    repo_type: str = "model",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
        allow_patterns=allow_patterns,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
    )
    return Path(repo_dir)


def download_artifact(
    target_type: Literal["base", "lora"],
    repo_id: str,
    *,
    filename: str | None = None,
    allow_patterns: list[str] | None = None,
    output_subdir: str | None = None,
    repo_type: str = "model",
) -> dict:
    token = require_hf_token()
    output_dir = _safe_output_dir(target_type, output_subdir)
    if target_type == "base":
        if not filename:
            raise ValueError("base 다운로드에는 filename이 필요합니다.")
        saved = download_base_model(
            output_dir=output_dir, token=token, repo_id=repo_id, filename=filename, repo_type=repo_type
        )
    else:
        saved = download_lora_repo(
            output_dir=output_dir,
            token=token,
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            repo_type=repo_type,
        )

    detected = _detect_downloaded_files(target_type, saved)
    warnings: list[str] = []
    if target_type == "lora" and not detected["adapter_model_safetensors"] and not detected["adapter_model_gguf"]:
        warnings.append("LoRA 핵심 파일(adapter_model.safetensors 또는 adapter_model.gguf)을 찾지 못했습니다.")
    if target_type == "lora" and not detected["adapter_config_json"]:
        warnings.append("adapter_config.json을 찾지 못했습니다.")

    return {
        "success": True,
        "target_type": target_type,
        "repo_id": repo_id,
        "resolved_path": _to_repo_relative(saved),
        "detected_files": detected,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-output",
        default="./artifacts/base",
        help="Base GGUF 저장 폴더",
    )
    parser.add_argument(
        "--lora-output",
        default="./artifacts/lora",
        help="LoRA 저장 폴더",
    )
    args = parser.parse_args()

    token = require_hf_token()
    base_path = download_base_model(Path(args.base_output), token)
    lora_path = download_lora_repo(Path(args.lora_output), token)

    print(f"BASE_MODEL_PATH={base_path}")
    print(f"LORA_REPO_PATH={lora_path}")
    print(
        "Hub에서 내려받은 LoRA는 보통 adapter_model.safetensors(PEFT)입니다. "
        "변환: pip install -r backend/requirements-lora-convert.txt 후 프로젝트 루트에서 npm run convert:lora"
    )


if __name__ == "__main__":
    main()

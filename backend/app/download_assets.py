import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, snapshot_download


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


def download_base_model(output_dir: Path, token: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_file = hf_hub_download(
        repo_id="unsloth/Qwen3.6-35B-A3B-GGUF",
        filename="Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf",
        token=token,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
    )
    return Path(model_file)


def download_lora_repo(output_dir: Path, token: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = snapshot_download(
        repo_id="Teluv/LLM-FineTuning",
        repo_type="model",
        token=token,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
    )
    return Path(repo_dir)


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

"""CLI: PEFT LoRA → adapter_model.gguf

프로젝트 루트가 아닌 `backend`에서 실행하세요:
  cd backend
  ..\\.venv\\Scripts\\python.exe -m app.convert_lora_cli
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from app.lora_gguf_convert import (
    LoraGgufConvertError,
    convert_peft_adapter,
    infer_base_dir_for_lora,
    repo_root,
)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rr = repo_root()
    parser = argparse.ArgumentParser(
        description="adapter_model.safetensors(PEFT)를 llama.cpp용 adapter_model.gguf로 변환합니다.",
    )
    parser.add_argument(
        "--lora-dir",
        type=Path,
        default=None,
        help=f"PEFT 디렉터리 (기본: {rr / 'artifacts' / 'lora'})",
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=None,
        help="베이스 HF 모델 디렉터리 (config.json). 생략 시 lora-dir에 config가 있으면 동일 경로 사용",
    )
    parser.add_argument(
        "--base-model-id",
        default=None,
        help="Hub 모델 ID (로컬 base 없을 때, adapter_config의 base_model_name_or_path 대신 강제)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="출력 .gguf 경로 (기본: <lora-dir>/adapter_model.gguf)",
    )
    parser.add_argument(
        "--outtype",
        default="f16",
        choices=["f32", "f16", "bf16", "q8_0", "auto"],
    )
    args = parser.parse_args()

    lora_dir = (args.lora_dir or rr / "artifacts" / "lora").resolve()
    base = args.base
    if base is None:
        base = infer_base_dir_for_lora(lora_dir)
    if base is None and not args.base_model_id:
        logging.warning(
            "base HF 모델 경로가 자동 감지되지 않았습니다. "
            "실패 시 `--base <HF_BASE_DIR>` 또는 `--base-model-id <MODEL_ID>`를 지정하세요."
        )

    try:
        out = convert_peft_adapter(
            lora_dir,
            base_dir=base,
            base_model_id=args.base_model_id,
            outfile=args.out,
            outtype=args.outtype,
        )
    except LoraGgufConvertError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    print(f"OK {out}")


if __name__ == "__main__":
    main()

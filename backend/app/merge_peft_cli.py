"""CLI: PEFT merge_and_unload -> merged.gguf.

프로젝트 루트가 아닌 `backend`에서 실행하세요:
  cd backend
  ..\\.venv\\Scripts\\python.exe -m app.merge_peft_cli
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.lora_gguf_convert import LoraGgufConvertError, convert_hf_to_gguf, repo_root


def _resolve_base_model_id(lora_dir: Path, base_model: str | None) -> str:
    if base_model:
        return base_model
    cfg_path = lora_dir / "adapter_config.json"
    if not cfg_path.is_file():
        raise LoraGgufConvertError(f"adapter_config.json 없음: {lora_dir}")
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise LoraGgufConvertError(f"adapter_config.json 파싱 실패: {cfg_path}") from exc
    base = str(cfg.get("base_model_name_or_path") or "").strip()
    if not base:
        raise LoraGgufConvertError(
            "base 모델 경로를 찾지 못했습니다. --base-model을 지정하세요."
        )
    return base


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rr = repo_root()
    parser = argparse.ArgumentParser(
        description="PEFT LoRA를 base 모델에 병합하고 merged.gguf를 생성합니다.",
    )
    parser.add_argument(
        "--lora-dir",
        type=Path,
        default=rr / "artifacts" / "lora",
        help="PEFT 어댑터 디렉터리 (adapter_model.safetensors + adapter_config.json)",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Hugging Face base 모델 경로/ID. 생략 시 adapter_config.json의 base_model_name_or_path 사용",
    )
    parser.add_argument(
        "--merged-hf-dir",
        type=Path,
        default=rr / "artifacts" / "merged-hf",
        help="merge_and_unload 결과 HF 디렉터리",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=rr / "artifacts" / "merged" / "merged.gguf",
        help="최종 GGUF 출력 경로",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="base 모델 로드 dtype",
    )
    parser.add_argument(
        "--gguf-outtype",
        default="f16",
        choices=["f32", "f16", "bf16", "q8_0", "auto"],
        help="convert_hf_to_gguf outtype",
    )
    args = parser.parse_args()

    lora_dir = args.lora_dir.resolve()
    if not lora_dir.is_dir():
        raise LoraGgufConvertError(f"LoRA 디렉터리가 없습니다: {lora_dir}")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(args.dtype)
    base_model_ref = _resolve_base_model_id(lora_dir, args.base_model)

    logging.info("base 모델 로드: %s", base_model_ref)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_ref,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_ref, use_fast=False)

    logging.info("PEFT 어댑터 로드: %s", lora_dir)
    peft_model = PeftModel.from_pretrained(model, str(lora_dir))
    logging.info("merge_and_unload 실행")
    merged_model = peft_model.merge_and_unload()

    merged_hf_dir = args.merged_hf_dir.resolve()
    merged_hf_dir.mkdir(parents=True, exist_ok=True)
    logging.info("병합 모델 저장: %s", merged_hf_dir)
    merged_model.save_pretrained(merged_hf_dir)
    tokenizer.save_pretrained(merged_hf_dir)

    out_path = convert_hf_to_gguf(
        merged_hf_dir,
        outfile=args.out.resolve(),
        outtype=args.gguf_outtype,
    )
    print(f"OK {out_path}")


if __name__ == "__main__":
    try:
        main()
    except LoraGgufConvertError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

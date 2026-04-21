"""PEFT LoRA(adapter_model.safetensors) → llama.cpp용 adapter_model.gguf 변환.

llama.cpp 저장소의 convert_lora_to_gguf.py를 subprocess로 실행한다.
저장소는 `.cache/llama_cpp`에 shallow clone 되거나 `LLAMA_CPP_ROOT`로 지정한다.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


class LoraGgufConvertError(RuntimeError):
    pass


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def llama_cpp_cache_dir() -> Path:
    return repo_root() / ".cache" / "llama_cpp"


MOE_PEFT_PATCH_MARKER = "# llm-lora-compere: MOE PEFT expert stub skip"


def _patch_convert_hf_moe_peft_expert_stubs(llama_root: Path) -> None:
    """Qwen3 MoE + PEFT 시 mlp.experts.base_layer / bare experts.weight 스텁이 _experts에 쌓여
    `Unprocessed experts`로 실패하는 문제를 회피한다 (upstream 미처리 구간).
    """
    path = llama_root / "convert_hf_to_gguf.py"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    if MOE_PEFT_PATCH_MARKER in text:
        return
    needle = (
        '        name = name.replace("language_model.", "") # InternVL\n\n'
        "        # NVFP4 expert weights are handled in _generate_nvfp4_tensors"
    )
    if needle not in text:
        logger.warning(
            "convert_hf_to_gguf.py에서 MOE PEFT 패치 앵커를 찾지 못했습니다. "
            "llama.cpp 버전이 달라졌을 수 있습니다."
        )
        return
    replacement = (
        '        name = name.replace("language_model.", "") # InternVL\n\n'
        f"        {MOE_PEFT_PATCH_MARKER}\n"
        '        if name.endswith(".mlp.experts.base_layer.weight"):\n'
        "            return\n"
        '        if re.match(r"^model\\.layers\\.\\d+\\.mlp\\.experts\\.weight$", name):\n'
        "            return\n\n"
        "        # NVFP4 expert weights are handled in _generate_nvfp4_tensors"
    )
    path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
    logger.info("convert_hf_to_gguf.py — Qwen MoE PEFT 스텁 스킵 패치 적용")


def _check_convert_dependencies() -> None:
    for mod in ("torch", "transformers", "safetensors", "numpy"):
        if importlib.util.find_spec(mod) is None:
            raise LoraGgufConvertError(
                f"변환에 필요한 패키지가 없습니다: {mod}. "
                "실행: pip install -r backend/requirements-lora-convert.txt"
            )


def ensure_llama_cpp_repo() -> Path:
    """convert_lora_to_gguf.py가 있는 llama.cpp 루트 경로."""
    env = os.getenv("LLAMA_CPP_ROOT", "").strip()
    if env:
        root = Path(env).expanduser().resolve()
        script = root / "convert_lora_to_gguf.py"
        if not script.is_file():
            raise LoraGgufConvertError(
                f"LLAMA_CPP_ROOT에 convert_lora_to_gguf.py가 없습니다: {root}"
            )
        _patch_convert_hf_moe_peft_expert_stubs(root)
        return root

    cache = llama_cpp_cache_dir()
    script = cache / "convert_lora_to_gguf.py"
    if script.is_file():
        _patch_convert_hf_moe_peft_expert_stubs(cache)
        return cache

    import shutil

    if shutil.which("git") is None:
        raise LoraGgufConvertError(
            "git이 PATH에 없어 llama.cpp를 자동으로 받을 수 없습니다. "
            "https://github.com/ggml-org/llama.cpp 를 클론한 뒤 LLAMA_CPP_ROOT를 설정하세요."
        )

    url = os.getenv("LLAMA_CPP_GIT_URL", "https://github.com/ggml-org/llama.cpp.git")
    ref = os.getenv("LLAMA_CPP_GIT_REF", "master")
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and not script.is_file():
        raise LoraGgufConvertError(
            f"llama.cpp 캐시가 불완전합니다. 폴더를 삭제한 뒤 다시 시도하세요: {cache}"
        )
    logger.info("llama.cpp shallow clone → %s (ref=%s)", cache, ref)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", ref, url, str(cache)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise LoraGgufConvertError(
            "llama.cpp git clone 실패. "
            f"stderr:\n{exc.stderr or exc.stdout}\n"
            "네트워크·방화벽을 확인하거나 LLAMA_CPP_ROOT로 로컬 클론 경로를 지정하세요."
        ) from exc

    if not script.is_file():
        raise LoraGgufConvertError(f"clone 후에도 convert_lora_to_gguf.py 없음: {cache}")
    _patch_convert_hf_moe_peft_expert_stubs(cache)
    return cache


def convert_peft_adapter(
    lora_dir: Path,
    *,
    base_dir: Path | None = None,
    base_model_id: str | None = None,
    outfile: Path | None = None,
    outtype: str = "f16",
) -> Path:
    """adapter_model.safetensors + adapter_config.json 이 있는 디렉터리를 GGUF로 변환."""
    _check_convert_dependencies()
    lora_dir = lora_dir.resolve()
    adapter_st = lora_dir / "adapter_model.safetensors"
    adapter_bin = lora_dir / "adapter_model.bin"
    if not (adapter_st.is_file() or adapter_bin.is_file()):
        raise LoraGgufConvertError(
            f"PEFT 가중치가 없습니다 (adapter_model.safetensors 또는 .bin): {lora_dir}"
        )
    if not (lora_dir / "adapter_config.json").is_file():
        raise LoraGgufConvertError(f"adapter_config.json 없음: {lora_dir}")

    if outfile is None:
        outfile = lora_dir / "adapter_model.gguf"
    outfile = outfile.resolve()

    llama_root = ensure_llama_cpp_repo()
    cmd: list[str] = [
        sys.executable,
        str(llama_root / "convert_lora_to_gguf.py"),
        str(lora_dir),
        "--outfile",
        str(outfile),
        "--outtype",
        outtype,
    ]
    if base_dir is not None:
        cmd.extend(["--base", str(base_dir.resolve())])
    if base_model_id:
        cmd.extend(["--base-model-id", base_model_id])

    logger.info("LoRA GGUF 변환 실행 (cwd=%s)", llama_root)
    proc = subprocess.run(
        cmd,
        cwd=str(llama_root),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        err = (proc.stderr or "") + (proc.stdout or "")
        raise LoraGgufConvertError(
            f"convert_lora_to_gguf.py 실패 (exit {proc.returncode}). 로그:\n{err[-8000:]}"
        )

    if not outfile.is_file():
        raise LoraGgufConvertError(f"변환 명령은 성공했으나 출력 파일이 없습니다: {outfile}")

    logger.info("LoRA GGUF 생성 완료: %s", outfile)
    return outfile


def convert_hf_to_gguf(
    model_dir: Path,
    *,
    outfile: Path,
    outtype: str = "f16",
) -> Path:
    """Hugging Face 모델 디렉터리를 llama.cpp GGUF로 변환한다."""
    _check_convert_dependencies()
    model_dir = model_dir.resolve()
    if not (model_dir / "config.json").is_file():
        raise LoraGgufConvertError(f"HF config.json 없음: {model_dir}")

    outfile = outfile.resolve()
    outfile.parent.mkdir(parents=True, exist_ok=True)

    llama_root = ensure_llama_cpp_repo()
    cmd: list[str] = [
        sys.executable,
        str(llama_root / "convert_hf_to_gguf.py"),
        str(model_dir),
        "--outfile",
        str(outfile),
        "--outtype",
        outtype,
    ]
    logger.info("HF → GGUF 변환 실행 (cwd=%s)", llama_root)
    proc = subprocess.run(
        cmd,
        cwd=str(llama_root),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        err = (proc.stderr or "") + (proc.stdout or "")
        raise LoraGgufConvertError(
            f"convert_hf_to_gguf.py 실패 (exit {proc.returncode}). 로그:\n{err[-8000:]}"
        )
    if not outfile.is_file():
        raise LoraGgufConvertError(f"변환 명령은 성공했으나 출력 파일이 없습니다: {outfile}")
    logger.info("HF GGUF 생성 완료: %s", outfile)
    return outfile


def infer_base_dir_for_lora(lora_dir: Path) -> Path | None:
    """같은 폴더에 베이스 config.json이 있으면 그 경로를 반환 (스냅샷 전체가 한 디렉터리일 때)."""
    lora_dir = lora_dir.resolve()
    if (lora_dir / "config.json").is_file():
        return lora_dir
    return None

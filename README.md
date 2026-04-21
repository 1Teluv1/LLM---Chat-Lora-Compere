# LLM LoRA Compare UI

동일 프롬프트와 동일 샘플링 파라미터로 `Base`와 `Base+LoRA` 출력을 한 번에 비교하는 프로젝트입니다.

## 1) 프로젝트 구조

- `frontend/`: Next.js 비교 UI
- `backend/`: FastAPI + llama.cpp 기반 비교 API
- `scripts/`: 실행/다운로드 PowerShell 스크립트

## 2) 사전 준비

1. `.env.example`을 `.env`로 복사하고 값을 채웁니다.
2. Python 3.11+, Node 20+ 환경을 준비합니다.

```powershell
Copy-Item .env.example .env
```

## 3) 모델/LoRA 다운로드

`HF_TOKEN`을 환경변수로 설정한 뒤 다운로드 스크립트를 실행합니다.

```powershell
$env:HF_TOKEN="hf_xxx_replace_me"
./scripts/download_assets.ps1
```

- Base 모델 파일은 `unsloth/Qwen3.6-35B-A3B-GGUF`의 `Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf`를 받습니다.
- LoRA는 `Teluv/LLM-FineTuning` 전체를 내려받습니다.
- 내려받은 LoRA 리포지토리에서 실제 어댑터 파일을 확인한 뒤 `.env`의 `LORA_ADAPTER_PATH`에 지정합니다.

### LoRA GGUF 변환 시 주의

- `adapter_model.safetensors`만 있을 때는 변환에 base HF 모델이 필요할 수 있습니다.
- 실패 시 다음처럼 base 경로를 명시해서 변환하세요.

```powershell
npm run convert:lora -- --lora-dir artifacts/lora --base D:\models\qwen-base-hf
```

## 4) 옵션 A: PEFT 병합 후 merged.gguf 비교

1. base HF 모델 + PEFT LoRA를 로드합니다.
2. `merge_and_unload()`로 병합 후 HF 디렉터리로 저장합니다.
3. `convert_hf_to_gguf.py`로 `merged.gguf`를 생성합니다.
4. 백엔드는 `base.gguf`와 `merged.gguf`를 각각 로드해 A/B 비교합니다.

```powershell
npm run merge:peft-to-gguf -- --lora-dir artifacts/lora --base-model Qwen/Qwen3-Base-Model
```

- 생성된 파일 기본 경로: `artifacts/merged/merged.gguf`
- 환경 변수 `MERGED_MODEL_GGUF`를 지정하면 해당 파일을 우선 사용합니다.
- `MERGED_MODEL_GGUF`가 있으면 런타임 LoRA 어댑터(`adapter_model.gguf`) 대신 merged GGUF를 비교 대상으로 사용합니다.

## 5) 실행 방법 (로컬)

### Backend

```powershell
cd backend
pip install -r requirements.txt
cd ..
./scripts/run_backend.ps1
```

### Frontend

```powershell
cd frontend
npm install
cd ..
./scripts/run_frontend.ps1
```

브라우저에서 `http://localhost:3000` 접속 후 프롬프트와 파라미터를 입력해 비교합니다.

## 6) API 스펙

- `POST /compare`
  - 입력: `prompt`, `seed`, `top_k`, `top_p`, `temperature`, `max_tokens`, `runtime`, `base_model_id`, `lora_id`, `lora_strategy`, `device_hint`
  - 동작: 동일 파라미터를 Base/LoRA 양쪽에 강제 적용
  - 출력: `base`, `lora`, `params`

## 7) 비교 재현성 체크 포인트

- 같은 `prompt`와 `seed`, `top_k`, `top_p`, `temperature`, `max_tokens`로 재요청
- 응답 `params`가 요청과 동일한지 확인
- Base와 LoRA만 다르고 나머지 조건이 동일한지 확인

## 8) 간단 검증 시나리오

1. `GET /health`가 `{"status":"ok"}`를 반환하는지 확인
2. `POST /compare`에 동일 요청 2회 전송 후 `params` 동일성 확인
3. UI에서 한 번 입력으로 좌/우 결과가 함께 렌더링되는지 확인

## 9) Docker Compose (선택)

`.env` 준비 후:

```powershell
docker compose up --build
```

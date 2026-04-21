# LLM LoRA Compare

동일한 프롬프트/샘플링 파라미터로 `Base` 모델과 `LoRA` 적용 모델 출력을 나란히 비교하는 프로젝트입니다.  
프론트엔드(Next.js)에서 비교 요청을 보내고, 백엔드(FastAPI)가 단일 요청 컨텍스트에서 Base/LoRA 추론을 수행합니다.

## 핵심 기능

- 단일 폼 입력으로 Base vs LoRA 비교 실행
- SSE(`POST /compare/stream`) 기반 진행 상태 및 결과 스트리밍
- 일반 비교(`POST /compare`) API 제공
- `llama_cpp` 기본 런타임 + `transformers` 선택 런타임
- LoRA 변환/병합 유틸리티 스크립트 제공

## 프로젝트 구조

```text
llm-lora-compere/
├─ frontend/                      # Next.js UI
│  └─ src/
│     ├─ app/
│     │  ├─ page.tsx              # 메인 대시보드 페이지
│     │  └─ api/artifacts-options/route.ts
│     ├─ features/compare/
│     │  ├─ components/           # 폼/진행/결과 컴포넌트
│     │  └─ hooks/useCompareStream.ts
│     └─ lib/api.ts               # 백엔드 API 연동
├─ backend/                       # FastAPI + 추론 런타임
│  ├─ app/
│  │  ├─ main.py                  # API 엔트리포인트
│  │  ├─ application/compare_service.py
│  │  ├─ infrastructure/
│  │  │  ├─ llama_cpp/runtime.py
│  │  │  └─ transformers/runtime.py
│  │  ├─ domain/                  # 모델/인터페이스
│  │  ├─ config/settings.py
│  │  └─ *.py                     # 변환/병합/다운로드 유틸
│  ├─ tests/test_main.py
│  ├─ requirements.txt
│  └─ requirements-lora-convert.txt
├─ scripts/                       # PowerShell 실행/유틸 스크립트
├─ .env.example                   # 환경 변수 템플릿
└─ package.json                   # 루트 실행 스크립트
```

## 요구사항

- Windows + PowerShell 기준 문서
- Python 3.11+
- Node.js 20+
- (권장) 프로젝트 루트 가상환경 `.venv`

## 빠른 시작

### 1) 환경 변수 준비

```powershell
Copy-Item .env.example .env
```

필수 확인 항목:

- `BASE_MODEL_PATH`: Base GGUF 파일 경로
- `LORA_ADAPTER_PATH`: LoRA 어댑터 파일 경로
- `NEXT_PUBLIC_BACKEND_URL`: 프론트엔드에서 접근할 백엔드 URL
- `INFERENCE_RUNTIME`: 기본값 `llama_cpp`

### 2) 의존성 설치

백엔드:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\backend\requirements.txt
```

프론트엔드:

```powershell
npm install
npm --prefix frontend install
```

### 3) 실행

한 번에 실행(루트):

```powershell
npm run start
```

또는 개별 실행:

```powershell
./scripts/run_backend.ps1
./scripts/run_frontend.ps1
```

접속:

- 프론트엔드: `http://localhost:3000`
- 백엔드 기본 포트: `http://127.0.0.1:8001`

## 환경 변수(.env) 가이드

`.env.example` 기준 주요 변수:

- `HF_TOKEN`: Hugging Face 다운로드 토큰(필요 시)
- `BASE_MODEL_PATH`: Base GGUF 파일
- `LORA_ADAPTER_PATH`: LoRA 어댑터 GGUF 파일
- `LLAMA_N_CTX`, `LLAMA_N_THREADS`, `LLAMA_N_GPU_LAYERS`: llama.cpp 런타임 옵션
- `NEXT_PUBLIC_BACKEND_URL`: 프론트엔드가 호출할 백엔드 주소
- `INFERENCE_RUNTIME`: `llama_cpp` 또는 `transformers`
- `HF_BASE_MODEL_ID`, `HF_LORA_MODEL_ID`: transformers 런타임용 모델 ID

## API 개요

### `GET /health`

- 서버 상태 확인
- 응답 예시: `{"status":"ok"}`

### `POST /compare`

- 요청 파라미터(주요):  
  `prompt`, `seed`, `top_k`, `top_p`, `temperature`, `max_tokens`,  
  `runtime`, `base_model_id`, `lora_id`, `lora_strategy`, `device_hint`
- 응답: `base`, `lora`, `params`, `debug`

### `POST /compare/stream`

- SSE 스트리밍 비교 API
- 이벤트 타입 예시: `meta`, `phase`, `delta`, `done`, `error`

### `GET /runtime/status`

- 런타임 로딩 상태 확인

## 모델/LoRA 준비

### 다운로드 스크립트 사용

```powershell
$env:HF_TOKEN="hf_xxx_replace_me"
./scripts/download_assets.ps1
```

다운로드 후 실제 파일 경로를 `.env`의 `BASE_MODEL_PATH`, `LORA_ADAPTER_PATH`에 반영하세요.

### LoRA 변환

```powershell
npm run convert:lora -- --lora-dir artifacts/lora --base D:\models\qwen-base-hf
```

### PEFT 병합 후 GGUF 변환

```powershell
npm run merge:peft-to-gguf -- --lora-dir artifacts/lora --base-model Qwen/Qwen3-Base-Model
```

참고:

- 기본 생성 경로: `artifacts/merged/merged.gguf`
- `MERGED_MODEL_GGUF`를 지정하면 해당 파일을 비교 대상으로 우선 사용

## 검증 체크리스트

1. `GET /health`가 정상 응답하는지 확인
2. 동일 입력으로 `POST /compare` 2회 호출 시 `params` 일관성 확인
3. UI에서 Base/LoRA 결과가 동시에 렌더링되는지 확인
4. `POST /compare/stream` 사용 시 단계별 이벤트가 순차적으로 수신되는지 확인

## 트러블슈팅

- 백엔드 연결 실패:
  - `.env`의 `NEXT_PUBLIC_BACKEND_URL`과 실제 백엔드 포트 일치 여부 확인
- 모델 로드 실패:
  - `BASE_MODEL_PATH`, `LORA_ADAPTER_PATH` 실제 파일 존재 여부 확인
- 실행 포트 충돌:
  - `npm run stop:ports` 후 재시작
- CUDA/llama.cpp 이슈:
  - 필요 시 `npm run install:llama-cuda` 사용

## 유용한 스크립트

- `npm run start`: 백엔드/프론트엔드 동시 실행
- `npm run build`: 프론트엔드 빌드
- `npm run stop:ports`: 포트 점유 프로세스 정리
- `npm run convert:lora`: LoRA 변환 CLI 실행
- `npm run merge:peft-to-gguf`: PEFT 병합 + GGUF 변환 CLI 실행

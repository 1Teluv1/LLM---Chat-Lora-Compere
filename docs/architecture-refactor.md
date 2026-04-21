# Windows 범용 추론 아키텍처

## 책임 분리
- `domain`: 추론 입력/출력 모델, 런타임 인터페이스
- `application`: 비교 유스케이스, 런타임 선택 오케스트레이션
- `infrastructure`: 런타임 어댑터(`llama_cpp`, `transformers`), 모델 경로 해석
- `interfaces/http`: 현재는 `main.py`가 컨트롤러 역할

## 런타임 전략
- `llama_cpp`: GGUF 기반 Base/LoRA 또는 merged GGUF 비교
- `transformers`: Hugging Face 모델 ID 기반 Base/Compare 모델 쌍 비교

## 확장 포인트
- 새 런타임 추가 시 `InferenceRuntime` 구현체 추가 후 `RuntimeRegistry` 등록
- LoRA 전략 고도화 시 `resolver`와 런타임별 로드 로직 분리 확장

## Windows 운영 원칙
- 경로는 Python `Path`로 정규화
- GPU 사용 실패 시 CPU fallback을 허용
- 런타임 상태/오류를 `/debug/inference`로 통합 노출

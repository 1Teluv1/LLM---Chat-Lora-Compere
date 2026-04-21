# llama-cpp-python을 CUDA(GGML_CUDA) 소스 빌드·재설치합니다.
#
# 사전 요구:
#   - NVIDIA 드라이버, CUDA Toolkit (v12.4 이상 권장; RTX 50xx Blackwell은 12.8+ 또는 13.0)
#   - Visual Studio 2022 C++ 빌드 도구
#   - git
#
# 중요:
#   pip sdist는 서브모듈 일부 파일(vendor/llama.cpp/tools/mtmd 등)이 누락되어
#   CUDA 빌드 중 "CMakeLists.txt 없음(MSB8066)" 오류가 납니다.
#   또 pip sdist의 vendor/llama.cpp가 오래되면 CUDA 13용 fattn 템플릿(.cu) 파일이
#   없어 "C1083: No such file or directory" 오류가 납니다.
#   해결: GitHub에서 git clone --recursive 로 서브모듈까지 완전히 가져와
#   최신 llama.cpp 소스로 빌드합니다.
#
# 사용법:
#   .\scripts\install-llama-cpp-cuda.ps1              # 릴리스 태그(v0.3.19) 소스 빌드
#   .\scripts\install-llama-cpp-cuda.ps1 -Ref main    # main 브랜치 최신 커밋 사용
#   .\scripts\install-llama-cpp-cuda.ps1 -CudaArch 89 # 특정 SM (예: RTX 4090=89, 5080=120)

param(
    [string]$Ref = "v0.3.19",
    [string]$CudaArch = "",          # 비워두면 CMake가 GPU 감지해서 자동 설정
    [string]$WorkDir = ""            # 비워두면 $env:TEMP\llama-cpp-python-build
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$venvPy = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPy)) {
    Write-Error ".venv 가 없습니다. 먼저 프로젝트 루트에서: py -3.11 -m venv .venv; .\.venv\Scripts\pip install -r backend\requirements.txt"
}

if (-not $WorkDir) {
    $WorkDir = Join-Path $env:TEMP "llama-cpp-python-build"
}

# GPU 감지 (참고용 출력)
Write-Host ">> 환경 확인" -ForegroundColor Cyan
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv | Out-Host
nvcc --version | Select-String "release" | Out-Host

# 기존 소스 제거 후 clone
if (Test-Path $WorkDir) {
    Write-Host ">> 기존 빌드 디렉터리 제거: $WorkDir" -ForegroundColor Yellow
    Remove-Item -Recurse -Force $WorkDir
}

Write-Host ">> git clone --recursive (서브모듈 포함) @ $Ref" -ForegroundColor Cyan
& git clone --recursive --branch $Ref --depth 1 https://github.com/abetlen/llama-cpp-python.git $WorkDir
if ($LASTEXITCODE -ne 0) {
    # 태그가 없는 경우(main 등) depth 제한 없이 다시 시도
    Write-Host ">> shallow clone 실패, 전체 clone 재시도" -ForegroundColor Yellow
    Remove-Item -Recurse -Force $WorkDir -ErrorAction SilentlyContinue
    & git clone --recursive https://github.com/abetlen/llama-cpp-python.git $WorkDir
    if ($LASTEXITCODE -ne 0) { Write-Error "git clone 실패" }
    Push-Location $WorkDir
    & git checkout $Ref
    & git submodule update --init --recursive
    Pop-Location
}

Push-Location $WorkDir
& git submodule update --init --recursive
Pop-Location

# CMake 인자: CUDA ON + (선택) 아키텍처 명시
$cmakeArgs = "-DGGML_CUDA=ON"
if ($CudaArch) {
    $cmakeArgs += " -DCMAKE_CUDA_ARCHITECTURES=$CudaArch"
}
$env:CMAKE_ARGS = $cmakeArgs
$env:FORCE_CMAKE = "1"

Write-Host ">> CMAKE_ARGS=$env:CMAKE_ARGS" -ForegroundColor Cyan

& $venvPy -m pip uninstall -y llama-cpp-python
Write-Host ">> 빌드·설치 시작 (수십 분 소요 가능)" -ForegroundColor Cyan
& $venvPy -m pip install --no-cache-dir --force-reinstall --no-deps $WorkDir
if ($LASTEXITCODE -ne 0) {
    Write-Error "빌드 실패. 로그 확인 후 -Ref main 으로 재시도하거나, CUDA/VS 빌드 도구 설치를 점검하세요."
}

Write-Host ">> 설치 성공. 검증:" -ForegroundColor Green
& $venvPy -c "import llama_cpp; print('version:', llama_cpp.__version__); print('supports_gpu_offload:', llama_cpp.llama_supports_gpu_offload())"

Write-Host "백엔드 재시작 후 UI의 '진행 상태' 패널에서 'GPU offload: 지원' 표시와 .env의 LLAMA_N_GPU_LAYERS=-1 가 실제로 적용되는지 확인하세요." -ForegroundColor Green

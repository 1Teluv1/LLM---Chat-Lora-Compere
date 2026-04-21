Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location "$PSScriptRoot/../backend"
python -m app.download_assets --base-output "./artifacts/base" --lora-output "./artifacts/lora"

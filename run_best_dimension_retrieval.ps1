param(
    [string]$ModelPath = "",
    [ValidateSet("cpu", "cuda")][string]$Device = "cpu"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    python -m venv .venv
}
$python = Join-Path $root ".venv\Scripts\python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install -r requirements.txt

$arguments = @("run_best_dimension_retrieval.py", "--device", $Device)
if ($ModelPath) {
    $arguments += @("--model-path", $ModelPath)
}
& $python @arguments

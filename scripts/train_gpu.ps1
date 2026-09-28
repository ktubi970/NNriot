<# Run an existing training script with the CUDA environment in WSL2. #>
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Distribution = 'NNriot-GPU',
    [string]$LinuxPython = '/opt/nnriot-venv/bin/python',
    [string]$Script = 'scripts/train_deepsets_v3.py',
    [switch]$Check,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$TrainingArgs
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$runnerArgs = @('scripts/run_gpu_training.py')
if ($Check) {
    $runnerArgs += '--check'
} else {
    $runnerArgs += $Script
    $runnerArgs += $TrainingArgs
}

& wsl.exe --distribution $Distribution --user root --cd $repoRoot --exec bash scripts/run_with_cuda.sh $LinuxPython @runnerArgs
exit $LASTEXITCODE

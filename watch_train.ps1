param(
  [Parameter(Mandatory=$true)][string]$Level,
  [string]$ActionSet = "simple",
  [int]$Seed = 0,
  [int]$NEnvs = 1,
  [int]$TotalSteps = 20000000,
  [double]$MaxHours = 4.5,
  [string]$RunName = ""
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$tag = "lvl" + $Level.Replace("-", "") + "_" + $ActionSet + "_seed" + $Seed
if ([string]::IsNullOrWhiteSpace($RunName)) { $RunName = $tag }
$runDir = Join-Path $root ("runs\" + $RunName)
$logDir = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ($RunName + ".watch.log")

$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:PYTHONUNBUFFERED = "1"

while ($true) {
  $statusPath = Join-Path $runDir "status.json"
  $steps = 0
  if (Test-Path $statusPath) {
    try { $steps = [int64](Get-Content $statusPath -Raw | ConvertFrom-Json).timesteps } catch {}
  }
  if ($steps -ge $TotalSteps) {
    Add-Content $log "$(Get-Date -Format o) target reached: $steps"
    break
  }

  Add-Content $log "$(Get-Date -Format o) starting/resuming at $steps"
  $args = @("src\train.py", "--level", $Level, "--action-set", $ActionSet,
            "--seed", "$Seed", "--n-envs", "$NEnvs", "--total-steps", "$TotalSteps",
            "--max-hours", "$MaxHours", "--run-name", $RunName)
  $p = Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $root `
    -WindowStyle Hidden -PassThru -Wait
  Add-Content $log "$(Get-Date -Format o) exited with code $($p.ExitCode); restarting after checkpoint"
  Start-Sleep -Seconds 10
}

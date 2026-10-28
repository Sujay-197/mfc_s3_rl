$ErrorActionPreference = "Continue"
$root = $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$logDir = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:PYTHONUNBUFFERED = "1"

$levels = @("8-1")
foreach ($level in $levels) {
  $runName = "lvl" + $level.Replace("-", "") + "_simple_seed0"
  $runDir = Join-Path $root ("runs\" + $runName)
  $watchLog = Join-Path $logDir ($runName + ".watch.log")
  $statusPath = Join-Path $runDir "status.json"
  $finished = $false

  while (-not $finished) {
    $steps = 0
    if (Test-Path $statusPath) {
      try { $steps = [int64](Get-Content $statusPath -Raw | ConvertFrom-Json).timesteps } catch {}
    }
    if ($steps -ge 20000000) { $finished = $true; break }

    Add-Content $watchLog "$(Get-Date -Format o) training $level from $steps"
    $args = @("src\train.py", "--level", $level, "--action-set", "simple",
              "--seed", "0", "--n-envs", "2", "--total-steps", "20000000",
              "--max-hours", "4.5", "--run-name", $runName)
    Push-Location $root
    try {
      # Invoke Python directly so PowerShell receives the real exit code from
      # train.py; Start-Process -Wait can remain attached to dead emulator
      # workers and prevent the watchdog from restarting the run.
      & $python @args *> (Join-Path $logDir ($runName + ".train.log"))
      $exitCode = $LASTEXITCODE
    } finally {
      Pop-Location
    }
    Add-Content $watchLog "$(Get-Date -Format o) training exited $exitCode; checkpoint resume will follow"
    Start-Sleep -Seconds 10
  }

  Add-Content $watchLog "$(Get-Date -Format o) training complete; generating outputs"
  & $python "src\report.py" "--level" $level *> (Join-Path $logDir ($runName + ".report.log"))
  & $python "src\build_page.py" "--level" $level *> (Join-Path $logDir ($runName + ".page.log"))
  Add-Content $watchLog "$(Get-Date -Format o) outputs complete"
}

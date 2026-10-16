# Launch the training runs for one level as detached processes.
# Detached (not child) processes so they survive this shell closing.
#
#   powershell -File run_all.ps1                        # 1-1, the original three runs
#   powershell -File run_all.ps1 -Level 1-2 -PrimaryActions complex
#   powershell -File run_all.ps1 -Level 1-2 -PrimaryActions complex -NEnvs 2 -Seeds 2
#   powershell -File run_all.ps1 -Level 1-2 -PrimaryActions complex -NEnvs 2 -Seeds 2 -Sequential
#
# 1-2 needs -PrimaryActions complex: the level ends by entering a pipe, and
# only the complex set contains a "down" action.
#
# --max-hours is deliberately set so each run stops on its own: a run that is
# killed externally never reaches the finally: block in train.py and so never
# writes final_model.zip, which report.py needs.
param(
  [string]$Level = "1-1",
  [string]$PrimaryActions = "simple",
  [string]$AblationActions = "right_only",
  [double]$MaxHours = 4.5,
  [int]$TotalSteps = 20000000,
  [int]$NEnvs = 4,
  [int]$Seeds = 2,
  [switch]$Sequential,
  [string]$Python = "$PSScriptRoot\.venv\Scripts\python.exe"
)

$train = "$PSScriptRoot\src\train.py"
$trainRel = "src\train.py"   # space-free; used with -WorkingDirectory below
New-Item -ItemType Directory -Force -Path "$PSScriptRoot\logs" | Out-Null

# non-default levels get a tag so their run directories cannot collide with 1-1's
$tag = if ($Level -eq "1-1") { "" } else { "lvl" + $Level.Replace("-", "") + "_" }

# One run = the PPO parent plus NEnvs emulator workers. Measured on a 16 GB
# laptop: 2.4 GB of commit for NEnvs=2, and the parent alone is ~1.1 GB because
# it holds torch and the CUDA context. Three concurrent runs therefore need
# roughly 7 GB of commit, which is fine on a desktop and out of reach on a
# machine whose commit headroom is a couple of GB. -Sequential runs the seeds
# one after another instead, trading wall clock for a footprint that fits.
$runs = @()
for ($seed = 0; $seed -lt [Math]::Max($Seeds, 1); $seed++) {
  $runs += @{actions = $PrimaryActions; seed = $seed}
}
if ($Seeds -ge 3) {
  $runs += @{actions = $AblationActions; seed = 2}
}

# BLAS spinning up a thread pool per emulator worker was enough to exhaust the
# commit limit on a small machine; the emulation is the bottleneck, not BLAS.
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
# redirected stdout is block-buffered by default; unbuffered logs stream live
$env:PYTHONUNBUFFERED = "1"

foreach ($r in $runs) {
  $name = "$tag$($r.actions)_seed$($r.seed)"
  $trainArgs = @("--level", $Level, "--action-set", $r.actions,
                 "--seed", $r.seed, "--n-envs", "$NEnvs",
                 "--total-steps", "$TotalSteps", "--max-hours", "$MaxHours",
                 "--run-name", $name)

  # The training script and the venv both live under a path with spaces
  # ("E:\Sem 3\..."). PowerShell's call operator does not quote a native
  # command's arguments, and Start-Process does not reliably preserve quotes
  # inside its ArgumentList elements either: an intermediate cmd.exe /c line
  # split the path at the first space and the run died instantly with an
  # argparse "unrecognized arguments" error (observed: "'E:\Sem' is not
  # recognized"). The fix is to keep every argument space-free: run the script
  # by its relative path (src\train.py) with -WorkingDirectory pointing at the
  # project root, so no token ever needs quoting, and let Start-Process's own
  # -RedirectStandardOutput/-RedirectStandardError capture the logs -- those
  # are real parameters, not shell redirection, so their spaces are fine.
  $log = "$PSScriptRoot\logs\$name.log"
  $errLog = "$PSScriptRoot\logs\$name.err.log"
  $launch = @($trainRel) + $trainArgs

  if ($Sequential) {
    Write-Host "running $name (level $Level, $($r.actions), seed $($r.seed), $NEnvs envs) -- log: $log"
    $p = Start-Process -FilePath $Python -ArgumentList $launch `
      -WorkingDirectory $PSScriptRoot `
      -RedirectStandardOutput $log -RedirectStandardError $errLog `
      -WindowStyle Hidden -PassThru -Wait
    Write-Host "finished $name (exit $($p.ExitCode))"
  } else {
    Start-Process -FilePath $Python -ArgumentList $launch `
      -WorkingDirectory $PSScriptRoot `
      -RedirectStandardOutput $log -RedirectStandardError $errLog `
      -WindowStyle Hidden
    Write-Host "launched $name in the background (level $Level, $($r.actions), seed $($r.seed), $NEnvs envs) -- log: $log"
  }
}

# Launch the three overnight training runs as detached processes.
# Detached (not child) processes so they survive this shell closing.
$py = "$PSScriptRoot\.venv\Scripts\python.exe"
$train = "$PSScriptRoot\src\train.py"
New-Item -ItemType Directory -Force -Path "$PSScriptRoot\logs" | Out-Null

$runs = @(
  @{name="simple_seed0";     actions="simple";     seed=0},
  @{name="simple_seed1";     actions="simple";     seed=1},
  @{name="right_only_seed2"; actions="right_only"; seed=2}
)

foreach ($r in $runs) {
  $args = @($train, "--action-set", $r.actions, "--seed", $r.seed,
            "--n-envs", "4", "--total-steps", "20000000",
            "--max-hours", "7.5", "--run-name", $r.name)
  Start-Process -FilePath $py -ArgumentList $args `
    -RedirectStandardOutput "$PSScriptRoot\logs\$($r.name).out.log" `
    -RedirectStandardError  "$PSScriptRoot\logs\$($r.name).err.log" `
    -WindowStyle Hidden
  Write-Host "launched $($r.name)"
  Start-Sleep -Seconds 3
}

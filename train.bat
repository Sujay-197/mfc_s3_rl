@echo off
REM ---------------------------------------------------------------------------
REM Training launcher for this project -- Windows.
REM Usage:
REM   train.bat                     trains 1-3: 2 seeds, 2 envs each, one seed
REM                                  at a time, up to 4.5 h, logs in logs/
REM   train.bat 1-2                 train 1-2 instead of 1-3
REM   train.bat 1-2 1 4             1 seed, 4 envs/seed
REM   train.bat 1-3 2 2 3.0         2 seeds, 2 envs/seed, 3.0 h max
REM   train.bat 1-3 2 2 4.5 p       parallel, 2 seeds at once (use with care)
REM 
REM Each run is detached, with its logs inside logs/. A run stops itself at
REM --max-hours and always saves its final model.
REM ---------------------------------------------------------------------------

setlocal enabledextensions
cd /d "%~dp0"

set LEVEL=1-3
set SEEDS=2
set NENVS=2
set MAXHOURS=4.5
set SEQUENTIAL=1

if "%~1" neq "" set LEVEL=%~1
if "%~2" neq "" set SEEDS=%~2
if "%~3" neq "" set NENVS=%~3
if "%~4" neq "" set MAXHOURS=%~4
if "%~5" equ "p" set SEQUENTIAL=0

if not exist .venv\Scripts\python.exe (
  echo ERROR: .venv not found. Run setup.bat first.
  exit /b 1
)
if not exist src\train.py (
  echo ERROR: src\train.py not found. Run setup.bat first.
  exit /b 1
)

set PY=.venv\Scripts\python.exe
if not exist logs mkdir logs

if "%SEQUENTIAL%"=="1" (
  for /l %%s in (0,1,%SEEDS%) do (
    echo ===============================
    echo seed %%s (level %LEVEL%, %NENVS% envs, max %.1f h)
    echo ============================
    "%PY%" src\train.py --level %LEVEL% --seed %%s --n-envs %NENVS% --max-hours %MAXHOURS% >> "%LOGROOT%\train_seed%%s.log" 2>&1
    if errorlevel 1 (
      echo ERROR: training failed at seed %%s -- see %LOGROOT%\train_seed%%s.log
      exit /b 1
    )
    echo done seed %%s
  )
) else (
  for /l %%s in (0,1,%SEEDS%) do (
    start "" /b "%PY%" src\train.py --level %LEVEL% --seed %%s --n-envs %NENVS% --max-hours %MAXHOURS% >> "%LOGROOT%\train_seed%%s.log" 2>&1
  )
  echo training started -- detached, per-seed logs in %LOGROOT%
)

endlocal
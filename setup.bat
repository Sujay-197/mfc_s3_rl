@echo off
REM ---------------------------------------------------------------------------
REM Setup for this project -- Windows.
REM Usage:  setup.bat
REM   Creates .venv if missing, then installs:
REM     * torch from the CUDA index (see README)
REM     * the pinned requirements in requirements.txt
REM ---------------------------------------------------------------------------

setlocal enabledextensions
cd /d "%~dp0"

:: 1) python must be on PATH
where python >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python 3.13+ not found on PATH.
  echo Install it from https://www.python.org/downloads/windows/ or use uv.
  exit /b 1
)

:: 2) create the venv if missing
if not exist .venv\Scripts\python.exe (
  echo [1/5] Creating .venv ...
  python -m venv .venv
)

:: 3) activate the venv
call .venv\Scripts\activate.bat
if errorlevel 1 (
  echo ERROR: could not activate .venv
  exit /b 1
)

:: 4) pip + torch
echo [3/5] Upgrading pip ...
python -m pip install --upgrade pip

echo [4/5] Installing torch (CUDA build)
pip install torch --index-url https://download.pytorch.org/whl/cu128

:: 5) pinned requirements
echo [5/5] Installing pinned requirements
pip install -r requirements.txt

echo.
echo Setup complete. The venv is active in this window. Run train.bat /
echo outputs.bat / build_math_pdf.py from here, or re-activate with
echo "call .venv\Scripts\activate.bat" afterwards.
endlocal
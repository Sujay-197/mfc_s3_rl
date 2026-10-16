@echo off
REM ---------------------------------------------------------------------------
REM Generate the submission outputs for one level -- Windows.
REM Usage:  outputs.bat <level>
REM   Example: outputs.bat 1-2
REM   Writes:
REM     results_<level>.md / results_<level>.json
REM     training_curve_<level>.png
REM     demo_<level>.mp4
REM     page_<level>.html
REM     artifacts\math_concepts.pdf (the math write-up)
REM ---------------------------------------------------------------------------

setlocal enabledextensions
cd /d "%~dp0"

set LEVEL=%~1
if "%LEVEL%" equ "" set LEVEL=1-2

set PY=.venv\Scripts\python.exe

echo [1/3] report.py --level %LEVEL%
"%PY%" src\report.py --level %LEVEL%

echo.
echo [2/3] build_page.py --level %LEVEL%
"%PY%" src\build_page.py --level %LEVEL%

echo.
echo [3/3] build_math_pdf.py
"%PY%" src\build_math_pdf.py

endlocal
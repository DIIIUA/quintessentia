@echo off
setlocal
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONUTF8=1"
set "SCRIPT_DIR=%~dp0"
if defined QUINTESSENTIA_PYTHON (
    set "PY=%QUINTESSENTIA_PYTHON%"
) else if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
    set "PY=%SCRIPT_DIR%.venv\Scripts\python.exe"
) else (
    set "PY=python"
)
"%PY%" -X utf8 -u "%SCRIPT_DIR%quintessentia.py" %*
exit /b %errorlevel%

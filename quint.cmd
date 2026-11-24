@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
if defined QUINTESSENTIA_PYTHON (
    set "PY=%QUINTESSENTIA_PYTHON%"
) else if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
    set "PY=%SCRIPT_DIR%.venv\Scripts\python.exe"
) else (
    set "PY=python"
)
"%PY%" "%SCRIPT_DIR%quintessentia.py" %*
exit /b %errorlevel%

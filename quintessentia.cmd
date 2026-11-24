@echo off
setlocal
rem Quintessentia: quintessentia.cmd <doctor|ask|chat|serve|discover> [options]
set "PYTHONDONTWRITEBYTECODE=1"
set "PY=python"
if defined QUINTESSENTIA_PYTHON (
    set "PY=%QUINTESSENTIA_PYTHON%"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY=%~dp0.venv\Scripts\python.exe"
)
"%PY%" -X utf8 -u "%~dp0quintessentia.py" %*
exit /b %ERRORLEVEL%

@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" goto check_environment

set "PYTHON="
for %%V in (3.14 3.13 3.12 3.11 3.10) do (
  if not defined PYTHON (
    py -%%V -c "import sys" >nul 2>&1
    if not errorlevel 1 set "PYTHON=py -%%V"
  )
)
if not defined PYTHON (
  where python >nul 2>&1
  if errorlevel 1 goto missing_python
  python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
  if errorlevel 1 goto missing_python
  set "PYTHON=python"
)
%PYTHON% -m venv .venv
if errorlevel 1 goto failed

echo Installing desktop dependencies. This only happens on first launch.
".venv\Scripts\python.exe" -m ensurepip --upgrade
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -e ".[desktop]"
if errorlevel 1 goto failed

:check_environment
".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if errorlevel 1 goto incompatible_environment

:run
".venv\Scripts\python.exe" -c "import webview" >nul 2>&1
if errorlevel 1 (
  echo Desktop dependencies are missing. Delete .venv and run this file again.
  goto failed
)

".venv\Scripts\python.exe" -m transformer_modeling.visual_app.desktop %*
if errorlevel 1 goto failed
exit /b 0

:missing_python
echo Python 3.10 or later was not found.
echo Install it from https://www.python.org/downloads/ and enable Add Python to PATH.
pause
exit /b 1

:incompatible_environment
echo The existing .venv uses Python older than 3.10.
echo Delete the .venv folder, then run this file again with Python 3.10 or later.
pause
exit /b 1

:failed
echo Failed to start the desktop application.
pause
exit /b 1

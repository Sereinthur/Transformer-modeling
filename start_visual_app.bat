@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" goto run

where py >nul 2>&1
if not errorlevel 1 (
  py -3 -m venv .venv
) else (
  where python >nul 2>&1
  if errorlevel 1 goto missing_python
  python -m venv .venv
)
if errorlevel 1 goto failed

echo Installing desktop dependencies. This only happens on first launch.
".venv\Scripts\python.exe" -m ensurepip --upgrade
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -e ".[desktop]"
if errorlevel 1 goto failed

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

:failed
echo Failed to start the desktop application.
pause
exit /b 1

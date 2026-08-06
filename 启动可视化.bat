@echo off
setlocal EnableExtensions
chcp 65001 >nul
set "PORT=8001"
set "URL=http://127.0.0.1:%PORT%/"
set "NO_BROWSER=0"
for %%A in (%*) do if /I "%%~A"=="--no-browser" set "NO_BROWSER=1"
cd /d "%~dp0"

powershell -NoProfile -Command "$listener = Get-NetTCPConnection -State Listen -LocalPort %PORT% -ErrorAction SilentlyContinue | Select-Object -First 1; if (-not $listener) { exit 10 }; $processInfo = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + $listener.OwningProcess); if ($processInfo.Name -match '^python(\.exe)?$' -and $processInfo.CommandLine -match 'transformer_modeling\.visual_app') { Write-Host ('Visual Modeling is already running (PID ' + $listener.OwningProcess + ').'); exit 0 }; Write-Host ('Port %PORT% is owned by ' + $processInfo.Name + '; refusing to reuse it.'); exit 2"
set "PORT_STATE=%ERRORLEVEL%"
if "%PORT_STATE%"=="0" (
  if "%NO_BROWSER%"=="0" start "" "%URL%"
  exit /b 0
)
if not "%PORT_STATE%"=="10" (
  pause
  exit /b %PORT_STATE%
)

where py >nul 2>&1
if not errorlevel 1 (
  set "PYTHON=py -3"
) else (
  set "PYTHON=python"
)
echo 启动可视化建模：%URL%
%PYTHON% -m transformer_modeling.visual_app %*
if errorlevel 1 pause

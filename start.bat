@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Runtime environment not found. Please run setup.bat first.
  pause
  exit /b 1
)

powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }" >nul 2>nul
if not errorlevel 1 (
  echo The system is already running at http://127.0.0.1:7860
  echo Opening the existing page instead of starting a second copy.
  start "" "http://127.0.0.1:7860"
  exit /b 0
)

set PYTHONUTF8=1
echo Starting MuAn Vision Fall Detection System...
echo Keep this window open while using the system.
".venv\Scripts\python.exe" app.py

if errorlevel 1 (
  echo.
  echo The application stopped with an error. See the message above.
  pause
)

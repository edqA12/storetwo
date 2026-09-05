@echo off
setlocal
cd /d "%~dp0"

echo [1/5] Checking Python...
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.10 to 3.13 and enable Add Python to PATH.
  pause
  exit /b 1
)

python -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)"
if errorlevel 1 (
  echo Unsupported Python version. Python 3.11 or 3.12 is recommended.
  pause
  exit /b 1
)

echo [2/5] Creating an isolated environment...
if not exist ".venv\Scripts\python.exe" python -m venv .venv
if errorlevel 1 goto :failed

echo [3/5] Installing dependencies. This may take several minutes...
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
if errorlevel 1 goto :failed
python -m pip install -r requirements.txt
if errorlevel 1 goto :failed

echo [4/5] Preparing the pose model...
python scripts\download_model.py
if errorlevel 1 goto :failed

echo [5/5] Running environment checks and tests...
python scripts\check_environment.py
if errorlevel 1 goto :failed
if not exist ".test_tmp" mkdir ".test_tmp"
set TEMP=%CD%\.test_tmp
set TMP=%CD%\.test_tmp
python -m pytest --basetemp "%CD%\.test_tmp\pytest-run"
if errorlevel 1 goto :failed

echo.
echo Setup completed. Double-click start.bat to launch the system.
pause
exit /b 0

:failed
echo.
echo Setup did not complete. Review the error message above and check the network connection.
pause
exit /b 1

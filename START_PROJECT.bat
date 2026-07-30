@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo Predictive Maintenance Anomaly Lab
echo ==========================================

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating virtual environment...
    py -3.11 -m venv .venv 2>nul
    if errorlevel 1 python -m venv .venv
)

call ".venv\Scripts\activate.bat"

echo [2/3] Installing requirements...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Installation failed.
    pause
    exit /b 1
)

echo [3/3] Running project...
python run_project.py

echo.
echo Project finished. Open outputs\REPORT.md
pause

@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found.
    echo Run START_PROJECT.bat first.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"
python run_project.py
pause

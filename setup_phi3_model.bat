@echo off
echo ===================================
echo Phi-3 Mini Model Setup and Fix Tool
echo ===================================
echo.
echo This script will download and set up the Microsoft Phi-3 Mini model
echo for title generation. This may take a few minutes on first run.
echo.
echo Press any key to continue or Ctrl+C to cancel...
pause > nul

echo.
echo Setting up Python environment...
cd %~dp0

REM Check if virtual environment exists
if exist venv\Scripts\activate.bat (
    echo Using virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo No virtual environment found, using system Python...
)

REM Install required packages
echo.
echo Installing required packages...
pip install --upgrade pip
pip install torch transformers huggingface_hub

echo.
echo Running Phi-3 model fix...
python fix_phi3_model.py

echo.
echo Testing title generation...
python test_phi3_simple.py

echo.
echo Setup complete! Press any key to exit...
pause > nul
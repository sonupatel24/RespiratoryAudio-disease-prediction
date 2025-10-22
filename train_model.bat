@echo off
echo ============================================================
echo RESPIRATORY DISEASE LSTM MODEL TRAINING
echo ============================================================
echo.

echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo.
echo Installing requirements...
pip install -r requirements.txt

echo.
echo Starting model training...
python train_model.py

echo.
echo Training completed! Press any key to continue...
pause

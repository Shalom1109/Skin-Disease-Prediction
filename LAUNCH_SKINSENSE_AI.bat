@echo off
echo ========================================
echo   SKINSENSE AI - CLINICAL SYSTEM
echo ========================================
echo Starting Unified Server...
echo Please wait for the "Model loaded" message.
echo.
echo Once the model is loaded, the system will open in your browser.
echo.
cd backend
start http://127.0.0.1:8000
python -m uvicorn app:app --host 127.0.0.1 --port 8000
pause

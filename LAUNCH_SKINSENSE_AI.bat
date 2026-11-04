@echo off
echo ========================================
echo   SKINSENSE AI - CLINICAL SYSTEM
echo ========================================
echo Starting server... the browser opens in ~15 seconds
echo (the PyTorch model needs a moment to load).
echo.
cd /d "%~dp0backend"
start "" cmd /c "timeout /t 15 /nobreak >nul & start http://127.0.0.1:8000"
python -m uvicorn app:app --host 127.0.0.1 --port 8000
pause

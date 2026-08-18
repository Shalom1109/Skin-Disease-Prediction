@echo off
echo ========================================
echo   Stopping SkinSense AI Background Server
echo ========================================
echo.
echo Searching for processes running on port 8000...

FOR /F "tokens=5" %%T IN ('netstat -a -n -o ^| findstr :8000') DO (
    echo Killing Process ID: %%T
    taskkill /f /pid %%T >nul 2>&1
)

echo.
echo Server has been successfully stopped.
pause

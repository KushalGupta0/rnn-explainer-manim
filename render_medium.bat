@echo off
setlocal
cd /d "%~dp0"

echo =======================================================================
echo  RNN Explainer Video - Rendering Standard Quality (720p, 30fps)
echo =======================================================================

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" render.py --quality medium %*
) else (
    python render.py --quality medium %*
)

if errorlevel 1 (
    echo.
    echo [ERROR] Render failed. Please check error messages above.
    pause
    exit /b %errorlevel%
)

echo.
echo [SUCCESS] Video rendered successfully!
pause

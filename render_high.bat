@echo off
setlocal
cd /d "%~dp0"

echo =======================================================================
echo  RNN Explainer Video - Rendering Full HD Quality (1080p, 60fps)
echo =======================================================================

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" render.py --quality high %*
) else (
    python render.py --quality high %*
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

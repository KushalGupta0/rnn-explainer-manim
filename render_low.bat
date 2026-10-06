@echo off
setlocal
cd /d "%~dp0"

echo =======================================================================
echo  RNN Explainer Video - Rendering Preview Quality (480p, 15fps)
echo =======================================================================

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" render.py --quality low %*
) else (
    python render.py --quality low %*
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

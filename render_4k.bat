@echo off
setlocal
cd /d "%~dp0"

echo =======================================================================
echo  RNN Explainer Video - Rendering Ultra HD 4K Quality (2160p, 60fps)
echo =======================================================================

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" render.py --quality 4k %*
) else (
    python render.py --quality 4k %*
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

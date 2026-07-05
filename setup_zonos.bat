@echo off
echo ============================================
echo   Alex Voice Cloner - Zonos Setup
echo ============================================
echo.

REM --- Check for uv ---
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo [INFO] uv not found. Installing uv...
    pip install -U uv
)

echo [Step 1] Cloning Zonos repository...
if exist "Zonos" (
    echo   -> Zonos directory already exists, pulling latest...
    cd Zonos
    git pull
    cd ..
) else (
    git clone https://github.com/Zyphra/Zonos.git
)

echo.
echo [Step 2] Installing Zonos dependencies with uv...
cd Zonos
uv sync
uv pip install -e .
cd ..

echo.
echo ============================================
echo   Setup complete!
echo.
echo   To run (from this directory):
echo   python clone_alex_zonos.py
echo ============================================
pause

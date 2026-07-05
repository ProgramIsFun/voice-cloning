@echo off
echo ============================================
echo   Alex Voice Cloner - Zonos Docker Setup
echo ============================================
echo.

REM --- Check for Docker ---
where docker >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Docker not found.
    echo Install Docker Desktop from: https://www.docker.com/products/docker-desktop/
    echo Make sure WSL2 backend is enabled.
    pause
    exit /b 1
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
echo [Step 2] Building Docker image (this will take a few minutes on first run)...
docker compose build

echo.
echo ============================================
echo   Setup complete!
echo.
echo   To run voice cloning:
echo   docker compose up
echo.
echo   To run in background:
echo   docker compose up -d
echo.
echo   To stop:
echo   docker compose down
echo ============================================
pause

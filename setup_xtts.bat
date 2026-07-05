@echo off
echo ============================================
echo   Alex Voice Cloner - Environment Setup
echo ============================================
echo.

REM --- Option 1: XTTS v2 (Recommended, Easiest) ---
echo [Option 1] XTTS v2 - Stable and easy
echo   Installing packages...
echo.

pip install -U "coqui-tts" torch torchaudio "transformers==4.36.2"

echo.
echo ============================================
echo   Setup complete!
echo.
echo   Next steps:
echo   1. Place a 6-12 second clip of Alex's voice
echo      as "alex_reference.wav" in this folder.
echo   2. Run: python clone_alex_xtts.py
echo ============================================
pause

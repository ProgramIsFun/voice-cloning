# Voice Cloning

AI voice cloning using XTTS v2 (Coqui TTS) with CUDA acceleration.

Clone any voice from just a few seconds of audio, running locally on your machine.

## Features

- Zero-shot voice cloning from short audio reference
- CUDA acceleration (RTX 4090)
- XTTS v2 multilingual model
- Speaker embedding caching (faster re-runs)
- Auto-play generated audio

## Requirements

- Python 3.11
- NVIDIA GPU with CUDA support
- ~2GB disk for model weights

## Setup

```bash
# Install dependencies
pip install -U "coqui-tts" torch torchaudio soundfile

# Install CUDA torch (if you have NVIDIA GPU)
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

## Usage

```powershell
cd C:\Users\whouse\Desktop\ref\voice-cloning
set COQUI_TOS_AGREED=1
py -3.11 clone_alex_xtts.py
```

1. Place a reference audio file (6-12 seconds of clean speech) in this folder
2. Update `REFERENCE_AUDIO` filename in `clone_alex_xtts.py`
3. Run the command above
4. Generated audio will be saved as `alex_output.wav` and auto-played

## How It Works

1. Loads XTTS v2 model (~2GB download on first run)
2. Extracts speaker embedding from reference audio (cached for future runs)
3. Generates speech using the cloned voice
4. Saves output as WAV and plays it

## Output

Generated audio is saved as `alex_output.wav` in the project folder.

## License

Non-commercial use only (CPML license from Coqui).

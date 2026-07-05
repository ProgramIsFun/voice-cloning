# Voice Cloning

AI voice cloning using XTTS v2 (Coqui TTS) with CUDA acceleration.

Clone any voice from just a few seconds of audio, running locally on your machine.

## Features

- Zero-shot voice cloning from short audio reference
- CUDA acceleration (RTX 4090)
- XTTS v2 multilingual model
- Post-processing effects

## Requirements

- Python 3.11
- NVIDIA GPU with CUDA support
- ~2GB disk for model weights

## Setup

```bash
# Install dependencies
pip install -U "coqui-tts" torch torchaudio soundfile pydub

# Install CUDA torch (if you have NVIDIA GPU)
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

## Usage

1. Place a reference audio file (6-12 seconds of clean speech) in this folder
2. Update the filename in `clone_alex_xtts.py`
3. Run:

```bash
set COQUI_TOS_AGREED=1
py -3.11 clone_alex_xtts.py
```

## Output

Generated audio will be saved as `alex_output.wav` in the project folder.

## License

Non-commercial use only (CPML license from Coqui).

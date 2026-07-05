# Voice Cloning

AI voice cloning using XTTS v2 (Coqui TTS) and Zonos v0.1 (Zyphra) with CUDA acceleration.

Clone any voice from just a few seconds of audio, running locally on your machine.

## Models

| Model | Quality | Setup |
|-------|---------|-------|
| **Zonos v0.1** | Much better — more natural, expressive, and consistent | Docker (recommended) |
| XTTS v2 | Good baseline | Direct pip install |

Zonos produces significantly better results than XTTS — the cloned voice sounds more natural, has better prosody, and handles longer sentences with more consistent tone.

## Features

- Zero-shot voice cloning from short audio reference
- CUDA acceleration
- Speaker embedding caching (faster re-runs)
- Auto-play generated audio

## Setup

### Zonos (Docker) — Recommended

```powershell
cd C:\Users\whouse\Desktop\ref\voice-cloning
setup_zonos.bat
```

Then run:
```powershell
docker compose up
```

### XTTS v2 (Direct)

```bash
pip install -U "coqui-tts" torch torchaudio soundfile
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

```powershell
set COQUI_TOS_AGREED=1
py -3.11 clone_alex_xtts.py
```

## Usage

1. Place a reference audio file (6-12 seconds of clean speech) in this folder
2. Update `REFERENCE_AUDIO` filename in the script
3. Run the appropriate command above
4. Generated audio will be saved and auto-played

## Reference Audio

The source reference audio is `49800b87-fe13-47ec-93bd-361e274c39fc.mp3`, extracted from the YouTube video *"Valerian and the City of a Thousand Planets | "Welcome" Clip | Own It Now"*.

## Output

- `alex_zonos_output.wav` — Zonos clone (English)
- `alex_zonos_output_ja.wav` — Zonos clone (Japanese)
- `alex_output.wav` — XTTS clone

## TODO

- [ ] Improve generation speed (currently slow on consumer GPUs)
- [ ] Improve Japanese voice clarity (currently lower quality than English)

## License

XTTS: Non-commercial use only (CPML license from Coqui).
Zonos: Apache 2.0 license.

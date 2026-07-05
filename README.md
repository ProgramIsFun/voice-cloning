# Voice Cloning

AI voice cloning using XTTS v2 (Coqui TTS), Zonos v0.1 (Zyphra), and F5-TTS with CUDA acceleration.

Clone any voice from just a few seconds of audio, running locally on your machine.

## Project Structure

```
voice-cloning/
├── 49800b87-...mp3          # Reference audio (shared)
├── clone_zonos.py           # Zonos script (best quality, slow)
├── zonos_output/            # Zonos output files
├── xtts/                    # XTTS v2 (good quality, medium speed)
│   ├── clone.py
│   └── output/
├── f5tts/                   # F5-TTS (good quality, fast) - coming soon
│   └── output/
├── Zonos/                   # Zonos library (git submodule)
└── tts_benchmark.py         # Speed comparison script
```

## Models

| Model | Quality | Speed | Setup |
|-------|---------|-------|-------|
| **Zonos v0.1** | Best — most natural, expressive | Slow (~30s+) | Docker |
| **F5-TTS** | Very good — fast + fine-tunable | Fast (real-time) | pip install |
| **XTTS v2** | Good baseline | Medium | pip install |

## Features

- Zero-shot voice cloning from short audio reference
- CUDA acceleration
- Speaker embedding caching (faster re-runs)
- Auto-play generated audio
- English + Japanese support (Zonos)

## Setup

### Zonos (Docker) — Best Quality

```powershell
cd C:\Users\whouse\Desktop\ref\voice-cloning
setup_zonos.bat
```

Then run:
```powershell
docker compose up
```

Or run directly:
```powershell
python clone_zonos.py
```

### XTTS v2 (Direct)

```bash
pip install -U "coqui-tts" torch torchaudio soundfile
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

```powershell
set COQUI_TOS_AGREED=1
cd xtts
py -3.11 clone.py
```

### F5-TTS (Coming Soon)

```powershell
cd f5tts
py clone.py
```

## Usage

1. Place a reference audio file (6-12 seconds of clean speech) in the root folder
2. Run the appropriate model script
3. Generated audio will be saved to the model's output folder

## Reference Audio

The source reference audio is `49800b87-fe13-47ec-93bd-361e274c39fc.mp3`, extracted from the YouTube video *"Valerian and the City of a Thousand Planets | "Welcome" Clip | Own It Now"*.

## Output

- `zonos_output/alex_zonos_output.wav` — Zonos clone (English)
- `zonos_output/alex_zonos_output_ja.wav` — Zonos clone (Japanese)
- `xtts/output/alex_output.wav` — XTTS clone

## Fast TTS Comparison

For speed comparison, run:
```powershell
pip install edge-tts
python tts_benchmark.py
```

edge-tts (Microsoft) completes in ~3s total vs Zonos taking much longer.

## TODO

- [ ] Set up F5-TTS
- [ ] Improve Zonos generation speed (currently slow on consumer GPUs)
  - [ ] Try `torch.compile` mode with `mode="reduce-overhead"` or `mode="max-autotune"`
  - [ ] Use half-precision (FP16/BF16) instead of FP32
  - [ ] Enable CUDA graph capture for repeated generation
  - [ ] Reduce `max_new_tokens` limit if not needed
  - [ ] Try the hybrid model (Zonos-v0.1-hybrid) — may be faster than transformer
  - [ ] Use TensorRT-ONNX export for inference
  - [ ] Batch multiple short sentences instead of one long one
- [ ] Improve Zonos Japanese voice clarity (currently lower quality than English)

## License

XTTS: Non-commercial use only (CPML license from Coqui).
Zonos: Apache 2.0 license.
F5-TTS: CC-BY-NC 4.0 license.

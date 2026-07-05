#!/usr/bin/env python3
"""
XTTS v2 Voice Cloning Script - Clone Alex from Valerian
"""
import sys
import os
import time

def main():
    print("=" * 60)
    print("  XTTS v2 - Alex Voice Cloner")
    print("=" * 60)
    sys.stdout.flush()

    REFERENCE_AUDIO = "49800b87-fe13-47ec-93bd-361e274c39fc.mp3"
    OUTPUT_FILE = "alex_output.wav"

    if not os.path.exists(REFERENCE_AUDIO):
        print(f"\n[ERROR] Reference audio not found: {REFERENCE_AUDIO}")
        sys.exit(1)

    print("[STEP 1/7] Checking GPU availability...")
    sys.stdout.flush()
    t0 = time.time()

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  -> Device: {device}")
        if device == "cpu":
            print("  -> WARNING: Running on CPU. This will be SLOW (10-30 min).")
        print(f"  -> Done in {time.time()-t0:.1f}s")
    except ImportError as e:
        print(f"\n[ERROR] torch import failed: {e}")
        sys.exit(1)

    print(f"\n[STEP 2/7] Importing TTS library...")
    sys.stdout.flush()
    t0 = time.time()
    try:
        from TTS.api import TTS
        print(f"  -> Done in {time.time()-t0:.1f}s")
    except ImportError as e:
        print(f"\n[ERROR] TTS import failed: {e}")
        sys.exit(1)

    print(f"\n[STEP 3/7] Loading XTTS v2 model (downloads ~2GB on first run)...")
    sys.stdout.flush()
    t0 = time.time()
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print(f"  -> Model loaded in {time.time()-t0:.1f}s")
    sys.stdout.flush()

    script_text = (
        "Intruder flight path locked. Welcome back, agent. "
        "All primary shields are stable at ninety-four percent."
    )

    print(f"\n[STEP 4/7] Loading reference audio: {REFERENCE_AUDIO}")
    sys.stdout.flush()
    t0 = time.time()
    import soundfile as sf
    info = sf.info(REFERENCE_AUDIO)
    print(f"  -> Duration: {info.duration:.2f}s, Sample rate: {info.samplerate}Hz, Channels: {info.channels}")
    print(f"  -> Done in {time.time()-t0:.1f}s")
    sys.stdout.flush()

    print(f"\n[STEP 5/7] Extracting voice fingerprint (speaker conditioning)...")
    sys.stdout.flush()
    t0 = time.time()
    import pickle
    cache_file = os.path.splitext(REFERENCE_AUDIO)[0] + "_embedding.pkl"

    if os.path.exists(cache_file):
        print(f"  -> Loading cached embedding from {cache_file}")
        with open(cache_file, "rb") as f:
            gpt_cond_latent, speaker_embedding = pickle.load(f)
    else:
        print("  -> Extracting embedding from reference audio...")
        gpt_cond_latent, speaker_embedding = tts.tts_model.get_conditioning_latents(audio_path=[REFERENCE_AUDIO])
        with open(cache_file, "wb") as f:
            pickle.dump((gpt_cond_latent, speaker_embedding), f)
        print(f"  -> Saved embedding to {cache_file}")
    print(f"  -> Done in {time.time()-t0:.1f}s")
    sys.stdout.flush()

    print(f"\n[STEP 6/7] Generating speech...")
    sys.stdout.flush()
    print(f"  -> Script: \"{script_text[:60]}...\"")
    t0 = time.time()

    out = tts.tts_model.inference(
        text=script_text,
        language="en",
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
    )

    import torchaudio
    torchaudio.save(OUTPUT_FILE, torch.tensor(out["wav"]).unsqueeze(0), 24000)
    print(f"  -> Speech generated in {time.time()-t0:.1f}s")
    sys.stdout.flush()

    print(f"\n[STEP 7/7] Saving output...")
    output_size = os.path.getsize(OUTPUT_FILE) / 1024
    print(f"  -> Saved to: {os.path.abspath(OUTPUT_FILE)}")
    print(f"  -> File size: {output_size:.1f} KB")
    print("=" * 60)
    print("  DONE! Your cloned Alex voice is ready.")
    print("=" * 60)
    sys.stdout.flush()

if __name__ == "__main__":
    main()

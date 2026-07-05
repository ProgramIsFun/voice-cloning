#!/usr/bin/env python3
"""
Zonos v0.1 Voice Cloning Script - Clone Alex from Valerian
Uses Zyphra's Zonos for high-fidelity zero-shot voice cloning.
"""
import sys
import os

def main():
    print("=" * 60)
    print("  Zonos v0.1 - Alex Voice Cloner")
    print("=" * 60)

    REFERENCE_AUDIO = "49800b87-fe13-47ec-93bd-361e274c39fc.mp3"
    OUTPUT_FILE = "alex_zonos_output.wav"

    if not os.path.exists(REFERENCE_AUDIO):
        print(f"\n[ERROR] Reference audio not found: {REFERENCE_AUDIO}")
        print("Please place a clean 6-12 second clip of Alex's voice as:")
        print(f"  -> {os.path.abspath(REFERENCE_AUDIO)}")
        sys.exit(1)

    try:
        import torch
        import torchaudio
        from zonos.model import Zonos
        from zonos.conditioning import make_cond_dict
        from zonos.utils import DEFAULT_DEVICE as device
    except ImportError:
        print("\n[ERROR] Required packages not installed.")
        print("Setup steps:")
        print("  1. Run setup_zonos.bat to clone and install Zonos")
        print("  2. Then re-run this script")
        sys.exit(1)

    print(f"\n[1/5] Loading Zonos transformer model on {device}...")
    print("      (First run will download model weights from HuggingFace)")
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)

    print(f"\n[2/5] Loading reference audio: {REFERENCE_AUDIO}")
    wav, sampling_rate = torchaudio.load(REFERENCE_AUDIO)

    print("[3/5] Extracting speaker embedding...")
    speaker = model.make_speaker_embedding(wav, sampling_rate)

    script_text = (
        "Diagnostics complete. The hyperdrive core is fully operational. "
        "All systems nominal. Awaiting further instructions."
    )

    print(f"\n[4/5] Generating speech...")
    print(f"       Script: \"{script_text[:60]}...\"")
    cond_dict = make_cond_dict(text=script_text, speaker=speaker, language="en-us")
    conditioning = model.prepare_conditioning(cond_dict)

    codes = model.generate(conditioning)
    wavs = model.autoencoder.decode(codes)

    print(f"[5/5] Saving output to: {OUTPUT_FILE}")
    torchaudio.save(OUTPUT_FILE, wavs[0].cpu(), model.autoencoder.sampling_rate)

    print(f"\nDone! Audio saved to: {os.path.abspath(OUTPUT_FILE)}")
    print("=" * 60)

if __name__ == "__main__":
    main()

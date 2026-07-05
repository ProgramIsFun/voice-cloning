#!/usr/bin/env python3
"""
Zonos v0.1 Voice Cloning Script - Clone Alex from Valerian
Uses Zyphra's Zonos for high-fidelity zero-shot voice cloning.
"""
import sys
import os

def patch_speaker_cloning():
    """Patch zonos/speaker_cloning.py to fix CUDA device mismatch in MelSpectrogram."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("zonos.speaker_cloning")
        if spec is None or spec.origin is None:
            print("  -> Warning: Could not find zonos.speaker_cloning module")
            return
        path = spec.origin
        with open(path, "r") as f:
            lines = f.readlines()

        new_lines = []
        i = 0
        patched_spk = False
        patched_lda = False
        while i < len(lines):
            line = lines[i]

            # Patch SpeakerEmbedding: remove with torch.device(device): and add self.to(device)
            if "with torch.device(device):" in line and not patched_spk:
                # Find the indent level of this line
                indent = len(line) - len(line.lstrip())
                inner_indent = indent + 4
                i += 1
                # Collect the indented block, dedent by one level
                block_lines = []
                while i < len(lines) and (
                    lines[i].startswith(" " * inner_indent) or lines[i].strip() == ""
                ):
                    if lines[i].startswith(" " * inner_indent):
                        block_lines.append(lines[i][4:])
                    elif lines[i].strip() == "":
                        block_lines.append(lines[i])
                    else:
                        break
                    i += 1
                new_lines.extend(block_lines)
                new_lines.append(" " * indent + "self.to(device)\n")
                patched_spk = True
                continue

            # Patch SpeakerEmbeddingLDA: remove with torch.device(device): wrapper
            if "with torch.device(device):" in line and not patched_lda:
                # Find the indent level of this line
                indent = len(line) - len(line.lstrip())
                inner_indent = indent + 4
                i += 1
                # Collect the indented block
                block_lines = []
                while i < len(lines) and lines[i].strip() != "" and (
                    lines[i].startswith(" " * inner_indent) or lines[i].strip() == ""
                ):
                    # Remove one level of indentation
                    if lines[i].startswith(" " * inner_indent):
                        block_lines.append(lines[i][4:])
                    else:
                        block_lines.append(lines[i])
                    i += 1
                new_lines.extend(block_lines)
                new_lines.append(" " * indent + "self.to(device)\n")
                patched_lda = True
                continue

            new_lines.append(line)
            i += 1

        if patched_spk or patched_lda:
            with open(path, "w") as f:
                f.writelines(new_lines)
            print(f"  -> Patched speaker_cloning.py (spk={patched_spk}, lda={patched_lda})")
        else:
            print("  -> speaker_cloning.py already patched or no changes needed")
    except Exception as e:
        print(f"  -> Warning: Could not patch speaker_cloning.py: {e}")

patch_speaker_cloning()

def main():
    import time
    t_start = time.time()

    print("=" * 60)
    print("  Zonos v0.1 - Alex Voice Cloner (English + Japanese)")
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

    print(f"\n[1/9] Loading Zonos transformer model on {device}...")
    print("      (First run will download model weights from HuggingFace)")
    t0 = time.time()
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
    print(f"  -> Done in {time.time()-t0:.1f}s")

    print(f"\n[2/9] Loading reference audio: {REFERENCE_AUDIO}")
    t0 = time.time()
    wav, sampling_rate = torchaudio.load(REFERENCE_AUDIO)
    print(f"  -> Done in {time.time()-t0:.1f}s")

    import pickle
    cache_file = os.path.splitext(REFERENCE_AUDIO)[0] + "_zonos_embedding.pkl"

    if os.path.exists(cache_file):
        print(f"\n[3/9] Loading cached speaker embedding from {cache_file}")
        t0 = time.time()
        with open(cache_file, "rb") as f:
            speaker = pickle.load(f)
    else:
        print(f"\n[3/9] Extracting speaker embedding...")
        t0 = time.time()
        speaker = model.make_speaker_embedding(wav, sampling_rate)
        with open(cache_file, "wb") as f:
            pickle.dump(speaker, f)
        print(f"  -> Saved embedding to {cache_file}")
    print(f"  -> Done in {time.time()-t0:.1f}s")

    print(f"\n[4/9] Warming up (torch.compile JIT)...")
    t0 = time.time()
    dummy_cond = make_cond_dict(text="warmup", speaker=speaker, language="en-us")
    dummy_conding = model.prepare_conditioning(dummy_cond)
    with torch.no_grad():
        _ = model.generate(dummy_conding, max_new_tokens=200, progress_bar=False)
    print(f"  -> Done in {time.time()-t0:.1f}s")

    # --- English generation ---
    script_text_en = (
        "Diagnostics complete. The hyperdrive core is fully operational. "
        "All systems nominal. Awaiting further instructions."
    )

    print(f"\n[5/9] Preparing conditioning (English)...")
    print(f"       Script: \"{script_text_en[:60]}...\"")
    t0 = time.time()
    cond_dict = make_cond_dict(text=script_text_en, speaker=speaker, language="en-us")
    conditioning = model.prepare_conditioning(cond_dict)
    print(f"  -> Done in {time.time()-t0:.1f}s")

    print(f"\n[6/9] Generating English speech...")
    t0 = time.time()
    codes = model.generate(conditioning)
    gen_time = time.time()-t0
    print(f"  -> Generated {codes.shape[2]} tokens in {gen_time:.1f}s ({codes.shape[2]/gen_time:.1f} tokens/s)")

    print(f"\n[7/9] Decoding and saving English output...")
    t0 = time.time()
    wavs = model.autoencoder.decode(codes)
    torchaudio.save(OUTPUT_FILE, wavs[0].cpu(), model.autoencoder.sampling_rate)
    print(f"  -> Done in {time.time()-t0:.1f}s")

    # --- Japanese generation ---
    script_text_ja = (
        "診断完了。ハイパードライブコアは完全に稼働している。"
        "全システム正常。さらなる指示を待っている。"
    )
    OUTPUT_FILE_JA = "alex_zonos_output_ja.wav"

    print(f"\n[8/9] Preparing conditioning (Japanese)...")
    print(f"       Script: \"{script_text_ja}\"")
    t0 = time.time()
    cond_dict_ja = make_cond_dict(
        text=script_text_ja, speaker=speaker, language="ja", speaking_rate=13.0
    )
    conditioning_ja = model.prepare_conditioning(cond_dict_ja)
    print(f"  -> Done in {time.time()-t0:.1f}s")

    print(f"\n[9/9] Generating and saving Japanese speech...")
    t0 = time.time()
    codes_ja = model.generate(conditioning_ja)
    gen_time = time.time()-t0
    print(f"  -> Generated {codes_ja.shape[2]} tokens in {gen_time:.1f}s ({codes_ja.shape[2]/gen_time:.1f} tokens/s)")

    wavs_ja = model.autoencoder.decode(codes_ja)
    torchaudio.save(OUTPUT_FILE_JA, wavs_ja[0].cpu(), model.autoencoder.sampling_rate)

    print(f"\nDone! Audio saved to:")
    print(f"  English: {os.path.abspath(OUTPUT_FILE)}")
    print(f"  Japanese: {os.path.abspath(OUTPUT_FILE_JA)}")
    print(f"Total time: {time.time()-t_start:.1f}s")
    print("=" * 60)

    if sys.platform == "win32":
        os.startfile(os.path.abspath(OUTPUT_FILE))
        os.startfile(os.path.abspath(OUTPUT_FILE_JA))

if __name__ == "__main__":
    main()

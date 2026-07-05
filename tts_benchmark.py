#!/usr/bin/env python3
"""
Simple TTS benchmark - compare Zonos with faster alternatives.
Uses pyttsx3 (offline, very fast) or edge-tts (Microsoft, fast + decent quality).
"""
import sys
import os
import time

TEXT_EN = (
    "Diagnostics complete. The hyperdrive core is fully operational. "
    "All systems nominal. Awaiting further instructions."
)
TEXT_JA = (
    "診断完了。ハイパードライブコアは完全に稼働している。"
    "全システム正常。さらなる指示を待っている。"
)

def run_pyttsx3():
    """pyttsx3 - offline, instant, but robotic voice."""
    import pyttsx3

    engine = pyttsx3.init()
    engine.setProperty("rate", 150)

    for lang, text, out in [("en", TEXT_EN, "fast_english.wav"), ("ja", TEXT_JA, "fast_japanese.wav")]:
        t0 = time.time()
        engine.save_to_file(text, out)
        engine.runAndWait()
        elapsed = time.time() - t0
        print(f"  pyttsx3 ({lang}): {elapsed:.2f}s -> {out}")

def run_edge_tts():
    """edge-tts - Microsoft voices, fast, decent quality."""
    import asyncio
    import edge_tts

    voices = {
        "en": ("en-US-GuyNeural", "fast_english.wav"),
        "ja": ("ja-JP-KeitaNeural", "fast_japanese.wav"),
    }

    async def generate():
        for lang, (voice, out) in voices.items():
            text = TEXT_EN if lang == "en" else TEXT_JA
            t0 = time.time()
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(out)
            elapsed = time.time() - t0
            print(f"  edge-tts ({lang}): {elapsed:.2f}s -> {out}")

    asyncio.run(generate())

def run_gtts():
    """gTTS - Google TTS, requires internet."""
    from gtts import gTTS

    for lang, text, out in [("en", TEXT_EN, "fast_english.mp3"), ("ja", TEXT_JA, "fast_japanese.mp3")]:
        t0 = time.time()
        tts = gTTS(text=text, lang=lang)
        tts.save(out)
        elapsed = time.time() - t0
        print(f"  gTTS ({lang}): {elapsed:.2f}s -> {out}")

if __name__ == "__main__":
    print("=" * 50)
    print("  Fast TTS Benchmark")
    print("=" * 50)

    available = []
    for name in ["pyttsx3", "edge_tts", "gtts"]:
        try:
            __import__(name if name != "gtts" else "gTTS")
            available.append(name)
        except ImportError:
            pass

    if not available:
        print("\nNo TTS libraries installed. Install one:")
        print("  pip install pyttsx3        # offline, instant")
        print("  pip install edge-tts       # fast, good quality")
        print("  pip install gTTS           # Google TTS")
        sys.exit(1)

    print(f"\nAvailable: {', '.join(available)}\n")

    t_total = time.time()

    if "pyttsx3" in available:
        print("[pyttsx3] Offline, instant, robotic voice:")
        try:
            run_pyttsx3()
        except Exception as e:
            print(f"  Error: {e}")

    if "edge_tts" in available:
        print("\n[edge-tts] Microsoft voices, fast + good quality:")
        try:
            run_edge_tts()
        except Exception as e:
            print(f"  Error: {e}")

    if "gtts" in available:
        print("\n[gTTS] Google TTS:")
        try:
            run_gtts()
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\nTotal: {time.time()-t_total:.2f}s")
    print("=" * 50)

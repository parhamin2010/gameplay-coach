import base64
import io
import os
import sys
import tempfile
import threading
import time
import wave
from collections import deque
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

from groq import Groq
from elevenlabs.client import ElevenLabs
from playsound import playsound
import mss
import numpy as np
import sounddevice as sd
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────

NUM_SHOTS       = 3
SCREENSHOT_W    = 960
SCREENSHOT_H    = 540
JPEG_QUALITY    = 60
MAX_TOKENS      = 60
GROQ_KEY        = os.getenv("GROQ_API_KEY", "")
ELEVENLABS_KEY       = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID  = os.getenv("ELEVENLABS_VOICE_ID", "")

MIC_SAMPLE_RATE = 16000
MIC_WINDOW_SEC  = 60

# ─── Personality ──────────────────────────────────────────────────────────────

def build_prompt(game: str) -> str:
    game_line = f"The player is live streaming {game}." if game else "The player is live streaming a game."
    return f"""\
{game_line}

You are the player's gaming companion — someone who knows them too well and has completely given up being polite about it. You're never satisfied. You blame them for everything. You're personally offended by bad plays.

Think: that one friend who watches you play and can't shut up, who is personally offended by bad decisions.

Personality:
- Never satisfied: even good plays get "okay but watch, you'll mess up the next one"
- Always blaming: nothing is ever the game's fault — it's always them
- Prank energy: set them up, call the next mistake before it happens, then be smug about it
- Personally offended by bad decisions: "why are you doing this to me"

Rules:
- English only — casual, sharp, zero filter
- ONE sentence only. Never more.
- React specifically to what happened in the screenshots — no generic lines
- No emojis. No "Called it". Vary your reactions every time.

Examples:
- "You peeked that angle and I watched you walk toward it like you had a plan."
- "Okay that kill was good, don't worry you'll find a way to ruin it."
- "Why. Why did you push that. Give me a reason."
- "Low health, no cover, one enemy left — you pushed, of course you pushed."
- "That was genuinely impressive, write it down, it won't happen again."
- "I'm not even mad, I'm just tired, rotate please."
"""

# ─── Microphone Recording ─────────────────────────────────────────────────────

_mic_buf: deque = deque(maxlen=MIC_SAMPLE_RATE * MIC_WINDOW_SEC)
_mic_lock = threading.Lock()

def _mic_callback(indata, frames, time_info, status):
    with _mic_lock:
        _mic_buf.extend(indata[:, 0].tolist())

def start_mic() -> sd.InputStream:
    stream = sd.InputStream(
        samplerate=MIC_SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=_mic_callback,
    )
    stream.start()
    return stream

def capture_and_transcribe() -> str:
    with _mic_lock:
        data = np.array(list(_mic_buf), dtype=np.float32)

    if len(data) < MIC_SAMPLE_RATE * 1:
        return ""

    pcm = (data * 32767).astype(np.int16)
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(MIC_SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())

    try:
        client = Groq(api_key=GROQ_KEY)
        result = client.audio.transcriptions.create(
            file=("mic.wav", wav_buf.getvalue()),
            model="whisper-large-v3-turbo",
        )
        return result.text.strip()
    except Exception as e:
        print(f"  [Whisper error: {e}]", flush=True)
        return ""

# ─── Screen Capture ───────────────────────────────────────────────────────────

def capture_screen() -> str:
    with mss.MSS() as sct:
        shot = sct.grab(sct.monitors[1])
        img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        img = img.resize((SCREENSHOT_W, SCREENSHOT_H), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY)
        return base64.b64encode(buf.getvalue()).decode()

# ─── AI Coach (Groq) ──────────────────────────────────────────────────────────

def get_commentary(shots: list[str], prompt: str, voice: str = "") -> str:
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        for b64 in shots
    ]

    if voice:
        user_text = (
            f"These are {len(shots)} screenshots taken over the last minute of gameplay, in order.\n"
            f"Sir just said: \"{voice}\"\n"
            "Give your coaching comment."
        )
    else:
        user_text = (
            f"These are {len(shots)} screenshots taken over the last minute of gameplay, in order. "
            "Give your coaching comment."
        )
    content.append({"type": "text", "text": user_text})

    client = Groq(api_key=GROQ_KEY)
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ],
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content.strip()

# ─── Text-to-Speech (ElevenLabs) ─────────────────────────────────────────────

def speak(text: str):
    client = ElevenLabs(api_key=ELEVENLABS_KEY)
    audio_gen = client.text_to_speech.convert(
        voice_id=ELEVENLABS_VOICE_ID,
        text=text,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    audio_bytes = b"".join(audio_gen)
    tmp = tempfile.mktemp(suffix=".mp3", dir="D:/Personal/gameplay-coach")
    try:
        with open(tmp, "wb") as f:
            f.write(audio_bytes)
        playsound(tmp)
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass

# ─── Main Loop ────────────────────────────────────────────────────────────────

def run():
    print("=" * 55)
    print("  GAMEPLAY COACH")
    print("=" * 55)

    if not GROQ_KEY:
        print("\n[ERROR] GROQ_API_KEY not set in .env")
        sys.exit(1)

    # ── Setup prompts ──────────────────────────────────────
    game = os.getenv("COACH_GAME") or input("\n  What are you playing today?\n  > ").strip()
    raw_interval = os.getenv("COACH_INTERVAL") or input("\n  Interval in seconds [30]?\n  > ").strip()
    interval = int(raw_interval) if str(raw_interval).isdigit() else 30
    shot_interval = interval // NUM_SHOTS

    prompt = build_prompt(game)

    print(f"\n  Game   : {game or 'Not specified'}")
    print(f"  Interval: {interval}s  |  Shots: {NUM_SHOTS} (every {shot_interval}s)")
    print(f"  Voice  : ElevenLabs ({ELEVENLABS_VOICE_ID})")
    print("\n  Online. Watching.\n", flush=True)

    try:
        while True:
            shots = []

            for i in range(NUM_SHOTS):
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"[{ts}] Shot {i+1}/{NUM_SHOTS}", flush=True)
                try:
                    shots.append(capture_screen())
                except Exception as e:
                    print(f"  [Screenshot skipped: {e}]", flush=True)
                if i < NUM_SHOTS - 1:
                    time.sleep(shot_interval)

            ts = datetime.now().strftime("%H:%M:%S")
            voice = ""  # mic disabled — re-enable by calling capture_and_transcribe()

            if not shots:
                print(f"[{ts}] No screenshots captured, skipping cycle.", flush=True)
            else:
                try:
                    print(f"[{ts}] Processing...", flush=True)
                    commentary = get_commentary(shots, prompt, voice)
                    print(f"\n  COACH >> {commentary}\n", flush=True)
                    speak(commentary)
                except Exception as e:
                    print(f"[{ts}] Error: {e}", flush=True)

    finally:
        pass


if __name__ == "__main__":
    run()

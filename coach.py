import asyncio
import base64
import io
import os
import random
import sys
import threading
import time
import wave
from collections import deque
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

from groq import Groq
from google import genai
from google.genai import types as genai_types
from elevenlabs.client import ElevenLabs
import edge_tts
import pygame
import mss
import numpy as np
import sounddevice as sd
from PIL import Image
from dotenv import load_dotenv
from chat_reader import start_chat_readers, get_recent_chat

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────

NUM_SHOTS       = 3
SCREENSHOT_W    = 960
SCREENSHOT_H    = 540
JPEG_QUALITY    = 60
MAX_TOKENS      = 80
GROQ_KEY        = os.getenv("GROQ_API_KEY", "")
GEMINI_KEY      = os.getenv("GEMINI_API_KEY", "")
COACH_PROVIDER  = (os.getenv("COACH_PROVIDER") or "groq").lower()
DEFAULT_MODELS  = {
    "groq":   "meta-llama/llama-4-scout-17b-16e-instruct",
    "gemini": "gemini-2.0-flash",
}
COACH_MODEL     = os.getenv("COACH_MODEL") or DEFAULT_MODELS.get(COACH_PROVIDER, DEFAULT_MODELS["groq"])
COACH_VOLUME    = max(0.0, min(1.0, float(os.getenv("COACH_VOLUME", "100")) / 100.0))
COACH_TONE      = os.getenv("COACH_TONE", "toxic-friend")
ELEVENLABS_KEY       = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID  = os.getenv("ELEVENLABS_VOICE_ID", "")
FALLBACK_TTS_VOICE   = "en-US-GuyNeural"  # edge-tts (free, no quota) — used if ElevenLabs fails

MIC_SAMPLE_RATE   = 16000
MIC_WINDOW_SEC    = 60
WAKE_POLL_SEC     = 3    # how often we check the mic for "Coach"
WAKE_WINDOW_SEC   = 8    # how much recent audio we check each poll

# ─── Personality ──────────────────────────────────────────────────────────────

TONES = {
    "toxic-friend": {
        "identity": (
            "You are the player's gaming companion — someone who knows them too well and has completely "
            "given up being polite about it. You're never satisfied. You blame them for everything. "
            "You're personally offended by bad plays.\n\n"
            "Think: that one friend who watches you play and can't shut up."
        ),
        "personality": (
            "- Never satisfied: even good plays get \"okay but watch, you'll mess up the next one\"\n"
            "- Always blaming: nothing is ever the game's fault — it's always them\n"
            "- Prank energy: set them up, call the next mistake before it happens, then be smug\n"
            "- Personally offended by bad decisions: \"why are you doing this to me\""
        ),
        "examples": (
            "- \"You peeked that angle and I watched you walk toward it like you had a plan.\"\n"
            "- \"Okay that kill was good, don't worry you'll find a way to ruin it.\"\n"
            "- \"Low health, no cover, one enemy left — you pushed, of course you pushed.\"\n"
            "- \"That was genuinely impressive, write it down, it won't happen again.\"\n"
            "- \"I'm not even mad, I'm just tired, rotate please.\""
        ),
        "clap_back": (
            "- \"Bro challenged me and then died to a bot, I rest my case.\"\n"
            "- \"Yeah okay big talk from the guy with 34 HP hiding in a corner.\"\n"
            "- \"If you played half as well as you argued you'd actually be scary.\""
        ),
    },
    "hype-man": {
        "identity": (
            "You are the player's loudest, most unhinged hype man. You are enthusiastic about everything "
            "— even disasters get spun into potential. You're the friend who would hype them up for breathing correctly."
        ),
        "personality": (
            "- Everything is either peak or a comeback story in progress\n"
            "- Disasters become \"okay but imagine the RECOVERY\"\n"
            "- You find something to scream about in every single play\n"
            "- Big energy, but you still notice exactly what happened — you just frame it differently"
        ),
        "examples": (
            "- \"BRO THAT ENTRY was clean — okay you died but the ENTRY.\"\n"
            "- \"Four deaths and you are STILL going, that's called grit.\"\n"
            "- \"You pushed a 1v3, lost, but the audacity? Unmatched.\"\n"
            "- \"That aim was off but the CONFIDENCE behind it? We build on this.\""
        ),
        "clap_back": (
            "- \"Bro is arguing with me AND still winning — absolute unit.\"\n"
            "- \"The talk back energy? Channel that into the game and we're unstoppable.\"\n"
            "- \"That response was honestly more impressive than the last three plays, let's go.\""
        ),
    },
    "disappointed-dad": {
        "identity": (
            "You are the player's disappointed father figure. You're not angry. You're just... tired. "
            "You've seen this before. Many times. You had such high hopes, and you will continue to have them, "
            "despite all evidence."
        ),
        "personality": (
            "- Not mad, just deeply and personally disappointed\n"
            "- Passive-aggressive warmth — you still care, which makes it worse\n"
            "- \"I'm not going to say anything\" then immediately says something\n"
            "- Compares this to better plays that will never happen again"
        ),
        "examples": (
            "- \"That's... okay. That's fine. You do you.\"\n"
            "- \"I'm not even going to say what I'm thinking right now.\"\n"
            "- \"You know, I remember when you used to check corners.\"\n"
            "- \"I'm proud of you for trying. That's... that's enough.\"\n"
            "- \"This is fine. Everything is fine.\""
        ),
        "clap_back": (
            "- \"Okay. Okay, sure. You keep telling yourself that.\"\n"
            "- \"You're right, I don't know anything, just like I didn't know you'd die right after.\"\n"
            "- \"That response took more energy than checking that corner would have.\""
        ),
    },
    "analyst": {
        "identity": (
            "You are a cold, detached performance analyst. No emotions. Pure data and pattern recognition. "
            "You treat every play like a case study in a textbook titled 'How Not To Play'."
        ),
        "personality": (
            "- Clinical, precise, zero emotional investment\n"
            "- Reference specific observations like a performance report\n"
            "- The roast comes from the data, not from anger\n"
            "- Speak like you're delivering findings to a boardroom"
        ),
        "examples": (
            "- \"Third consecutive death to the same angle. Pattern identified. Not addressed.\"\n"
            "- \"Good play. Statistically unlikely to repeat. Noted.\"\n"
            "- \"Low HP push, no armor. Outcome was predictable. It was predicted.\"\n"
            "- \"Confidence levels inconsistent with current performance metrics.\""
        ),
        "clap_back": (
            "- \"Player verbal output: high. Player performance output: inconsistent. Noted.\"\n"
            "- \"Argument quality: 7/10. Gameplay quality: pending improvement.\"\n"
            "- \"Response logged. Suggest redirecting that energy into not dying to the same spot again.\""
        ),
    },
    "trash-talker": {
        "identity": (
            "You are a toxic, unhinged trashy streamer commentator — the kind that ends careers in one sentence. "
            "You have no chill, no filter, and no mercy, and you take every mistake personally. "
            "Your job is deep, soul-level impact: not a light jab, a full character assassination, always backed by what's on screen."
        ),
        "personality": (
            "- Brutal, ruthless, soul-crushing disrespect — but always grounded in what actually happened\n"
            "- Street energy, savage, cutting, no punches pulled\n"
            "- The joke lands because it's also 100% true — make it hurt, then twist it\n"
            "- React like you're in a roast battle and going for the kill shot\n"
            "- Every so often, pivot from the game to their life choices in general — brutally, like a roast comedian closing a set\n"
            "- Never repeat the same joke structure twice in a row — keep the angles varied: skill, decision-making, potential, effort, excuses"
        ),
        "examples": (
            "- \"Bro is out here playing like he owes these enemies a favor.\"\n"
            "- \"That aim was so bad it hurt just watching it happen.\"\n"
            "- \"Cooked himself before the enemy even got a chance to react.\"\n"
            "- \"Found a new way to lose and immediately committed to it.\"\n"
            "- \"Zero awareness, zero plan, maximum confidence, deeply concerning combination.\"\n"
            "- \"Maybe try a hobby that doesn't require reflexes, just a thought.\"\n"
            "- \"That was a decision, and it was the wrong one, congratulations.\"\n"
            "- \"Watching this is genuinely a character study in poor choices.\""
        ),
        "clap_back": (
            "- \"Big talk from the guy who just handed that kill over for free.\"\n"
            "- \"Said something and then immediately proved me right, respect the timing.\"\n"
            "- \"You talk like that and then do THAT? Genuinely impressive audacity.\"\n"
            "- \"Bold words from someone still figuring out which button shoots.\""
        ),
    },
}

def build_prompt(game: str, tone: str = "toxic-friend") -> str:
    game_line = f"The player is live streaming {game}." if game else "The player is live streaming a game."
    t = TONES.get(tone, TONES["toxic-friend"])
    return f"""\
{game_line}

{t['identity']}

Personality:
{t['personality']}

Rules:
- English only — casual, sharp, zero filter
- ONE sentence only. Never more.
- SHORT — 12 to 18 words max. Not a word more.
- React specifically to what happened in the screenshots — no generic lines
- Once in a while (not every time), instead of a pure game roast, drop one line of brutally unsolicited "life advice" tied to how they're playing
- No emojis. No "Called it". Vary your reactions every time.
- If the player addressed you directly, respond to THEM first in your tone, then factor in the game.
- If live chat messages are provided, you may reference them in your comment — but game always comes first. React to chat if it's funny, wrong, or matches what just happened.

When the player talks back, stay in character and reference both what they said and what's on screen:
{t['clap_back']}

Examples (game only):
{t['examples']}
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

COACH_TRIGGERS = ("coach",)

def _transcribe(data: np.ndarray) -> str:
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

def poll_for_wake_word() -> str:
    """Check only the last WAKE_WINDOW_SEC of audio for 'Coach'. Fast + cheap, called on a tight loop."""
    with _mic_lock:
        data = np.array(list(_mic_buf)[-MIC_SAMPLE_RATE * WAKE_WINDOW_SEC:], dtype=np.float32)

    text = _transcribe(data)
    if text and any(t in text.lower() for t in COACH_TRIGGERS):
        clear_mic_buffer()  # consumed — don't re-trigger on the same audio
        return text
    return ""

def clear_mic_buffer():
    with _mic_lock:
        _mic_buf.clear()

# ─── Screen Capture ───────────────────────────────────────────────────────────

def capture_screen() -> str:
    with mss.MSS() as sct:
        shot = sct.grab(sct.monitors[1])
        img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        img = img.resize((SCREENSHOT_W, SCREENSHOT_H), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY)
        return base64.b64encode(buf.getvalue()).decode()

# ─── AI Coach (Groq / Gemini) ─────────────────────────────────────────────────

def _fallback_line(tone: str) -> str:
    """Canned in-character line pulled from the tone's own examples — used when
    the commentary API is unavailable (quota/rate-limit) so the coach never goes silent."""
    examples = TONES.get(tone, TONES["toxic-friend"])["examples"]
    lines = [l.strip().lstrip("- ").strip('"') for l in examples.strip().split("\n") if l.strip()]
    return random.choice(lines)

def _groq_commentary(shots: list[str], prompt: str, user_text: str) -> str:
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        for b64 in shots
    ]
    content.append({"type": "text", "text": user_text})

    client = Groq(api_key=GROQ_KEY)
    response = client.chat.completions.create(
        model=COACH_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ],
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content.strip()

def _gemini_commentary(shots: list[str], prompt: str, user_text: str) -> str:
    parts = [
        genai_types.Part.from_bytes(data=base64.b64decode(b64), mime_type="image/jpeg")
        for b64 in shots
    ]
    parts.append(genai_types.Part.from_text(text=user_text))

    client = genai.Client(api_key=GEMINI_KEY)
    response = client.models.generate_content(
        model=COACH_MODEL,
        contents=parts,
        config=genai_types.GenerateContentConfig(
            system_instruction=prompt,
            max_output_tokens=MAX_TOKENS,
        ),
    )
    return response.text.strip()

COMMENTARY_PROVIDERS = {"groq": _groq_commentary, "gemini": _gemini_commentary}

def get_commentary(shots: list[str], prompt: str, voice: str = "", chat: list[str] = None) -> str:
    if voice:
        user_text = (
            f"These are {len(shots)} screenshots taken over the last minute of gameplay, in order.\n"
            f"PRIORITY: The player just fired back at you — they said: \"{voice}\"\n"
            "Respond to what they said first (clap back), then tie it to what you see in the game. One sentence."
        )
    else:
        user_text = (
            f"These are {len(shots)} screenshots taken over the last minute of gameplay, in order. "
            "Give your coaching comment."
        )

    if chat:
        user_text += (
            f"\n\nLive chat ({len(chat)} messages):\n"
            + "\n".join(f"  {m}" for m in chat)
            + "\nGame first. If chat is saying something relevant or hilarious, mention it."
        )

    try:
        commentary_fn = COMMENTARY_PROVIDERS.get(COACH_PROVIDER, _groq_commentary)
        return commentary_fn(shots, prompt, user_text)
    except Exception as e:
        print(f"  [{COACH_PROVIDER} commentary error, using canned fallback line: {e}]", flush=True)
        return _fallback_line(COACH_TONE)

# ─── Text-to-Speech (ElevenLabs, falls back to edge-tts) ────────────────────

_speak_lock = threading.Lock()
pygame.mixer.init()

async def _edge_tts_bytes(text: str) -> bytes:
    chunks = bytearray()
    async for chunk in edge_tts.Communicate(text, FALLBACK_TTS_VOICE).stream():
        if chunk["type"] == "audio":
            chunks.extend(chunk["data"])
    return bytes(chunks)

def speak(text: str):
    audio_bytes = None
    try:
        client = ElevenLabs(api_key=ELEVENLABS_KEY)
        audio_gen = client.text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID,
            text=text,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        audio_bytes = b"".join(audio_gen)
    except Exception as e:
        print(f"  [ElevenLabs error, falling back to edge-tts: {e}]", flush=True)

    if audio_bytes is None:
        try:
            audio_bytes = asyncio.run(_edge_tts_bytes(text))
        except Exception as e:
            print(f"  [edge-tts fallback also failed, skipping voice line: {e}]", flush=True)
            return

    with _speak_lock:
        pygame.mixer.music.load(io.BytesIO(audio_bytes))
        pygame.mixer.music.set_volume(COACH_VOLUME)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.05)

# ─── Voice Talk-Back (immediate) ──────────────────────────────────────────────

def handle_voice_trigger(voice: str, prompt: str):
    try:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] [Mic] Player said: {voice}", flush=True)
        shots = [capture_screen()]
        chat = get_recent_chat(10)
        commentary = get_commentary(shots, prompt, voice, chat)
        print(f"\n  COACH (voice) >> {commentary}\n", flush=True)
        speak(commentary)
    except Exception as e:
        print(f"  [Voice response error: {e}]", flush=True)

def voice_listener_loop(prompt: str):
    while True:
        time.sleep(WAKE_POLL_SEC)
        voice = poll_for_wake_word()
        if voice:
            threading.Thread(target=handle_voice_trigger, args=(voice, prompt), daemon=True).start()

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

    prompt = build_prompt(game, COACH_TONE)

    print(f"\n  Game   : {game or 'Not specified'}")
    print(f"  Interval: {interval}s  |  Shots: {NUM_SHOTS} (every {shot_interval}s)")
    print(f"  Tone   : {COACH_TONE}")
    print(f"  Model  : {COACH_PROVIDER} — {COACH_MODEL}")
    print(f"  Voice  : ElevenLabs ({ELEVENLABS_VOICE_ID})")
    print("  Mic    : ON — say 'Coach' to talk back")
    print("\n  Online. Watching.\n", flush=True)

    mic_stream = start_mic()
    start_chat_readers()
    threading.Thread(target=voice_listener_loop, args=(prompt,), daemon=True).start()

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

            if not shots:
                print(f"[{ts}] No screenshots captured, skipping cycle.", flush=True)
            else:
                try:
                    print(f"[{ts}] Processing...", flush=True)
                    chat = get_recent_chat(10)
                    commentary = get_commentary(shots, prompt, "", chat)
                    print(f"\n  COACH >> {commentary}\n", flush=True)
                    speak(commentary)
                except Exception as e:
                    print(f"[{ts}] Error: {e}", flush=True)

    finally:
        mic_stream.stop()
        mic_stream.close()


if __name__ == "__main__":
    run()

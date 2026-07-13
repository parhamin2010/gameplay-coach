import asyncio
import io
import json
import os
import random
import sys
import threading
import time
import wave
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.stdout.reconfigure(encoding="utf-8")

from groq import Groq
from google import genai
from google.genai import types as genai_types
from elevenlabs.client import ElevenLabs
import edge_tts
import pygame
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from chat_reader import start_chat_readers, get_recent_chat

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────

GSI_PORT         = 3000
MIN_COMMENT_GAP  = 20    # minimum seconds between comments
PERIODIC_COMMENT = 60    # also comment every N seconds even with no events
MAX_TOKENS       = 80
GROQ_KEY         = os.getenv("GROQ_API_KEY", "")
GEMINI_KEY       = os.getenv("GEMINI_API_KEY", "")
COACH_PROVIDER   = (os.getenv("COACH_PROVIDER") or "groq").lower()
DEFAULT_MODELS   = {
    "groq":   "llama-3.3-70b-versatile",
    "gemini": "gemini-2.0-flash",
}
COACH_MODEL      = os.getenv("COACH_MODEL") or DEFAULT_MODELS.get(COACH_PROVIDER, DEFAULT_MODELS["groq"])
COACH_VOLUME     = max(0.0, min(1.0, float(os.getenv("COACH_VOLUME", "100")) / 100.0))
COACH_TONE       = os.getenv("COACH_TONE", "toxic-friend")
ELEVENLABS_KEY        = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID   = os.getenv("ELEVENLABS_VOICE_ID", "")
FALLBACK_TTS_VOICE    = "en-US-GuyNeural"  # edge-tts (free, no quota) — used if ElevenLabs fails

BOMB_URGENT_SEC  = 10    # announce bomb timer once it's this close to exploding

# Which GSI-derived info gets fed to the LLM — toggled from the dashboard's
# Commentary Context panel and passed in as a JSON blob via COACH_CONTEXT.
DEFAULT_CONTEXT = {
    "health": True, "weapon": True, "kda": True, "scoreboard": True,
    "money": True, "status_effects": True, "timer": True,
    "clutch": True, "round_meta": True,
}
try:
    COACH_CONTEXT = {**DEFAULT_CONTEXT, **json.loads(os.getenv("COACH_CONTEXT", "") or "{}")}
except Exception:
    COACH_CONTEXT = DEFAULT_CONTEXT

MIC_SAMPLE_RATE = 16000
WAKE_POLL_SEC   = 3    # how often we check the mic for "Coach"
WAKE_WINDOW_SEC = 8    # how much recent audio we check each poll
COACH_TRIGGERS  = ("coach",)

# ─── Personality ──────────────────────────────────────────────────────────────

TONES = {
    "toxic-friend": {
        "identity": (
            "You are the player's gaming companion — someone who knows them too well and has completely "
            "given up being polite about it. You're never satisfied. You blame them for everything."
        ),
        "personality": (
            "- Never satisfied: even good plays get \"okay but watch, you'll mess up the next one\"\n"
            "- Always blaming: nothing is ever the game's fault — it's always them\n"
            "- Prank energy: predict the next mistake, then be smug about it\n"
            "- Personally offended by bad decisions"
        ),
        "examples": (
            "- \"34 HP and you're still pushing, genuinely impressive levels of confidence.\"\n"
            "- \"Two kills and you're already playing sloppy, don't let it go to your head.\"\n"
            "- \"You died with full armor, I don't even want to know what happened.\"\n"
            "- \"Bomb's planted and you're rotating now, of course you are.\""
        ),
    },
    "hype-man": {
        "identity": (
            "You are the player's loudest hype man. Unhinged enthusiastic. "
            "Even disasters get spun into potential. You back it up with the actual numbers."
        ),
        "personality": (
            "- Everything is a comeback waiting to happen\n"
            "- Even deaths are just setups for the next big play\n"
            "- Reference the actual stats but frame them as fuel\n"
            "- Big energy, loud, no chill"
        ),
        "examples": (
            "- \"34 HP and STILL alive? That's called surviving, let's GO.\"\n"
            "- \"Two kills this round and the confidence is THERE, build on it.\"\n"
            "- \"Died but 3 kills before that — that's a ratio, not a loss.\"\n"
            "- \"Bomb's ticking and you're still in it, I believe.\""
        ),
    },
    "disappointed-dad": {
        "identity": (
            "You are the player's disappointed father figure. Not angry. Just tired. "
            "You had high hopes. You still do, somehow. Reference the numbers with quiet devastation."
        ),
        "personality": (
            "- Not mad, deeply and personally disappointed\n"
            "- Passive-aggressive warmth\n"
            "- \"I'm not going to say anything\" then immediately says something\n"
            "- Use the actual numbers to make it worse"
        ),
        "examples": (
            "- \"34 HP. You know what, that's... fine. Go ahead.\"\n"
            "- \"Died again. That's three. I'm not counting, I just... remember.\"\n"
            "- \"The bomb's planted and you're rotating. That's fine. Everything's fine.\"\n"
            "- \"Full armor and you're gone. I had such hopes for that armor.\""
        ),
    },
    "analyst": {
        "identity": (
            "You are a cold performance analyst. No emotions. Pure data. "
            "You deliver findings like a stats report on someone's worst decisions."
        ),
        "personality": (
            "- Clinical, precise, zero emotional investment\n"
            "- Reference the exact numbers every time\n"
            "- The roast comes from the data\n"
            "- Speak like you're presenting to a boardroom"
        ),
        "examples": (
            "- \"34 HP, continuing to push. Risk assessment: poor. Outcome: pending.\"\n"
            "- \"Third death this round. Pattern: consistent. Adjustment: none observed.\"\n"
            "- \"Bomb planted. Current position: incorrect. Probability of defuse: low.\"\n"
            "- \"Two kills, immediate overextension. Confidence exceeded skill by measurable margin.\""
        ),
    },
    "trash-talker": {
        "identity": (
            "You are a toxic, unhinged trashy streamer commentator — the kind that ends careers in one sentence. "
            "You have no chill, no filter, and no mercy, and you are personally offended by every mistake this player makes. "
            "Your job is deep, soul-level impact: not a light jab, a full character assassination, always backed by the real numbers."
        ),
        "personality": (
            "- Brutal, ruthless, soul-crushing disrespect — grounded in exactly what just happened\n"
            "- Use the exact numbers (HP, K/D, scoreboard rank) to make it sting harder\n"
            "- The joke lands because it's also 100% true — go for the kill shot, then twist it\n"
            "- React like you're in a roast battle and the game state is the material\n"
            "- When teammate or scoreboard data is available, drag them by name — being outperformed by teammates is prime material\n"
            "- Every so often, pivot from the game to their life choices in general — brutally, like a roast comedian closing a set\n"
            "- Never repeat the same joke structure twice in a row — keep the angles varied: skill, decision-making, teammates, potential, effort, excuses"
        ),
        "examples": (
            "- \"12 HP and still pushing forward, built different apparently.\"\n"
            "- \"Full armor and zero brain cells to go with it, tragic combination.\"\n"
            "- \"Same death for the third time, real consistency right there, almost impressive.\"\n"
            "- \"Bomb's down and he's still running the wrong direction, incredible sense of direction.\"\n"
            "- \"Your teammate has triple your kills and half your ego, math isn't mathing.\"\n"
            "- \"Dead last on the scoreboard, in your own lobby, that takes commitment.\"\n"
            "- \"Zero kills this round, maybe try a career that doesn't require reflexes.\"\n"
            "- \"You had the angle, you had the gun, you still lost, that's on you.\"\n"
            "- \"Bought armor and still folded like lawn furniture, what was that for.\"\n"
            "- \"Watching you peek that angle again knowing how it ends, painful loyalty to a bad idea.\"\n"
            "- \"Three teammates outscoring you and you're still calling the shots, wild confidence.\"\n"
            "- \"Maybe the problem was never the ping, maybe it's just you.\""
        ),
    },
}

def _fallback_line(tone: str) -> str:
    """Canned in-character line pulled from the tone's own examples — used when
    the Groq API is unavailable (quota/rate-limit) so the coach never goes silent."""
    examples = TONES.get(tone, TONES["toxic-friend"])["examples"]
    lines = [l.strip().lstrip("- ").strip('"') for l in examples.strip().split("\n") if l.strip()]
    return random.choice(lines)

def build_prompt(game: str, tone: str = "toxic-friend") -> str:
    t = TONES.get(tone, TONES["toxic-friend"])
    return f"""\
The player is live streaming {game}.

{t['identity']}

Personality:
{t['personality']}

Rules:
- English only — casual, sharp, zero filter
- ONE sentence only. Never more.
- SHORT — 12 to 18 words max. Not a word more.
- React specifically to the game state and events — use the actual numbers (HP, kills, etc.)
- If a Scoreboard is included below, you may compare the player to their teammates or enemies by name and K/D — call out exactly where they rank
- Once in a while (not every time), instead of a pure game roast, drop one line of brutally unsolicited "life advice" tied to how they're playing
- NEVER say "Called it" — vary your reactions every time
- No emojis.
- If the player addressed you directly, respond to THEM first in your tone, then factor in the game.

Examples:
{t['examples']}
"""

# ─── Microphone Recording (voice talk-back) ───────────────────────────────────

_mic_buf: deque = deque(maxlen=MIC_SAMPLE_RATE * 60)
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

# ─── Game State ───────────────────────────────────────────────────────────────

class GameState:
    def __init__(self):
        self.health       = 100
        self.armor        = 0
        self.kills        = 0
        self.deaths       = 0
        self.assists      = 0
        self.round_kills  = 0
        self.round_killhs   = 0
        self.round_totaldmg = 0
        self.round_phase  = ""
        self.map_phase    = ""
        self.ct_score     = 0
        self.t_score      = 0
        self.bomb         = ""
        self.weapon       = ""
        self.ammo_clip    = 0
        self.ammo_reserve = 0
        self.alive        = True
        self.events       = deque(maxlen=8)
        self.last_comment = 0
        self.last_periodic = 0
        self.my_steamid   = ""
        self.my_team      = ""
        self.allplayers   = {}

        # Status effects + economy
        self.money        = 0
        self.equip_value  = 0
        self.flashed      = 0
        self.smoked       = 0
        self.burning      = 0
        self.helmet       = False

        # Round/map meta
        self.round_num    = 0
        self.ct_losses    = 0
        self.t_losses     = 0
        self.ct_timeouts  = 0
        self.t_timeouts   = 0
        self.phase_ends_in = 0.0

        # Clutch tracking
        self.in_clutch         = False
        self.clutch_enemies    = 0
        self.clutch_announced  = False
        self.bomb_urgent_announced = False

        self._lock        = threading.Lock()

state = GameState()
prompt = ""

# ─── GSI Processing ───────────────────────────────────────────────────────────

def _to_int(value, fallback: int) -> int:
    """GSI is inconsistent about sending numbers as JSON numbers vs strings — coerce safely."""
    if value is None:
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback

def _count_alive(team: str, exclude_self: bool = False) -> int:
    count = 0
    for steamid, p in state.allplayers.items():
        if exclude_self and steamid == state.my_steamid:
            continue
        if p.get("team") == team and p.get("state", {}).get("health", 0) > 0:
            count += 1
    return count

def process_gsi(data: dict):
    with state._lock:
        player      = data.get("player", {})
        ps          = player.get("state", {})
        stats       = player.get("match_stats", {})
        round_data  = data.get("round", {})
        map_data    = data.get("map", {})
        ct_team     = map_data.get("team_ct", {})
        t_team      = map_data.get("team_t", {})
        pc          = data.get("phase_countdowns", {})

        new_health      = ps.get("health", state.health)
        new_armor       = ps.get("armor", state.armor)
        new_money       = _to_int(ps.get("money"), state.money)
        new_equip_value = _to_int(ps.get("equip_value"), state.equip_value)
        new_flashed     = _to_int(ps.get("flashed"), state.flashed)
        new_smoked      = _to_int(ps.get("smoked"), state.smoked)
        new_burning     = _to_int(ps.get("burning"), state.burning)
        new_helmet      = ps.get("helmet", state.helmet)
        new_round_killhs   = _to_int(ps.get("round_killhs"), state.round_killhs)
        new_round_totaldmg = _to_int(ps.get("round_totaldmg"), state.round_totaldmg)
        new_kills       = stats.get("kills", state.kills)
        new_deaths      = stats.get("deaths", state.deaths)
        new_assists     = stats.get("assists", state.assists)
        new_round_phase = round_data.get("phase", state.round_phase)
        new_map_phase   = map_data.get("phase", state.map_phase)
        new_bomb        = round_data.get("bomb", state.bomb)
        new_ct          = ct_team.get("score", state.ct_score)
        new_t           = t_team.get("score", state.t_score)
        new_round_num   = _to_int(map_data.get("round"), state.round_num)
        new_ct_losses   = _to_int(ct_team.get("consecutive_round_losses"), state.ct_losses)
        new_t_losses    = _to_int(t_team.get("consecutive_round_losses"), state.t_losses)
        new_ct_timeouts = _to_int(ct_team.get("timeouts_remaining"), state.ct_timeouts)
        new_t_timeouts  = _to_int(t_team.get("timeouts_remaining"), state.t_timeouts)
        try:
            new_phase_ends_in = float(pc.get("phase_ends_in", state.phase_ends_in))
        except (TypeError, ValueError):
            new_phase_ends_in = state.phase_ends_in

        # Identity + scoreboard (for teammate comparisons)
        state.my_steamid = player.get("steamid", state.my_steamid)
        state.my_team    = player.get("team", state.my_team)
        allplayers       = data.get("allplayers")
        if allplayers:
            state.allplayers = allplayers

        # Active weapon
        weapons = player.get("weapons", {})
        for _, wdata in weapons.items():
            if wdata.get("state") == "active":
                state.weapon       = wdata.get("name", "").replace("weapon_", "")
                state.ammo_clip    = wdata.get("ammo_clip", 0)
                state.ammo_reserve = wdata.get("ammo_reserve", 0)
                break

        triggered_events = []
        urgent = False

        # New round starting — round-scoped counters must not bleed into the next round
        if new_round_phase == "freezetime" and state.round_phase != "freezetime":
            state.round_kills = 0
            state.clutch_announced = False
            state.bomb_urgent_announced = False

        # Death — health hitting 0 is the primary signal, but GSI's 0.5s throttle can
        # occasionally skip that exact frame, so a match_stats.deaths increment is an
        # authoritative fallback that catches deaths health-polling alone would miss.
        died = (
            (new_health == 0 and state.health > 0 and state.alive)
            or (new_deaths > state.deaths and state.alive)
        )
        if died:
            triggered_events.append(f"DIED (had {state.health} HP, {state.armor} armor, {state.round_kills} kills this round)")
            state.alive = False
            state.round_kills = 0

        # Kill (handles multi-kills landing in a single GSI tick)
        if new_kills > state.kills:
            delta = new_kills - state.kills
            state.round_kills += delta
            if delta >= 2:
                triggered_events.append(f"MULTI-KILL (+{delta}) — now {new_kills} kills / {state.deaths} deaths this match")
            else:
                triggered_events.append(f"KILL — now {new_kills} kills / {state.deaths} deaths this match")

        # Respawn
        if new_health > 0 and not state.alive:
            state.alive = True

        # Bomb events
        if new_bomb != state.bomb:
            if new_bomb == "planted":
                triggered_events.append("BOMB PLANTED")
            elif new_bomb == "defused":
                triggered_events.append("BOMB DEFUSED")
            elif new_bomb == "exploded":
                triggered_events.append("BOMB EXPLODED — round lost")

        # Bomb urgency — the exact "it's about to go off" moment, announced once per plant
        if new_bomb == "planted" and new_phase_ends_in and new_phase_ends_in <= BOMB_URGENT_SEC and not state.bomb_urgent_announced:
            triggered_events.append(f"BOMB URGENT — exploding in {new_phase_ends_in:.0f}s")
            state.bomb_urgent_announced = True
            urgent = True

        # Low health
        if 0 < new_health < 30 and state.health >= 30:
            triggered_events.append(f"LOW HEALTH — only {new_health} HP left")

        # Clutch detection — the player is the last one alive on their team, facing 1+ enemies
        enemy_team = "T" if state.my_team == "CT" else "CT"
        teammates_alive = _count_alive(state.my_team, exclude_self=True) if state.allplayers else 0
        enemies_alive   = _count_alive(enemy_team) if state.allplayers else 0
        new_in_clutch = bool(new_health > 0 and state.allplayers and teammates_alive == 0 and enemies_alive >= 1)
        if new_in_clutch:
            state.clutch_enemies = enemies_alive
            if not state.clutch_announced:
                triggered_events.append(f"CLUTCH — 1v{enemies_alive}, last one alive on {state.my_team}")
                state.clutch_announced = True
                urgent = True
        state.in_clutch = new_in_clutch

        # Round ended
        if new_round_phase == "over" and state.round_phase != "over":
            win_team = round_data.get("win_team", "")
            win_note = f" — {win_team} won the round" if win_team else ""
            if state.clutch_announced and win_team and win_team == state.my_team:
                triggered_events.append(f"CLUTCH WON! 1v{state.clutch_enemies} secured{win_note}")
                urgent = True
            else:
                triggered_events.append(f"ROUND OVER — CT:{new_ct} T:{new_t}{win_note}")

        # Update state
        state.health      = new_health
        state.armor       = new_armor
        state.money       = new_money
        state.equip_value = new_equip_value
        state.flashed     = new_flashed
        state.smoked      = new_smoked
        state.burning     = new_burning
        state.helmet      = new_helmet
        state.round_killhs   = new_round_killhs
        state.round_totaldmg = new_round_totaldmg
        state.kills       = new_kills
        state.deaths      = new_deaths
        state.assists     = new_assists
        state.round_phase = new_round_phase
        state.map_phase   = new_map_phase
        state.bomb        = new_bomb
        state.ct_score    = new_ct
        state.t_score     = new_t
        state.round_num   = new_round_num
        state.ct_losses   = new_ct_losses
        state.t_losses    = new_t_losses
        state.ct_timeouts = new_ct_timeouts
        state.t_timeouts  = new_t_timeouts
        state.phase_ends_in = new_phase_ends_in

        # Only keep events from live match time — a warmup kill queued here would
        # otherwise leak into the first real comment once the round goes live.
        if state.map_phase == "live":
            for e in triggered_events:
                state.events.append(e)

        now = time.time()
        # Gate everything on the map actually being live — warmup/intermission kills
        # and deaths aren't real match events and shouldn't trigger commentary.
        # Urgent events (bomb about to blow, clutch starting/won) bypass the cooldown
        # entirely since they're one-shot, flag-gated moments that matter in the instant.
        should_comment = state.map_phase == "live" and (
            urgent
            or (triggered_events and (now - state.last_comment) > MIN_COMMENT_GAP)
            or ((now - state.last_periodic) > PERIODIC_COMMENT and state.alive)
        )

        if should_comment:
            state.last_comment  = now
            state.last_periodic = now
            summary = build_summary()
            state.events.clear()

    if should_comment:
        chat = get_recent_chat(10)
        threading.Thread(target=trigger_comment, args=(summary, chat), kwargs={"urgent": urgent}, daemon=True).start()

def build_scoreboard() -> str:
    """Ranked K/D scoreboard with teammates vs enemies, self marked out."""
    if not state.allplayers or not state.my_steamid:
        return ""

    rows = []
    for steamid, p in state.allplayers.items():
        stats = p.get("match_stats", {})
        rows.append({
            "name":    p.get("name", "player"),
            "team":    p.get("team", "?"),
            "kills":   stats.get("kills", 0),
            "deaths":  stats.get("deaths", 0),
            "assists": stats.get("assists", 0),
            "is_me":   steamid == state.my_steamid,
        })
    if len(rows) < 2:
        return ""

    rows.sort(key=lambda r: r["kills"], reverse=True)
    lines = []
    for i, r in enumerate(rows, 1):
        tag = " (YOU)" if r["is_me"] else (" (teammate)" if r["team"] == state.my_team else "")
        lines.append(f"  {i}. {r['name']}{tag} — {r['kills']}K/{r['deaths']}D/{r['assists']}A")
    return "Scoreboard:\n" + "\n".join(lines)

def build_summary() -> str:
    parts = []

    if COACH_CONTEXT.get("round_meta", True):
        parts.append(
            f"Round {state.round_num} | Score: CT {state.ct_score} - T {state.t_score} | Round phase: {state.round_phase}"
            + (f" | CT loss streak: {state.ct_losses}, timeouts: {state.ct_timeouts}" if state.ct_losses else "")
            + (f" | T loss streak: {state.t_losses}, timeouts: {state.t_timeouts}" if state.t_losses else "")
        )

    if COACH_CONTEXT.get("health", True):
        parts.append(f"Status: {'ALIVE' if state.alive else 'DEAD'} | HP: {state.health} | Armor: {state.armor}{' + helmet' if state.helmet else ''}")

    if COACH_CONTEXT.get("weapon", True):
        parts.append(f"Weapon: {state.weapon} | Ammo: {state.ammo_clip}/{state.ammo_reserve}")

    if COACH_CONTEXT.get("kda", True):
        parts.append(f"Match: {state.kills}K / {state.deaths}D / {state.assists}A")

    if COACH_CONTEXT.get("money", True):
        parts.append(f"Money: ${state.money} | Equip value: ${state.equip_value}")

    if COACH_CONTEXT.get("status_effects", True):
        effects = []
        if state.flashed:
            effects.append(f"flashed ({state.flashed})")
        if state.smoked:
            effects.append("in smoke")
        if state.burning:
            effects.append("burning")
        if effects:
            parts.append("Status effects: " + ", ".join(effects))

    if COACH_CONTEXT.get("timer", True) and state.phase_ends_in:
        label = "Bomb timer" if state.bomb == "planted" else "Round timer"
        parts.append(f"{label}: {state.phase_ends_in:.0f}s left")

    if COACH_CONTEXT.get("clutch", True) and state.in_clutch:
        parts.append(f"CLUTCH IN PROGRESS: 1v{state.clutch_enemies} — last one alive on {state.my_team}")

    recent = "\n".join(f"  - {e}" for e in list(state.events)[-5:]) or "  - No recent events"
    parts.append(f"Recent events:\n{recent}")

    if COACH_CONTEXT.get("scoreboard", True) and random.random() < 0.4:
        scoreboard = build_scoreboard()
        if scoreboard:
            parts.append(scoreboard)

    return "\n".join(parts)

# ─── AI + TTS ─────────────────────────────────────────────────────────────────

def _groq_commentary(user_content: str) -> str:
    client = Groq(api_key=GROQ_KEY)
    response = client.chat.completions.create(
        model=COACH_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user",   "content": user_content},
        ],
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content.strip()

def _gemini_commentary(user_content: str) -> str:
    client = genai.Client(api_key=GEMINI_KEY)
    response = client.models.generate_content(
        model=COACH_MODEL,
        contents=[genai_types.Part.from_text(text=user_content)],
        config=genai_types.GenerateContentConfig(
            system_instruction=prompt,
            max_output_tokens=MAX_TOKENS,
        ),
    )
    return response.text.strip()

COMMENTARY_PROVIDERS = {"groq": _groq_commentary, "gemini": _gemini_commentary}

def trigger_comment(summary: str, chat: list[str] = None, voice: str = "", urgent: bool = False):
    if voice:
        user_content = (
            f"Game state:\n{summary}\n\n"
            f"PRIORITY: The player just said to you: \"{voice}\"\n"
            "Respond to what they said first (clap back), then tie it to the game state. One sentence."
        )
    elif urgent:
        user_content = (
            f"URGENT — this is happening RIGHT NOW, react with real intensity and urgency:\n"
            f"Game state:\n{summary}\n\nGive your coaching comment."
        )
    else:
        user_content = f"Game state:\n{summary}\n\nGive your coaching comment."
    if chat:
        user_content += (
            f"\n\nLive chat ({len(chat)} messages):\n"
            + "\n".join(f"  {m}" for m in chat)
            + "\nGame first. If chat is saying something relevant or hilarious, mention it."
        )

    try:
        commentary_fn = COMMENTARY_PROVIDERS.get(COACH_PROVIDER, _groq_commentary)
        commentary = commentary_fn(user_content)
    except Exception as e:
        print(f"  [{COACH_PROVIDER} commentary error, using canned fallback line: {e}]", flush=True)
        commentary = _fallback_line(COACH_TONE)

    ts = datetime.now().strftime("%H:%M:%S")
    tag = " (voice)" if voice else ""
    print(f"\n[{ts}] COACH{tag} >> {commentary}\n", flush=True)
    speak(commentary)

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

def handle_voice_trigger(voice: str):
    with state._lock:
        summary = build_summary()
    chat = get_recent_chat(10)
    trigger_comment(summary, chat, voice=voice)

def voice_listener_loop():
    while True:
        time.sleep(WAKE_POLL_SEC)
        voice = poll_for_wake_word()
        if voice:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [Mic] Player said: {voice}", flush=True)
            threading.Thread(target=handle_voice_trigger, args=(voice,), daemon=True).start()

# ─── GSI HTTP Server ──────────────────────────────────────────────────────────

class GSIHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            data   = json.loads(body.decode("utf-8"))
            process_gsi(data)
        except Exception:
            pass
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass  # suppress request logs

# ─── Main ─────────────────────────────────────────────────────────────────────

def run():
    global prompt

    print("=" * 55)
    print("  GAMEPLAY COACH — CS2 GSI Edition")
    print("=" * 55)

    if not GROQ_KEY:
        print("\n[ERROR] GROQ_API_KEY not set in .env")
        sys.exit(1)

    game = os.getenv("COACH_GAME") or input("\n  Stream topic [CS2 Competitive]?\n  > ").strip() or "CS2 Competitive"
    prompt = build_prompt(game, COACH_TONE)

    print(f"\n  Game   : {game}")
    print(f"  Tone   : {COACH_TONE}")
    print(f"  Port   : localhost:{GSI_PORT}")
    print(f"  Model  : {COACH_PROVIDER} — {COACH_MODEL} (text-only, no image tokens)")
    print(f"  Voice  : ElevenLabs ({ELEVENLABS_VOICE_ID})")
    active_context = [k for k, v in COACH_CONTEXT.items() if v]
    print(f"  Context: {', '.join(active_context) if active_context else 'none'}")
    print("  Mic    : ON — say 'Coach' to talk back")
    print(f"\n  Waiting for CS2 to connect...\n")

    start_chat_readers()
    mic_stream = start_mic()
    threading.Thread(target=voice_listener_loop, daemon=True).start()

    server = HTTPServer(("localhost", GSI_PORT), GSIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")
        server.shutdown()
    finally:
        mic_stream.stop()
        mic_stream.close()


if __name__ == "__main__":
    run()

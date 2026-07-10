import json
import os
import sys
import tempfile
import threading
import time
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.stdout.reconfigure(encoding="utf-8")

from groq import Groq
from elevenlabs.client import ElevenLabs
from playsound import playsound
from dotenv import load_dotenv

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────

GSI_PORT         = 3000
MIN_COMMENT_GAP  = 20    # minimum seconds between comments
PERIODIC_COMMENT = 60    # also comment every N seconds even with no events
MAX_TOKENS       = 60
GROQ_KEY         = os.getenv("GROQ_API_KEY", "")
ELEVENLABS_KEY        = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID   = os.getenv("ELEVENLABS_VOICE_ID", "")

# ─── Personality ──────────────────────────────────────────────────────────────

def build_prompt(game: str) -> str:
    return f"""\
The player is live streaming {game}.

You are the player's gaming companion — someone who knows them too well and has completely given up being polite about it. You're never satisfied. You blame them for everything. You're personally offended by bad plays.

Personality:
- Never satisfied: even good plays get "okay but watch, you'll mess up the next one"
- Always blaming: nothing is ever the game's fault — it's always them
- Prank energy: predict the next mistake, then be smug about it
- Personally offended by bad decisions

Rules:
- English only — casual, sharp, zero filter
- ONE sentence only. Never more.
- React specifically to the game state and events — use the actual numbers (HP, kills, etc.)
- NEVER say "Called it" — vary your reactions every time
- No emojis.

Examples:
- "34 HP and you're still pushing, this is genuinely impressive levels of confidence."
- "Two kills and you're already playing sloppy, don't let it go to your head."
- "You died with full armor, I don't even want to know what happened."
- "Bomb's planted and you're rotating now, of course you are."
- "3 deaths in and nothing's changed about how you're playing, respect the consistency I guess."
"""

# ─── Game State ───────────────────────────────────────────────────────────────

class GameState:
    def __init__(self):
        self.health       = 100
        self.armor        = 0
        self.kills        = 0
        self.deaths       = 0
        self.assists      = 0
        self.round_kills  = 0
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
        self._lock        = threading.Lock()

state = GameState()
prompt = ""

# ─── GSI Processing ───────────────────────────────────────────────────────────

def process_gsi(data: dict):
    with state._lock:
        player      = data.get("player", {})
        ps          = player.get("state", {})
        stats       = player.get("match_stats", {})
        round_data  = data.get("round", {})
        map_data    = data.get("map", {})

        new_health      = ps.get("health", state.health)
        new_armor       = ps.get("armor", state.armor)
        new_kills       = stats.get("kills", state.kills)
        new_deaths      = stats.get("deaths", state.deaths)
        new_assists     = stats.get("assists", state.assists)
        new_round_phase = round_data.get("phase", state.round_phase)
        new_map_phase   = map_data.get("phase", state.map_phase)
        new_bomb        = round_data.get("bomb", state.bomb)
        new_ct          = map_data.get("team_ct", {}).get("score", state.ct_score)
        new_t           = map_data.get("team_t", {}).get("score", state.t_score)

        # Active weapon
        weapons = player.get("weapons", {})
        for _, wdata in weapons.items():
            if wdata.get("state") == "active":
                state.weapon       = wdata.get("name", "").replace("weapon_", "")
                state.ammo_clip    = wdata.get("ammo_clip", 0)
                state.ammo_reserve = wdata.get("ammo_reserve", 0)
                break

        triggered_events = []

        # Death
        if new_health == 0 and state.health > 0 and state.alive:
            triggered_events.append(f"DIED (had {state.health} HP, {state.armor} armor, {state.round_kills} kills this round)")
            state.alive = False
            state.round_kills = 0

        # Kill
        if new_kills > state.kills:
            state.round_kills += (new_kills - state.kills)
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

        # Low health
        if 0 < new_health < 30 and state.health >= 30:
            triggered_events.append(f"LOW HEALTH — only {new_health} HP left")

        # Round ended
        if new_round_phase == "over" and state.round_phase != "over":
            triggered_events.append(f"ROUND OVER — CT:{new_ct} T:{new_t}")

        # Update state
        state.health      = new_health
        state.armor       = new_armor
        state.kills       = new_kills
        state.deaths      = new_deaths
        state.assists     = new_assists
        state.round_phase = new_round_phase
        state.map_phase   = new_map_phase
        state.bomb        = new_bomb
        state.ct_score    = new_ct
        state.t_score     = new_t

        for e in triggered_events:
            state.events.append(e)

        now = time.time()
        should_comment = (
            triggered_events and (now - state.last_comment) > MIN_COMMENT_GAP
        ) or (
            (now - state.last_periodic) > PERIODIC_COMMENT and state.alive and state.map_phase == "live"
        )

        if should_comment:
            state.last_comment  = now
            state.last_periodic = now
            summary = build_summary()

    if should_comment:
        threading.Thread(target=trigger_comment, args=(summary,), daemon=True).start()

def build_summary() -> str:
    recent = "\n".join(f"  - {e}" for e in list(state.events)[-5:]) or "  - No recent events"
    return (
        f"Score: CT {state.ct_score} - T {state.t_score} | Round: {state.round_phase}\n"
        f"Status: {'ALIVE' if state.alive else 'DEAD'} | HP: {state.health} | Armor: {state.armor}\n"
        f"Weapon: {state.weapon} | Ammo: {state.ammo_clip}/{state.ammo_reserve}\n"
        f"Match: {state.kills}K / {state.deaths}D / {state.assists}A\n"
        f"Recent events:\n{recent}"
    )

# ─── AI + TTS ─────────────────────────────────────────────────────────────────

def trigger_comment(summary: str):
    try:
        client = Groq(api_key=GROQ_KEY)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",   # text-only — fast, cheap, sharp
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user",   "content": f"Game state:\n{summary}\n\nGive your coaching comment."},
            ],
            max_tokens=MAX_TOKENS,
        )
        commentary = response.choices[0].message.content.strip()
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{ts}] COACH >> {commentary}\n", flush=True)
        speak(commentary)
    except Exception as e:
        print(f"  [Error: {e}]", flush=True)

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
    prompt = build_prompt(game)

    print(f"\n  Game   : {game}")
    print(f"  Port   : localhost:{GSI_PORT}")
    print(f"  Model  : llama-3.3-70b (text-only, no image tokens)")
    print(f"  Voice  : ElevenLabs ({ELEVENLABS_VOICE_ID})")
    print(f"\n  Waiting for CS2 to connect...\n")

    server = HTTPServer(("localhost", GSI_PORT), GSIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")
        server.shutdown()


if __name__ == "__main__":
    run()

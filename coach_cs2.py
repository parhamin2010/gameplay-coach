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
from chat_reader import start_chat_readers, get_recent_chat

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
            "You are a pure, unfiltered trash talker. Zero chill. "
            "You roast everything and always back it up with the actual game state numbers."
        ),
        "personality": (
            "- Zero filter, full disrespect — grounded in what just happened\n"
            "- Use the exact numbers to make it sting\n"
            "- The joke lands because it's also 100% true\n"
            "- React like you're in a roast battle and the game state is the material"
        ),
        "examples": (
            "- \"34 HP and still pushing, bro is built different from the neck down.\"\n"
            "- \"Died with full armor, I genuinely want to understand the thought process.\"\n"
            "- \"3 deaths in and the strategy is just: same thing again, got it.\"\n"
            "- \"Bomb's planted and he's rotating, this guy is incredible.\""
        ),
    },
}

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
- React specifically to the game state and events — use the actual numbers (HP, kills, etc.)
- NEVER say "Called it" — vary your reactions every time
- No emojis.

Examples:
{t['examples']}
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
        chat = get_recent_chat(10)
        threading.Thread(target=trigger_comment, args=(summary, chat), daemon=True).start()

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

def trigger_comment(summary: str, chat: list[str] = None):
    try:
        user_content = f"Game state:\n{summary}\n\nGive your coaching comment."
        if chat:
            user_content += (
                f"\n\nLive chat ({len(chat)} messages):\n"
                + "\n".join(f"  {m}" for m in chat)
                + "\nGame first. If chat is saying something relevant or hilarious, mention it."
            )
        client = Groq(api_key=GROQ_KEY)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user",   "content": user_content},
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
    tone = os.getenv("COACH_TONE", "toxic-friend")
    prompt = build_prompt(game, tone)

    print(f"\n  Game   : {game}")
    print(f"  Tone   : {tone}")
    print(f"  Port   : localhost:{GSI_PORT}")
    print(f"  Model  : llama-3.3-70b (text-only, no image tokens)")
    print(f"  Voice  : ElevenLabs ({ELEVENLABS_VOICE_ID})")
    print(f"\n  Waiting for CS2 to connect...\n")

    start_chat_readers()

    server = HTTPServer(("localhost", GSI_PORT), GSIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")
        server.shutdown()


if __name__ == "__main__":
    run()

import json
import os
import queue
import subprocess
import sys
import threading
import webbrowser

sys.stdout.reconfigure(encoding="utf-8")
from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

PYTHON      = sys.executable
COACH_DIR   = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(COACH_DIR, "last_session.json")

_log_queue      = queue.Queue()
_coach_process  = None
_process_lock   = threading.Lock()

VOICES = [
    {"id": "onwK4e9ZLuTAKqWW03F9", "name": "Daniel — Steady Broadcaster"},
    {"id": "SOYHLrjzK2X1ezoPC6cr", "name": "Harry — Fierce Warrior"},
    {"id": "N2lVS1w4EtoT3dr4eOWO", "name": "Callum — Husky Trickster"},
    {"id": "IKne3meq5aSn9XLyUdCD", "name": "Charlie — Deep & Energetic"},
    {"id": "TX3LPaxmHKxFdv7VOQHJ", "name": "Liam — Social Media Creator"},
    {"id": "pNInz6obpgDQGcFmaJgB", "name": "Adam — Dominant & Firm"},
]

# ─── Config persistence ───────────────────────────────────────────────────────

def _save_config(data: dict):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

def _load_config() -> dict:
    try:
        with open(CONFIG_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

# ─── Core launch logic ────────────────────────────────────────────────────────

def _launch_coach(data: dict):
    global _coach_process
    mode     = data.get("mode", "screenshot")
    game     = data.get("game", "CS2 Competitive")
    interval = str(data.get("interval", 60))
    voice_id = data.get("voice_id", os.getenv("ELEVENLABS_VOICE_ID", ""))
    tone     = data.get("tone", "toxic-friend")
    chat     = data.get("chat", {})

    script = "coach_cs2.py" if mode == "cs2" else "coach.py"

    env = {
        **os.environ,
        "COACH_GAME":           game,
        "COACH_INTERVAL":       interval,
        "COACH_TONE":           tone,
        "ELEVENLABS_VOICE_ID":  voice_id,
        "TWITCH_TOKEN":         chat.get("twitch_token", ""),
        "TWITCH_CHANNEL":       chat.get("twitch_channel", ""),
        "YOUTUBE_API_KEY":      chat.get("youtube_api_key", ""),
        "YOUTUBE_VIDEO_ID":     chat.get("youtube_video_id", ""),
        "KICK_CHANNEL":         chat.get("kick_channel", ""),
        "PYTHONIOENCODING":     "utf-8",
    }

    _coach_process = subprocess.Popen(
        [PYTHON, "-u", script],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=COACH_DIR,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    def _read():
        for line in _coach_process.stdout:
            _log_queue.put(line.rstrip())
        _log_queue.put("__STOPPED__")

    threading.Thread(target=_read, daemon=True).start()
    _log_queue.put(f"Started in {mode.upper()} mode — {game} — every {interval}s [{tone}]")

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    current_voice = os.getenv("ELEVENLABS_VOICE_ID", "onwK4e9ZLuTAKqWW03F9")
    return render_template(
        "index.html",
        voices=VOICES,
        current_voice=current_voice,
        twitch_token=os.getenv("TWITCH_TOKEN", ""),
        twitch_channel=os.getenv("TWITCH_CHANNEL", ""),
        youtube_api_key=os.getenv("YOUTUBE_API_KEY", ""),
        youtube_video_id=os.getenv("YOUTUBE_VIDEO_ID", ""),
        kick_channel=os.getenv("KICK_CHANNEL", ""),
    )

@app.route("/api/start", methods=["POST"])
def start():
    with _process_lock:
        if _coach_process and _coach_process.poll() is None:
            return jsonify({"error": "Already running"}), 400
        data = request.json or {}
        _save_config(data)
        _launch_coach(data)
    return jsonify({"status": "started"})

@app.route("/api/stop", methods=["POST"])
def stop():
    global _coach_process
    with _process_lock:
        if _coach_process:
            _coach_process.terminate()
            _coach_process = None
            _log_queue.put("__STOPPED__")
    return jsonify({"status": "stopped"})

@app.route("/api/status")
def status():
    running = _coach_process is not None and _coach_process.poll() is None
    return jsonify({"running": running})

@app.route("/api/config")
def config():
    return jsonify(_load_config())

@app.route("/api/logs")
def logs():
    def generate():
        yield "data: __CONNECTED__\n\n"
        while True:
            try:
                line = _log_queue.get(timeout=15)
                yield f"data: {json.dumps(line)}\n\n"
            except queue.Empty:
                yield "data: __PING__\n\n"
    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# ─── Launch ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    autostart = "--autostart" in sys.argv
    url = "http://localhost:5000"
    print(f"\n  Gameplay Coach UI → {url}")

    if autostart:
        cfg = _load_config()
        if cfg:
            print(f"  Auto-starting: {cfg.get('mode','screenshot').upper()} | {cfg.get('game','')} | {cfg.get('tone','')}\n")
            with _process_lock:
                _launch_coach(cfg)
        else:
            print("  No saved session — open the dashboard to configure.\n")
    else:
        print()

    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host="localhost", port=5000, debug=False, threaded=True)

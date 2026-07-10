# Gameplay Coach

An AI voiceover coach that watches your gameplay in real time and trash-talks you like that one friend who can't shut up. Pick your poison — toxic friend, disappointed dad, hype man, cold analyst, or pure trash talker. Talk back and it claps back.

![Dashboard](https://img.shields.io/badge/UI-Flask%20Dashboard-00e5ff?style=flat-square) ![AI](https://img.shields.io/badge/AI-Groq%20LLaMA-orange?style=flat-square) ![TTS](https://img.shields.io/badge/TTS-ElevenLabs-purple?style=flat-square)

---

## Modes

### Screenshot Mode
Takes 3 screenshots every N seconds and sends them to a Groq vision model. Works with any game.

### CS2 GSI Mode
Uses Counter-Strike 2's built-in Game State Integration to receive live game data (health, kills, bomb state, scores) over HTTP. Event-driven, no screenshots needed — reacts instantly to deaths, kills, bomb plants, low HP, and round ends.

---

## Features

### Voice Response (Talk Back)
Say **"Coach"** into your mic at any point and the coach will prioritize your response in the next comment — clapping back at what you said before reacting to the game. The mic buffer clears after each use so the same line never repeats.

> You: "Coach, if you could do it better come and do it."
> Coach: "Big talk from the guy who just handed that kill over for free."

### Coach Tones
Pick the personality that fits your stream from the dashboard:

| Tone | Vibe |
|------|------|
| Toxic Friend | Never satisfied, always blaming, prank energy |
| Hype Man | Unhinged enthusiast — spins every disaster into potential |
| Disappointed Dad | Not angry, just tired. Passive-aggressive warmth. |
| The Analyst | Cold, clinical, delivers roasts like a stats report |
| Trash Talker | Zero filter, full disrespect, always backed by what just happened |

Each tone also has its own clap-back style when you address the coach directly.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install flask elevenlabs
```

### 2. Add API keys

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

- **Groq** — free at [console.groq.com](https://console.groq.com)
- **ElevenLabs** — free at [elevenlabs.io](https://elevenlabs.io)

### 3. (CS2 GSI only) Install the GSI config

Copy `gamestate_integration_coach.cfg` to your CS2 cfg folder:

```
Steam/steamapps/common/Counter-Strike Global Offensive/game/csgo/cfg/
```

---

## Run

### First time
Double-click `start.bat` — configure your settings in the dashboard, hit Start. Everything is saved automatically.

### Every stream after that
Double-click `quickstart.bat` — coach launches immediately with your last session settings. Browser opens showing the live feed. Zero clicks needed.

### Manual
```bash
python app.py           # normal start
python app.py --autostart  # auto-launch with last saved settings
```

All settings (mode, game, interval, tone, voice, chat credentials) are remembered between sessions — you never have to configure twice.

---

## Dashboard

- **Mode** — switch between Screenshot and CS2 GSI
- **Game / Topic** — tells the coach what you're playing
- **Interval** — how often it comments (15–120 seconds)
- **Tone** — pick the coach personality
- **Voice** — pick from 6 ElevenLabs voices
- **Live Feed** — real-time log of every coach comment and game event

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Vision AI | Groq `meta-llama/llama-4-scout-17b-16e-instruct` |
| Text AI (CS2) | Groq `llama-3.3-70b-versatile` |
| Speech-to-Text | Groq Whisper `whisper-large-v3-turbo` |
| Text-to-Speech | ElevenLabs `eleven_multilingual_v2` |
| Screen Capture | `mss` |
| UI | Flask + SSE |
| Audio Playback | `playsound` |

---

## Chat Integration

Connect your live chat so the coach can react to your viewers too.

| Platform | What you need |
|----------|--------------|
| Twitch | OAuth token from [twitchapps.com/tmi](https://twitchapps.com/tmi) + channel name |
| YouTube | YouTube Data API v3 key (Google Cloud Console) + live stream Video ID |
| Kick | Channel slug only (e.g. `shroud`) — no API key needed |

Toggle each platform ON in the dashboard and fill in the fields. The coach pulls recent messages each cycle and weaves them into commentary — reacting to what chat is saying, calling out when they're right or wrong, roasting viewers and player at the same time.

Install `websockets` for Kick support:

```bash
pip install websockets
```

---

## Notes

- Screenshot mode uses ~7,500 tokens per cycle — stay mindful of Groq's free tier limits
- CS2 GSI mode uses ~200 tokens per comment — much cheaper
- Voice response only activates when you say "Coach" — ambient audio is ignored

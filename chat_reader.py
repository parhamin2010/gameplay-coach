import json
import os
import socket
import threading
import time
import urllib.request
from collections import deque

MAX_CHAT = 30

_chat_buf: deque = deque(maxlen=MAX_CHAT)


def get_recent_chat(n: int = 10) -> list[str]:
    return list(_chat_buf)[-n:]


# ─── Twitch IRC ───────────────────────────────────────────────────────────────

class TwitchChat:
    def __init__(self, token: str, channel: str):
        self.token   = token if token.startswith("oauth:") else f"oauth:{token}"
        self.channel = channel.lower().lstrip("#")
        self._sock   = None

    def start(self):
        try:
            self._sock = socket.socket()
            self._sock.connect(("irc.chat.twitch.tv", 6667))
            self._sock.send(f"PASS {self.token}\r\n".encode())
            self._sock.send(b"NICK coachbot\r\n")
            self._sock.send(f"JOIN #{self.channel}\r\n".encode())
            threading.Thread(target=self._read, daemon=True).start()
            print(f"[Chat] Twitch connected -> #{self.channel}", flush=True)
        except Exception as e:
            print(f"[Chat] Twitch error: {e}", flush=True)

    def _read(self):
        buf = ""
        while True:
            try:
                data = self._sock.recv(2048).decode("utf-8", errors="replace")
                buf += data
                while "\r\n" in buf:
                    line, buf = buf.split("\r\n", 1)
                    if line.startswith("PING"):
                        self._sock.send(b"PONG :tmi.twitch.tv\r\n")
                    elif "PRIVMSG" in line:
                        try:
                            user = line.split("!")[0][1:]
                            msg  = line.split("PRIVMSG")[1].split(":", 1)[1].strip()
                            _chat_buf.append(f"[Twitch] {user}: {msg}")
                        except Exception:
                            pass
            except Exception as e:
                print(f"[Chat] Twitch read error: {e}", flush=True)
                break

    def stop(self):
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass


# ─── YouTube Live Chat ────────────────────────────────────────────────────────

class YouTubeChat:
    _VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
    _CHAT_URL   = "https://www.googleapis.com/youtube/v3/liveChat/messages"

    def __init__(self, api_key: str, video_id: str):
        self.api_key      = api_key
        self.video_id     = video_id
        self._chat_id     = None
        self._page_token  = None
        self._seen: set   = set()

    def _fetch(self, url: str) -> dict:
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.loads(r.read())

    def _get_chat_id(self) -> str | None:
        url  = f"{self._VIDEOS_URL}?part=liveStreamingDetails&id={self.video_id}&key={self.api_key}"
        data = self._fetch(url)
        items = data.get("items", [])
        if items:
            return items[0].get("liveStreamingDetails", {}).get("activeLiveChatId")
        return None

    def start(self):
        try:
            self._chat_id = self._get_chat_id()
            if not self._chat_id:
                print("[Chat] YouTube: no active live chat found for this video ID", flush=True)
                return
            threading.Thread(target=self._poll, daemon=True).start()
            print(f"[Chat] YouTube connected -> {self.video_id}", flush=True)
        except Exception as e:
            print(f"[Chat] YouTube error: {e}", flush=True)

    def _poll(self):
        while True:
            try:
                url = (
                    f"{self._CHAT_URL}?liveChatId={self._chat_id}"
                    f"&part=snippet,authorDetails&key={self.api_key}"
                )
                if self._page_token:
                    url += f"&pageToken={self._page_token}"
                data = self._fetch(url)
                for item in data.get("items", []):
                    iid = item["id"]
                    if iid not in self._seen:
                        self._seen.add(iid)
                        author = item["authorDetails"]["displayName"]
                        msg    = item["snippet"]["displayMessage"]
                        _chat_buf.append(f"[YouTube] {author}: {msg}")
                self._page_token = data.get("nextPageToken")
                interval = data.get("pollingIntervalMillis", 5000) / 1000
                time.sleep(max(interval, 3))
            except Exception as e:
                print(f"[Chat] YouTube poll error: {e}", flush=True)
                time.sleep(15)

    def stop(self):
        pass


# ─── Kick (Pusher WebSocket) ──────────────────────────────────────────────────

class KickChat:
    _API_URL    = "https://kick.com/api/v1/channels"
    _PUSHER_URL = (
        "wss://ws-us2.pusher.com/app/eb1d5f283081a78b932c"
        "?protocol=7&client=js&version=7.6.0&flash=false"
    )

    def __init__(self, channel: str):
        self.channel      = channel.lower()
        self._chatroom_id = None

    def _get_chatroom_id(self) -> int | None:
        url = f"{self._API_URL}/{self.channel}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        return data.get("chatroom", {}).get("id")

    def start(self):
        try:
            self._chatroom_id = self._get_chatroom_id()
            if not self._chatroom_id:
                print("[Chat] Kick: could not get chatroom ID", flush=True)
                return
            threading.Thread(target=self._run, daemon=True).start()
            print(f"[Chat] Kick connected -> {self.channel}", flush=True)
        except Exception as e:
            print(f"[Chat] Kick error: {e}", flush=True)

    def _run(self):
        try:
            import asyncio
            import websockets  # noqa: F401
            asyncio.run(self._connect())
        except ImportError:
            print("[Chat] Kick requires websockets: pip install websockets", flush=True)
        except Exception as e:
            print(f"[Chat] Kick connection error: {e}", flush=True)

    async def _connect(self):
        import websockets
        async with websockets.connect(self._PUSHER_URL) as ws:
            await ws.send(json.dumps({
                "event": "pusher:subscribe",
                "data": {"auth": "", "channel": f"chatrooms.{self._chatroom_id}.v2"},
            }))
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                    if msg.get("event") == "App\\Events\\ChatMessageEvent":
                        inner = json.loads(msg["data"])
                        user  = inner["sender"]["username"]
                        text  = inner["content"]
                        _chat_buf.append(f"[Kick] {user}: {text}")
                except Exception:
                    pass

    def stop(self):
        pass


# ─── Bootstrap ────────────────────────────────────────────────────────────────

def start_chat_readers() -> list:
    readers = []

    twitch_token   = os.getenv("TWITCH_TOKEN", "")
    twitch_channel = os.getenv("TWITCH_CHANNEL", "")
    yt_key         = os.getenv("YOUTUBE_API_KEY", "")
    yt_video       = os.getenv("YOUTUBE_VIDEO_ID", "")
    kick_channel   = os.getenv("KICK_CHANNEL", "")

    if twitch_token and twitch_channel:
        r = TwitchChat(twitch_token, twitch_channel)
        r.start()
        readers.append(r)

    if yt_key and yt_video:
        r = YouTubeChat(yt_key, yt_video)
        r.start()
        readers.append(r)

    if kick_channel:
        r = KickChat(kick_channel)
        r.start()
        readers.append(r)

    return readers

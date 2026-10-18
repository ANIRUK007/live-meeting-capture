"""Live meeting capture server.

    python -m server.main          # then open http://<laptop-lan-ip>:8443/

Single laptop, single meeting at a time: this is a recording tool for one
room, not a multi-tenant service.

Flow: phones open /join?role=<role>, grant mic access, and register over a
websocket. The operator clicks Start on the display page; every connected
phone gets a start signal and begins tagging 500ms PCM16 chunks with a
sequence number. The server mixes matched sequence numbers across roles
(silence-filling a late/missing side) and appends to a growing raw PCM file.
On Stop, the file is wrapped into a WAV and handed to Gemini for transcript
+ structured markdown notes in the background.

getUserMedia requires a secure context for anything that is not literally
"localhost", and phones reach this server by LAN IP, so this always runs over
HTTPS with a self-signed cert (see run.sh). Expect one "not secure" click-
through per phone, once. ponytail: self-signed cert; swap for a CA-trusted
one (mkcert, Tailscale cert) if the warning becomes a real problem.
"""

from __future__ import annotations

import io
import json
import os
import socket
import struct
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import qrcode
from fastapi import BackgroundTasks, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from server.gemini import summarize, transcribe
from server.mixer import Mixer, resample

ROLES = [r.strip() for r in os.getenv("ROLES", "board,audience").split(",") if r.strip()]
TARGET_RATE = 16000
CHUNK_MS = 500
STATIC = Path(__file__).parent / "static"
DATA_DIR = Path(__file__).parent.parent / "data" / "meetings"

app = FastAPI()


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class Meeting:
    def __init__(self):
        self.sockets: dict[str, set[WebSocket]] = {r: set() for r in ROLES}
        self.sample_rates: dict[WebSocket, int] = {}
        self.mixer: Mixer | None = None
        self.raw_path: Path | None = None
        self.raw_file = None
        self.meeting_id: str | None = None
        self.started = False
        self.processing = False
        self.done = False

    def connected(self) -> dict[str, bool]:
        return {r: len(s) > 0 for r, s in self.sockets.items()}


meeting = Meeting()


@app.get("/", response_class=HTMLResponse)
async def display():
    return (STATIC / "display.html").read_text()


@app.get("/join", response_class=HTMLResponse)
async def join():
    return (STATIC / "join.html").read_text()


@app.get("/qr/{role}.png")
async def qr(role: str):
    url = f"https://{lan_ip()}:{os.getenv('PORT', '8443')}/join?role={role}"
    img = qrcode.make(url)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


@app.get("/status")
async def status():
    return JSONResponse(
        {
            "roles": ROLES,
            "connected": meeting.connected(),
            "started": meeting.started,
            "processing": meeting.processing,
            "done": meeting.done,
            "meetingId": meeting.meeting_id,
        }
    )


@app.post("/start")
async def start():
    if meeting.started:
        return JSONResponse({"error": "already started"}, status_code=400)
    meeting.meeting_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    meeting_dir = DATA_DIR / meeting.meeting_id
    meeting_dir.mkdir(parents=True, exist_ok=True)
    meeting.raw_path = meeting_dir / "raw.pcm"
    meeting.raw_file = open(meeting.raw_path, "wb")
    meeting.mixer = Mixer(roles=ROLES, sample_rate=TARGET_RATE, chunk_ms=CHUNK_MS, start_time=time.monotonic())
    meeting.started = True

    for sockets in meeting.sockets.values():
        for ws in sockets:
            await ws.send_text(json.dumps({"type": "start"}))

    return {"meetingId": meeting.meeting_id}


@app.post("/stop")
async def stop(background_tasks: BackgroundTasks):
    if not meeting.started:
        return JSONResponse({"error": "not started"}, status_code=400)
    meeting.started = False

    for sockets in meeting.sockets.values():
        for ws in sockets:
            await ws.send_text(json.dumps({"type": "stop"}))

    for chunk in meeting.mixer.finish():
        meeting.raw_file.write(chunk)
    meeting.raw_file.close()

    meeting_dir = meeting.raw_path.parent
    wav_path = meeting_dir / "combined.wav"
    pcm_bytes = meeting.raw_path.read_bytes()
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TARGET_RATE)
        w.writeframes(pcm_bytes)

    meeting.processing = True
    background_tasks.add_task(run_pipeline, wav_path, meeting_dir)
    return {"meetingId": meeting.meeting_id, "wav": str(wav_path)}


def run_pipeline(wav_path: Path, meeting_dir: Path):
    try:
        transcript = transcribe(wav_path)
        (meeting_dir / "transcript.txt").write_text(transcript, encoding="utf-8")
        notes = summarize(transcript)
        (meeting_dir / "notes.md").write_text(notes, encoding="utf-8")
    finally:
        meeting.processing = False
        meeting.done = True


@app.get("/meetings/{meeting_id}/notes.md")
async def get_notes(meeting_id: str):
    path = DATA_DIR / meeting_id / "notes.md"
    if not path.exists():
        return JSONResponse({"error": "not ready"}, status_code=404)
    return FileResponse(path, media_type="text/markdown")


@app.websocket("/ws/{role}")
async def ws_endpoint(ws: WebSocket, role: str):
    if role not in ROLES:
        await ws.close(code=4404)
        return
    await ws.accept()
    meeting.sockets[role].add(ws)
    try:
        while True:
            msg = await ws.receive()
            if "text" in msg and msg["text"] is not None:
                handshake = json.loads(msg["text"])
                meeting.sample_rates[ws] = int(handshake["sampleRate"])
            elif "bytes" in msg and msg["bytes"] is not None:
                if not meeting.started or meeting.mixer is None:
                    continue
                data = msg["bytes"]
                seq = struct.unpack(">I", data[:4])[0]
                pcm = np.frombuffer(data[4:], dtype=np.int16)
                src_rate = meeting.sample_rates.get(ws, TARGET_RATE)
                pcm = resample(pcm, src_rate, TARGET_RATE)
                for mixed in meeting.mixer.add_chunk(role, seq, pcm):
                    meeting.raw_file.write(mixed)
    except WebSocketDisconnect:
        pass
    finally:
        meeting.sockets[role].discard(ws)
        meeting.sample_rates.pop(ws, None)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8443"))
    certs = Path(__file__).parent.parent / "certs"
    print(f"Display page: https://{lan_ip()}:{port}/  (phones scan the QR codes shown there)")
    uvicorn.run(
        "server.main:app",
        host="0.0.0.0",
        port=port,
        ssl_keyfile=str(certs / "key.pem"),
        ssl_certfile=str(certs / "cert.pem"),
    )

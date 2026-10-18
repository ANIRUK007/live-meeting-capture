# live-meeting-capture

Capture a live technical meeting from two or more phones in a room, merge the
audio in real time, and turn it into a markdown wiki page — decisions,
technical approaches, action items, open questions — via Gemini.

## How it works

1. **Join.** Laptop runs the server and shows one QR code per role (default
   `board`, `audience` — set `ROLES=board,audience,...` to change). Each
   phone scans its QR, opens the page, taps "Join as mic", grants microphone
   access. No app install.
2. **Start.** Operator clicks "Start meeting" on the laptop. Every connected
   phone gets a synchronized start signal and begins streaming 500ms PCM16
   chunks over WebSocket, each tagged with a sequence number computed from
   elapsed time since that signal.
3. **Merge.** The server pairs chunks by sequence number across roles and
   averages them into one mono stream, filling silence for a side that's
   late or missing (jitter window: 2s). Appended continuously to a raw PCM
   file.
4. **Stop.** Operator clicks "Stop meeting". The raw file is wrapped into a
   WAV, then handed to Gemini: first for a verbatim transcript, then for the
   structured markdown summary.
5. **Notes.** Saved to `data/meetings/<timestamp>/notes.md`, linked from the
   display page once ready.

## Run it

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=...
./run.sh
```

Open the printed `https://<lan-ip>:8443/` on the laptop — that's the display
page with the QR codes. Phones must be on the same LAN/Wi-Fi.

**Why HTTPS:** `getUserMedia` only works over a secure context, and phones
reach the laptop by LAN IP (not `localhost`), so a self-signed cert is
generated on first run. Each phone's browser will show a one-time "not
secure" warning — tap through it once per phone.

## Known limits (deliberate, cheap to lift later)

- One meeting at a time, single process, no auth — this is a LAN tool for a
  room, not a hosted service.
- Self-signed TLS cert — swap for `mkcert` or a Tailscale cert if the
  per-phone warning becomes a real problem.
- Audio quality/transcription depend on Gemini reachability — there's no
  offline fallback (deliberately dropped local Whisper for this, per request).
- `ScriptProcessorNode` for mic capture (deprecated API, but one line and
  works everywhere today) — move to an `AudioWorklet` if a browser drops it.

## Files

- `server/main.py` — FastAPI app: pages, websockets, start/stop, pipeline kickoff.
- `server/mixer.py` — jitter-buffered sequence-paired mixer (`python3 server/mixer.py` runs its self-check).
- `server/gemini.py` — transcription + summarization via the Gemini API.
- `server/static/{display,join}.html` — laptop control page / phone mic page.

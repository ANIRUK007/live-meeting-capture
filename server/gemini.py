"""Transcription and summarization, both via the Gemini API.

Gemini accepts audio directly, so there is no local speech-to-text model to
install or run: upload the meeting WAV, ask for a plain transcript, then feed
that transcript back for the structured markdown summary. Needs GEMINI_API_KEY.
"""

from __future__ import annotations

import os
from pathlib import Path

from google import genai

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

TRANSCRIBE_PROMPT = (
    "Transcribe this meeting recording verbatim as plain text. "
    "Do not label speakers. Do not summarize or omit anything. "
    "Preserve technical terms, acronyms, and numbers exactly as spoken."
)

SUMMARY_PROMPT = """You are writing a wiki page for a technical meeting from its raw transcript below.

Extract, as markdown with these exact headings:

## Key Decisions
## Technical Approaches & Methods Discussed
## Action Items
## Open Questions

Only include what was actually said. Do not infer or invent anything not in the transcript.
If a section has nothing, write "None noted." under it. Use bullet points.

Transcript:
---
{transcript}
---
"""


def _client() -> genai.Client:
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def transcribe(wav_path: Path) -> str:
    client = _client()
    audio_file = client.files.upload(file=str(wav_path))
    response = client.models.generate_content(model=MODEL, contents=[audio_file, TRANSCRIBE_PROMPT])
    return response.text or ""


def summarize(transcript: str) -> str:
    client = _client()
    prompt = SUMMARY_PROMPT.format(transcript=transcript)
    response = client.models.generate_content(model=MODEL, contents=[prompt])
    return response.text or ""

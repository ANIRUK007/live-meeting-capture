"""Jitter-buffered, seq-indexed mixer for two or more mono PCM16 streams.

Each role sends fixed-duration chunks tagged with a sequence number (elapsed
time since the synchronized start, divided by chunk duration). The mixer
waits a bounded jitter window for every role's chunk at a given seq, then
mixes what arrived and fills any missing role with silence. Pure function of
wall-clock time in, so it is testable without asyncio or real sockets.
"""

from __future__ import annotations

import time

import numpy as np


def resample(pcm: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    if from_rate == to_rate or len(pcm) == 0:
        return pcm
    n_new = max(1, round(len(pcm) * to_rate / from_rate))
    x_old = np.arange(len(pcm))
    x_new = np.linspace(0, len(pcm) - 1, n_new)
    return np.interp(x_new, x_old, pcm).astype(np.int16)


class Mixer:
    def __init__(
        self,
        roles: list[str],
        sample_rate: int = 16000,
        chunk_ms: int = 500,
        jitter_window_s: float = 2.0,
        start_time: float | None = None,
    ):
        self.roles = list(roles)
        self.sample_rate = sample_rate
        self.chunk_samples = sample_rate * chunk_ms // 1000
        self.chunk_dur_s = chunk_ms / 1000
        self.jitter_window_s = jitter_window_s
        self.start_time = start_time if start_time is not None else time.monotonic()
        self.pending: dict[int, dict[str, np.ndarray]] = {}
        self.next_seq = 0

    def _fit(self, pcm: np.ndarray) -> np.ndarray:
        if len(pcm) == self.chunk_samples:
            return pcm
        out = np.zeros(self.chunk_samples, dtype=np.int16)
        n = min(len(pcm), self.chunk_samples)
        out[:n] = pcm[:n]
        return out

    def _mix(self, entry: dict[str, np.ndarray]) -> bytes:
        silence = np.zeros(self.chunk_samples, dtype=np.int32)
        acc = sum((self._fit(entry[r]).astype(np.int32) if r in entry else silence) for r in self.roles)
        mixed = (acc // len(self.roles)).astype(np.int16)
        return mixed.tobytes()

    def add_chunk(self, role: str, seq: int, pcm: np.ndarray) -> list[bytes]:
        self.pending.setdefault(seq, {})[role] = pcm
        return self._flush(time.monotonic())

    def tick(self) -> list[bytes]:
        return self._flush(time.monotonic())

    def finish(self) -> list[bytes]:
        """Flush everything up to the last seq seen, ignoring deadlines (meeting ended)."""
        if not self.pending:
            return []
        max_seq = max(self.pending)
        out = []
        while self.next_seq <= max_seq:
            out.append(self._mix(self.pending.pop(self.next_seq, None) or {}))
            self.next_seq += 1
        return out

    def _flush(self, now: float) -> list[bytes]:
        out = []
        while True:
            entry = self.pending.get(self.next_seq)
            complete = entry is not None and set(entry) >= set(self.roles)
            deadline = self.start_time + (self.next_seq + 1) * self.chunk_dur_s + self.jitter_window_s
            if complete or now >= deadline:
                out.append(self._mix(entry or {}))
                self.pending.pop(self.next_seq, None)
                self.next_seq += 1
                continue
            break
        return out


def demo():
    m = Mixer(roles=["board", "audience"], sample_rate=80, chunk_ms=100, jitter_window_s=0.01, start_time=time.monotonic())

    out = m.add_chunk("board", 0, np.array([10, 10, 10, 10, 10, 10, 10, 10], dtype=np.int16))
    assert out == [], "should wait for the other role before flushing seq 0"

    out = m.add_chunk("audience", 0, np.array([20, 20, 20, 20, 20, 20, 20, 20], dtype=np.int16))
    assert len(out) == 1, "both roles present, seq 0 should flush immediately"
    mixed = np.frombuffer(out[0], dtype=np.int16)
    assert (mixed == 15).all(), f"expected average of 10 and 20, got {mixed}"

    time.sleep(0.25)
    out = m.add_chunk("board", 1, np.array([40] * 8, dtype=np.int16))
    assert len(out) == 1, "audience never sent seq 1, should flush as silence after jitter window"
    mixed = np.frombuffer(out[0], dtype=np.int16)
    assert (mixed == 20).all(), f"expected average of 40 and 0 (silence), got {mixed}"

    m.add_chunk("board", 5, np.array([8] * 8, dtype=np.int16))
    rest = m.finish()
    assert len(rest) == 4, f"finish() should flush seq 2,3,4,5 (4 chunks), got {len(rest)}"

    rs = resample(np.array([0, 100, 200, 300], dtype=np.int16), from_rate=8, to_rate=4)
    assert len(rs) == 2, f"downsample by half should halve sample count, got {len(rs)}"

    print("mixer self-check passed")


if __name__ == "__main__":
    demo()

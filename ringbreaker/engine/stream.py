"""F2 stream replay inside the API process.

Feeds held-back payments, in timestamp order, through ``Engine.score`` — the
same code path as ``POST /score``. Controlled from the dashboard (start, pause,
step, speed) so a demo does not need a second terminal. The standalone
``python -m ringbreaker.stream.replay`` still replays over HTTP.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

from ringbreaker.engine.engine import Engine, EngineError

log = logging.getLogger("ringbreaker.stream")


class StreamRunner:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.position = 0
        self.rate = 8.0
        self.pause_on_block = True
        self.running = False
        self.last_event: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._wake = threading.Event()
        self._lock = threading.Lock()

    @property
    def total(self) -> int:
        return len(self.engine.stream_rows)

    def status(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "position": self.position,
            "total": self.total,
            "rate": self.rate,
            "pause_on_block": self.pause_on_block,
            "clock": self.engine.clock.isoformat() if self.engine.clock else None,
            "last_event": self.last_event,
            "finished": self.position >= self.total,
        }

    def configure(self, rate: Optional[float] = None, pause_on_block: Optional[bool] = None) -> None:
        if rate is not None:
            self.rate = max(0.5, min(float(rate), 200.0))
        if pause_on_block is not None:
            self.pause_on_block = bool(pause_on_block)

    def start(self) -> None:
        with self._lock:
            if self.running or self.position >= self.total:
                return
            self.running = True
            self.last_event = None
            self._thread = threading.Thread(target=self._loop, name="ringbreaker-stream", daemon=True)
            self._thread.start()

    def pause(self) -> None:
        self.running = False
        self._wake.set()

    def step(self, n: int = 1) -> int:
        done = 0
        for _ in range(max(1, min(n, 500))):
            if self.position >= self.total:
                break
            self._one()
            done += 1
        return done

    def reset(self) -> None:
        self.pause()
        if self._thread:
            self._thread.join(timeout=5)
        self.engine.reset()
        self.position = 0
        self.last_event = None

    def _one(self) -> Optional[Dict[str, Any]]:
        row = self.engine.stream_rows[self.position]
        self.position += 1
        try:
            return self.engine.score(row, source="stream")
        except EngineError as exc:  # duplicate after a manual /score, etc.
            log.warning("stream payment %s skipped: %s", row.get("transaction_id"), exc)
            return None

    def _loop(self) -> None:
        while self.running and self.position < self.total:
            out = self._one()
            if out and out["action"] == "BLOCK" and self.pause_on_block:
                self.running = False
                self.last_event = f"Paused on BLOCK: {out['alert_id']}"
                break
            self._wake.wait(1.0 / self.rate)
            self._wake.clear()
        self.running = False

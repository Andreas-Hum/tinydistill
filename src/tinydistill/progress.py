"""How far a long step is: printed progress with rate and time left, and a status file per run."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path


def duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


class Progress:
    """Count work done; every ``every`` seconds (and at the end) print `stage: done/total (pct), rate, time
    left` and write the same to ``status`` (``RUN/status.json``), which `tinydistill status` reads."""

    def __init__(
        self,
        stage: str,
        total: int | None,
        log: Callable[[str], None],
        status: Path | None = None,
        unit: str = "texts",
        every: float = 10.0,
    ):
        self.stage, self.total, self.log, self.status, self.unit, self.every = (
            stage,
            total,
            log,
            status,
            unit,
            every,
        )
        self.done = 0
        self.t0 = self.last = time.perf_counter()
        self._write()

    def update(self, n: int = 1) -> None:
        self.done += n
        now = time.perf_counter()
        if now - self.last >= self.every:
            self.last = now
            self.log(self.line())
            self._write()

    def finish(self) -> None:
        self.log(self.line(final=True))
        self._write(final=True)

    def rate(self) -> float:
        elapsed = time.perf_counter() - self.t0
        return self.done / elapsed if elapsed > 0 else 0.0

    def eta(self) -> float | None:
        r = self.rate()
        return (self.total - self.done) / r if self.total and r > 0 else None

    def line(self, final: bool = False) -> str:
        r, eta = self.rate(), self.eta()
        of = f"/{self.total} ({self.done / self.total:.0%})" if self.total else ""
        tail = (
            f"done in {duration(time.perf_counter() - self.t0)}"
            if final
            else (f"{duration(eta)} left" if eta is not None else "")
        )
        return f"{self.stage}: {self.done}{of} {self.unit}, {r:.1f}/s" + (f", {tail}" if tail else "")

    def _write(self, final: bool = False) -> None:
        if self.status is None:
            return
        eta = self.eta()
        record = {
            "stage": self.stage,
            "done": self.done,
            "total": self.total,
            "unit": self.unit,
            "rate_per_s": round(self.rate(), 3),
            "eta_s": None if final or eta is None else round(eta),
            "finished": final,
            "updated": datetime.now(UTC).isoformat(timespec="seconds"),
            "pid": os.getpid(),
        }
        tmp = self.status.with_suffix(".tmp")
        tmp.write_text(json.dumps(record))
        tmp.replace(self.status)  # atomic: `status` never reads a half-written file


def describe_status(path: Path) -> str:
    """A human-readable summary of RUN/status.json, including whether the process is still running."""
    if not path.exists():
        return "no status yet: nothing has run in this directory with tinydistill 0.3.0 or later"
    s = json.loads(path.read_text())
    age = (datetime.now(UTC) - datetime.fromisoformat(s["updated"])).total_seconds()
    try:
        os.kill(s["pid"], 0)
        alive = True
    except (ProcessLookupError, PermissionError):
        alive = False
    except OSError:
        alive = False
    total = f"/{s['total']}" if s.get("total") else ""
    pct = f" ({s['done'] / s['total']:.0%})" if s.get("total") else ""
    lines = [f"{s['stage']}: {s['done']}{total}{pct} {s['unit']}, {s['rate_per_s']}/s"]
    if s.get("finished"):
        lines.append(f"finished {duration(age)} ago")
    elif alive:
        left = f", about {duration(s['eta_s'])} left" if s.get("eta_s") is not None else ""
        lines.append(f"running (pid {s['pid']}), last update {duration(age)} ago{left}")
    else:
        lines.append(
            f"not running: the process (pid {s['pid']}) stopped {duration(age)} ago without finishing"
        )
    return "\n".join(lines)

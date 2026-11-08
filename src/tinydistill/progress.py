"""How far a long step is: printed progress with rate and time left, and a status file per run."""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
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


_drawn: Progress | None = None  # the bar drawn on the terminal's current line, if any


def use_bar() -> bool:
    """Whether progress is a bar redrawn in place (on a terminal) or a line every few seconds (when the output
    goes to a file or a pipe); TINYDISTILL_PROGRESS=bar or lines chooses."""
    mode = os.environ.get("TINYDISTILL_PROGRESS", "auto")
    if mode in ("bar", "lines"):
        return mode == "bar"
    return sys.stderr.isatty() and os.environ.get("TERM") != "dumb"


def clear_bar() -> None:
    """Take a drawn bar off the terminal's line, so that a log line prints on a line of its own; the bar is
    drawn again at its next update."""
    global _drawn
    if _drawn is not None:
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()
        _drawn = None


TRACE_POINTS = 300  # at most this many (seconds, done) points of a stage in status.json, for the live page


class Progress:
    """Count work done; every ``every`` seconds (and at the end) print `stage: done/total (pct), rate, time
    left`, and every ``write_every`` seconds write the same to ``status`` (``RUN/status.json``), which
    `tinydistill status` reads, with a trace of (seconds, done) points for the live page's speed chart."""

    def __init__(
        self,
        stage: str,
        total: int | None,
        log: Callable[[str], None],
        status: Path | None = None,
        unit: str = "texts",
        every: float = 10.0,
        write_every: float = 2.0,
        bar: bool | None = None,
    ):
        self.stage, self.total, self.log, self.status, self.unit, self.every = (
            stage,
            total,
            log,
            status,
            unit,
            every,
        )
        self.write_every = write_every
        self.bar = use_bar() if bar is None else bar  # None: on a terminal
        self.note = ""  # shown after the bar (train: the last epoch's loss)
        self.last_draw = 0.0
        self.done = 0
        self.t0 = self.last = self.last_write = time.perf_counter()
        self.trace: list[list[float]] = []  # (seconds since the start, done), thinned to TRACE_POINTS
        self._lock, self._stop = threading.Lock(), threading.Event()
        self._write()
        if status is not None and write_every > 0:  # the status stays current while one batch takes long
            threading.Thread(target=self._beat, daemon=True).start()

    def _beat(self) -> None:
        while not self._stop.wait(self.write_every):
            if time.perf_counter() - self.last_write >= self.write_every:
                try:
                    self._write()
                except OSError:  # the run's directory is gone
                    return

    def update(self, n: int = 1) -> None:
        self.done += n
        now = time.perf_counter()
        if self.bar:
            if now - self.last_draw >= 0.1:
                self.draw()
        elif now - self.last >= self.every:
            self.last = now
            self.log(self.line())
        if now - self.last_write >= self.write_every:
            self._write()

    def finish(self) -> None:
        global _drawn
        if self.bar:  # the full bar stays on its line
            self.draw(final=True)
            sys.stderr.write("\n")
            sys.stderr.flush()
            _drawn = None
        else:
            self.log(self.line(final=True))
        self._stop.set()
        self._write(final=True)
        if self.status is not None:  # the stages done, for the live page (status --live)
            record = {
                "stage": self.stage,
                "done": self.done,
                "total": self.total,
                "unit": self.unit,
                "seconds": round(time.perf_counter() - self.t0, 1),
                "finished": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            with self.status.with_name("status-log.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

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

    def draw(self, final: bool = False) -> None:
        """Draw the bar on the terminal's current line:
        `stage  done/total [=====>....]  pct  rate  time left  note`."""
        global _drawn
        if _drawn is not None and _drawn is not self:
            clear_bar()
        self.last_draw = time.perf_counter()
        width = shutil.get_terminal_size((100, 20)).columns
        eta, elapsed = self.eta(), time.perf_counter() - self.t0
        tail = f"  {self.rate():.1f} {self.unit}/s" if elapsed >= 1 or final else ""  # a rate needs some time
        if final:
            tail += f"  done in {duration(elapsed)}"
        elif eta is not None and elapsed >= 1:
            tail += f"  ETA {duration(eta)}"
        if self.note:
            tail += f"  {self.note}"
        if self.total:
            frac = min(1.0, self.done / self.total)
            head = f"{self.stage}  {self.done:>{len(str(self.total))}}/{self.total} "  # a bar that stays put
            n = max(10, min(30, width - len(head) - len(tail) - 8))
            full = int(n * frac)
            body = "=" * full + (">" + "." * (n - full - 1) if full < n else "")
            line = f"{head}[{body}] {frac:4.0%}{tail}"
        else:
            line = f"{self.stage}  {self.done} {self.unit}{tail}"
        sys.stderr.write("\r\033[K" + line[: width - 1])
        sys.stderr.flush()
        _drawn = self

    def _write(self, final: bool = False) -> None:
        if self.status is None:
            return
        with self._lock:
            self._write_status(self.status, final)

    def _write_status(self, status: Path, final: bool) -> None:
        self.last_write = time.perf_counter()
        if final or not self.trace or self.trace[-1][1] != self.done:  # a point when there is progress
            self.trace.append([round(self.last_write - self.t0, 1), self.done])
        if len(self.trace) > TRACE_POINTS:  # every other point, keeping the latest
            self.trace = self.trace[-1::-2][::-1]
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
            "trace": self.trace,
        }
        tmp = status.with_suffix(".tmp")
        tmp.write_text(json.dumps(record), encoding="utf-8")
        tmp.replace(status)  # atomic: `status` never reads a half-written file


def _alive(pid: int) -> bool:
    """Whether a process is still running, without signalling it."""
    if os.name == "nt":  # os.kill(pid, 0) would send Ctrl+C on Windows
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]  # windll exists only on Windows
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)  # signal 0: only checks that the process exists
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, but belongs to another user
        return True
    return True


def describe_status(path: Path) -> str:
    """A human-readable summary of RUN/status.json, including whether the process is still running."""
    if not path.exists():
        return "no status yet: nothing has run in this directory with tinydistill 0.3.0 or later"
    s = json.loads(path.read_text(encoding="utf-8"))
    age = (datetime.now(UTC) - datetime.fromisoformat(s["updated"])).total_seconds()
    alive = _alive(s["pid"])
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

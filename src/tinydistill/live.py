"""A live page for a run (`status RUN --live`, `run --live`): a small HTTP server on 127.0.0.1 whose page
polls /state, which reads the files a run writes as it goes: status.json (the current stage), status-log.jsonl
(the stages done), history-live.json (the loss per epoch) and results.json. It only reads them, so it can
watch a run started anywhere, and stopping it never stops the run."""

from __future__ import annotations

import json
import os
import sys
import threading
import webbrowser
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .progress import _alive

PAGE = Path(__file__).with_name("live.html")
RESULT_KEYS = ("name", "accuracy", "accuracy_std", "macro_f1", "latency_ms_b1", "size_mb", "device")


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # missing, or replaced while read
        return None


def state(run: Path) -> dict:
    """What the page shows, from the run's files."""
    now = datetime.now(UTC)
    status = _json(run / "status.json")
    if status:
        status["age_s"] = round((now - datetime.fromisoformat(status["updated"])).total_seconds())
        status["alive"] = _alive(status["pid"])
    stages = []
    if (run / "status-log.jsonl").exists():
        for line in (run / "status-log.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                stages.append(json.loads(line))
            except ValueError:  # a line being written
                continue
    results = None
    res = run / "results.json"
    started = min(
        (datetime.fromisoformat(s["finished"]).timestamp() - s["seconds"] for s in stages), default=0
    )
    if res.exists() and res.stat().st_mtime >= started:  # not the results of an earlier run in this directory
        r = _json(res) or {}
        results = [{k: m.get(k) for k in RESULT_KEYS} for m in r.get("models", [])]
    return {
        "run": str(run),
        "status": status,
        "stages": stages,
        "history": _json(run / "history-live.json") or {},
        "labels": _json(run / "labels-live.json"),
        "command": _json(run / "command.json"),
        "results_md": (run / "results.md").exists() and results is not None,
        "results": results,
    }


class _Server(ThreadingHTTPServer):
    # on Windows, SO_REUSEADDR lets a second server bind a port that is in use: then the next port is tried
    allow_reuse_address = os.name != "nt"
    daemon_threads = True


def serve(run: Path, port: int = 8765, open_browser: bool = True) -> ThreadingHTTPServer:
    """Start the page's server in a background thread (on the first free port from ``port``) and return it."""
    page = PAGE.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (the http.server name)
            if self.path.split("?")[0] == "/state":
                body, kind = json.dumps(state(run)).encode(), "application/json"
            elif self.path == "/results.md" and (run / "results.md").exists():
                body, kind = (run / "results.md").read_bytes(), "text/plain; charset=utf-8"
            elif self.path in ("/", "/index.html"):
                body, kind = page, "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:  # quiet: the run's own log is what the terminal shows
            pass

    for p in range(port, port + 20):
        try:
            server = _Server(("127.0.0.1", p), Handler)
            break
        except OSError:  # in use
            continue
    else:
        raise SystemExit(f"no free port from {port} to {port + 19} for the live page")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"live page: {url}", file=sys.stderr, flush=True)
    if open_browser:
        webbrowser.open(url)
    return server

"""A small JSON HTTP endpoint for an exported model (standard library only).

POST /predict  {"texts": ["...", ...]}  ->  {"predictions": [{"label": ..., "confidence": ...}, ...]}
GET  /health                            ->  {"status": "ok", "labels": [...]}
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_BODY = 1_000_000  # bytes
MAX_TEXTS = 1000


def make_server(model, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    """``model`` is a Predictor or a Cascade."""
    from .export import Cascade

    lock = threading.Lock()  # one prediction at a time: the model already uses every thread it was given
    labels = model.student.labels if isinstance(model, Cascade) else model.labels

    def predict(texts: list[str]) -> list[dict]:
        with lock:
            if isinstance(model, Cascade):
                return [r | {"confidence": round(r["confidence"], 4)} for r in model.predict(texts)]
            probs = model.predict_proba(texts)
        return [{"label": labels[int(p.argmax())], "confidence": round(float(p.max()), 4)} for p in probs]

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802 (http.server naming)
            if self.path == "/health":
                self._send(200, {"status": "ok", "labels": labels})
            else:
                self._send(404, {"error": "use GET /health or POST /predict"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/predict":
                return self._send(404, {"error": "use POST /predict"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._send(400, {"error": "bad Content-Length"})
            if length > MAX_BODY:
                return self._send(413, {"error": f"body larger than {MAX_BODY} bytes"})
            try:
                texts = json.loads(self.rfile.read(length) or b"{}").get("texts")
            except (json.JSONDecodeError, AttributeError):
                return self._send(400, {"error": 'send JSON like {"texts": ["..."]}'})
            if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
                return self._send(400, {"error": '"texts" must be a list of strings'})
            if len(texts) > MAX_TEXTS:
                return self._send(413, {"error": f"at most {MAX_TEXTS} texts per request"})
            self._send(200, {"predictions": predict(texts) if texts else []})

        def log_message(self, format: str, *args) -> None:  # quiet by default
            pass

    return ThreadingHTTPServer((host, port), Handler)

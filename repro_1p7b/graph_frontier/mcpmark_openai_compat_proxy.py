"""Minimal OpenAI transport shim for MCPMark with native-context checkpoints.

The upstream MCPMarkAgent requests 32768 output tokens on every turn. Qwen3-1.7B
checkpoints here have a 40960-token native context; the second tool turn can
therefore be rejected even when its actual completion would be short. This
shim changes only the OpenAI request's advertised output budget, identically
for both checkpoints. It does not touch messages, tools, responses or scoring.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def clamp_request(body: bytes, max_tokens: int) -> bytes:
    request = json.loads(body)
    if request.get("max_tokens") is not None:
        request["max_tokens"] = min(int(request["max_tokens"]), max_tokens)
    return json.dumps(request, separators=(",", ":")).encode("utf-8")


def make_handler(upstream: str, max_tokens: int):
    class Handler(BaseHTTPRequestHandler):
        def _forward(self, body: bytes | None = None) -> None:
            if self.path not in ("/v1/chat/completions", "/v1/models"):
                self.send_error(404)
                return
            if body is not None and self.path == "/v1/chat/completions":
                body = clamp_request(body, max_tokens)
            headers = {"Content-Type": "application/json"}
            if auth := self.headers.get("Authorization"):
                headers["Authorization"] = auth
            req = Request(
                upstream + self.path,
                data=body,
                headers=headers,
                method=self.command,
            )
            try:
                with urlopen(req, timeout=1200) as response:
                    status, payload = response.status, response.read()
            except HTTPError as exc:
                status, payload = exc.code, exc.read()
            except URLError as exc:
                status = 502
                payload = json.dumps({"error": str(exc.reason)}).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            self._forward()

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            self._forward(self.rfile.read(length))

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    upstream = args.upstream.rstrip("/")
    ThreadingHTTPServer(
        ("127.0.0.1", args.port),
        make_handler(upstream, args.max_tokens),
    ).serve_forever()


if __name__ == "__main__":
    main()

"""Streaming output support"""
import json
import time

import httpx

from .. import retry
from ..base import mask


class _Masked:
    """Streamed text with the key replaced by its masked form before it is printed (a server
    may echo the key). The end of the text that could be the start of the key is held back
    until the next chunk shows whether it is."""

    def __init__(self, secret: str):
        self.secret = secret if secret and len(secret) >= 8 else ""
        self.held = ""

    def feed(self, text: str) -> str:
        if not self.secret:
            return text
        s = (self.held + text).replace(self.secret, mask(self.secret))
        keep = next((k for k in range(min(len(s), len(self.secret) - 1), 0, -1)
                     if self.secret.startswith(s[-k:])), 0)
        self.held = s[len(s) - keep:]
        return s[:len(s) - keep]

    def flush(self) -> str:
        out, self.held = self.held, ""
        return out


def error_text(status: int, text: str) -> str:
    """'Error: HTTP <status>: <message>' from an Anthropic error body (or the raw text)."""
    try:
        data = json.loads(text)
        err = data.get("error", data) if isinstance(data, dict) else data
        msg = err.get("message", err) if isinstance(err, dict) else err
    except ValueError:
        msg = text
    return f"Error: HTTP {status}: {str(msg)[:300]}"


def stream_claude(api_key: str, model: str, prompt: str, timeout: int = 120, max_output_tokens: int = 16384,
                  url: str = "https://api.anthropic.com/v1/messages",
                  waits=retry.WAITS, sleep=time.sleep):
    """Claude streaming, printing the text as it arrives (with the key masked).

    Returns (content, input_tokens, output_tokens, error); error is "" or "Error: ...".
    A rate limit, a server error or a timeout before any text is tried again (llm/retry.py).
    """
    content = ""
    input_tokens = 0
    output_tokens = 0
    shown = _Masked(api_key)

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    body = {
        "model": model,
        "max_tokens": max_output_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True
    }

    with httpx.Client(timeout=timeout) as client:
        for attempt in range(len(waits) + 1):
            last = attempt == len(waits)
            try:
                with client.stream("POST", url, headers=headers, json=body) as resp:
                    if resp.status_code != 200:
                        # a timeout while reading a final error's text is not retried
                        text = retry.complete(resp).content.decode("utf-8", "replace")
                        if last or not retry.retryable(resp.status_code):
                            return content, input_tokens, output_tokens, error_text(resp.status_code, text)
                        wait = retry.wait_before(attempt, resp, waits)
                    else:
                        for line in resp.iter_lines():
                            if not line or not line.startswith("data: "):
                                continue
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            try:
                                event = json.loads(data)
                            except json.JSONDecodeError:
                                continue
                            if not isinstance(event, dict):
                                continue
                            event_type = event.get("type")

                            if event_type == "content_block_delta":
                                text = (event.get("delta") or {}).get("text", "")
                                content += text
                                print(shown.feed(text), end="", flush=True)

                            elif event_type == "message_start":
                                usage = (event.get("message") or {}).get("usage") or {}
                                input_tokens = usage.get("input_tokens", 0)

                            elif event_type == "message_delta":
                                usage = event.get("usage") or {}
                                output_tokens = usage.get("output_tokens", 0)

                            elif event_type == "error":
                                print(shown.flush())
                                return content, input_tokens, output_tokens, \
                                    error_text(200, json.dumps(event))
                        break
            except httpx.TimeoutException:
                if last or content:          # text was already shown: do not repeat it
                    raise
                wait = retry.wait_before(attempt, None, waits)
            sleep(wait)

    print(shown.flush())  # the held-back end, then a newline
    return content, input_tokens, output_tokens, ""

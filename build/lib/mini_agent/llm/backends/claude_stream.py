"""Streaming output support"""
import httpx
import json
import sys


def stream_claude(api_key: str, model: str, prompt: str, timeout: int = 120, max_output_tokens: int = 16384):
    """Claude streaming; returns (content, input_tokens, output_tokens)"""
    content = ""
    input_tokens = 0
    output_tokens = 0

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }

    with httpx.Client(timeout=timeout) as client:
        with client.stream(
            "POST",
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json={
                "model": model,
                "max_tokens": max_output_tokens,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True
            }
        ) as resp:
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                    event_type = event.get("type")

                    if event_type == "content_block_delta":
                        text = event.get("delta", {}).get("text", "")
                        content += text
                        print(text, end="", flush=True)

                    elif event_type == "message_start":
                        usage = event.get("message", {}).get("usage", {})
                        input_tokens = usage.get("input_tokens", 0)

                    elif event_type == "message_delta":
                        usage = event.get("usage", {})
                        output_tokens = usage.get("output_tokens", 0)

                except json.JSONDecodeError:
                    pass

    print()  # newline
    return content, input_tokens, output_tokens

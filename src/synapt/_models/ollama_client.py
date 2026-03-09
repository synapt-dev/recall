from __future__ import annotations

import json
import urllib.request
from typing import List

from .base import Message, ModelClient


class OllamaClient(ModelClient):
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")
        self.max_tokens: int | None = None

    def chat(
        self,
        model: str,
        messages: List[Message],
        temperature: float = 0.2,
        **kwargs,
    ) -> str:
        url = f"{self.base_url}/api/chat"
        options = {"temperature": temperature}
        if "max_tokens" in kwargs and kwargs["max_tokens"] is not None:
            options["num_predict"] = int(kwargs["max_tokens"])
        elif self.max_tokens is not None:
            options["num_predict"] = int(self.max_tokens)
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": options,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as response:
            raw = response.read().decode("utf-8")
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            if "message" in parsed and isinstance(parsed["message"], dict):
                return parsed["message"].get("content", "")
            if "response" in parsed:
                return parsed["response"]
        return ""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Message:
    role: str
    content: str


class ModelClient:
    def chat(
        self,
        model: str,
        messages: List[Message],
        temperature: float = 0.2,
        **kwargs,
    ) -> str:
        raise NotImplementedError

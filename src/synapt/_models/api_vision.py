from __future__ import annotations

import base64
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VisionRequest:
    """Request for vision model analysis."""
    prompt: str
    image_paths: list[str] = field(default_factory=list)
    image_bytes: list[bytes] = field(default_factory=list)
    max_tokens: int = 1024


@dataclass
class VisionResponse:
    """Response from vision model."""
    text: str
    model: str
    provider: str
    tokens_used: int | None = None
    cost: float | None = None


class VisionProvider(ABC):
    """Abstract base class for vision providers."""

    @abstractmethod
    def analyze(self, request: VisionRequest) -> VisionResponse:
        """Analyze images with the given prompt."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available (API keys set, etc.)."""
        pass

    def _load_image_as_base64(self, path: str) -> str:
        """Load an image file and return as base64."""
        with open(path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")

    def _get_mime_type(self, path: str) -> str:
        """Get MIME type from file extension."""
        ext = Path(path).suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        return mime_types.get(ext, "image/png")


class OpenAIVisionProvider(VisionProvider):
    """OpenAI GPT-4 Vision provider."""

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY", "")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def analyze(self, request: VisionRequest) -> VisionResponse:
        if not self.is_available():
            raise RuntimeError("OPENAI_API_KEY is not set")

        try:
            import openai
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")

        client = openai.OpenAI(api_key=self.api_key)

        # Build content with images
        content: list[dict[str, Any]] = [{"type": "text", "text": request.prompt}]

        for path in request.image_paths:
            base64_image = self._load_image_as_base64(path)
            mime_type = self._get_mime_type(path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
            })

        for img_bytes in request.image_bytes:
            base64_image = base64.standard_b64encode(img_bytes).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            })

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=request.max_tokens,
        )

        text = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else None

        return VisionResponse(
            text=text,
            model=self.model,
            provider="openai",
            tokens_used=tokens,
        )


class AnthropicVisionProvider(VisionProvider):
    """Anthropic Claude Vision provider."""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        self.model = model
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def analyze(self, request: VisionRequest) -> VisionResponse:
        if not self.is_available():
            raise RuntimeError("ANTHROPIC_API_KEY is not set")

        try:
            import anthropic
        except ImportError:
            raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

        client = anthropic.Anthropic(api_key=self.api_key)

        # Build content with images
        content: list[dict[str, Any]] = []

        for path in request.image_paths:
            base64_image = self._load_image_as_base64(path)
            mime_type = self._get_mime_type(path)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": base64_image,
                },
            })

        for img_bytes in request.image_bytes:
            base64_image = base64.standard_b64encode(img_bytes).decode("utf-8")
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64_image,
                },
            })

        content.append({"type": "text", "text": request.prompt})

        response = client.messages.create(
            model=self.model,
            max_tokens=request.max_tokens,
            messages=[{"role": "user", "content": content}],
        )

        text = response.content[0].text if response.content else ""
        tokens = response.usage.input_tokens + response.usage.output_tokens if response.usage else None

        return VisionResponse(
            text=text,
            model=self.model,
            provider="anthropic",
            tokens_used=tokens,
        )


class OllamaVisionProvider(VisionProvider):
    """Ollama local vision provider (Qwen3-VL, LLaVA, etc.)."""

    def __init__(self, model: str = "qwen3-vl:4b"):
        self.model = model
        self.base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    def is_available(self) -> bool:
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def analyze(self, request: VisionRequest) -> VisionResponse:
        try:
            import requests
        except ImportError:
            raise RuntimeError("requests package not installed. Run: pip install requests")

        # Collect base64 images
        images: list[str] = []
        for path in request.image_paths:
            images.append(self._load_image_as_base64(path))
        for img_bytes in request.image_bytes:
            images.append(base64.standard_b64encode(img_bytes).decode("utf-8"))

        payload = {
            "model": self.model,
            "prompt": request.prompt,
            "images": images,
            "stream": False,
        }

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()

        return VisionResponse(
            text=data.get("response", ""),
            model=self.model,
            provider="ollama",
            tokens_used=data.get("eval_count"),
        )


class QwenVisionProvider(VisionProvider):
    """Qwen3-VL cloud vision provider via Dashscope/OpenAI-compatible API."""

    def __init__(self, model: str = "qwen3-vl:235b-instruct-cloud"):
        self.model = model
        self.api_key = os.getenv("DASHSCOPE_API_KEY", "") or os.getenv("QWEN_API_KEY", "")
        self.base_url = os.getenv("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def analyze(self, request: VisionRequest) -> VisionResponse:
        if not self.is_available():
            raise RuntimeError("DASHSCOPE_API_KEY or QWEN_API_KEY is not set")

        try:
            import openai
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")

        # Use OpenAI-compatible client with Dashscope endpoint
        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        # Build content with images
        content: list[dict[str, Any]] = [{"type": "text", "text": request.prompt}]

        for path in request.image_paths:
            base64_image = self._load_image_as_base64(path)
            mime_type = self._get_mime_type(path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
            })

        for img_bytes in request.image_bytes:
            base64_image = base64.standard_b64encode(img_bytes).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            })

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=request.max_tokens,
        )

        text = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else None

        return VisionResponse(
            text=text,
            model=self.model,
            provider="qwen",
            tokens_used=tokens,
        )


class VisionClient:
    """Multi-provider vision client with automatic fallback."""

    def __init__(self, provider: str = "auto"):
        """
        Initialize vision client.

        Args:
            provider: "openai", "anthropic", "ollama", "qwen", or "auto" (try in order)
        """
        self.provider_name = provider
        self._providers: dict[str, VisionProvider] = {
            "openai": OpenAIVisionProvider(),
            "anthropic": AnthropicVisionProvider(),
            "ollama": OllamaVisionProvider(),
            "qwen": QwenVisionProvider(),
        }

    def _get_provider(self) -> VisionProvider:
        """Get the appropriate provider."""
        if self.provider_name != "auto":
            provider = self._providers.get(self.provider_name)
            if provider and provider.is_available():
                return provider
            raise RuntimeError(f"Provider {self.provider_name} is not available")

        # Auto-select: try in order of preference (local first, then cloud)
        for name in ["ollama", "qwen", "anthropic", "openai"]:
            provider = self._providers[name]
            if provider.is_available():
                return provider

        raise RuntimeError("No vision provider available. Set API keys or start Ollama.")

    def analyze(self, request: VisionRequest) -> str:
        """
        Analyze images with the configured provider.

        Returns the text response for backward compatibility.
        """
        provider = self._get_provider()
        response = provider.analyze(request)
        return response.text

    def analyze_full(self, request: VisionRequest) -> VisionResponse:
        """Analyze images and return full response with metadata."""
        provider = self._get_provider()
        return provider.analyze(request)

    def is_available(self) -> bool:
        """Check if any vision provider is available."""
        if self.provider_name != "auto":
            provider = self._providers.get(self.provider_name)
            return provider.is_available() if provider else False
        return any(p.is_available() for p in self._providers.values())


def create_vision_client(provider: str = "auto") -> VisionClient:
    """Factory function to create a vision client."""
    return VisionClient(provider=provider)

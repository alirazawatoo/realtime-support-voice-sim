import sys
import types
from pathlib import Path
from typing import Dict, Iterator

import pytest


def _install_fake_groq_module() -> None:
    """
    Install a lightweight fake `groq` module into sys.modules if it is missing.

    This keeps imports of `src.llm_module` working in test environments
    where the real Groq SDK is not installed, without affecting runtime code.
    """
    if "groq" in sys.modules:
        return

    fake_groq = types.ModuleType("groq")

    class FakeGroqClient:
        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - trivial
            pass

    fake_groq.Groq = FakeGroqClient
    sys.modules["groq"] = fake_groq


_install_fake_groq_module()


@pytest.fixture
def personas_dir() -> Path:
    """
    Fixture pointing at the real personas directory in the project.

    Tests that exercise persona loading can use this to avoid hard-coded paths.
    """
    return Path(__file__).resolve().parent.parent / "config" / "personas"


class _FakeGroqChatCompletions:
    def __init__(self, response_text: str) -> None:
        self._response_text = response_text

    def create(self, **kwargs) -> "_FakeGroqResponse":
        # Minimal response object with the shape used by LLMClient.
        return _FakeGroqResponse(self._response_text)


class _FakeGroqChat:
    def __init__(self, response_text: str) -> None:
        self.completions = _FakeGroqChatCompletions(response_text)


class _FakeGroqClient:
    def __init__(self, response_text: str = "hello") -> None:
        self.chat = _FakeGroqChat(response_text)


class _FakeChunkChoiceDelta:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChunkChoice:
    def __init__(self, content: str) -> None:
        self.delta = _FakeChunkChoiceDelta(content)


class _FakeStreamChunk:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChunkChoice(content)]


class _FakeGroqStreamingClient(_FakeGroqClient):
    def __init__(self, tokens: Iterator[str]) -> None:
        super().__init__()
        self._tokens = list(tokens)

    class _StreamingCompletions:
        def __init__(self, tokens: Iterator[str]) -> None:
            self._tokens = list(tokens)

        def create(self, **kwargs):
            for tok in self._tokens:
                yield _FakeStreamChunk(tok)

    class _StreamingChat:
        def __init__(self, tokens: Iterator[str]) -> None:
            self.completions = _FakeGroqStreamingClient._StreamingCompletions(tokens)

    def __post_init__(self) -> None:
        self.chat = self._StreamingChat(iter(self._tokens))


@pytest.fixture
def fake_groq_client() -> _FakeGroqClient:
    """Provide a fake Groq client that returns a fixed completion."""
    return _FakeGroqClient("test completion")


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


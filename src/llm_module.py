import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple

from groq import Groq


def approx_tokens(text: str) -> int:
    """
    Roughly estimate the number of tokens in a string.

    This uses a simple heuristic of ~4 characters per token and always
    returns at least 1 to avoid downstream division-by-zero issues.
    """
    return max(1, int(len(text) / 4))


@dataclass(slots=True)
class LLMConfig:
    """Configuration for LLM calls."""

    model: str
    temperature: float
    max_tokens: int
    seed: Optional[int]


class LLMClient:
    """
    Thin wrapper around the Groq chat completions API.

    The underlying Groq client can be injected for testing by passing a
    preconfigured instance via the `client` parameter.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.4,
        max_tokens: int = 180,
        client: Optional[Groq] = None,
    ) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        self.client: Groq = client or Groq(api_key=api_key)

        model_name = model or os.getenv("GROQ_LLM_MODEL", "penai/gpt-oss-20b")
        seed_str = os.getenv("SEED", "")
        try:
            seed = int(seed_str) if seed_str.strip() != "" else None
        except Exception:
            seed = None

        self.config = LLMConfig(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )

    def _common_params(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Build the parameter dictionary shared by streaming and non-streaming calls."""
        return {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "seed": self.config.seed,
        }

    def stream_chat(self, messages: List[Dict[str, str]]) -> Tuple[Generator[str, None, None], float]:
        """
        Stream tokens from the LLM for a chat-style conversation.

        Returns a tuple of (token_generator, first_token_latency_ms).
        """
        t0 = time.perf_counter()
        params = self._common_params(messages)
        try:
            stream = self.client.chat.completions.create(
                **params,
                stream=True,
            )
        except Exception as exc:
            raise RuntimeError("Failed to start streaming chat completion") from exc

        first_latency: Optional[float] = None

        def gen() -> Generator[str, None, None]:
            nonlocal first_latency
            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta.content or ""
                except Exception:
                    delta = ""
                if delta and first_latency is None:
                    first_latency = (time.perf_counter() - t0) * 1000
                if delta:
                    yield delta

        return gen(), (first_latency or (time.perf_counter() - t0) * 1000)

    def complete(self, messages: List[Dict[str, str]]) -> Tuple[str, float, Optional[Dict[str, Any]]]:
        """
        Get a full (non-streaming) chat completion response from the LLM.

        Returns a tuple of (text, latency_ms, usage_metadata_dict_or_none).
        """
        t0 = time.perf_counter()
        params = self._common_params(messages)
        try:
            resp = self.client.chat.completions.create(
                **params,
                stream=False,
            )
        except Exception as exc:
            raise RuntimeError("Failed to get chat completion") from exc

        latency_ms = (time.perf_counter() - t0) * 1000
        txt = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        return txt, latency_ms, usage

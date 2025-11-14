from typing import Any, Callable, Dict, Iterable, List

import numpy as np

from src.simple_voice_handler import SimpleVoiceHandler
from src.state_manager import ConversationState
from src.voice_client import VoiceClient


class _FakeASR:
    def __init__(self, text: str = "hello from user") -> None:
        self.text = text

    def transcribe_wav_bytes(self, wav_bytes: bytes) -> tuple[str, float]:
        assert isinstance(wav_bytes, (bytes, bytearray))
        return self.text, 123.0

    def streaming_listen(self, vad_stream, on_partial: Callable[[str], None] | None = None, **_: Any) -> tuple[str, float, float]:
        if on_partial:
            on_partial(self.text)
        return self.text, 111.0, 1.0


class _FakeLLM:
    def __init__(self, text: str = "assistant reply.") -> None:
        self.text = text

    def complete(self, messages: List[Dict[str, str]]) -> tuple[str, float, Dict[str, Any] | None]:
        assert messages and messages[0]["role"] == "system"
        return self.text, 222.0, None

    def stream_chat(self, messages: List[Dict[str, str]]):
        assert messages and messages[0]["role"] == "system"

        def gen() -> Iterable[str]:
            for tok in ["assistant ", "reply."]:
                yield tok

        return gen(), 200.0


class _FakeTTS:
    def __init__(self) -> None:
        self.spoken: List[str] = []

    def synthesize_sentence(self, text: str) -> bytes:
        self.spoken.append(text)
        return b"fake-wav"

    def speak_sentences(self, sentences: Iterable[str], stop_flag: Callable[[], bool]) -> float:
        for s in sentences:
            if stop_flag():
                break
            self.spoken.append(s)
        return 333.0

    class playback:  # type: ignore[override]
        @staticmethod
        def play_wav(wav_bytes: bytes) -> None:
            assert isinstance(wav_bytes, (bytes, bytearray))


class _FakeLogger:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def log_turn(self, turn: int, persona: str, asr_ms: float, llm_ms: float, tts_ms: float, total_ms: float, *_, **__):
        self.rows.append(
            {
                "turn": turn,
                "persona": persona,
                "asr_ms": asr_ms,
                "llm_ms": llm_ms,
                "tts_ms": tts_ms,
                "total_ms": total_ms,
            }
        )


def test_simple_voice_handler_happy_path(monkeypatch) -> None:
    persona = {"name": "Test Persona", "system_prompt": "Be helpful."}
    callbacks: Dict[str, Any] = {"status": [], "user_text": [], "assistant_text": [], "metrics": []}

    def _append_cb(key: str):
        def inner(value):
            callbacks[key].append(value)

        return inner

    handler = SimpleVoiceHandler(
        persona,
        callbacks={
            "status": _append_cb("status"),
            "user_text": _append_cb("user_text"),
            "assistant_text": _append_cb("assistant_text"),
            "metrics": _append_cb("metrics"),
        },
        asr_client=_FakeASR("hello world"),
        llm_client=_FakeLLM("hi there."),
        tts_client=_FakeTTS(),
    )

    # Seed some fake audio data in the buffer so that process_voice_input
    # does not attempt to read from a real microphone.
    fake_chunk = np.zeros((1600, 1), dtype=np.int16)
    handler.audio_buffer.put(fake_chunk)

    metrics = handler.process_voice_input()

    assert "error" not in metrics
    assert metrics["asr_ms"] == 123.0
    assert metrics["llm_ms"] >= 0
    assert metrics["tts_ms"] >= 0
    assert metrics["total_ms"] >= metrics["asr_ms"]

    assert callbacks["user_text"][-1] == "hello world"
    assert callbacks["assistant_text"][-1] == "hi there."
    assert callbacks["metrics"], "Expected metrics callback to be invoked"


class _TestVoiceClient(VoiceClient):
    """Subclass VoiceClient to avoid touching real audio devices in tests."""

    def __init__(self, persona: Dict[str, Any]) -> None:
        super().__init__(
            persona=persona,
            session_id="test-session",
            callbacks={},
            asr=_FakeASR("hello from user"),
            llm=_FakeLLM("assistant reply."),
            tts=_FakeTTS(),
            vad_stream=None,
        )

    def listen_once(self) -> tuple[str, float, float]:  # type: ignore[override]
        return "hello from user", 111.0, 1.0

    def monitor_barge_in(self) -> None:  # type: ignore[override]
        # No-op in tests
        return

    def stop_barge_in_monitor(self) -> None:  # type: ignore[override]
        return


def test_voice_client_run_turn_logs_and_updates_state(monkeypatch, tmp_path) -> None:
    persona = {"name": "Customer", "system_prompt": "Be concise."}
    vc = _TestVoiceClient(persona=persona)
    logger_obj = _FakeLogger()

    class DummyFeedback:
        @staticmethod
        def evaluate(state: ConversationState) -> str:
            return f"turns={len(state.turns)}"

    vc.run(max_turns=1, logger_obj=logger_obj, feedback=DummyFeedback)

    assert len(logger_obj.rows) == 1
    assert logger_obj.rows[0]["persona"] == "Customer"
    assert len(vc.state.turns) == 2  # user + assistant


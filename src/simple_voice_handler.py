"""
Simple Voice Handler for manual recording control.

Provides start/stop recording interface for the Streamlit UI.
"""
import queue
import re
import time
from typing import Any, Callable, Dict, Optional

import numpy as np
import sounddevice as sd
from loguru import logger

from .asr_module import ASRClient, pcm16_to_wav_bytes
from .llm_module import LLMClient
from .state_manager import ConversationState
from .tts_module import KokoroTTSClient


class SimpleVoiceHandler:
    """
    Simplified voice handler with manual recording controls.

    Dependencies (ASR, LLM, TTS) can be injected for testing.
    """

    def __init__(
        self,
        persona: Dict[str, Any],
        callbacks: Optional[Dict[str, Callable]] = None,
        asr_client: Optional[ASRClient] = None,
        llm_client: Optional[LLMClient] = None,
        tts_client: Optional[KokoroTTSClient] = None,
    ) -> None:
        self.persona = persona
        self.callbacks: Dict[str, Callable] = callbacks or {}

        self.asr_client = asr_client or ASRClient()
        self.llm_client = llm_client or LLMClient()
        self.tts_client = tts_client or KokoroTTSClient()

        self.state = ConversationState(
            session_id="streamlit_session",
            persona_name=persona.get("name", "Assistant"),
        )

        self.sample_rate = 16000
        self.audio_buffer: "queue.Queue" = queue.Queue()
        self.stream: Optional[sd.InputStream] = None
        self.is_recording = False

    def _audio_callback(self, indata, frames, time_info, status) -> None:  # type: ignore[override]
        if self.is_recording:
            self.audio_buffer.put(indata.copy())

    def start_recording(self) -> None:
        """Start recording audio."""
        try:
            while not self.audio_buffer.empty():
                self.audio_buffer.get()

            self.is_recording = True
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                callback=self._audio_callback,
            )
            self.stream.start()

            if self.callbacks.get("status"):
                self.callbacks["status"]("🎤 Recording...")

            logger.info("Recording started")
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            if self.callbacks.get("error"):
                self.callbacks["error"](str(e))

    def stop_recording(self) -> None:
        """Stop recording audio."""
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        logger.info("Recording stopped")

    def _emit(self, key: str, *args, **kwargs) -> None:
        cb = self.callbacks.get(key)
        if cb:
            try:
                cb(*args, **kwargs)
            except Exception:
                pass

    def process_voice_input(self) -> Dict[str, Any]:
        """
        Process recorded audio through ASR -> LLM -> TTS pipeline.

        Returns a metrics dict, or a dict with an ``error`` key on failure.
        """
        metrics: Dict[str, Any] = {}
        start_total = time.time()

        try:
            self.stop_recording()

            self._emit("status", "🎯 Transcribing...")

            audio_chunks = []
            while not self.audio_buffer.empty():
                audio_chunks.append(self.audio_buffer.get())

            if not audio_chunks:
                return {"error": "No audio recorded"}

            audio_data = np.concatenate(audio_chunks, axis=0)

            pcm_bytes = audio_data.flatten().tobytes()
            wav_bytes = pcm16_to_wav_bytes(pcm_bytes, self.sample_rate)
            user_text, asr_ms = self.asr_client.transcribe_wav_bytes(wav_bytes)
            metrics["asr_ms"] = asr_ms

            if not user_text or not user_text.strip():
                return {"error": "No speech detected"}

            logger.info(f"User said: {user_text}")

            self.state.add_turn("user", user_text)
            self._emit("user_text", user_text)

            self._emit("status", "🤖 Thinking...")
            self._emit("llm_start", user_text)

            start_llm = time.time()
            messages = self.state.as_messages(
                self.persona.get("system_prompt", "You are a helpful assistant."),
            )

            assistant_text, llm_ms, usage = self.llm_client.complete(messages)
            _ = usage  # usage is optional; retained for future metrics
            metrics["llm_ms"] = llm_ms

            if not assistant_text.strip():
                return {"error": "No response generated"}

            logger.info(f"Assistant said: {assistant_text}")

            self.state.add_turn("assistant", assistant_text)
            self._emit("assistant_text", assistant_text)

            self._emit("status", "🔊 Speaking...")

            start_tts = time.time()
            sentences = re.split(r"(?<=[.!?])\s+", assistant_text)
            for sentence in sentences:
                if sentence.strip():
                    wav_bytes = self.tts_client.synthesize_sentence(sentence)
                    self.tts_client.playback.play_wav(wav_bytes)
            tts_ms = (time.time() - start_tts) * 1000
            metrics["tts_ms"] = tts_ms

            total_ms = (time.time() - start_total) * 1000
            metrics["total_ms"] = total_ms

            logger.info(
                f"Metrics: ASR={asr_ms:.0f}ms, LLM={llm_ms:.0f}ms, "
                f"TTS={tts_ms:.0f}ms, Total={total_ms:.0f}ms",
            )

            self._emit("metrics", metrics)

            return metrics

        except Exception as e:
            logger.error(f"Error processing voice input: {e}")
            self._emit("error", str(e))
            return {"error": str(e)}

    def reset_conversation(self) -> None:
        """Reset conversation history."""
        self.state.turns = []
        logger.info("Conversation reset")

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.tts_client.playback.stop()
        logger.info("Cleanup complete")

import io
import os
import time
import wave
from typing import Callable, Iterable, Optional

import numpy as np
import simpleaudio as sa
from kokoro import KPipeline


def _read_wav_params(wav_bytes: bytes):
    bio = io.BytesIO()
    with wave.open(bio, "rb") as wf:
        params = wf.getparams()
        frames = wf.readframes(wf.getnframes())
        return params, frames


class PlaybackController:
    """
    Handle playback of WAV audio, including an interruptible variant for barge-in.
    """

    def __init__(self) -> None:
        self._current: Optional[sa.PlayObject] = None

    def stop(self) -> None:
        if self._current is not None:
            try:
                self._current.stop()
            except Exception:
                pass
            self._current = None

    def play_wav(self, wav_bytes: bytes) -> None:
        params, frames = _read_wav_params(wav_bytes)
        play = sa.play_buffer(frames, params.nchannels, params.sampwidth, params.framerate)
        self._current = play
        play.wait_done()
        self._current = None

    def play_wav_interruptible(self, wav_bytes: bytes, stop_flag: Callable[[], bool]) -> None:
        params, frames = _read_wav_params(wav_bytes)
        play = sa.play_buffer(frames, params.nchannels, params.sampwidth, params.framerate)
        self._current = play
        try:
            while play.is_playing():
                if stop_flag():
                    try:
                        play.stop()
                    except Exception:
                        pass
                    break
                time.sleep(0.02)
        finally:
            self._current = None


class KokoroTTSClient:
    """
    Text-to-speech client backed by Kokoro, with optional macOS `say` fallback.
    """

    def __init__(self) -> None:
        self.voice = os.getenv("KOKORO_VOICE", "af_sky")
        self.lang_code = os.getenv("KOKORO_LANG_CODE", "a")  # 'a' = American English
        self.sample_rate = 24000  # Kokoro outputs at 24kHz
        self.allow_fallback = os.getenv("ALLOW_FALLBACK_TTS", "0") == "1"
        self.playback = PlaybackController()

        try:
            self.pipeline = KPipeline(lang_code=self.lang_code)
            self.use_kokoro = True
        except Exception as e:
            print(f"Warning: Failed to initialize Kokoro pipeline: {e}")
            self.pipeline = None
            self.use_kokoro = False

    def _synthesize_with_kokoro(self, text: str) -> Optional[bytes]:
        if not (self.use_kokoro and self.pipeline):
            return None
        generator = self.pipeline(text, voice=self.voice, speed=1.0)
        audio_chunks = []
        for _, _, audio in generator:
            audio_chunks.append(audio)
        if not audio_chunks:
            return None

        full_audio = np.concatenate(audio_chunks)
        audio_int16 = (full_audio * 32767).astype(np.int16)

        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())
        wav_io.seek(0)
        return wav_io.read()

    def _fallback_tts(self, text: str) -> Optional[bytes]:
        if not self.allow_fallback:
            return None
        import subprocess

        tmp = text.replace("\n", " ")
        subprocess.run(["say", tmp])
        wav = io.BytesIO()
        with wave.open(wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(b"\x00" * int(self.sample_rate * 0.2) * 2)
        return wav.getvalue()

    def synthesize_sentence(self, text: str) -> bytes:
        """
        Convert a sentence of text into WAV bytes.

        Preference order:
        1. Kokoro pipeline (when available)
        2. Optional OS-level fallback (macOS `say`) with dummy audio padding
        """
        try:
            wav = self._synthesize_with_kokoro(text)
            if wav is not None:
                return wav
        except Exception as e:
            print(f"Kokoro TTS error: {e}")
            if not self.allow_fallback:
                raise

        wav = self._fallback_tts(text)
        if wav is not None:
            return wav

        raise RuntimeError("Kokoro TTS not configured and fallback disabled")

    def speak_sentences(self, sentences: Iterable[str], stop_flag: Callable[[], bool]) -> float:
        """
        Speak a sequence of sentences, honoring a stop flag for barge-in.
        """
        t0 = time.perf_counter()
        for s in sentences:
            if stop_flag():
                break
            wav = self.synthesize_sentence(s)
            if stop_flag():
                break
            self.playback.play_wav_interruptible(wav, stop_flag)
        return (time.perf_counter() - t0) * 1000

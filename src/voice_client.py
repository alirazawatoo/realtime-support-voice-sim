import os
import queue
import re
import threading
import time
from typing import Callable, Dict, Iterable, List, Optional

import sounddevice as sd
from loguru import logger

from .asr_module import ASRClient, VADStream
from .llm_module import LLMClient, approx_tokens
from .state_manager import ConversationState
from .tts_module import KokoroTTSClient


class AudioCapture:
    def __init__(self, sample_rate: int = 16000, frame_ms: int = 30) -> None:
        self.sample_rate = sample_rate
        self.blocksize = int(sample_rate * (frame_ms / 1000.0))
        self.q: "queue.Queue" = queue.Queue()
        self.stream: Optional[sd.InputStream] = None

    def _cb(self, indata, frames, time_info, status) -> None:  # type: ignore[override]
        self.q.put(indata.copy())

    def start(self) -> None:
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.blocksize,
            callback=self._cb,
        )
        self.stream.start()

    def read(self):
        return self.q.get()


def split_sentences(
    text_stream: Iterable[str],
    stop_flag: Callable[[], bool] = lambda: False,
    on_partial: Optional[Callable[[str], None]] = None,
) -> Iterable[str]:
    buf = ""
    for tok in text_stream:
        if stop_flag():
            break
        if on_partial:
            on_partial(tok)
        buf += tok
        while True:
            m = re.search(r"([\s\S]*?[\.\!\?])\s", buf)
            if not m:
                break
            s = m.group(1)
            yield s
            buf = buf[len(s) :].lstrip()
    if not stop_flag() and buf.strip():
        yield buf.strip()


class VoiceClient:
    """
    Full duplex voice client for CLI use, orchestrating ASR → LLM → TTS with barge-in.
    """

    def __init__(
        self,
        persona: Dict,
        session_id: str,
        callbacks: Optional[Dict[str, Callable]] = None,
        asr: Optional[ASRClient] = None,
        llm: Optional[LLMClient] = None,
        tts: Optional[KokoroTTSClient] = None,
        vad_stream: Optional[VADStream] = None,
    ) -> None:
        self.sample_rate = 16000
        self.persona = persona
        self.state = ConversationState(session_id=session_id, persona_name=persona.get("name", "Customer"))
        self.asr = asr or ASRClient(sample_rate=self.sample_rate)
        self.llm = llm or LLMClient()
        self.tts = tts or KokoroTTSClient()
        self.vad_stream = vad_stream or VADStream(sample_rate=self.sample_rate)
        self.barge_in_flag = threading.Event()
        self.stop_event = threading.Event()
        self.callbacks: Dict[str, Callable] = callbacks or {}

    def emit(self, name: str, *args, **kwargs) -> None:
        cb = self.callbacks.get(name)
        if cb:
            try:
                cb(*args, **kwargs)
            except Exception:
                pass

    def start(self) -> None:
        self.vad_stream.start()

    def listen_once(self) -> tuple[str, float, float]:
        partial_last = [0.0]

        def on_partial(text: str) -> None:
            now = time.perf_counter()
            if (now - partial_last[0]) * 1000 >= 400:
                print(f"ASR partial: {text}")
                partial_last[0] = now
                self.emit("asr_partial", text)

        final_text, asr_ms, asr_secs = self.asr.streaming_listen(self.vad_stream, on_partial=on_partial)
        self.emit("asr_final", final_text)
        return final_text, asr_ms, asr_secs

    def monitor_barge_in(self) -> None:
        self.barge_in_flag.clear()

        def run() -> None:
            vad = self.vad_stream.vad
            frame_bytes = self.vad_stream.frame_bytes
            streak = 0
            threshold_frames = 5  # ~150ms at 30ms per frame
            while not self.barge_in_flag.is_set():
                data = self.vad_stream.stream.read(int(frame_bytes / 2))[0].tobytes()  # type: ignore[union-attr]
                if len(data) < frame_bytes:
                    continue
                if vad.is_speech(data, self.sample_rate):
                    streak += 1
                else:
                    streak = 0
                if streak >= threshold_frames:
                    self.barge_in_flag.set()
                    break

        th = threading.Thread(target=run, daemon=True)
        th.start()

    def stop_barge_in_monitor(self) -> None:
        self.barge_in_flag.set()

    def run_turn(self, system_prompt: str, turn_idx: int, logger_obj, live_hints=None) -> None:
        logger.info("Listening...")
        self.emit("status", "Listening")
        user_text, asr_ms, asr_secs = self.listen_once()
        print(f"You: {user_text}")
        self.state.add_turn("user", user_text)

        msgs = self.state.as_messages(system_prompt)
        stream, _ = self.llm.stream_chat(msgs)
        t0_llm = time.perf_counter()
        llm_done_time = [None]

        def token_stream_with_done():
            for tok in stream:
                yield tok
            llm_done_time[0] = time.perf_counter()

        self.monitor_barge_in()
        print("Customer (streaming): ", end="", flush=True)
        output_sents: List[str] = []

        def on_llm_partial(tok: str) -> None:
            print(tok, end="", flush=True)
            self.emit("llm_partial", tok)

        sentences = split_sentences(
            token_stream_with_done(),
            stop_flag=lambda: self.barge_in_flag.is_set(),
            on_partial=on_llm_partial,
        )

        def sentences_with_capture():
            for s in sentences:
                output_sents.append(s)
                self.emit("assistant_sentence", s)
                yield s

        def stop_flag() -> bool:
            return self.barge_in_flag.is_set()

        self.emit("status", "Speaking")
        tts_ms = self.tts.speak_sentences(sentences_with_capture(), stop_flag)
        print("")
        self.stop_barge_in_monitor()
        end_ref = llm_done_time[0] or time.perf_counter()
        llm_ms = (end_ref - t0_llm) * 1000

        output_text = " ".join(output_sents).strip()
        self.state.add_turn("assistant", output_text)
        if output_text:
            print(f"Customer (final): {output_text}")
            self.emit("assistant_final", output_text)

        cost_est = None
        tokens_in = approx_tokens(user_text + "\n" + system_prompt)
        tokens_out = approx_tokens(output_text)
        try:
            price_in = float(os.getenv("LLM_PRICE_IN_PER_1K", "0") or 0)
            price_out = float(os.getenv("LLM_PRICE_OUT_PER_1K", "0") or 0)
            cost_est = (tokens_in / 1000.0) * price_in + (tokens_out / 1000.0) * price_out
        except Exception:
            cost_est = None
        total_ms = asr_ms + llm_ms + tts_ms
        logger_obj.log_turn(
            turn_idx,
            self.persona.get("name", "Customer"),
            asr_ms,
            llm_ms,
            tts_ms,
            total_ms,
            len(user_text),
            len(output_text),
            tokens_in,
            tokens_out,
            asr_secs,
            len(output_text),
            cost_est,
            None,
        )
        self.emit(
            "turn_metrics",
            {
                "turn": turn_idx,
                "asr_ms": asr_ms,
                "llm_ms": llm_ms,
                "tts_ms": tts_ms,
                "total_ms": total_ms,
                "input_chars": len(user_text),
                "output_chars": len(output_text),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "asr_secs": asr_secs,
                "tts_chars": len(output_text),
                "cost_est": cost_est,
            },
        )

    def run(self, max_turns: int, logger_obj, feedback=None) -> None:
        self.start()
        system_prompt = self.persona.get("system_prompt", "")
        for i in range(1, max_turns + 1):
            if self.stop_event.is_set():
                break
            self.run_turn(system_prompt, i, logger_obj)
        if feedback:
            fb = feedback.evaluate(self.state)
            print(fb)

    def request_stop(self) -> None:
        self.stop_event.set()
        self.barge_in_flag.set()

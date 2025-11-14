from pathlib import Path

from src.feedback import evaluate
from src.logger import LatencyLogger
from src.state_manager import ConversationState


def test_latency_logger_writes_header_and_row(tmp_path) -> None:
    path = tmp_path / "latency.csv"
    logger = LatencyLogger(str(path))

    logger.log_turn(
        turn=1,
        persona="Test",
        asr_ms=10.0,
        llm_ms=20.0,
        tts_ms=30.0,
        total_ms=60.0,
        input_chars=5,
        output_chars=10,
        llm_tokens_in=3,
        llm_tokens_out=4,
        asr_secs=1.2,
        tts_chars=10,
        cost_est_usd=0.001,
        error=None,
    )

    content = path.read_text().strip().splitlines()
    assert len(content) == 2
    header = content[0].split(",")
    assert header[0] == "ts"
    assert header[1] == "turn"


def test_feedback_evaluate_heuristics() -> None:
    state = ConversationState(session_id="s", persona_name="p")
    state.add_turn("user", "hello there")
    state.add_turn("assistant", "I understand, I am sorry about that.")
    state.add_turn("assistant", "Let us block your card and reissue a new one.")

    report = evaluate(state)
    assert "Greeting present" in report
    assert "Empathy detected" in report
    assert "Resolution provided" in report


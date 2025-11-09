"""
The page as Streamlit actually renders it, in text mode against the scripted
agent.

`AppTest` runs the script headlessly, so these catch what the pure-logic tests
in test_streamlit_app.py cannot: a widget that raises on first draw, an empty
state that never shows, a review button that does nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

testing = pytest.importorskip("streamlit.testing.v1")

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


@pytest.fixture
def text_mode(monkeypatch):
    monkeypatch.setenv("VOICE_IO", "text")
    monkeypatch.setenv("LLM_PROVIDER", "scripted")
    monkeypatch.setenv("STT_PROVIDER", "unavailable")
    monkeypatch.setenv("TTS_PROVIDER", "silent")


def _run(seed: str, monkeypatch) -> testing.AppTest:
    monkeypatch.setenv("SEED_DEMO_CALL", seed)
    app = testing.AppTest.from_file(APP, default_timeout=30)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def _page_text(app: testing.AppTest) -> str:
    return "\n".join(block.value for block in app.markdown)


def test_a_fresh_call_shows_the_empty_state(text_mode, monkeypatch) -> None:
    app = _run("0", monkeypatch)

    assert "No turns yet" in _page_text(app)
    assert len(app.chat_message) == 0


def test_the_seeded_call_renders_three_exchanges(text_mode, monkeypatch) -> None:
    app = _run("1", monkeypatch)

    assert [m.name for m in app.chat_message] == ["user", "assistant"] * 3
    assert "Turn latency" in _page_text(app)


def test_a_typed_line_becomes_a_new_turn(text_mode, monkeypatch) -> None:
    app = _run("0", monkeypatch)

    app.chat_input[0].set_value("I've lost my card.").run()

    assert not app.exception
    assert [m.name for m in app.chat_message] == ["user", "assistant"]


def test_the_review_says_when_it_is_only_a_heuristic(text_mode, monkeypatch) -> None:
    app = _run("1", monkeypatch)

    review = next(b for b in app.button if b.label == "Review this call")
    review.click().run()

    assert not app.exception
    assert any("keyword heuristic" in info.value for info in app.info)


def test_each_stand_in_is_labelled_in_words(text_mode, monkeypatch) -> None:
    # Status is carried by text as well as colour, so it survives a screen
    # reader and a colour-blind reviewer.
    app = _run("0", monkeypatch)
    sidebar = "\n".join(block.value for block in app.sidebar.markdown)

    # Offline: no recogniser, a scripted model that does answer, no speaker.
    assert sidebar.count(">stand-in<") == 2
    assert sidebar.count(">ready<") == 1

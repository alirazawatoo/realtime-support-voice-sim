"""The CLI entry point on a machine without the audio extras."""

from __future__ import annotations

import sys

import pytest

import main
from src.audio_devices import AudioUnavailable


def test_missing_audio_exits_with_the_fix_not_a_traceback(monkeypatch, tmp_path) -> None:
    class NoMicrophone:
        def __init__(self, **kwargs) -> None:
            raise AudioUnavailable("install the audio extras")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "VoiceClient", NoMicrophone)
    monkeypatch.setattr(main, "init_logger", lambda: None)
    monkeypatch.setattr(sys, "argv", ["main.py", "--turns", "1"])

    with pytest.raises(SystemExit) as exited:
        main.main()

    assert "install the audio extras" in str(exited.value)

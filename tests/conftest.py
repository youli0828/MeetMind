import socket

import pytest

from meetmind.demo import DemoSummarizer, demo_transcript


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Tests must not connect to the network")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def transcript():
    return demo_transcript()


@pytest.fixture
def minutes(transcript):
    return DemoSummarizer().summarize(transcript)

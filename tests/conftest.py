import socket
import threading

import pytest

from meetmind.demo import DemoSummarizer, demo_transcript


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    original_connect = socket.socket.connect
    original_socketpair = socket.socketpair
    local = threading.local()

    def blocked(sock, address):
        # Windows implements asyncio's internal socketpair through a loopback connection.
        if getattr(local, "in_socketpair", False):
            return original_connect(sock, address)
        raise AssertionError("Tests must not connect to the network")

    def socketpair(*args, **kwargs):
        local.in_socketpair = True
        try:
            return original_socketpair(*args, **kwargs)
        finally:
            local.in_socketpair = False

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "socketpair", socketpair)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def transcript():
    return demo_transcript()


@pytest.fixture
def minutes(transcript):
    return DemoSummarizer().summarize(transcript)

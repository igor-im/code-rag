"""Shared test fixtures."""

import pytest


@pytest.fixture(autouse=True)
def _no_torch_compile(monkeypatch):
    """Prevent torch.compile from tracing mock objects in all tests."""
    monkeypatch.setattr("torch.compile", lambda m, **kw: m)


@pytest.fixture(autouse=True)
def _no_load_dotenv(monkeypatch):
    """Prevent load_dotenv() from re-injecting .env values over monkeypatch."""
    monkeypatch.setattr("eval.cli.load_dotenv", lambda **kw: None)

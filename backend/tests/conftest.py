import os

import pytest

# Never send traces from test runs to a real Langfuse project, even if .env has keys.
os.environ["LANGFUSE_TRACING_ENABLED"] = "false"


@pytest.fixture(autouse=True)
def screen_stores_everything(monkeypatch):
    """Keep tests offline: the poisoning screen is an LLM call, so by default it
    approves every candidate. Tests of the screen itself override this."""
    from app.memory import screen

    monkeypatch.setattr(screen, "screen_fact", lambda candidate, source: screen.ScreenDecision(
        decision="STORE", category="none", reason="test default"))

import os

import pytest


@pytest.fixture(autouse=True)
def _default_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guarantee safe offline defaults in local runs and GitHub Actions CI."""
    if not os.getenv("OPENAI_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-ci-openai-key")
    if not os.getenv("ANTHROPIC_API_KEY"):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-ci-anthropic-key")
    monkeypatch.setenv("ORCHESTRATOR_STORAGE_MODE", "in_memory")
    monkeypatch.setenv("ORCHESTRATOR_LIVE_LLM", "false")
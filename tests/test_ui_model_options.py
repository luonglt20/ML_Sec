import pytest

from medqa_multiagent.config import RunConfig
from medqa_multiagent.ui.model_options import live_model_config


def _config() -> RunConfig:
    return RunConfig(model="deepseek-chat", temperature=0.0)


def test_flash_profile_enables_provider_thinking_without_mutating_base_config():
    base = _config()
    selected = live_model_config(base, "deepseek-v4-flash")
    assert selected.model == "deepseek-v4-flash"
    assert selected.enable_thinking is True
    assert base.model == "deepseek-chat"
    assert base.enable_thinking is False


def test_chat_profile_disables_thinking_and_rejects_unknown_model():
    selected = live_model_config(_config(), "deepseek-chat")
    assert selected.enable_thinking is False
    with pytest.raises(ValueError, match="Unsupported demo model"):
        live_model_config(_config(), "other-model")

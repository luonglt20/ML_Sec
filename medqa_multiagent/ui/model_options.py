"""Model profiles used by the live Streamlit security demo."""

from __future__ import annotations

from dataclasses import replace

from ..config import RunConfig


DEMO_MODELS = ("deepseek-chat", "deepseek-v4-flash")
MODEL_LABELS = {
    "deepseek-chat": "DeepSeek Chat",
    "deepseek-v4-flash": "DeepSeek V4 Flash (thinking)",
}


def live_model_config(base_config: RunConfig, model: str) -> RunConfig:
    """Apply the selected live-demo model without writing the config file.

    V4 Flash is always invoked in thinking mode so that its provider-returned
    ``reasoning_content`` is captured by the audit log.  Chat retains the
    standard non-thinking call mode.
    """
    if model not in DEMO_MODELS:
        raise ValueError(f"Unsupported demo model: {model!r}")
    return replace(
        base_config,
        model=model,
        enable_thinking=model == "deepseek-v4-flash",
    )

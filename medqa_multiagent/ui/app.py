"""Streamlit demo UI over the black-box `answer_question` entrypoint.

A minimal single-page app for manually testing/demoing the pipeline. Lets a
user type/paste a question stem and its A-D options, pick a variant (read
dynamically from `entrypoint.SUPPORTED_VARIANTS`, so the UI needs no code
change as V1-V4 land) and a config file, then run the question through
`answer_question` and see the predicted answer, explanation, invalid-
response flag, and -- when present -- the per-agent trace.

UI-only: adds no new answer-producing logic beyond what the CLI already
calls, and has no test-set access anywhere -- only ad hoc questions typed
in by the user. Reuses the exact same LLM client stack as the CLI
(`client_factory.build_llm_client`, i.e. `create_llm_client` +
`OnDiskLLMCache` + `LoggingLLMClient`), so demo runs are cached exactly like
CLI runs.

Run with:

    streamlit run medqa_multiagent/ui/app.py
"""

from __future__ import annotations

from typing import Any

import streamlit as st

# Absolute imports (not relative) because Streamlit executes this file
# directly, rather than importing it as part of the `medqa_multiagent`
# package -- relative imports would fail with "no known parent package".
# The package must be installed (e.g. `pip install -e .`) for these to
# resolve.
from medqa_multiagent.client_factory import DEFAULT_CACHE_DIR
from medqa_multiagent.entrypoint import SUPPORTED_VARIANTS
from medqa_multiagent.env_file import load_env_file
from medqa_multiagent.ui.logic import DEFAULT_CONFIG_PATH, OPTION_LETTERS, get_answer, load_config

DEFAULT_ENV_FILE = ".env"


def render_trace(trace: dict) -> None:
    """Render a per-agent trace dict generically, whatever keys it contains.

    Deliberately has no per-variant/per-key special-casing -- V1-V4 can add
    whatever trace keys they like (router query, retrieved passages,
    reasoner output, verifier decision, ...) without any UI code change.
    """
    if not trace:
        st.caption("No per-agent trace for this variant.")
        return
    for key, value in trace.items():
        st.markdown(f"**{key}**")
        _render_trace_value(value)


def _render_trace_value(value: Any) -> None:
    if isinstance(value, (dict, list)):
        st.json(value)
    else:
        st.write(value)


def main() -> None:
    st.set_page_config(page_title="MedQA Multi-Agent Demo", layout="wide")
    st.title("MedQA-USMLE Multi-Agent Demo")
    st.caption(
        "Manual testing/demo UI over the black-box `answer_question` "
        "entrypoint. No test-set access -- type or paste any ad hoc "
        "question below."
    )

    with st.sidebar:
        st.header("Run configuration")
        env_file_path = st.text_input(
            "Env file path",
            value=DEFAULT_ENV_FILE,
            help=(
                "Optional .env file of KEY=VALUE provider API keys "
                "(e.g. OPENAI_API_KEY=...). Loaded before every call; "
                "values already exported in the shell always take "
                "precedence."
            ),
        )
        load_env_file(env_file_path)

        config_path = st.text_input("Config file path", value=DEFAULT_CONFIG_PATH)
        cache_dir = st.text_input("Cache directory", value=DEFAULT_CACHE_DIR)

        config, config_error = load_config(config_path)
        if config_error or config is None:
            st.error(config_error)
            st.stop()
            return

        st.text_input("Model", value=config.model, disabled=True)
        st.number_input("Temperature", value=config.temperature, disabled=True)

    if not SUPPORTED_VARIANTS:
        st.error("No variants are implemented yet.")
        st.stop()
    variant = st.selectbox("Variant", SUPPORTED_VARIANTS)

    question = st.text_area("Question stem", height=150)

    st.subheader("Options")
    columns = st.columns(2)
    options = {}
    for index, letter in enumerate(OPTION_LETTERS):
        with columns[index % 2]:
            options[letter] = st.text_input(f"Option {letter}", key=f"option_{letter}")

    if st.button("Get answer", type="primary"):
        if not question.strip():
            st.warning("Enter a question stem first.")
        elif not all(text.strip() for text in options.values()):
            st.warning("Fill in all four options first.")
        else:
            with st.spinner("Calling the model..."):
                outcome = get_answer(question, options, variant, config, cache_dir)

            if outcome.error is not None or outcome.result is None:
                st.error(outcome.error)
            else:
                result = outcome.result
                st.subheader("Result")
                st.metric("Predicted answer", result.answer or "(none)")
                if not result.is_valid:
                    st.warning(
                        "The model's response did not parse to exactly one "
                        "valid option letter."
                    )

                st.markdown("**Explanation**")
                st.write(result.explanation)

                with st.expander("Per-agent trace"):
                    render_trace(result.trace)

                with st.expander("Raw model response"):
                    st.text(result.raw_response)


if __name__ == "__main__":
    main()

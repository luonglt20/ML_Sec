"""Streamlit-free helper logic for the demo UI.

Kept separate from `app.py` (which imports `streamlit`) so this module's
behavior -- config loading and the answer_question call itself, including
its error handling -- can be unit-tested without a `streamlit` install,
using the same `FakeLLMClient`-style test doubles as the rest of the
project.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple, Union

from ..client_factory import DEFAULT_CACHE_DIR, build_llm_client
from ..config import RunConfig
from ..entrypoint import AnswerResult, answer_question

#: Default config file the UI loads on first load -- the dev configuration
#: (`gpt-4o-mini`), matching this project's convention that demo runs
#: default to dev, never test-set, settings.
DEFAULT_CONFIG_PATH = "config.example.json"

#: MedQA-USMLE (US, 4-option) answer letters, fixed by the dataset shape.
OPTION_LETTERS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class AnswerOutcome:
    """Either a successful `AnswerResult` or a readable error message.

    Exactly one of `result`/`error` is set, so the UI layer never has to
    handle a raw exception/stack trace itself.
    """

    result: Optional[AnswerResult] = None
    error: Optional[str] = None


def load_config(path: Union[str, Path]) -> Tuple[Optional[RunConfig], Optional[str]]:
    """Load a `RunConfig` from `path`, returning `(config, None)` on success.

    Returns `(None, error_message)` -- never raises -- if the file is
    missing, isn't valid JSON, or doesn't parse to a valid `RunConfig`, so
    the UI can surface a readable message instead of a stack trace.
    """
    try:
        return RunConfig.from_json_file(path), None
    except (OSError, ValueError) as exc:
        return None, f"Could not load config file {str(path)!r}: {exc}"


def get_answer(
    question: str,
    options: Mapping[str, str],
    variant: str,
    config: RunConfig,
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
) -> AnswerOutcome:
    """Answer one question through the exact same client stack the CLI uses.

    Builds the standard cache- and logging-wrapped `LLMClient` for
    `config.model` (via `client_factory.build_llm_client`) and calls the
    black-box `answer_question` entrypoint. Never raises: a missing/invalid
    provider API key, an unsupported variant, or any other failure is
    reported back as `AnswerOutcome.error` instead of propagating as an
    exception, so the UI layer can render it directly.
    """
    try:
        client = build_llm_client(config, cache_dir)
    except Exception as exc:  # noqa: BLE001 - surfaced as a readable UI message
        return AnswerOutcome(error=str(exc))

    try:
        result = answer_question(question, options, variant, config, client)
    except NotImplementedError as exc:
        return AnswerOutcome(error=str(exc))
    except Exception as exc:  # noqa: BLE001 - surfaced as a readable UI message
        return AnswerOutcome(error=f"The LLM call failed: {exc}")

    return AnswerOutcome(result=result)

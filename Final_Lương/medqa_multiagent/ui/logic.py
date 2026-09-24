"""Streamlit-free helper logic for the demo UI.

Kept separate from `app.py` (which imports `streamlit`) so this module's
behavior -- config loading and the answer_question call itself, including
its error handling -- can be unit-tested without a `streamlit` install,
using the same `FakeLLMClient`-style test doubles as the rest of the
project.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple, Union

from ..client_factory import DEFAULT_CACHE_DIR, build_llm_client
from ..config import RunConfig
from ..entrypoint import AnswerResult, answer_question

#: Default config file the UI loads on first load -- the dev configuration
#: (`gpt-4o-mini`), matching this project's convention that demo runs
#: default to dev, never test-set, settings.
DEFAULT_CONFIG_PATH = "config.example.json"

#: MedQA-USMLE (US, 4-option) answer letters, fixed by the dataset shape.
OPTION_LETTERS = ("A", "B", "C", "D")

#: Default on-disk location for the UI's "saved questions" scratchpad --
#: under `.cache/` (already entirely git-ignored, like the LLM/embedding/
#: retrieval caches) since this is local, ad hoc convenience state, not a
#: project data artifact. Purely a UI convenience for re-testing a
#: particular hard question multiple times later; never read by the
#: CLI/harness, and holds no official test-set data (only whatever the
#: user typed/loaded into the form and chose to save).
DEFAULT_SAVED_QUESTIONS_PATH = ".cache/ui/saved_questions.jsonl"


@dataclass(frozen=True)
class AnswerOutcome:
    """Either a successful `AnswerResult` or a readable error message.

    Exactly one of `result`/`error` is set, so the UI layer never has to
    handle a raw exception/stack trace itself.
    """

    result: Optional[AnswerResult] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class SavedQuestion:
    """One question a user chose to keep around for repeated manual testing.

    Attributes:
        question: The question stem text.
        options: Mapping of option letter to option text.
        saved_at: ISO-8601 UTC timestamp of when this was saved.
        expected_answer: The dataset's recorded correct answer, if this was
            saved while showing an unedited "Load random example" question;
            `None` for a purely ad hoc question with no known answer.
        question_id: The original dataset question id, if `expected_answer`
            came from an unedited "Load random example" question; `None`
            otherwise (including for purely ad hoc questions).
    """

    question: str
    options: Dict[str, str]
    saved_at: str
    expected_answer: Optional[str] = None
    question_id: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SavedQuestion":
        return cls(
            question=data["question"],  # type: ignore[arg-type]
            options=dict(data["options"]),  # type: ignore[arg-type]
            saved_at=data["saved_at"],  # type: ignore[arg-type]
            expected_answer=data.get("expected_answer"),  # type: ignore[union-attr]
            question_id=data.get("question_id"),  # type: ignore[union-attr]
        )


def load_saved_questions(
    path: Union[str, Path] = DEFAULT_SAVED_QUESTIONS_PATH,
) -> List[SavedQuestion]:
    """Read back every question saved so far, oldest first.

    Returns an empty list -- never raises -- if `path` doesn't exist yet
    (the common case before anything has ever been saved).
    """
    path = Path(path)
    if not path.exists():
        return []
    saved: List[SavedQuestion] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                saved.append(SavedQuestion.from_dict(json.loads(line)))
    return saved


def save_question(
    question: str,
    options: Mapping[str, str],
    expected_answer: Optional[str] = None,
    path: Union[str, Path] = DEFAULT_SAVED_QUESTIONS_PATH,
    question_id: Optional[str] = None,
) -> SavedQuestion:
    """Append one question/options pair to the on-disk saved-questions file.

    A thin, ad hoc convenience for re-testing a question that turned out to
    be hard multiple times (e.g. across variants, or after a prompt tweak)
    without having to retype it -- appends only; never deduplicates, since
    the user may deliberately want to save near-duplicate variations.
    Creates parent directories as needed.
    """
    saved = SavedQuestion(
        question=question,
        options=dict(options),
        saved_at=datetime.now(timezone.utc).isoformat(),
        expected_answer=expected_answer,
        question_id=question_id,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(saved.to_dict(), ensure_ascii=True))
        fh.write("\n")
    return saved


def delete_saved_question(
    index: int, path: Union[str, Path] = DEFAULT_SAVED_QUESTIONS_PATH
) -> List[SavedQuestion]:
    """Remove the saved question at `index` (as returned by `load_saved_questions`).

    Rewrites the file with every other saved question, preserving order,
    and returns the resulting (post-deletion) list. Raises `IndexError` if
    `index` is out of range -- there's nothing sensible to delete.
    """
    saved = load_saved_questions(path)
    if not 0 <= index < len(saved):
        raise IndexError(f"No saved question at index {index}; have {len(saved)}.")
    remaining = saved[:index] + saved[index + 1 :]
    _write_saved_questions(remaining, path)
    return remaining


def clear_saved_questions(path: Union[str, Path] = DEFAULT_SAVED_QUESTIONS_PATH) -> None:
    """Delete every saved question, e.g. to clear out stale demo scratch data."""
    _write_saved_questions([], path)


def _write_saved_questions(saved: List[SavedQuestion], path: Union[str, Path]) -> None:
    path = Path(path)
    if not saved:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for item in saved:
            fh.write(json.dumps(item.to_dict(), ensure_ascii=True))
            fh.write("\n")


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

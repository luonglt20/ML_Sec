"""medqa_multiagent: MedQA-USMLE multi-agent evaluation harness.

Exposes:

- The pure-logic foundation shared by every pipeline variant (V0-V4): run
  configuration, deterministic dev/test sampling, the shared strict-regex
  output parser, and cache-key derivation. No LLM or network calls.
- The V0 Direct-LLM baseline vertical slice: a provider-abstracting LLM
  client interface, an on-disk cache wrapping it, the single black-box
  entrypoint (`answer_question`) and its CLI wrapper, and a durable
  prediction/trace file writer.
"""

from .cache import OnDiskLLMCache
from .cache_key import compute_cache_key
from .client_factory import DEFAULT_CACHE_DIR, build_llm_client
from .config import RunConfig
from .data import Question, load_questions
from .entrypoint import AnswerResult, answer_question
from .env_file import load_env_file
from .llm_client import (
    LLMClient,
    LLMResponse,
    LoggingLLMClient,
    OpenAICompatibleClient,
    UnknownModelError,
    create_llm_client,
)
from .official_eval import sample_official_test_set
from .parsing import DEFAULT_VALID_OPTIONS, ParsedAnswer, parse_final_answer, strip_final_answer
from .pipeline import run_dev_evaluation
from .records import PredictionRecord, read_prediction_records, write_prediction_records
from .sampling import deterministic_sample, sample_dev_set

__all__ = [
    "RunConfig",
    "ParsedAnswer",
    "DEFAULT_VALID_OPTIONS",
    "parse_final_answer",
    "strip_final_answer",
    "deterministic_sample",
    "sample_dev_set",
    "sample_official_test_set",
    "compute_cache_key",
    "Question",
    "load_questions",
    "LLMClient",
    "LLMResponse",
    "LoggingLLMClient",
    "OpenAICompatibleClient",
    "UnknownModelError",
    "create_llm_client",
    "OnDiskLLMCache",
    "DEFAULT_CACHE_DIR",
    "build_llm_client",
    "load_env_file",
    "AnswerResult",
    "answer_question",
    "run_dev_evaluation",
    "PredictionRecord",
    "write_prediction_records",
    "read_prediction_records",
]

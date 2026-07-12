# medqa-multiagent

MedQA-USMLE multi-agent evaluation harness — a V0-V4 variant ladder
(Direct LLM → RAG-only → Multi-agent → Full system → Full system minus
Verifier) evaluated with shared, deterministic scoring and statistics.

See `SPEC.md` and `DESIGN.md` for the full design and requirements.

## Status

Work in progress, built ticket by ticket. Currently implemented:

- **Foundational config, sampling, and output-parsing utilities**
  (`medqa_multiagent/config.py`, `sampling.py`, `official_eval.py`,
  `parsing.py`, `cache_key.py`) — pure logic, no LLM or network calls.
- **V0 Direct-LLM baseline via the black-box entrypoint**
  (`medqa_multiagent/llm_client.py`, `cache.py`, `prompts.py`,
  `entrypoint.py`, `pipeline.py`, `records.py`, `cli.py`) — a
  provider-abstracting LLM client (OpenAI `gpt-4o-mini` for development,
  DeepSeek `deepseek-chat` for reported evaluation, selected purely by
  `RunConfig.model`), an on-disk cache keyed via #1's cache-key logic, the
  single stable `answer_question(question, options, variant, config,
  client)` entrypoint, a thin CLI wrapper, and a durable JSONL
  prediction/trace file writer.

## Data

`data/dev.jsonl` and `data/test.jsonl` hold the official MedQA-USMLE (US,
4-option) `dev`/`test` splits, one JSON object per line:
`{"question_id", "question", "options": {"A": ..., ...}, "answer"}`.
Sourced from the public `GBaker/MedQA-USMLE-4-options-hf` mirror of the
original Jin et al. (2020) release (CC-BY-4.0); see
`scripts/download_medqa.py` to regenerate them.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

## Running the CLI

Set the provider API key for whichever model your `RunConfig` names
(`OPENAI_API_KEY` for `gpt-4o-mini`, `DEEPSEEK_API_KEY` for
`deepseek-chat`), then:

```bash
# Answer a single question through the black-box entrypoint.
medqa-multiagent answer \
  --config config.json \
  --question "..." \
  --options '{"A": "...", "B": "...", "C": "...", "D": "..."}'

# Run this run's configured dev sample end-to-end and write a
# prediction/trace file (one record per question).
medqa-multiagent run \
  --config config.json \
  --data data/dev.jsonl \
  --output predictions/v0_dev.jsonl
```

`config.json` is a `RunConfig` (see `medqa_multiagent/config.py`);
`config.example.json` is a ready-to-copy starting point.

Every LLM call is cached on disk (default `.cache/llm/`, override with
`--cache-dir`) and logged with its model, temperature, and exact rendered
prompt, so re-running the same invocation spends no additional API cost
and yields identical predictions.

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
- **Streamlit demo UI** (`medqa_multiagent/ui/`) — a manual
  testing/demoing convenience over the same black-box `answer_question`
  entrypoint, with no new answer-producing logic and no test-set access.

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
`deepseek-chat`). Either export it directly:

```bash
export OPENAI_API_KEY=sk-...
```

or put it in a `.env` file instead -- copy `.env.example` to `.env` and
fill in the key(s) you need:

```bash
cp .env.example .env
# then edit .env: OPENAI_API_KEY=sk-...
```

Both the CLI (default `--env-file .env`) and the demo UI load `.env`
automatically before every call; a value already exported in your shell
always takes precedence over the `.env` entry of the same name, and
`.env` is git-ignored so it's safe to keep real keys there.

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
and yields identical predictions. Override the `.env` path (or opt out of
it) with `--env-file path/to/other.env`.

## Running the demo UI

A minimal Streamlit app (`medqa_multiagent/ui/app.py`) for manually
testing/demoing the pipeline: type/paste a question stem and its A-D
options, pick a variant (only ever the ones `entrypoint.SUPPORTED_VARIANTS`
currently reports as implemented) and a config file, and see the predicted
answer, explanation, invalid-response flag, and per-agent trace returned by
the same black-box `answer_question` entrypoint the CLI calls. There's no
test-set access anywhere in this UI -- only ad hoc questions you type in.

### Step by step

1. **Install the extra UI dependency** (one-time setup, in addition to the
   `Development` step above):

   ```bash
   .venv/bin/pip install -e ".[ui]"
   ```

2. **Set the provider API key** for whichever model your config names
   (`OPENAI_API_KEY` for `gpt-4o-mini`, `DEEPSEEK_API_KEY` for
   `deepseek-chat`). Easiest is a `.env` file, so you only do this once:

   ```bash
   cp .env.example .env
   # then edit .env: OPENAI_API_KEY=sk-...
   ```

   The app loads `.env` automatically on every run (the sidebar's "Env
   file path" field defaults to `.env`; point it elsewhere if needed). A
   value already exported in your shell always takes precedence, so
   `export OPENAI_API_KEY=sk-...` still works too if you'd rather not use
   a file.

3. **Launch the app**:

   ```bash
   .venv/bin/streamlit run medqa_multiagent/ui/app.py
   ```

   Streamlit prints a local URL (default `http://localhost:8501`); open it
   in a browser. It should open automatically in most setups.

4. **Confirm the run configuration** in the left sidebar:
   - "Config file path" defaults to `config.example.json` (the dev
     configuration, `gpt-4o-mini`). Point it at your own `config.json` to
     change the model/temperature the UI uses -- the resolved model and
     temperature are shown read-only right below the field, so you can
     confirm before running anything.
   - "Cache directory" defaults to `.cache/llm/`, the same on-disk cache
     the CLI uses. Leave it as-is to share cache hits with CLI runs, or
     point it elsewhere to isolate demo traffic.

5. **Pick a variant** from the dropdown (only variants
   `entrypoint.SUPPORTED_VARIANTS` currently reports as implemented are
   offered -- just `V0` for now).

6. **Type/paste a question stem** and fill in all four options (A-D) in
   the main panel. This is always an ad hoc question you type in -- the UI
   never samples from `data/dev.jsonl`/`data/test.jsonl`.

7. **Click "Get answer"**. On success you'll see:
   - the predicted answer letter,
   - a warning banner if the response didn't parse to exactly one valid
     option letter,
   - the explanation text,
   - an expandable "Per-agent trace" section (empty for `V0`, populated
     for later variants without any UI code change),
   - an expandable "Raw model response" section.

   If the configured model's provider API key isn't set (or is otherwise
   invalid), you'll see a readable error message in place of the result --
   never a raw stack trace.

8. **Re-submit the exact same question/options/variant/model** to confirm
   caching: the call should return instantly and show an identical
   answer/explanation, since it's served from the on-disk cache rather
   than spending a new API call (visible in the terminal running Streamlit
   -- no new request is logged).

Stop the app with `Ctrl+C` in the terminal it's running in.

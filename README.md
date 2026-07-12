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
- **V1 RAG-only variant** (`medqa_multiagent/rag/`) — the same one-call
  black-box entrypoint, now with the prompt augmented by the top-`k`
  MedCPT-embedded, FAISS-flat-indexed textbook passages retrieved for the
  question's raw text (see "RAG index" below to build the index this
  needs). Retrieved passages are recorded in the prediction/trace record's
  `trace.retrieved_passages`. Both the embedding and retrieval calls are
  on-disk cached, exactly like the LLM client.
- **V2 multi-agent variant (Router → Reasoner → Verifier)** — three LLM
  calls per question, reusing the same RAG module as V1. The Router
  formulates a retrieval query from the raw question (rather than
  retrieving on the question's raw text, unlike V1); the Reasoner produces
  a candidate answer/explanation grounded in the passages that query
  retrieved; the Verifier reviews that candidate exactly once and either
  approves it or overrides it with its own corrected answer/explanation --
  single-pass only, with no revision loop back to the Reasoner. The
  prediction/trace record's final answer is always the Verifier's
  decision, and `trace` records the Router's query (`router_query`), the
  retrieved passages (`retrieved_passages`), the Reasoner's candidate
  (`reasoner_candidate`), and the Verifier's decision (`verifier_decision`,
  `"approve"` or `"override"`).

## Data

`data/dev.jsonl` and `data/test.jsonl` hold the official MedQA-USMLE (US,
4-option) `dev`/`test` splits, one JSON object per line:
`{"question_id", "question", "options": {"A": ..., ...}, "answer"}`.
Sourced from the public `GBaker/MedQA-USMLE-4-options-hf` mirror of the
original Jin et al. (2020) release (CC-BY-4.0); see
`scripts/download_medqa.py` to regenerate them.

## RAG index (required for V1 and later RAG-retaining variants)

V1 (and every later variant that retains RAG) retrieves passages from a
local FAISS flat index built once, offline, by `scripts/build_rag_index.py`
-- exactly analogous to how `scripts/download_medqa.py` populates
`data/dev.jsonl`/`data/test.jsonl`. This is **not** done automatically;
run it yourself before using V1:

```bash
.venv/bin/pip install -e ".[rag]"   # faiss-cpu, transformers, torch, numpy
.venv/bin/python scripts/build_rag_index.py --config config.json
```

This downloads the `MedRAG/textbooks` corpus (18 medical textbooks, the
canonical retrieval corpus paired with this benchmark), chunks it at your
config's `rag_chunk_size` (~256 tokens by convention), embeds every chunk
with MedCPT (`ncbi/MedCPT-Article-Encoder`), and writes a FAISS flat index
plus passage metadata to your config's `rag_index_dir` (default
`data/rag_index/`). It's a genuinely expensive one-time step (~125,000
source rows, CPU-only MedCPT embedding, no GPU required but correspondingly
slow) -- use `--max-rows N` for a **development-only** smoke-test index
while iterating, never for the index actually reported on. See the
script's module docstring for full details.

Once built, nothing at pipeline run-time touches the network or re-does
this work: `rag.client_factory.build_retriever` just loads the index, and
both embedding and retrieval calls are on-disk cached (`.cache/embeddings/`,
`.cache/retrieval/` by default) exactly like the LLM client's cache.

If you only need V0, none of this is required -- `config.example.json`'s
`rag_index_dir` field has a default and V0 never reads it.

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

Pass `--variant V1` to either subcommand to run the RAG-only variant
instead of V0's default (requires the RAG index to already be built --
see "RAG index" above). `run`'s prediction/trace records then include a
`trace.retrieved_passages` field with each question's exact top-`k`
retrieved passages, and both embedding and retrieval calls are cached on
disk (`.cache/embeddings/`, `.cache/retrieval/` by default) just like the
LLM client -- override those locations with `--rag-index-dir`,
`--embedding-cache-dir`, and `--retrieval-cache-dir` if needed.

Pass `--variant V2` to run the 3-agent Router->Reasoner->Verifier
multi-agent pipeline (also requires the RAG index -- V2 reuses V1's RAG
module via the same retriever stack, so the same `--rag-index-dir`/
`--embedding-cache-dir`/`--retrieval-cache-dir` flags apply). Each of the
three agent calls is cached/logged independently; `run`'s prediction/trace
records include `trace.router_query`, `trace.retrieved_passages`,
`trace.reasoner_candidate`, and `trace.verifier_decision`, and the
recorded `predicted_answer`/`explanation` are always the Verifier's
decision, not the Reasoner's raw candidate.

## Running the demo UI

A minimal Streamlit app (`medqa_multiagent/ui/app.py`) for manually
testing/demoing the pipeline: type/paste a question stem and its A-D
options (or load a random one from the dev pool as a shortcut), pick a
variant (only ever the ones `entrypoint.SUPPORTED_VARIANTS` currently
reports as implemented) and a config file, and see the predicted answer,
explanation, invalid-response flag, and per-agent trace returned by the
same black-box `answer_question` entrypoint the CLI calls. There's no
official test-set access anywhere in this UI -- `data/test.jsonl` is never
read.

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
   - "Saved questions file" defaults to `.cache/ui/saved_questions.jsonl`
     -- see step 6b below for what it's for. Git-ignored, like the rest of
     `.cache/`.

5. **Pick a variant** from the dropdown (only variants
   `entrypoint.SUPPORTED_VARIANTS` currently reports as implemented are
   offered -- `V0`, `V1`, and `V2` so far). Picking `V1` or `V2` requires
   the RAG index to already be built (see "RAG index" above) -- if it
   isn't, you'll see a readable error instead of a stack trace.

6. **Type/paste a question stem** and fill in all four options (A-D) in
   the main panel, or click "🎲 Load random example" to fill them in
   automatically from a random question in `data/dev.jsonl` (a typing
   shortcut only -- edit the fields afterwards if you like). The official
   test split (`data/test.jsonl`) is never read by this UI. When a loaded
   example is showing unedited, an info box displays that question's
   dataset-recorded expected answer.

   **6b. Save/reload a question for repeated testing.** Found a question
   that's hard (e.g. the model keeps getting it wrong, or you want to
   compare variants/prompts on it) and want to run it again later without
   retyping it? Click "💾 Save this question" (next to "Get answer") to
   append the current question/options to a local, on-disk file (the
   "Saved questions file" path from the sidebar -- `.cache/ui/
   saved_questions.jsonl` by default). If it was an unedited loaded
   example, its dataset-recorded expected answer is saved alongside it too.
   Saved questions persist across app restarts (it's a plain file, not
   just in-memory session state) and show up in the "💾 Saved questions"
   dropdown next to "Load random example": pick one and click "📂 Load
   selected" to refill the form with it (including its expected answer,
   if it has one), "🗑️ Delete selected" to remove just that one, or
   "🧹 Clear all saved questions" to empty the whole file. This is purely
   a manual-testing convenience -- saved questions are never read by the
   CLI/evaluation harness.

7. **Click "Get answer"**. On success you'll see:
   - the predicted answer letter (shown side by side with the dataset's
     expected answer, plus a match/mismatch note, if you're still looking
     at an unedited loaded example),
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

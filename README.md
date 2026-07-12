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

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

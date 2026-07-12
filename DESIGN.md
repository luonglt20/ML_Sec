# MedQA-USMLE Multi-Agent System — Design Document

Status: **Design agreed, not yet implemented.**

> **Note on location:** This project is intended to live in its **own standalone repository**, independent of `MedAgentBench/` (which is an unrelated FHIR-EHR agent benchmark and must not be modified as part of this work). It currently lives at `MedAgentBench/medqa-multiagent/` only because no other project root was available in this workspace at design time. **Before implementation begins, move/re-init this directory as its own repo root** (e.g. `medqa-multiagent/`), separate from `MedAgentBench/`.

---

## 0. Foundational Scope Note

The original plan (see "Source Plan" section below) describes a MedQA-USMLE multiple-choice QA system. The checked-out `MedAgentBench` repository in this workspace is a **different**, unrelated benchmark (FHIR virtual-EHR agent tasks, not multiple-choice QA — confirmed by inspecting `MedAgentBench/data/medagentbench/test_data_v1.json`). This project builds the MedQA-USMLE system fresh, from scratch, and does not reuse or depend on any `MedAgentBench` code or data.

---

## 1. Project Motivation & Learning Objectives

(Unchanged from source plan.)

- Build a medical multi-agent LLM system that improves factual accuracy on MedQA-USMLE.
- Evaluate system design, not clinical deployment.
- Learning objectives: modular multi-agent architecture, reproducible evaluation pipeline, comparison vs. direct-LLM baseline, component-contribution analysis.

## 2. System Task

- **Input:** one MedQA-USMLE multiple-choice question (4 options, standard US split).
- **Output:** exactly one final answer option (letter) + a short explanation (**1-3 sentences**, prompt-enforced, not hard-truncated).

## 3. Models

- **One model used consistently across all agent roles and all variants within a given run** — architecture is the isolated variable under test, not model capability.
- **Development / debugging phase:** `gpt-4o-mini` (OpenAI).
- **Reported evaluation runs:** `deepseek-chat` (DeepSeek-V3) — chosen for cost efficiency; supports `temperature` normally (unlike `deepseek-reasoner`, which was considered and rejected because it ignores `temperature` and adds a separate reasoning-content field, conflicting with the determinism/simplicity goals below).
- **Temperature = 0** for every agent, every role, both phases — prioritizes reproducibility over generation diversity. Known caveat: provider-side determinism at temp=0 is not perfectly guaranteed; disclose as a limitation in the report.

## 4. Data & Evaluation Scale

- Dataset: MedQA-USMLE, standard US 4-option split (canonical test set = 1273 questions; official release also ships `train`/`dev`/`test` splits — use the official `dev` split directly rather than carving one out of `train`).
- **Dev set:** sampled from the official `dev` split. Size is a **config parameter** (not hardcoded) — default small for fast/cheap iteration (e.g. `N=30`), trivially scalable to 100-150 (or more) later purely via config, no code changes.
- **Official test-set touch:** a **small smoke test** (~20-30 questions, config-driven), sampled **once**, used only to prove the full pipeline runs end-to-end. Explicitly **not** treated as a statistically powered accuracy claim — report this honestly (wide CIs, low power, "pipeline validation" framing) rather than as a headline benchmark number.
- **Hard invariant:** the official test split is touched only for this one smoke-test pass. All debugging/tuning/prompt iteration happens exclusively on dev data. The long-term memory store (below) is built exclusively from dev data and frozen before any test-set run — no test-set leakage.
- Harness must be size-agnostic throughout: sample size N is always a runtime config value, never assumed/hardcoded, so it can be scaled up later (e.g. to satisfy future report requirements of 100-150+ questions) without redesign.

## 5. Architecture — Variant Ladder (V0–V4)

Custom, minimal Python orchestration (explicit state object passed between agent functions). **No** external multi-agent framework (LangGraph/AutoGen/CrewAI) — chosen for transparency, reproducibility, and precise control over exact prompts/call sequences.

| Variant | Definition | LLM calls/question |
|---|---|---|
| **V0** Direct LLM | Single LLM call, question → answer + explanation. No RAG, no agents, no memory. | 1 |
| **V1** RAG-only | Single LLM call, prompt augmented with top-3 retrieved textbook passages. Still one call, no separate agents, no memory. | 1 |
| **V2** Multi-agent, no memory | Full 3-agent pipeline: **Router** (formulates retrieval query) → **Reasoner** (produces candidate answer using retrieved context) → **Verifier** (checks/finalizes). RAG retained. No long-term case memory. | 3 |
| **V3** Full system | Same as V2, plus Reasoner also receives top-3 few-shot exemplars from the frozen long-term case-memory store. | 3 |
| **V4** Full system, no verifier (optional ablation) | Same as V3, but Verifier step is skipped — Reasoner's output passed straight through as final. | 2 |

Known limitation to note in report: V0→V2 comparison entangles "adds RAG" and "adds multi-agent workflow" together (V2 retains RAG from V1, by explicit decision), since V1 isn't in the formal paired-comparison set (see §8). This was a deliberate tradeoff, not an oversight.

No evolutionary optimization component (explicitly descoped as optional; note as future work in report).

## 6. RAG Module

- Corpus: **MedQA "textbooks" corpus** (18 medical textbooks, the canonical retrieval corpus paired with this exact benchmark in the literature).
- Embeddings: **MedCPT** (NCBI's biomedical query/article encoders) — free, local, domain-appropriate, standard choice in MedRAG-style setups.
- Vector store: **FAISS flat index** (exact nearest-neighbor search — corpus is small enough that approximate indexing isn't needed).
- Retrieval settings (fixed, for reproducibility and to keep cost/latency comparable across all questions): **top-k = 3** passages, **~256-token** chunks.

## 7. Memory

- **Short-term memory:** intra-question working state passed between Router → Reasoner → Verifier within a single pipeline run (retrieved passages, intermediate reasoning, verifier feedback). Resets every question. Intrinsic to the pipeline — cannot be meaningfully "removed" without breaking the multi-agent workflow itself.
- **Long-term memory:** a persistent case-memory store, built as follows:
  1. Run the pipeline (V2 configuration — no memory yet) once over the dev set during development.
  2. For any dev question where the final answer matches the known correct answer, store that run's actual `(question, reasoning trace, answer)` as a case-memory entry.
  3. Freeze the store (read-only) before any test-set evaluation — no online updates from test-set ground truth, ever.
  4. At inference time (V3 and V4 — both retain memory; V4 only removes the Verifier, see table above), the Reasoner retrieves the **top-k = 3** most similar past cases via embedding similarity, used as few-shot exemplars.
- Rationale for self-bootstrapped (vs. LLM-generated hindsight rationales, vs. answer-only memory): avoids hindsight bias in fabricated rationales, transfers the system's own genuine reasoning style, and is nearly free to build since a dev-set pass is needed anyway for tuning.

## 8. Verifier Behavior

- **Single-pass only.** Reviews the Reasoner's candidate answer + explanation exactly once; either approves it or overrides it with its own corrected answer/explanation. **No revision loop** back to the Reasoner.
- Chosen to keep per-question LLM call count fixed and deterministic per variant (clean cost/latency comparisons; V4 is simply "drop the Verifier call" with no other side effects).

## 9. Output Format & Invalid-Response Handling

- Prompted format convention (not provider-enforced JSON schema/function-calling): e.g. `Final Answer: <letter>` followed by a 1-3 sentence explanation.
- Parsing: **strict regex**, no fallback re-parse attempt. A response is "invalid" if it fails to yield exactly one valid option letter.
- Deliberate choice: preserves genuine variance in the required **Invalid Response Rate** metric across variants (provider-enforced schemas would drive this to ~0% everywhere and make the metric uninformative; a fallback reparse would blur what's actually being measured).

## 10. Evaluation Metrics & Statistics

- Accuracy = Correct / Total.
- Invalid Response Rate = Invalid / Total.
- Accuracy Gain = Accuracy(V3) − Accuracy(V0).
- Ablation comparison (see paired comparisons below).
- Win/Loss/Tie analysis, computed per pair.
- Bootstrap confidence intervals: percentile bootstrap, 10,000 resamples (standard, low-controversy default).
- McNemar test: exact variant for small discordant-pair counts (e.g. via `statsmodels.stats.contingency_tables.mcnemar`).
- Cost: **token-based estimation** — log prompt/completion token usage from API responses, multiply by published per-token pricing for the active model. Fully reproducible from our own logs (no dependency on provider billing dashboards).
- Latency: wall-clock timestamps around each LLM call and each full pipeline run.

### Paired comparisons (curated set of 5, each isolating one component's marginal effect)

| Pair | Isolates |
|---|---|
| V0 vs V1 | Effect of RAG alone |
| V0 vs V2 | Effect of multi-agent workflow (RAG already present — entangled, see §5 caveat) |
| V2 vs V3 | Effect of long-term memory |
| V3 vs V4 | Effect of the Verifier |
| V0 vs V3 | Headline total accuracy gain |

All 5 pairs get: accuracy delta, bootstrap CI, McNemar test, win/loss/tie breakdown.

## 11. Required Result Tables

- **Leaderboard table:** rows = V0-V4, columns = accuracy, invalid rate, cost, latency, (accuracy gain vs. V0 where applicable).
- **Paired comparison table:** the 5 curated pairs above, with accuracy delta / CI / McNemar p-value / win-loss-tie.
- **Error analysis table:** **DEFERRED** — content/structure not yet decided. (Leading candidate discussed: pipeline-stage failure attribution — categorize each incorrect V2/V3/V4 prediction by where it originated, e.g. Reasoner-wrong-Verifier-missed-it, Verifier-introduced-error, retrieval-irrelevant, memory-misleading-exemplar, invalid-format — supplemented with a few illustrative row-level qualitative examples. To be finalized before implementation of this table.)

## 12. Engineering / Implementation Requirements

- **Caching:** full on-disk cache for all LLM and embedding calls, keyed by a hash of exact request parameters (agent role, full prompt, model, temperature, retrieved-context hash). Safe given temperature=0 — a cache hit only occurs for bit-for-bit identical inputs. Purpose: cost/time control across repeated dev-set iteration (debugging, prompt tuning, building the case-memory store all require multiple passes over dev data).
- **Report format:** Markdown in the repo (e.g. `REPORT.md`), with tables/figures generated by evaluation scripts and checked in as artifacts (`.md`/`.csv`/image files). Convertible to PDF via Pandoc at the end if a polished submission copy is needed later.
- **Forward-compatibility for the final-project connection (attack/defense reuse):** expose exactly **one minimal, stable black-box entrypoint**, e.g. `answer_question(question: str, options: dict) -> {"answer": str, "explanation": str}`, plus a thin CLI wrapper. Deliberately **no** speculative attack-surface hooks (e.g. RAG-poisoning APIs, memory-injection APIs) built preemptively — the specific attack vectors for the follow-up project are unknown at this time, and guessing wrong would waste effort or bias the attack surface unrealistically. The modular internal design (separate Router/Reasoner/Verifier, separate RAG index, separate memory store) already makes internals inspectable/swappable if the next project needs deeper access.

## 13. Reproducibility Requirements (to report explicitly)

- Model(s) used per phase (dev vs. eval), exact names/versions.
- All prompts (verbatim, per agent role).
- Temperature (0, everywhere).
- Retrieval settings: RAG top-k=3 / ~256-token chunks; long-term memory top-k=3.
- Hardware/API: which provider endpoints, any local compute used for embeddings (MedCPT, FAISS).
- Random seed: default `42` (used for dev/test sampling and any tie-breaking) — low-stakes choice, changeable if needed.
- Runtime configuration: all sample sizes (dev N, smoke-test N), cache behavior, and model selection must be defined in a single config file/mechanism, not scattered magic numbers.

## 14. Minimum Acceptance Criteria

(Unchanged from source plan.) Runnable baseline and full system, official prediction files, evaluation results, at least one ablation (we have 5 curated pairs, exceeding minimum), and report.

## 15. Final-Project Connection

The submitted system will be reused as the target system for a later attack/defense project. Addressed via the minimal stable black-box entrypoint in §12 — no further speculative design added now.

---

## Open Items (deferred, to resolve before/during implementation)

1. **Error analysis table** — exact structure/dimensions not yet decided (see §11).
2. Concrete repo folder layout — not yet drafted.
3. Exact default sample sizes as literal starting numbers (e.g. dev=30, smoke=20) — placeholders, trivially changed via config.
4. Bootstrap resample count (10,000) and McNemar exact-test threshold — standard defaults chosen, not deeply debated.

## Decision Log (chronological, for traceability)

1. Confirmed target = build MedQA-USMLE system fresh; ignore `MedAgentBench` repo (unrelated FHIR benchmark).
2. New standalone repo, separate from `MedAgentBench`.
3. Single model across all roles/variants (isolates architecture as the variable under test).
4. Full 1273-question official test set will NOT be run; harness built to be capable of it, actual run is a small subset.
5. Official-style run = small smoke test (20-30 Q), not a powered accuracy claim.
6. Reported metrics/statistics: sample size made fully configurable; will scale to 100-150 later per report needs — resolved as an engineering requirement (config-driven N), not a fixed number.
7. Orchestration: custom minimal Python, no external multi-agent framework.
8. RAG corpus: MedQA textbooks corpus.
9. RAG embeddings/index: MedCPT + FAISS flat index.
10. Multi-agent pattern: linear pipeline (Router → Reasoner → Verifier), not debate/ensemble.
11. Memory semantics: short-term = intra-pipeline state; long-term = frozen cross-question case memory built from dev data.
12. Evolutionary optimization: skipped (explicitly optional in source plan).
13. Verifier behavior: single-pass, no revision loop.
14. Paired comparisons: curated set of 5 (not all 10 combinations).
15. Output format strictness: prompted format + strict regex, no provider-enforced schema, no fallback reparse (preserves invalid-rate metric variance).
16. Temperature: 0 everywhere.
17. Long-term memory content source: self-bootstrapped from verified-correct dev-set pipeline runs (not LLM-generated hindsight rationales, not answer-only).
18. Cost/latency measurement: token-based cost estimate + wall-clock latency, derived from our own logs.
19. Explanation length: soft prompt constraint (1-3 sentences), not hard-truncated.
20. Caching: full on-disk cache keyed by exact request params.
21. Report format: Markdown in repo.
22. Error analysis table: deferred (open item).
23. RAG retrieval settings: top-k=3, ~256-token chunks.
24. Variant ladder V0-V4: exact definitions locked (table in §5); V2 explicitly retains RAG.
25. Long-term memory retrieval settings: top-k=3 (parallels RAG).
26. Model choice: `gpt-4o-mini` for dev, `deepseek-chat` for reported evaluation runs.
27. Final-project forward-compatibility: minimal black-box entrypoint only, no speculative attack hooks.

---

## Source Plan (as originally provided, for reference)

1. **Project Motivation:** Build a medical multi-agent LLM system that improves factual accuracy on MedQA-USMLE. The project evaluates system design rather than clinical deployment.
2. **Learning Objectives:** Design a modular multi-agent architecture, implement a reproducible evaluation pipeline, compare against a direct LLM baseline, and analyze component contributions.
3. **System Task:** Input: one MedQA-USMLE multiple-choice question. Output: exactly one final answer option plus a short explanation.
4. **Required System Components:** (1) Direct LLM baseline, (2) Medical reasoning agent, (3) RAG module, (4) Short-term memory, (5) Long-term memory, (6) Multi-agent workflow (≥3 agents), (7) Optional evolutionary optimization.
5. **Benchmark Protocol:** Development: 100-150 public MedQA training/dev questions. Official evaluation: full MedQA-USMLE test set. Development data may be used only for debugging and tuning. Official test set is used once for final evaluation.
6. **Required System Variants:** V0 Direct LLM, V1 RAG-only, V2 Multi-agent without memory, V3 Full system, V4 Full system without verifier (optional).
7. **Evaluation Metrics:** Accuracy = Correct/Total; official Accuracy = Correct/1273; Invalid Response Rate; Accuracy Gain = Accuracy(V3) − Accuracy(V0); Ablation comparison; Win/Loss/Tie analysis; Bootstrap CI and McNemar test (recommended); Cost and latency.
8. **Required Result Tables:** Leaderboard table, paired comparison table, error analysis table.
9. **Implementation Requirements:** Runnable source code, documented repository structure, prediction files, evaluation script, and report.
10. **Reproducibility Requirements:** Report model, prompts, temperature, retrieval settings, hardware/API, random seed, and runtime configuration.
11. **Final-Project Connection:** The submitted system will be reused as the target system for the final attack/defense project.
12. **Minimum Acceptance Criteria:** Runnable baseline and full system, official prediction files, evaluation results, at least one ablation, and report.

**Verified Equations:**
- Accuracy = Correct Predictions / Total Number of Questions
- Official benchmark accuracy = Correct Predictions / Dataset size
- Invalid Response Rate = Invalid Responses / Total Number of Questions
- Accuracy Gain = Accuracy(V3) − Accuracy(V0)

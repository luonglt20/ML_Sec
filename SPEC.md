# Spec: MedQA-USMLE Multi-Agent System (V0–V4 Variant Ladder)

**Status:** Ready for agent
**Source design doc:** `medqa-multiagent/DESIGN.md`
**Suggested labels:** `ready-for-agent`

> Note: this spec could not be published to a project issue tracker automatically. See **Further Notes** for why, and what's needed to do so.

---

## Problem Statement

As the owner of this coursework project, I need to demonstrate — with evidence, not assertion — how much of a medical multi-agent LLM system's factual accuracy on MedQA-USMLE comes from *which* architectural decision (retrieval augmentation, decomposing reasoning into multiple agents, giving the system memory of past cases, adding a verification step). Right now I have a direct-LLM baseline and a pile of individually plausible-sounding architecture ideas, but no way to run them side-by-side, no shared scoring logic, and no statistically defensible way to say "component X contributed Y accuracy points." I also can't currently reproduce any given result later (no fixed config, no logging of exactly which model/prompt/settings produced it), and I have no cheap way to iterate without burning API budget re-generating identical responses on every debug pass.

Separately, whatever system comes out of this needs to be usable, later, as a fixed target by a different project (an attack/defense exercise) — without that project's author needing to understand this system's internals.

## Solution

Build one evaluation harness plus five increasingly-capable *variants* of the same underlying pipeline (V0 Direct LLM → V1 RAG-only → V2 Multi-agent → V3 Full system → V4 Full system minus Verifier), where each variant differs from its predecessor by exactly one architectural component. Run all variants over the same held-out questions, score them with shared, deterministic logic, and produce a leaderboard table and a curated set of statistically-backed paired comparisons that each isolate one component's marginal contribution. Every run is config-driven (model, sample size, seed, retrieval settings) and cached, so any prior result can be reproduced or re-derived without re-spending API cost. The whole system is reachable through a single, stable black-box function so it can be handed to the follow-up attack/defense project without exposing internals.

## User Stories

### Harness configuration & data
1. As a developer, I want to configure the dev-set sample size via a config value, so that I can iterate cheaply during development and scale up later without touching code.
2. As a developer, I want to configure the official test-set smoke-test sample size via the same config mechanism, so that I control the cost/time of the pipeline-validation run independently of the dev-set size.
3. As a developer, I want dev-set and test-set subsets sampled using a fixed random seed, so that "which questions did we evaluate on" is reproducible across runs.
4. As a developer, I want the harness to load MedQA-USMLE's official `dev` and `test` splits directly (rather than carving a dev set out of `train`), so that our numbers are comparable to published work using the same benchmark.
5. As a developer, I want the test split to be readable only through a distinctly-labeled "official evaluation" code path, so that I can't accidentally use test questions during tuning/debugging.

### Model & determinism
6. As a developer, I want to select which model (e.g. `gpt-4o-mini` for development, `deepseek-chat` for reported evaluation) powers every agent role via a single config value, so that switching phases doesn't require touching agent code.
7. As a developer, I want temperature fixed at 0 for every LLM call in every agent role, so that pipeline outputs are as deterministic as the provider allows.
8. As a developer, I want every LLM call to log its model name, temperature, and the exact rendered prompt, so that any past prediction can be traced back to precisely what produced it.

### V0 — Direct LLM baseline
9. As a developer, I want a Direct-LLM variant (V0) that answers a MedQA question with exactly one LLM call and no retrieval/agents/memory, so that I have a minimal reference point for every other variant's accuracy gain.
10. As a developer, I want V0's output to follow the same required output format as every other variant, so all variants can be scored by one shared parser.

### V1 — RAG-only
11. As a developer, I want a RAG-only variant (V1) that augments the single LLM call's prompt with retrieved textbook passages, so I can measure retrieval's isolated effect versus V0.
12. As a developer, I want the retrieval corpus to be the MedQA "textbooks" corpus, embedded with MedCPT and indexed in a local FAISS flat index, so retrieval is domain-appropriate and requires no external hosted service.
13. As a developer, I want retrieval to always return exactly top-3 passages of a fixed chunk size (~256 tokens), so that prompt size — and thus per-question cost — is consistent and comparable across every question and variant that uses RAG.
14. As a developer, I want retrieved passages recorded in the per-question trace, so I can audit whether retrieval found relevant content when investigating an error later.

### V2 — Multi-agent workflow (no memory)
15. As a developer, I want a 3-agent variant (V2) — Router, Reasoner, Verifier — that reuses the same RAG module as V1, so I can measure the isolated effect of decomposing reasoning into multiple specialized agent roles.
16. As a developer, I want the Router agent to formulate a retrieval query from the raw question rather than passing the question through verbatim, so retrieval quality isn't limited to whatever phrasing the question happens to use.
17. As a developer, I want the Reasoner agent to receive the Router's retrieved context and produce a candidate answer plus explanation, so its output is grounded in retrieved evidence.
18. As a developer, I want the Verifier agent to review the Reasoner's candidate answer exactly once, and either approve it or override it with its own corrected answer/explanation, so verification is a single-pass, cost-predictable step with no open-ended revision loop.
19. As a developer, I want every agent hand-off (Router's query, retrieved context, Reasoner's candidate answer, Verifier's decision) recorded in a structured per-question trace, so pipeline-stage error attribution is possible later without re-running anything.

### V3 — Full system (adds long-term memory)
20. As a developer, I want a long-term case-memory store built exclusively from dev-set pipeline runs whose final answer matched the known correct answer, so memory content reflects the system's own genuine successful reasoning rather than fabricated hindsight explanations.
21. As a developer, I want the case-memory store frozen (read-only) before any test-set evaluation, so no test-set ground truth can ever leak into the system's behavior.
22. As a developer, I want the Reasoner to retrieve the top-3 most similar past cases from the frozen case-memory store as few-shot exemplars, so it can reason by analogy to previously-solved, verified-correct cases.
23. As a developer, I want a full-system variant (V3) that is V2 plus this memory retrieval step, so I can measure memory's isolated marginal effect versus V2.

### V4 — Full system without Verifier (optional ablation)
24. As a developer, I want a variant (V4) identical to V3 but with the Verifier step skipped entirely (Reasoner's output passed straight through), so I can measure the Verifier's isolated marginal effect versus V3.

### Output parsing & format
25. As a developer, I want every variant's raw model output parsed by one shared strict-regex parser, so "is this response valid" is judged identically regardless of which variant produced it.
26. As a developer, I want a response counted as invalid whenever the parser can't extract exactly one valid option letter, with no fallback re-parse attempt, so the invalid-response-rate metric reflects genuine model/architecture behavior rather than parser leniency.
27. As a developer, I want every answer-producing prompt to instruct a 1-3 sentence explanation, so explanation length — and its contribution to token cost — stays roughly comparable across variants.

### Evaluation, metrics & statistics
28. As a developer, I want to compute accuracy, invalid response rate, and accuracy gain (`Accuracy(V3) - Accuracy(V0)`) for any completed evaluation run, so I can report the required headline numbers.
29. As a developer, I want a curated set of 5 paired comparisons — V0-V1, V0-V2, V2-V3, V3-V4, V0-V3 — each computed with a bootstrap confidence interval, an exact McNemar test, and a win/loss/tie breakdown, so accuracy differences are attributed to specific architectural components with statistical backing, not just eyeballed deltas.
30. As a developer, I want per-call token usage and wall-clock latency logged for every LLM call, so I can compute an estimated dollar cost and latency per variant purely from my own logs, without depending on a provider billing dashboard.
31. As a developer, I want a leaderboard table generated from a completed evaluation run, showing accuracy, invalid rate, cost, and latency for all 5 variants at a glance.
32. As a developer, I want a paired-comparison table generated from the 5 curated pairs' statistics, presented alongside the leaderboard.
33. As a developer, I want predictions, gold labels, and full per-question traces written to a durable prediction file for every run, so evaluation metrics can be re-derived later without re-calling any LLM.

### Reproducibility & cost control
34. As a developer, I want every LLM and embedding call cached on disk, keyed by a hash of its exact request parameters (role, full prompt, model, temperature, retrieved-context), so re-running an evaluation after a non-prompt code change doesn't re-spend API cost regenerating identical responses.
35. As a developer, I want the cache to produce a hit only for bit-for-bit identical requests, so a cached result never silently masks a real behavioral change I introduced.
36. As a developer, I want a single run configuration (model, temperature, sample sizes, retrieval top-k, memory top-k, seed) captured per run, so any prior run's exact conditions can be reconstructed later.
37. As a developer, I want the report to state the model, prompts, temperature, retrieval settings, hardware/API, random seed, and full runtime configuration for a given run, so a reader can either reproduce it or understand precisely why they can't.

### Black-box entrypoint (final-project connection)
38. As a downstream developer building the future attack/defense project, I want one stable function/CLI entrypoint that takes a question and its answer options and returns a final answer plus explanation, so I can integrate against this system without needing to understand its internal agent architecture.
39. As a downstream developer, I want that entrypoint's behavior to be selectable by variant (V0-V4) via configuration, so I can target a specific, documented configuration of the system in later experiments.

### Reporting & review
40. As a reviewer/grader, I want a Markdown report checked into the repo, containing the leaderboard table and paired-comparison table as generated artifacts, so I can verify the required deliverables without running any code myself.
41. As a reviewer/grader, I want the report to explicitly disclose that the official test-set run is a small smoke test (not the full 1273-question benchmark), so I don't misread it as a statistically powered accuracy claim.
42. As a reviewer/grader, I want the report to disclose the known V0→V2 entanglement (RAG and multi-agent workflow effects combined in that one comparison), so component-attribution claims are read with the appropriate caveat.

### Scope control
43. As a developer, I want the evolutionary-optimization component explicitly marked out of scope in both code and docs, so effort isn't split against a descoped, optional requirement.
44. As a developer, I want the error-analysis table intentionally left unimplemented (structure not yet decided), so I don't build a table whose design will be revisited and likely reworked.

### Demo UI (optional convenience, non-pipeline)
45. As a developer, I want a minimal Streamlit demo UI wrapping the black-box entrypoint, so I can manually test/demo the pipeline interactively in a browser, reusing the exact same cached LLM client stack as the CLI.
46. As a developer, I want the demo UI's variant selector to read `entrypoint.SUPPORTED_VARIANTS` live, so adding a new variant (V1-V4) requires no UI code change.
47. As a developer, I want the demo UI to have zero official test-set access and no dev/test evaluation-pool sampling of its own, so it can never be mistaken for, or misused as, part of the formal evaluation harness.

## Implementation Decisions

- **Single black-box seam.** The entire system is reachable through exactly one entrypoint — conceptually `answer_question(question, options, variant) -> {answer, explanation}` plus a thin CLI wrapper — which is also the primary seam used for testing (see Testing Decisions). All five variants (V0-V4) are configurations of the same underlying pipeline machinery, not five separate code paths.
- **Variant-as-configuration, not variant-as-branch.** A "variant" is a declarative selection of which components are active (RAG on/off, single-call vs. 3-agent workflow, long-term memory on/off, Verifier on/off). The pipeline reads this configuration and assembles the active components accordingly, rather than each variant being its own hand-written script.
- **Agent roles** (active only under the multi-agent configurations, V2-V4): **Router** (formulates a retrieval query from the raw question), **Reasoner** (produces a candidate answer + explanation from retrieved context, and — in V3/V4 — from retrieved case-memory exemplars), **Verifier** (single-pass review/override of the Reasoner's output; entirely skipped in V4). Each is a thin wrapper around the shared LLM-calling interface with a role-specific prompt template.
- **LLM access is behind a single client interface** used by every agent role, abstracting the underlying provider. Two concrete configurations exist: development (`gpt-4o-mini`, OpenAI) and reported-evaluation (`deepseek-chat`, DeepSeek). Selecting between them is a config change, not a code change. The interface must surface per-call token usage (for cost accounting) and must support temperature=0.
- **RAG module** consists of: a one-time corpus-indexing step (MedQA textbooks corpus → MedCPT embeddings → FAISS flat index) and a query-time retrieval interface returning a fixed top-k=3 passages of ~256-token chunks. No approximate indexing (flat/exact search only, corpus is small enough).
- **Long-term memory module** consists of: an offline "memory-building" pass that runs the pipeline (in its V2 configuration) over the dev set once, filters to verified-correct runs, and persists their `(question, reasoning trace, answer)` as case-memory entries; and a query-time retrieval interface returning the top-k=3 most similar cases by embedding similarity. The store has an explicit frozen/read-only state that must be set before any test-set run — this is a hard invariant, not a convention, and should be structurally enforced (e.g., the store refuses writes once frozen) rather than merely documented.
- **Short-term memory** is not a separate persisted module — it is the in-memory state object threaded through the Router → Reasoner → Verifier hand-off for a single question, and is implicitly always present whenever the multi-agent configuration is active (it cannot be independently toggled off without disabling the multi-agent workflow itself).
- **Output parsing** is a single shared, pure function used by every variant: given raw model text, it extracts exactly one option letter via strict regex (from the `Final Answer: <letter>` convention) or reports the response as invalid. No provider-enforced structured output/JSON schema is used anywhere, and no fallback re-parse attempt exists.
- **Caching layer** wraps both the LLM client interface and the embedding/retrieval interface. Cache key = hash of (agent role, fully-rendered prompt, model identifier, temperature, retrieved-context identifier where applicable). Cache is content-addressed and safe under temperature=0 (a hit requires bit-for-bit identical inputs).
- **Run configuration** is a single, explicit object/file per run capturing: model, temperature, dev/test sample sizes, random seed, RAG top-k and chunk size, memory top-k, cache location, and which variant(s) to execute. No sampling size, seed, or retrieval setting should be a hardcoded literal inside pipeline logic.
- **Prediction/trace file schema** — one record per (variant, question_id), containing at minimum: question id, variant identifier, predicted answer, explanation text, correct answer, correctness flag, invalid-response flag, token usage, latency, and the full per-agent intermediate trace (router query, retrieved passages, reasoner output, verifier decision) needed for later pipeline-stage error attribution.
- **Evaluation/statistics module** is decoupled from the pipeline entirely — it operates only on prediction/trace files, computing accuracy, invalid rate, accuracy gain, the 5 curated paired comparisons (bootstrap CI via percentile method with 10,000 resamples, exact McNemar test, win/loss/tie), and cost/latency aggregates, then renders the leaderboard and paired-comparison tables as checked-in Markdown/CSV artifacts.
- **Error-analysis table is explicitly not implemented in this spec's scope** — its structure (leading candidate: pipeline-stage failure attribution) is deferred to a follow-up decision/spec.
- **No external multi-agent framework** (LangGraph/AutoGen/CrewAI) is used; orchestration is custom Python for full control over exact prompts and call sequences.
- **No evolutionary optimization** is implemented; it is explicitly out of scope for this system.
- **Demo UI is additive, not part of the pipeline seam.** A Streamlit app (`medqa_multiagent/ui/`) offers the same black-box entrypoint through a browser instead of the CLI, purely for live demos/manual testing. It introduces no new answer-producing logic and shares the CLI's exact LLM client stack via one factored-out `client_factory.build_llm_client`. It has no access to the official test split and performs no dev/test sampling of its own -- an optional "load a random dev-pool example" convenience is unseeded, ad hoc, reads only `data/dev.jsonl`, and produces no persisted prediction/trace record.

## Testing Decisions

- **What makes a good test here:** tests exercise the pipeline strictly through the single black-box entrypoint (`answer_question`, parameterized by variant configuration), asserting on externally observable outcomes — the returned answer, explanation, invalid-response flag, and (where relevant) the persisted prediction/trace record — never on which private helper function was called or how many times an internal method fired. The LLM client and retrieval/memory interfaces are the dependency-injection points: tests substitute deterministic fakes (scripted response sequences) for these collaborators rather than mocking internals of the agents themselves. This keeps the number of test seams to effectively **one** (the entrypoint), with the injectable client interfaces acting as the boundary that makes that one seam sufficient.
- **Modules to test directly (pure logic, no seam needed):**
  - The output parser (strict-regex extraction / invalid detection) — pure function, exhaustively testable with valid/malformed/edge-case strings.
  - The statistics module (bootstrap CI, McNemar, win/loss/tie) — pure functions over small, hand-constructed prediction arrays with known expected outputs.
  - The cache key derivation and hit/miss logic — verify identical requests hit, any single differing parameter misses.
  - The sampling module — given a fixed seed, verify deterministic, non-overlapping dev/test subset selection.
- **Modules tested only through the single pipeline seam (fake LLM/retriever/memory-store injected):**
  - Each variant's end-to-end behavior (V0 through V4) — given scripted fake responses for each active agent role, assert the final parsed answer/explanation and the shape of the persisted trace match expectations for that variant's configuration.
  - The frozen/read-only invariant on the long-term memory store — assert a write attempt after freezing is rejected, exercised through the memory-building pass's public interface, not by inspecting internal state.
  - The Verifier's single-pass approve/override behavior — assert no second Reasoner call occurs regardless of the Verifier's decision.
- **Prior art:** none — this is a greenfield project with no existing test suite to follow. This spec's approach (one primary seam, injectable collaborators, pure-function unit tests for anything that doesn't need the seam) should be treated as the precedent for all subsequent specs in this project.

## Out of Scope

- Running the full 1273-question official MedQA-USMLE test set (only a small, config-driven smoke test is in scope; the harness must be *capable* of the full run, but actually executing it is not part of this spec).
- Evolutionary/genetic prompt optimization (explicitly descoped as optional in the source design).
- Verifier revision loops (multi-round self-correction between Reasoner and Verifier) — only single-pass verification is in scope.
- Debate/ensemble multi-agent patterns (multiple parallel reasoning agents + judge) — only the linear Router→Reasoner→Verifier pipeline is in scope.
- Provider-enforced structured output / JSON-schema function calling for answer formatting.
- The error-analysis table's concrete structure and implementation (deferred to a follow-up spec).
- Any speculative attack-surface hooks for the future attack/defense project (e.g., RAG-poisoning APIs, memory-injection APIs) beyond the single black-box entrypoint.
- Clinical-grade validation, deployment, or production hardening of any kind — this system is a research/evaluation artifact only.
- Non-US or 5-option MedQA variants, and any language other than English.
- Publishing this spec to an actual issue tracker (see Further Notes).
- Any UI-driven evaluation, scoring, or batch/sampled question runs -- the demo UI (see "Demo UI" user stories above) answers exactly one ad hoc question at a time and is not an alternate evaluation harness.

## Further Notes

- **Issue-tracker publishing could not be completed.** This project (`medqa-multiagent`) does not yet exist as its own repository — it currently lives as a subfolder inside `MedAgentBench/`, whose only configured git remote (`origin`) points to the unrelated upstream `stanfordmlgroup/MedAgentBench` repository. Filing an issue there would be incorrect (wrong repo, no ownership, and it's explicitly out of scope to touch that project). Once `medqa-multiagent` is spun out as its own repository with its own issue tracker, this file's contents should be filed as an issue there and given the `ready-for-agent` label. Until then, this file *is* the record of the spec.
- This spec covers the full V0-V4 variant ladder and the shared evaluation harness as **one** unit of work, consistent with the "fewest seams" principle — splitting it further would fragment the single black-box seam this whole design is built around.
- The error-analysis table (deferred) and the concrete repo folder layout (not yet drafted) remain open items from `DESIGN.md` and should be resolved via a follow-up spec before or during implementation of reporting.
- Default literal values (dev sample size, smoke-test size, bootstrap resample count, seed) are placeholders per `DESIGN.md` §"Open Items" and are expected to change via configuration, not code, as report requirements firm up.

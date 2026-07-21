"""The single black-box entrypoint: `answer_question`.

Given a question and its answer options, plus a variant selection, returns a
final answer and explanation. This is the one stable seam every later
variant (V1-V4) and the future downstream attack/defense project call
through -- `answer_question`'s signature and `AnswerResult`'s shape do not
change as later tickets add variants; only the internal behavior selected
by `variant` does. The one exception is the optional, backward-compatible
`retriever` keyword argument added alongside V1: existing callers (V0,
the CLI's `answer`/`run` commands as written for #2, the demo UI) never
pass it and are entirely unaffected, since it defaults to `None` and a
real one is then built lazily, on first use, from `config.rag_index_dir`.

Currently implements V0 (Direct-LLM baseline), V1 (RAG-only), and V2
(3-agent Router -> Reasoner -> Verifier). V0 is exactly one LLM call per
question with no retrieval/agents/memory; V1 is still exactly one LLM
call, with the prompt augmented by the top-`k` retrieved textbook passages
for the question's raw text; V2 is exactly three LLM calls -- the Router
formulates a retrieval query from the raw question, the Reasoner produces
a candidate answer grounded in the passages that query retrieved, and the
Verifier reviews that candidate exactly once (approving or overriding it)
with no revision loop back to the Reasoner. V3-V4 raise
`NotImplementedError` until their respective tickets land.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from .config import RunConfig
from .llm_client import LLMClient
from .parsing import parse_final_answer, strip_final_answer
from .prompts import (
    render_direct_prompt,
    render_rag_prompt,
    render_reasoner_prompt,
    render_router_prompt,
    render_verifier_prompt,
    render_researcher_feedback_prompt,
    render_router_reformulate_prompt,
    render_researcher_synthesis_prompt,
    render_verifier_review_prompt,
    render_reasoner_debate_prompt,
    render_verifier_final_consensus_prompt,
    render_multi_query_prompt,
    render_memory_agent_prompt,
)
from .agents import (
    MemoryAgent,
    ReasonerAgent,
    ResearcherAgent,
    RouterAgent,
    VerifierAgent,
)
from .memory import LongTermMemory, create_default_memory_store
from .rag.retriever import Passage, Retriever


#: Variants implemented: V0 (Direct), V1 (RAG-only), V2 (Multi-agent), V3 (Full 5-Agent + Memory), V4 (Full w/o Verifier)
SUPPORTED_VARIANTS = ("V0", "V1", "V2", "V3", "V4")



@dataclass(frozen=True)
class AnswerResult:
    """Return shape of `answer_question` -- stable across every variant.

    Attributes:
        answer: The predicted option letter, or `None` if the response was invalid.
        explanation: A short explanation of the answer (the model's free
            text, with the `Final Answer: <letter>` line itself stripped out).
        is_valid: Whether the raw response parsed to exactly one valid option letter.
        raw_response: The unparsed model text, kept for tracing.
        variant: Which variant configuration produced this result.
        prompt_tokens: Total prompt/input tokens spent answering this question.
        completion_tokens: Total completion/output tokens spent answering this question.
        total_tokens: Total tokens (prompt + completion) spent answering this question.
        latency_seconds: Total wall-clock time spent on LLM calls for this question.
        trace: Per-agent intermediate hand-off state. Empty for V0 (no
            retrieval/agent hand-offs). For V1, contains the retrieved
            passages only (`retrieved_passages`). For V2 (and later
            variants that retain the multi-agent pipeline), additionally
            contains the Router's query (`router_query`), the Reasoner's
            candidate (`reasoner_candidate`), and the Verifier's decision
            (`verifier_decision`).
    """

    answer: Optional[str]
    explanation: str
    is_valid: bool
    raw_response: str
    variant: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float
    trace: Dict[str, Any] = field(default_factory=dict)


def answer_question(
    question: str,
    options: Mapping[str, str],
    variant: str,
    config: RunConfig,
    client: LLMClient,
    retriever: Optional[Retriever] = None,
) -> AnswerResult:
    """Answer one MedQA-USMLE question under the given variant configuration.

    This is the single stable seam through which every variant (V0-V4) is
    invoked, and the one function the future attack/defense project
    integrates against. Its signature and `AnswerResult` return shape do not
    change for any later variant -- only the internal behavior selected by
    `variant` does.

    Args:
        question: The question stem text.
        options: Mapping of option letter to option text.
        variant: Which variant configuration to run (e.g. `"V0"`, `"V1"`).
        config: This run's configuration (model, temperature, etc).
        client: The `LLMClient` (typically cache- and logging-wrapped) used
            for every LLM call this question requires.
        retriever: The `Retriever` used by variants that need retrieval
            (currently V1 and V2). Optional and backward-compatible: existing V0
            callers never pass it; when a RAG-using variant needs one and
            none was injected, a real one is built lazily from
            `config.rag_index_dir` (via `rag.client_factory.build_retriever`)
            and reused across calls. Tests inject a scripted fake instead
            (see `tests/fakes.py`), per this project's dependency-injection
            testing convention.

    Raises:
        NotImplementedError: if `variant` isn't one of `SUPPORTED_VARIANTS` yet.
    """
    if variant not in SUPPORTED_VARIANTS:
        raise NotImplementedError(
            f"Variant {variant!r} is not yet implemented. "
            f"Implemented variants: {SUPPORTED_VARIANTS}."
        )
    if variant == "V0":
        return _answer_question_v0(question, options, config, client)
    if variant == "V1":
        return _answer_question_v1(question, options, config, client, retriever)
    if variant == "V2":
        return _answer_question_v2(question, options, config, client, retriever)
    if variant == "V3":
        return _answer_question_v3(question, options, config, client, retriever)
    return _answer_question_v4(question, options, config, client, retriever)



def _answer_question_v0(
    question: str,
    options: Mapping[str, str],
    config: RunConfig,
    client: LLMClient,
) -> AnswerResult:
    prompt = render_direct_prompt(question, options)
    response = client.complete(
        role="direct",
        prompt=prompt,
        model=config.model,
        temperature=config.temperature,
    )
    parsed = parse_final_answer(response.text)
    return AnswerResult(
        answer=parsed.answer,
        explanation=strip_final_answer(response.text),
        is_valid=parsed.is_valid,
        raw_response=response.text,
        variant="V0",
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        total_tokens=response.total_tokens,
        latency_seconds=response.latency_seconds,
        trace={},
    )


def _compute_retrieved_context_id(passages: List[Passage]) -> str:
    """Derive a stable identifier for a retrieved-passages set.

    Used as the LLM cache key's `retrieved_context_id` component (see
    `cache_key.compute_cache_key`) -- a hash of the passage ids in their
    retrieved (best-first) order, so a differing retrieval result is a
    guaranteed cache miss even in the (extremely unlikely) case that it
    happened to render to an identical prompt string.
    """
    canonical = json.dumps([passage.passage_id for passage in passages], ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _default_retriever(config: RunConfig) -> Retriever:
    # Imported lazily so that importing this module never requires the
    # `rag` extra's heavy dependencies (faiss/transformers/torch) unless a
    # RAG-using variant is actually run without an injected retriever.
    from .rag.client_factory import build_retriever

    return build_retriever(config)


def _parse_router_query(router_text: str) -> str:
    """Extract the ``Search Query`` from a structured Router agent response.

    The updated ``render_router_prompt`` asks the Router to respond in a
    structured 3-line format::

        Question Type: ...
        Key Entities: ...
        Search Query: <query>

    This function extracts only the ``Search Query:`` line to drive FAISS
    retrieval. Falls back to the full stripped text when the structured
    format is absent (e.g. legacy cache hits from the old single-line
    router) so existing `.cache/` entries remain usable without invalidation.
    """
    for line in router_text.splitlines():
        if line.strip().lower().startswith("search query:"):
            return line.split(":", 1)[1].strip()
    # Fallback: treat entire response as the query (backward-compat).
    return router_text.strip()


def _answer_question_v1(
    question: str,
    options: Mapping[str, str],
    config: RunConfig,
    client: LLMClient,
    retriever: Optional[Retriever],
) -> AnswerResult:
    if retriever is None:
        retriever = _default_retriever(config)

    passages = retriever.retrieve(question, config.rag_top_k)
    prompt = render_rag_prompt(
        question, options, passages,
        max_passage_tokens=config.rag_max_passage_tokens,
    )
    response = client.complete(
        role="rag",
        prompt=prompt,
        model=config.model,
        temperature=config.temperature,
        retrieved_context_id=_compute_retrieved_context_id(passages),
    )
    parsed = parse_final_answer(response.text)
    return AnswerResult(
        answer=parsed.answer,
        explanation=strip_final_answer(response.text),
        is_valid=parsed.is_valid,
        raw_response=response.text,
        variant="V1",
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        total_tokens=response.total_tokens,
        latency_seconds=response.latency_seconds,
        trace={"retrieved_passages": [asdict(passage) for passage in passages]},
    )


def _answer_question_v2(
    question: str,
    options: Mapping[str, str],
    config: RunConfig,
    client: LLMClient,
    retriever: Optional[Retriever],
) -> AnswerResult:
    if retriever is None:
        retriever = _default_retriever(config)

    # If both features are disabled, run the legacy baseline V2 (exactly 3 calls: router -> reasoner -> verifier)
    # to maintain 100% backward compatibility with existing tests.
    if not config.rag_enable_backtracking and not config.rag_enable_debate:
        # Router: formulate a retrieval query
        router_prompt = render_router_prompt(question)
        router_response = client.complete(
            role="router",
            prompt=router_prompt,
            model=config.model,
            temperature=config.temperature,
        )
        router_query = _parse_router_query(router_response.text)

        passages = retriever.retrieve(router_query, config.rag_top_k)
        retrieved_context_id = _compute_retrieved_context_id(passages)

        # Reasoner: propose candidate answer
        reasoner_prompt = render_reasoner_prompt(
            question, options, passages,
            max_passage_tokens=config.rag_max_passage_tokens,
        )
        reasoner_response = client.complete(
            role="reasoner",
            prompt=reasoner_prompt,
            model=config.model,
            temperature=config.temperature,
            retrieved_context_id=retrieved_context_id,
        )
        candidate_parsed = parse_final_answer(reasoner_response.text)
        candidate_explanation = strip_final_answer(reasoner_response.text)

        # Verifier: review candidate
        verifier_prompt = render_verifier_prompt(
            question, options, passages, candidate_parsed.answer, candidate_explanation,
            max_passage_tokens=config.rag_max_passage_tokens,
        )
        verifier_response = client.complete(
            role="verifier",
            prompt=verifier_prompt,
            model=config.model,
            temperature=config.temperature,
            retrieved_context_id=retrieved_context_id,
        )
        verifier_parsed = parse_final_answer(verifier_response.text)
        verifier_explanation = strip_final_answer(verifier_response.text)

        decision = (
            "approve"
            if candidate_parsed.answer is not None
            and verifier_parsed.answer == candidate_parsed.answer
            else "override"
        )

        return AnswerResult(
            answer=verifier_parsed.answer,
            explanation=verifier_explanation,
            is_valid=verifier_parsed.is_valid,
            raw_response=verifier_response.text,
            variant="V2",
            prompt_tokens=(
                router_response.prompt_tokens
                + reasoner_response.prompt_tokens
                + verifier_response.prompt_tokens
            ),
            completion_tokens=(
                router_response.completion_tokens
                + reasoner_response.completion_tokens
                + verifier_response.completion_tokens
            ),
            total_tokens=(
                router_response.total_tokens
                + reasoner_response.total_tokens
                + verifier_response.total_tokens
            ),
            latency_seconds=(
                router_response.latency_seconds
                + reasoner_response.latency_seconds
                + verifier_response.latency_seconds
            ),
            trace={
                "router_query": router_query,
                "retrieved_passages": [asdict(passage) for passage in passages],
                "reasoner_candidate": {
                    "answer": candidate_parsed.answer,
                    "explanation": candidate_explanation,
                    "is_valid": candidate_parsed.is_valid,
                    "raw_response": reasoner_response.text,
                },
                "verifier_decision": {
                    "decision": decision,
                    "answer": verifier_parsed.answer,
                    "explanation": verifier_explanation,
                    "is_valid": verifier_parsed.is_valid,
                    "raw_response": verifier_response.text,
                },
            },
        )

    # Initialize tracking variables for LLM usage
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    latency_seconds = 0.0
    backtrack_history: List[dict] = []
    debate_history: List[dict] = []
    bypassed_router = False

    def log_response(resp):
        nonlocal prompt_tokens, completion_tokens, total_tokens, latency_seconds
        prompt_tokens += resp.prompt_tokens
        completion_tokens += resp.completion_tokens
        total_tokens += resp.total_tokens
        latency_seconds += resp.latency_seconds

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 0: Adaptive Routing (Check raw question first for high confidence)
    # ─────────────────────────────────────────────────────────────────────────
    passages = []
    router_query = ""

    if config.rag_adaptive_routing:
        # Dry-run search using raw question stem and options
        raw_hits = retriever.retrieve(question, config.rag_top_k, options)
        if raw_hits and len(raw_hits) > 0:
            best_score = raw_hits[0].score
            # If RRF score is extremely high, bypass Router Agent entirely
            if best_score >= 0.030:
                print(f"[Adaptive Routing] High confidence raw hit score {best_score:.4f}. Bypassing Router.", flush=True)
                passages = raw_hits
                router_query = "Bypassed (Adaptive Routing)"
                bypassed_router = True

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Router - Generate Initial Query (if not bypassed)
    # ─────────────────────────────────────────────────────────────────────────
    if not bypassed_router:
        if config.rag_use_multi_query:
            router_prompt = render_multi_query_prompt(question)
            router_response = client.complete(
                role="router_multi",
                prompt=router_prompt,
                model=config.model,
                temperature=config.temperature,
            )
            log_response(router_response)

            # Parse 3 queries from lines starting with "Search Query \d:"
            queries = []
            for line in router_response.text.splitlines():
                if "search query" in line.lower():
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        queries.append(parts[1].strip())

            # Fallback to single raw question if parsing fails
            if not queries:
                queries = [question]

            print(f"[Multi-Query] Formulated queries: {queries}", flush=True)
            router_query = " || ".join(queries)

            # Retrieve for all 3 queries and merge
            all_passages = []
            for q in queries:
                all_passages.extend(retriever.retrieve(q, config.rag_top_k, options))

            # Deduplicate and sort by RRF score descending
            seen_ids = set()
            unique_passages = []
            for p in sorted(all_passages, key=lambda x: -x.score):
                if p.passage_id not in seen_ids:
                    seen_ids.add(p.passage_id)
                    unique_passages.append(p)

            # Respect dynamic top-k sizing inside retriever.retrieve() by slicing
            # to the minimum dynamic top-k computed
            passages = unique_passages[:config.rag_top_k]
        else:
            router_prompt = render_router_prompt(question)
            router_response = client.complete(
                role="router",
                prompt=router_prompt,
                model=config.model,
                temperature=config.temperature,
            )
            log_response(router_response)
            router_query = _parse_router_query(router_response.text)
            passages = retriever.retrieve(router_query, config.rag_top_k, options)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Researcher - Quality Check (with Backtracking, only if not bypassed)
    # ─────────────────────────────────────────────────────────────────────────
    retrieval_loop = 1
    if not bypassed_router and config.rag_enable_backtracking:
        while retrieval_loop < config.rag_max_retrieval_loops:
            eval_prompt = render_researcher_feedback_prompt(question, router_query, passages)
            eval_response = client.complete(
                role="researcher_eval",
                prompt=eval_prompt,
                model=config.model,
                temperature=config.temperature,
            )
            log_response(eval_response)

            relevance = "sufficient"
            reason = ""
            suggested_focus = ""
            for line in eval_response.text.splitlines():
                line_str = line.strip().lower()
                if line_str.startswith("relevance:"):
                    relevance = line.split(":", 1)[1].strip().lower()
                elif line_str.startswith("reason:"):
                    reason = line.split(":", 1)[1].strip()
                elif line_str.startswith("suggested focus:"):
                    suggested_focus = line.split(":", 1)[1].strip()

            backtrack_history.append({
                "loop": retrieval_loop,
                "query": router_query,
                "relevance": relevance,
                "reason": reason,
                "suggested_focus": suggested_focus,
                "passages_count": len(passages),
            })

            if "insufficient" in relevance and len(passages) > 0:
                reform_prompt = render_router_reformulate_prompt(question, router_query, reason or suggested_focus)
                reform_response = client.complete(
                    role="router_reformulate",
                    prompt=reform_prompt,
                    model=config.model,
                    temperature=config.temperature,
                )
                log_response(reform_response)

                new_query = _parse_router_query(reform_response.text)
                print(f"[Backtrack] Loop {retrieval_loop}: Reformulated query '{router_query}' -> '{new_query}' due to: {reason}", flush=True)

                router_query = new_query
                passages = retriever.retrieve(router_query, config.rag_top_k, options)
                retrieval_loop += 1
            else:
                break

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Researcher - Synthesize into a Research Brief
    # ─────────────────────────────────────────────────────────────────────────
    research_brief = ""
    if passages:
        if config.rag_heuristic_compression:
            # 0ms / 0-LLM token synthesis: join heuristically compressed sentences
            lines = ["Clinical Evidence (Heuristically Compressed):"]
            for index, passage in enumerate(passages, start=1):
                lines.append(f"[{index}] ({passage.source}) {passage.text.strip()}")
            research_brief = "\n".join(lines)
        else:
            synthesis_prompt = render_researcher_synthesis_prompt(question, passages)
            synthesis_response = client.complete(
                role="researcher_synthesis",
                prompt=synthesis_prompt,
                model=config.model,
                temperature=config.temperature,
            )
            log_response(synthesis_response)
            research_brief = synthesis_response.text
    else:
        research_brief = "No clinical evidence retrieved."

    if config.rag_heuristic_compression and passages:
        from .rag.retriever import TokenBudgetManager
        passages = TokenBudgetManager.compress_passages(passages, max_total_words=350)

    retrieved_context_id = _compute_retrieved_context_id(passages)


    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: Reasoner - Propose Initial Diagnosis & Answer (Turn 1)
    # ─────────────────────────────────────────────────────────────────────────
    reasoner_prompt = render_reasoner_prompt(
        question,
        options,
        passages,
        max_passage_tokens=config.rag_max_passage_tokens,
        research_brief=research_brief,
    )
    reasoner_response = client.complete(
        role="reasoner",
        prompt=reasoner_prompt,
        model=config.model,
        temperature=config.temperature,
        retrieved_context_id=retrieved_context_id,
    )
    log_response(reasoner_response)
    candidate_parsed = parse_final_answer(reasoner_response.text)
    candidate_explanation = strip_final_answer(reasoner_response.text)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5: Verifier - Peer Review and Debate Loop (Turns 2-4)
    # ─────────────────────────────────────────────────────────────────────────
    verifier_review_prompt = render_verifier_review_prompt(
        question,
        options,
        research_brief,
        candidate_parsed.answer,
        candidate_explanation,
    )
    verifier_review_response = client.complete(
        role="verifier_review",
        prompt=verifier_review_prompt,
        model=config.model,
        temperature=config.temperature,
        retrieved_context_id=retrieved_context_id,
    )
    log_response(verifier_review_response)

    decision = "agree"
    review_memo = ""
    for line in verifier_review_response.text.splitlines():
        line_str = line.strip().lower()
        if line_str.startswith("decision:"):
            decision = line.split(":", 1)[1].strip().lower()
        elif line_str.startswith("review memo:"):
            review_memo = line.split(":", 1)[1].strip()

    final_answer = candidate_parsed.answer
    final_explanation = candidate_explanation
    final_raw_response = reasoner_response.text

    # Debate loop if Verifier disagrees
    if "disagree" in decision and config.rag_enable_debate:
        debate_prompt = render_reasoner_debate_prompt(
            question,
            options,
            research_brief,
            candidate_explanation,
            review_memo or verifier_review_response.text,
        )
        debate_response = client.complete(
            role="reasoner_debate",
            prompt=debate_prompt,
            model=config.model,
            temperature=config.temperature,
            retrieved_context_id=retrieved_context_id,
        )
        log_response(debate_response)
        debate_parsed = parse_final_answer(debate_response.text)
        debate_explanation = strip_final_answer(debate_response.text)

        consensus_prompt = render_verifier_final_consensus_prompt(
            question,
            options,
            research_brief,
            candidate_parsed.answer,
            review_memo or verifier_review_response.text,
            debate_response.text,
        )
        consensus_response = client.complete(
            role="verifier_consensus",
            prompt=consensus_prompt,
            model=config.model,
            temperature=config.temperature,
            retrieved_context_id=retrieved_context_id,
        )
        log_response(consensus_response)
        consensus_parsed = parse_final_answer(consensus_response.text)
        consensus_explanation = strip_final_answer(consensus_response.text)

        debate_history.append({
            "initial_candidate": candidate_parsed.answer,
            "review_memo": review_memo,
            "reasoner_debate_response": debate_response.text,
            "consensus_decision": consensus_parsed.answer,
        })

        final_answer = consensus_parsed.answer
        final_explanation = consensus_explanation
        final_raw_response = consensus_response.text
        decision_label = "debate_consensus"
    else:
        decision_label = "immediate_agreement"
        if "disagree" in decision and not config.rag_enable_debate:
            fallback_parsed = parse_final_answer(verifier_review_response.text)
            fallback_explanation = strip_final_answer(verifier_review_response.text)
            final_answer = fallback_parsed.answer
            final_explanation = fallback_explanation
            final_raw_response = verifier_review_response.text
            decision_label = "immediate_override"

    final_parsed = parse_final_answer(final_raw_response)

    return AnswerResult(
        answer=final_answer,
        explanation=final_explanation,
        is_valid=final_parsed.is_valid,
        raw_response=final_raw_response,
        variant="V2",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_seconds=latency_seconds,
        trace={
            "router_query": router_query,
            "retrieved_passages": [asdict(passage) for passage in passages],
            "research_brief": research_brief,
            "decision_flow": decision_label,
            "backtrack_history": backtrack_history,
            "debate_history": debate_history,
            "reasoner_candidate": {
                "answer": candidate_parsed.answer,
                "explanation": candidate_explanation,
                "is_valid": candidate_parsed.is_valid,
                "raw_response": reasoner_response.text,
            },
        },
    )


def _answer_question_v3(
    question: str,
    options: Mapping[str, str],
    config: RunConfig,
    client: LLMClient,
    retriever: Optional[Retriever],
    memory_store: Optional[LongTermMemory] = None,
) -> AnswerResult:
    """Full 5-Agent system (V3): Router -> Researcher -> Memory Specialist -> Reasoner -> Verifier Audit."""
    if memory_store is None:
        memory_store = create_default_memory_store()

    # Step 1: Agent 3 (Memory Specialist Agent) retrieves exemplars from long-term memory
    exemplars = memory_store.retrieve_exemplars(question, top_k=config.memory_top_k)
    mem_prompt = render_memory_agent_prompt(question, exemplars)
    mem_response = client.complete(
        role="memory_specialist",
        prompt=mem_prompt,
        model=config.model,
        temperature=config.temperature,
    )
    memory_brief = mem_response.text

    # Step 2-5: Execute Multi-Agent RAG flow with RAG + Reasoning + Verifier Audit
    v2_result = _answer_question_v2(question, options, config, client, retriever)

    # Wrap as V3 result including Memory Specialist Agent stats & trace
    total_prompt_tokens = v2_result.prompt_tokens + mem_response.prompt_tokens
    total_completion_tokens = v2_result.completion_tokens + mem_response.completion_tokens
    total_tokens = v2_result.total_tokens + mem_response.total_tokens
    latency_seconds = v2_result.latency_seconds + mem_response.latency_seconds

    trace = dict(v2_result.trace)
    trace["memory_brief"] = memory_brief
    trace["memory_exemplars_count"] = len(exemplars)

    return AnswerResult(
        answer=v2_result.answer,
        explanation=v2_result.explanation,
        is_valid=v2_result.is_valid,
        raw_response=v2_result.raw_response,
        variant="V3",
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_tokens=total_tokens,
        latency_seconds=latency_seconds,
        trace=trace,
    )


def _answer_question_v4(
    question: str,
    options: Mapping[str, str],
    config: RunConfig,
    client: LLMClient,
    retriever: Optional[Retriever],
    memory_store: Optional[LongTermMemory] = None,
) -> AnswerResult:
    """Ablation System (V4): Full 5-Agent Flow WITHOUT Verifier Agent."""
    if retriever is None:
        retriever = _default_retriever(config)

    # Step 1: Agent 3 (Memory Specialist Agent)
    mem_agent = MemoryAgent(config, client, memory_store)
    mem_out = mem_agent.process_memory(question)

    # Step 2: Agent 1 (Router Agent)
    router_agent = RouterAgent(config, client)
    router_out = router_agent.formulate_query(question)

    # Step 3: Agent 2 (Researcher Agent RAG)
    researcher_agent = ResearcherAgent(config, client, retriever)
    passages = researcher_agent.retrieve(router_out.query, options=options)
    synthesis_out = researcher_agent.synthesize_brief(question, passages)
    retrieved_context_id = _compute_retrieved_context_id(passages)

    # Step 4: Agent 4 (Reasoner Agent) proposes final answer directly (No Verifier Agent)
    reasoner_agent = ReasonerAgent(config, client)
    reasoner_out = reasoner_agent.propose_candidate(
        question,
        options,
        passages,
        research_brief=synthesis_out.research_brief,
        memory_brief=mem_out.memory_brief,
        retrieved_context_id=retrieved_context_id,
    )

    return AnswerResult(
        answer=reasoner_out.candidate_answer,
        explanation=reasoner_out.candidate_explanation,
        is_valid=reasoner_out.is_valid,
        raw_response=reasoner_out.raw_response,
        variant="V4",
        prompt_tokens=(
            mem_out.prompt_tokens
            + router_out.prompt_tokens
            + synthesis_out.prompt_tokens
            + reasoner_out.prompt_tokens
        ),
        completion_tokens=(
            mem_out.completion_tokens
            + router_out.completion_tokens
            + synthesis_out.completion_tokens
            + reasoner_out.completion_tokens
        ),
        total_tokens=(
            mem_out.total_tokens
            + router_out.total_tokens
            + synthesis_out.total_tokens
            + reasoner_out.total_tokens
        ),
        latency_seconds=(
            mem_out.latency_seconds
            + router_out.latency_seconds
            + synthesis_out.latency_seconds
            + reasoner_out.latency_seconds
        ),
        trace={
            "router_query": router_out.query,
            "retrieved_passages": [asdict(p) for p in passages],
            "research_brief": synthesis_out.research_brief,
            "memory_brief": mem_out.memory_brief,
            "verifier_omitted": True,
        },
    )






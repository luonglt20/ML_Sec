"""Prompt rendering.

Every answer-producing prompt instructs a 1-3 sentence explanation followed
by the shared `Final Answer: <letter>` convention (see `parsing.py`), so
explanation length -- and its contribution to token cost -- stays roughly
comparable across variants. These functions return the *exact* string sent
to the model, so callers can log it verbatim.

Token budget
------------
All functions that include retrieved passages accept an optional
``max_passage_tokens`` keyword argument (default 200 words). Passages
exceeding this limit are word-split, truncated, and appended with a
``[Truncated]`` marker so the LLM is aware that context is partial. This
caps the per-question RAG context at ~600 tokens (3 passages × 200 words)
and meaningfully reduces API costs without dropping high-signal content.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

from .rag.retriever import Passage

_ANSWER_FORMAT_INSTRUCTIONS = (
    'Respond with a concise, 1-3 sentence explanation, then end your '
    'response with a line in exactly this format (no extra words on that '
    'line): "Final Answer: <letter>", where <letter> is the single letter '
    "of the option you have chosen."
)

_DEFAULT_MAX_PASSAGE_TOKENS = 200


def _truncate_passage(text: str, max_tokens: int = _DEFAULT_MAX_PASSAGE_TOKENS) -> str:
    """Truncate ``text`` to at most ``max_tokens`` whitespace-delimited words.

    Uses the same zero-dependency whitespace approximation as
    ``chunking.approximate_token_count``. Appends ``[Truncated]`` when
    truncation occurs so the LLM knows the passage context is partial.
    No-op when ``text`` is already within the limit.
    """
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens]) + " [Truncated]"


def render_direct_prompt(question: str, options: Mapping[str, str]) -> str:
    """Render the V0 direct-LLM prompt: raw question + lettered options.

    Used verbatim as the single LLM call's prompt in the V0 (Direct-LLM
    baseline) variant -- no retrieval, no agent hand-offs.
    """
    lines = [
        "You are a medical expert answering a USMLE-style multiple-choice question.",
        "",
        "Question:",
        question.strip(),
        "",
        "Options:",
    ]
    for letter in sorted(options):
        lines.append(f"{letter}. {options[letter]}")
    lines.append("")
    lines.append(_ANSWER_FORMAT_INSTRUCTIONS)
    return "\n".join(lines)


def render_rag_prompt(
    question: str,
    options: Mapping[str, str],
    passages: Sequence[Passage],
    max_passage_tokens: int = _DEFAULT_MAX_PASSAGE_TOKENS,
) -> str:
    """Render the V1 RAG-augmented prompt: retrieved passages + question + options.

    Used as the single LLM call's prompt in the V1 (RAG-only) variant --
    the same one-call-per-question shape as `render_direct_prompt`, just
    with a reference-passages section prepended. `passages` is expected in
    the retriever's own ranked (best-first) order, and is rendered in that
    same order without re-sorting.

    Each passage is truncated to ``max_passage_tokens`` words (with a
    ``[Truncated]`` marker) to bound total context size and token cost.
    """
    lines = [
        "You are a medical expert answering a USMLE-style multiple-choice question.",
        "",
        "Relevant reference passages:",
    ]
    if passages:
        for index, passage in enumerate(passages, start=1):
            truncated = _truncate_passage(passage.text.strip(), max_passage_tokens)
            lines.append(f"[{index}] ({passage.source}) {truncated}")
    else:
        lines.append("(none retrieved)")
    lines.append("")
    lines.append("Question:")
    lines.append(question.strip())
    lines.append("")
    lines.append("Options:")
    for letter in sorted(options):
        lines.append(f"{letter}. {options[letter]}")
    lines.append("")
    lines.append(_ANSWER_FORMAT_INSTRUCTIONS)
    return "\n".join(lines)


def render_router_prompt(question: str) -> str:
    """Render the V2 Router agent's prompt: raw question -> structured retrieval plan.

    Used as the Router agent's single LLM call in the V2 (and later,
    memory-augmented) multi-agent pipeline. The Router's job is to:

    1. Classify the question type (Diagnosis / Treatment / Mechanism /
       Pharmacology / Anatomy / Other) to focus retrieval on the right
       knowledge domain.
    2. Extract the key medical entities (diseases, drugs, symptoms, organs,
       pathways, lab values) as a comma-separated list -- these entities
       form the semantic core of what needs to be retrieved.
    3. Produce a concise keyword-focused ``Search Query`` optimised for
       textbook-style dense retrieval (not a natural-language question).

    The caller (`entrypoint._parse_router_query`) extracts only the
    ``Search Query:`` line to drive FAISS retrieval, so the Question Type
    and Key Entities fields are for the Router's own chain-of-thought -- they
    improve query quality without changing the retrieval interface.
    """
    lines = [
        "You are the Router agent in a medical question-answering pipeline.",
        "Analyze the question below and produce a structured retrieval plan.",
        "",
        "Question:",
        question.strip(),
        "",
        "Respond in EXACTLY this 3-line format (no extra text before or after):",
        "Question Type: [Diagnosis / Treatment / Mechanism / Pharmacology / Anatomy / Other]",
        "Key Entities: [comma-separated medical terms: diseases, drugs, symptoms, organs, pathways]",
        "Search Query: [concise keyword query optimised for medical textbook retrieval]",
    ]
    return "\n".join(lines)


def render_reasoner_prompt(
    question: str,
    options: Mapping[str, str],
    passages: Sequence[Passage],
    max_passage_tokens: int = _DEFAULT_MAX_PASSAGE_TOKENS,
    research_brief: Optional[str] = None,
    memory_brief: Optional[str] = None,
) -> str:
    """Render the Reasoner agent's prompt.

    Accepts raw passages or research_brief plus optional long-term case memory exemplars (memory_brief).
    Instructs the model to think step-by-step according to medical guidelines.
    """
    lines = [
        "You are the Reasoner agent in a medical question-answering pipeline.",
        "Your task is to analyze the medical question and propose a candidate answer.",
        "A separate Verifier agent will review your reasoning, so provide a detailed step-by-step clinical analysis.",
        "",
    ]
    if research_brief:
        lines.append("Clinical Evidence Summary (Prepared by Researcher Agent):")
        lines.append(research_brief.strip())
        lines.append("")
    else:
        lines.append("Relevant reference passages:")
        if passages:
            for index, passage in enumerate(passages, start=1):
                truncated = _truncate_passage(passage.text.strip(), max_passage_tokens)
                lines.append(f"[{index}] ({passage.source}) {truncated}")
        else:
            lines.append("(none retrieved)")
        lines.append("")

    if memory_brief:
        lines.append("Long-Term Case Memory Exemplars (Prepared by Memory Specialist Agent):")
        lines.append(memory_brief.strip())
        lines.append("")

    lines.append("Question:")
    lines.append(question.strip())
    lines.append("")
    lines.append("Options:")
    for letter in sorted(options):
        lines.append(f"{letter}. {options[letter]}")
    lines.append("")
    lines.append(
        "For your explanation, think step-by-step:\n"
        "1. Identify the core clinical presentation and key findings.\n"
        "2. Discuss the pathophysiology of the condition.\n"
        "3. Evaluate each multiple-choice option, explaining why it is correct or incorrect.\n"
        "4. Conclude with a clear reasoning chain.\n"
        "5. Cite reference passages (e.g. [1], [2]) to ground every clinical claim and prevent hallucination."
    )
    lines.append("")
    lines.append(_ANSWER_FORMAT_INSTRUCTIONS)
    return "\n".join(lines)


def render_memory_agent_prompt(
    question: str,
    exemplars: Sequence[Any],
) -> str:
    """Render prompt for Agent 3 (Memory Specialist Agent) to summarize case exemplars."""
    lines = [
        "You are the Memory Specialist Agent in a medical question-answering pipeline.",
        "Synthesize the retrieved long-term case memory exemplars below for the target question.",
        "Extract key diagnostic rules and clinical pearls from similar past cases.",
        "",
        "Target Question:",
        question.strip(),
        "",
        "Similar Past Cases:",
    ]
    if exemplars:
        for idx, ex in enumerate(exemplars, start=1):
            q_text = getattr(ex, "question", "")
            ans = getattr(ex, "correct_answer", "")
            exp = getattr(ex, "explanation", "")
            lines.append(f"Case [{idx}]: {q_text}")
            lines.append(f"Correct Answer Option: {ans}")
            lines.append(f"Clinical Rationale: {exp}")
            lines.append("")
    else:
        lines.append("(no similar past cases found)")
        lines.append("")

    lines.append(
        "Summarize in 2-3 bullet points the most relevant clinical lessons for the Reasoner Agent:"
    )
    return "\n".join(lines)




def render_verifier_prompt(
    question: str,
    options: Mapping[str, str],
    passages: Sequence[Passage],
    candidate_answer: Optional[str],
    candidate_explanation: str,
    max_passage_tokens: int = _DEFAULT_MAX_PASSAGE_TOKENS,
    research_brief: Optional[str] = None,
) -> str:
    """Render the V2 Verifier agent's prompt: review the Reasoner's candidate."""
    lines = [
        "You are the Verifier agent in a medical question-answering pipeline.",
        "A separate Reasoner agent already proposed a candidate answer below.",
        "Review their reasoning against the medical evidence. If you agree, restate the answer and explanation.",
        "If you disagree, override it with your own corrected answer and explanation.",
        "",
    ]
    if research_brief:
        lines.append("Clinical Evidence Summary:")
        lines.append(research_brief.strip())
    else:
        lines.append("Relevant reference passages:")
        if passages:
            for index, passage in enumerate(passages, start=1):
                truncated = _truncate_passage(passage.text.strip(), max_passage_tokens)
                lines.append(f"[{index}] ({passage.source}) {truncated}")
        else:
            lines.append("(none retrieved)")
    lines.append("")
    lines.append("Question:")
    lines.append(question.strip())
    lines.append("")
    lines.append("Options:")
    for letter in sorted(options):
        lines.append(f"{letter}. {options[letter]}")
    lines.append("")
    lines.append("Reasoner's candidate answer:")
    lines.append(candidate_answer if candidate_answer is not None else "(none -- invalid/unparseable response)")
    lines.append("Reasoner's candidate explanation:")
    lines.append(candidate_explanation.strip() if candidate_explanation.strip() else "(none)")
    lines.append("")
    lines.append(_ANSWER_FORMAT_INSTRUCTIONS)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Backtracking & Active Query Reformulation Prompts
# ─────────────────────────────────────────────────────────────────────────────

def render_researcher_feedback_prompt(
    question: str,
    query: str,
    passages: Sequence[Passage],
) -> str:
    """Ask the Researcher Agent to evaluate the relevance of retrieved passages.

    If they are irrelevant or do not cover the key medical entities in the
    question, the Researcher should issue a feedback message pointing out the
    gap and requesting a new query from the Router.
    """
    lines = [
        "You are the Researcher Agent in a medical multi-agent system.",
        "Evaluate if the retrieved textbook passages below are relevant and sufficient ",
        "to answer the given question.",
        "",
        "Question:",
        question.strip(),
        "",
        "Query used:",
        f"'{query}'",
        "",
        "Retrieved passages:",
    ]
    for index, passage in enumerate(passages, start=1):
        lines.append(f"[{index}] ({passage.source}) {passage.text[:200]}...")
    lines.append("")
    lines.append(
        "Respond in EXACTLY this format:\n"
        "Relevance: [Sufficient / Insufficient]\n"
        "Reason: [Short explanation of why it is sufficient or what critical information/entities are missing]\n"
        "Suggested Focus: [Specific keyword focus or entities that the next search query should prioritize]"
    )
    return "\n".join(lines)


def render_router_reformulate_prompt(
    question: str,
    failed_query: str,
    feedback_reason: str,
) -> str:
    """Ask the Router Agent to write a new query based on Researcher feedback."""
    lines = [
        "You are the Router Agent in a medical question-answering pipeline.",
        "Your previous search query failed to retrieve relevant medical textbook passages.",
        "Reformulate the query using the feedback from the Researcher Agent.",
        "",
        "Question:",
        question.strip(),
        "",
        "Failed Query:",
        f"'{failed_query}'",
        "",
        "Feedback from Researcher Agent:",
        feedback_reason.strip(),
        "",
        "Respond on a single line in this format (no extra text):",
        "Search Query: [new keyword query focusing on the suggested terms]"
    ]
    return "\n".join(lines)


def render_researcher_synthesis_prompt(
    question: str,
    passages: Sequence[Passage],
) -> str:
    """Ask the Researcher Agent to synthesize raw passages into a Research Brief."""
    lines = [
        "You are the Researcher Agent in a medical multi-agent system.",
        "Synthesize the following textbook passages into a concise Research Brief ",
        "tailored for the clinical question below. Do not analyze the question or choose an option; ",
        "only summarize the medical facts.",
        "",
        "Question:",
        question.strip(),
        "",
        "Retrieved Passages:",
    ]
    for index, passage in enumerate(passages, start=1):
        lines.append(f"[{index}] ({passage.source}) {passage.text.strip()}")
    lines.append("")
    lines.append(
        "Structure your Research Brief with these bullet points:\n"
        "- Clinical Findings: [key symptoms, signs, lab markers mentioned in y văn]\n"
        "- Pathophysiology: [biological mechanisms, drug modes of action, causes]\n"
        "- Treatment Guidelines: [first-line therapy, secondary options, contraindications]"
    )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Collaborative Debate Prompts
# ─────────────────────────────────────────────────────────────────────────────

def render_verifier_review_prompt(
    question: str,
    options: Mapping[str, str],
    research_brief: str,
    reasoner_candidate: Optional[str],
    reasoner_explanation: str,
) -> str:
    """Verifier agent peer-reviews the Reasoner's proposal.

    If they disagree, they write a Review Memo explaining why, acting as feedback.
    """
    lines = [
        "You are the Verifier agent (peer reviewer) in a medical team.",
        "Review the Reasoner's proposed candidate answer against the medical evidence brief.",
        "If you find errors in their clinical reasoning, diagnosis, or drug choice, write a detailed Review Memo.",
        "If you fully agree, write 'Decision: Agree' and state why.",
        "",
        "Clinical Evidence Brief:",
        research_brief.strip(),
        "",
        "Question:",
        question.strip(),
        "",
        "Options:",
    ]
    for letter in sorted(options):
        lines.append(f"{letter}. {options[letter]}")
    lines.append("")
    lines.append(f"Reasoner's Proposed Answer: {reasoner_candidate}")
    lines.append(f"Reasoner's Logic:\n{reasoner_explanation.strip()}")
    lines.append("")
    lines.append(
        "Respond in EXACTLY this format:\n"
        "Decision: [Agree / Disagree]\n"
        "Review Memo: [If Disagree, detail logical flaws, diagnostic errors, missed findings, or hallucinated claims not backed by evidence [1],[2]. If Agree, brief support.]"
    )
    return "\n".join(lines)



def render_reasoner_debate_prompt(
    question: str,
    options: Mapping[str, str],
    research_brief: str,
    original_explanation: str,
    review_memo: str,
) -> str:
    """Reasoner agent reads the peer feedback (Review Memo) and revises or defends."""
    lines = [
        "You are the Reasoner agent in a medical team.",
        "Your clinical peer (the Verifier agent) has disagreed with your proposed diagnosis/answer ",
        "and issued the Review Memo below. Re-evaluate your reasoning carefully using the evidence brief.",
        "You can either agree and change your answer option, or defend your original choice if you believe it is correct.",
        "",
        "Clinical Evidence Brief:",
        research_brief.strip(),
        "",
        "Question:",
        question.strip(),
        "",
        "Options:",
    ]
    for letter in sorted(options):
        lines.append(f"{letter}. {options[letter]}")
    lines.append("")
    lines.append("Your Original Logic:")
    lines.append(original_explanation.strip())
    lines.append("")
    lines.append("Peer Reviewer's Memo:")
    lines.append(review_memo.strip())
    lines.append("")
    lines.append("Think step-by-step, address their criticisms, and output your final candidate.")
    lines.append("")
    lines.append(_ANSWER_FORMAT_INSTRUCTIONS)
    return "\n".join(lines)


def render_verifier_final_consensus_prompt(
    question: str,
    options: Mapping[str, str],
    research_brief: str,
    reasoner_initial_proposal: Optional[str],
    review_memo: str,
    reasoner_debate_response: str,
) -> str:
    """Verifier makes the final executive decision after reading the debate round."""
    lines = [
        "You are the Verifier agent making the final decision for a medical team.",
        "Review the entire discussion history and clinical evidence to choose the final best option.",
        "",
        "Clinical Evidence Brief:",
        research_brief.strip(),
        "",
        "Question:",
        question.strip(),
        "",
        "Options:",
    ]
    for letter in sorted(options):
        lines.append(f"{letter}. {options[letter]}")
    lines.append("")
    lines.append(f"Reasoner's Initial Proposal: {reasoner_initial_proposal}")
    lines.append(f"Your Review Memo: {review_memo.strip()}")
    lines.append("Reasoner's Response to Memo:")
    lines.append(reasoner_debate_response.strip())
    lines.append("")
    lines.append("Synthesize the debate and output the final consensus answer of the team.")
    lines.append("")
    lines.append(_ANSWER_FORMAT_INSTRUCTIONS)
    return "\n".join(lines)


def render_multi_query_prompt(question: str) -> str:
    """Render prompt for the Router Agent to generate 3 distinct search queries.

    Each query targets a different clinical aspect (symptoms, pharmacology, pathology).
    Outputs 3 lines, each starting with 'Search Query:'.
    """
    lines = [
        "You are the Router Agent in a medical question-answering pipeline.",
        "Your task is to generate exactly three (3) distinct search queries to retrieve relevant textbook pages for the clinical question.",
        "Each query should focus on a different aspect of the case:",
        "1. Symptoms & Diagnosis: clinical signs, markers, presentation.",
        "2. Pathogenesis & Biological mechanism: causes, micro-organisms, cells, pathways.",
        "3. Pharmacology & Treatment: drug classes, mechanism of action, contraindications.",
        "",
        "Question:",
        question.strip(),
        "",
        "Respond with exactly three lines in this format (no extra introductory or concluding text):",
        "Search Query 1: [keywords for symptoms/diagnosis]",
        "Search Query 2: [keywords for pathogenesis/mechanism]",
        "Search Query 3: [keywords for pharmacology/treatment]"
    ]
    return "\n".join(lines)



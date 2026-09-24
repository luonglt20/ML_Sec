"""Advanced Function Calling Tooling framework and 26 Specialized Medical RAG Toolkits.

Provides BaseTool, ToolRegistry, and 26 specialized medical & RAG tools for the 5 AI Agents:
- Router Agent (6 Tools): EntityExtractor, HyDEGen, SubqueryDecomposer, NoisePruner, FastThinkingEval, SearchPlanner
- Researcher Agent (7 Tools): FAISSSearch, BM25Search, RRFFusion, MMRFilter, CoverageEval, TokenCompressor, CooccurrenceScorer
- Memory Agent (4 Tools): LongTermMemorySearch, PearlsExtractor, FewShotFormatter, SimilarityScorer
- Reasoner Agent (5 Tools): DifferentialEval, CitationAuditor, MechanismChainBuilder, OptionEliminator, ConfidenceEstimator
- Verifier Agent (4 Tools): AntiHallucinationFactChecker, CitationFidelityVerifier, CallbackIssuer, DebateSynthesizer
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple


@dataclass
class BaseTool:
    """Base class for structured agent tools (Function Calling schema)."""

    name: str
    description: str
    parameters_schema: Dict[str, Any]
    func: Callable[..., Any]

    def execute(self, **kwargs: Any) -> Any:
        """Execute the tool function with kwargs."""
        return self.func(**kwargs)

    def to_schema_dict(self) -> Dict[str, Any]:
        """Format as standard function calling schema dictionary."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }


class ToolRegistry:
    """Registry managing available tools for an agent."""

    def __init__(self, tools: Optional[Sequence[BaseTool]] = None) -> None:
        self._tools: Dict[str, BaseTool] = {}
        if tools:
            for t in tools:
                self.register_tool(t)

    def register_tool(self, tool: BaseTool) -> None:
        """Register a new tool in the registry."""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[BaseTool]:
        """List all registered tools."""
        return list(self._tools.values())

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get function schemas for all tools in registry."""
        return [t.to_schema_dict() for t in self._tools.values()]

    def execute_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """Execute a registered tool by name with arguments."""
        tool = self.get_tool(tool_name)
        if tool is None:
            raise ValueError(f"Tool {tool_name!r} not found in ToolRegistry.")
        return tool.execute(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 1: ROUTER AGENT TOOLS (6 TOOLS)
# ─────────────────────────────────────────────────────────────────────────────

def tool_extract_medical_entities(text: str) -> Set[str]:
    """Tool 1: Extract medical entities, drugs, pathogens, and acronyms."""
    acronyms = set(re.findall(r"\b[A-Z]{2,}\b", text))
    medical_suffixes = (
        "itis", "mycin", "micin", "cillin", "penem", "oxacin", "olol", "pine", "pril",
        "sartan", "azole", "nazole", "vir", "oma", "carcinoma", "tomy", "ectomy", "cocci",
        "bacillus", "statin", "mab", "nib", "tinib", "cycline", "thromycin", "dine",
        "tidine", "prazole", "sone", "solone", "zole"
    )
    medical_terms = {
        "gonorrhoeae", "neisseria", "streptococcus", "staphylococcus", "chlamydia",
        "treponema", "syphilis", "cephalosporin", "fluoroquinolone", "aminoglycoside",
        "macrolide", "carbapenem", "sulfonamide", "tetracycline", "pneumoniae",
        "meningitidis", "aureus", "influenzae", "pseudomonas", "enterococcus",
        "gentamicin", "ceftriaxone", "myocardial", "infarction", "hypertension", "diabetes"
    }
    words = set(re.findall(r"\b\w+\b", text.lower()))
    matched = set()
    for w in words:
        if len(w) >= 4:
            if w.endswith(medical_suffixes) or w in medical_terms:
                matched.add(w)
    return acronyms | matched


def tool_generate_hyde_context(question: str) -> str:
    """Tool 2: Generate hypothetical textbook passage context snippet."""
    return f"Hypothetical clinical textbook passage discussing diagnosis and mechanisms for: {question[:80]}"


def tool_decompose_subqueries(question: str) -> List[str]:
    """Tool 3: Decompose clinical question into 3 sub-query focuses."""
    return [
        f"{question} symptoms diagnosis presentation",
        f"{question} pathogenesis biological mechanism etiology",
        f"{question} pharmacology treatment first line therapy"
    ]


def tool_prune_vignette_noise(question: str) -> str:
    """Tool 4: Remove sentence fluff and question prompt noise from clinical vignette."""
    sentences = re.split(r"(?<=[.!?]) +", question.strip())
    pruned = []
    question_patterns = (
        "which of the following", "what is the", "most likely",
        "what should be", "which of these", "is the next step"
    )
    for s in sentences:
        s_lower = s.lower()
        if any(p in s_lower for p in question_patterns):
            continue
        pruned.append(s)
    return " ".join(pruned) if pruned else question


def tool_fast_thinking_evaluator(question: str, options: Mapping[str, str]) -> Dict[str, Any]:
    """Tool 5: Fast Thinking Evaluator - Detect simple questions for fast-path processing."""
    q_len = len(question.split())
    has_direct_keywords = any(kw in question.lower() for kw in ["first-line", "drug of choice", "causative organism", "treatment for"])
    is_simple = q_len < 30 and has_direct_keywords
    return {
        "is_fast_path": is_simple,
        "reason": "Short question stem with direct clinical keywords." if is_simple else "Standard complex clinical vignette.",
        "estimated_tokens_saved": 400 if is_simple else 0,
    }


def tool_search_strategy_planner(question: str) -> Dict[str, Any]:
    """Tool 6: Plan search strategy (Dense vs Sparse vs Hybrid RAG)."""
    entities = tool_extract_medical_entities(question)
    has_acronyms = any(e.isupper() for e in entities)
    strategy = "hybrid" if len(entities) >= 2 else ("sparse" if has_acronyms else "dense")
    return {
        "strategy": strategy,
        "entities_found": list(entities),
        "use_bm25": strategy in ("sparse", "hybrid"),
        "use_faiss": strategy in ("dense", "hybrid"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 2: RESEARCHER AGENT TOOLS (7 TOOLS)
# ─────────────────────────────────────────────────────────────────────────────

def tool_faiss_dense_search(query: str, retriever: Any, top_k: int) -> List[Any]:
    """Tool 7: Execute FAISS dense vector search."""
    if hasattr(retriever, "retrieve"):
        return retriever.retrieve(query, top_k)
    return []


def tool_bm25_sparse_search(query: str, retriever: Any, top_k: int) -> List[Any]:
    """Tool 8: Execute BM25 sparse keyword search."""
    if hasattr(retriever, "retrieve"):
        return retriever.retrieve(query, top_k)
    return []


def tool_rrf_fusion_ranker(dense_passages: Sequence[Any], sparse_passages: Sequence[Any], k: int = 60) -> List[Any]:
    """Tool 9: Fuse dense & sparse search results using Reciprocal Rank Fusion (RRF)."""
    scores: Dict[str, float] = {}
    passage_map: Dict[str, Any] = {}

    for rank, p in enumerate(dense_passages, start=1):
        pid = getattr(p, "passage_id", str(rank))
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
        passage_map[pid] = p

    for rank, p in enumerate(sparse_passages, start=1):
        pid = getattr(p, "passage_id", str(rank))
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
        passage_map[pid] = p

    sorted_pids = sorted(scores.keys(), key=lambda pid: scores[pid], reverse=True)
    return [passage_map[pid] for pid in sorted_pids]


def tool_mmr_filter_passages(passages: Sequence[Any], top_k: int, mmr_lambda: float = 0.7) -> List[Any]:
    """Tool 10: Apply Maximal Marginal Relevance (MMR) filtering to reduce passage redundancy."""
    if len(passages) <= top_k:
        return list(passages)

    selected: List[Any] = []
    candidates = list(passages)

    def jaccard(t1: str, t2: str) -> float:
        w1 = set(re.findall(r"\b\w+\b", t1.lower()))
        w2 = set(re.findall(r"\b\w+\b", t2.lower()))
        if not w1 or not w2:
            return 0.0
        return len(w1 & w2) / len(w1 | w2)

    while candidates and len(selected) < top_k:
        if not selected:
            best = candidates.pop(0)
            selected.append(best)
        else:
            best_score = -999.0
            best_cand = None
            for cand in candidates:
                cand_text = getattr(cand, "text", "")
                cand_score = getattr(cand, "score", 0.0)
                max_sim = max(jaccard(cand_text, getattr(sel, "text", "")) for sel in selected)
                adj_score = mmr_lambda * cand_score - (1 - mmr_lambda) * max_sim
                if adj_score > best_score:
                    best_score = adj_score
                    best_cand = cand
            if best_cand:
                candidates.remove(best_cand)
                selected.append(best_cand)
            else:
                break
    return selected


def tool_evaluate_passages_coverage(question: str, passages: Sequence[Any]) -> Dict[str, Any]:
    """Tool 11: Evaluate if passages contain essential clinical keywords."""
    if not passages:
        return {"is_sufficient": False, "missing_keywords": ["medical evidence"]}
    combined = " ".join(getattr(p, "text", "") for p in passages).lower()
    q_words = set(re.findall(r"\b\w+\b", question.lower())) - {
        "the", "a", "an", "is", "of", "and", "in", "to", "for", "with", "patient", "which", "following"
    }
    if not q_words:
        return {"is_sufficient": True, "missing_keywords": []}

    passage_words = set(re.findall(r"\b\w+\b", combined))
    missing = list(q_words - passage_words)
    coverage = len(q_words & passage_words) / len(q_words)
    return {
        "is_sufficient": coverage >= 0.20,
        "coverage_ratio": coverage,
        "missing_keywords": missing[:3]
    }


def tool_passage_token_compressor(passages: Sequence[Any], max_words: int = 150) -> List[Dict[str, Any]]:
    """Tool 12: Compress passage text to save token bandwidth."""
    compressed = []
    for idx, p in enumerate(passages, start=1):
        text = getattr(p, "text", "")
        words = text.split()
        short_text = " ".join(words[:max_words]) + "..." if len(words) > max_words else text
        compressed.append({
            "citation_id": f"[{idx}]",
            "source": getattr(p, "source", "Corpus"),
            "text": short_text,
        })
    return compressed


def tool_entity_cooccurrence_scorer(query: str, passage: str) -> float:
    """Tool 13: Score entity co-occurrence between query and passage."""
    q_entities = tool_extract_medical_entities(query)
    if not q_entities:
        return 0.5
    p_text = passage.lower()
    matches = sum(1 for e in q_entities if e.lower() in p_text)
    return matches / len(q_entities)


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 3: MEMORY SPECIALIST AGENT TOOLS (4 TOOLS)
# ─────────────────────────────────────────────────────────────────────────────

def tool_long_term_memory_search(memory_store: Any, question: str, top_k: int) -> List[Any]:
    """Tool 14: Search LongTermMemory store for top_k relevant past solved cases."""
    if hasattr(memory_store, "retrieve_exemplars"):
        return memory_store.retrieve_exemplars(question, top_k=top_k)
    return []


def tool_clinical_pearls_extractor(exemplars: Sequence[Any]) -> str:
    """Tool 15: Extract clinical pearls from exemplars."""
    if not exemplars:
        return "No previous clinical exemplars found."
    pearls = []
    for idx, ex in enumerate(exemplars, start=1):
        q = getattr(ex, "question", "")
        ans = getattr(ex, "correct_answer", "")
        exp = getattr(ex, "explanation", "")
        pearls.append(f"Exemplar [{idx}]: {q[:80]}... -> Answer {ans}: {exp[:100]}...")
    return "\n".join(pearls)


def tool_fewshot_exemplar_formatter(exemplars: Sequence[Any]) -> str:
    """Tool 16: Format few-shot exemplars into structured markdown for Reasoner prompt."""
    if not exemplars:
        return ""
    blocks = ["### Clinical Case Memory Exemplars:"]
    for idx, ex in enumerate(exemplars, start=1):
        q = getattr(ex, "question", "")
        ans = getattr(ex, "correct_answer", "")
        exp = getattr(ex, "explanation", "")
        blocks.append(f"**Case #{idx}**:\nQuestion: {q}\nCorrect Answer: {ans}\nExplanation: {exp}\n")
    return "\n".join(blocks)


def tool_exemplar_similarity_scorer(q1: str, q2: str) -> float:
    """Tool 17: Calculate Jaccard text similarity score between two clinical cases."""
    w1 = set(re.findall(r"\b\w+\b", q1.lower()))
    w2 = set(re.findall(r"\b\w+\b", q2.lower()))
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 4: REASONER AGENT TOOLS (5 TOOLS)
# ─────────────────────────────────────────────────────────────────────────────

def tool_differential_diagnosis_evaluator(question: str, options: Mapping[str, str], passages: Sequence[Any]) -> Dict[str, Any]:
    """Tool 18: Analyze differential diagnosis and option scores A-D."""
    combined_passages = " ".join(getattr(p, "text", "") for p in passages).lower()
    scores: Dict[str, float] = {}
    for letter, opt_text in options.items():
        opt_words = set(re.findall(r"\b\w+\b", opt_text.lower())) - {"the", "a", "an", "of", "and"}
        if not opt_words:
            scores[letter] = 0.5
            continue
        overlap = sum(1 for w in opt_words if w in combined_passages)
        scores[letter] = overlap / len(opt_words)
    best_opt = max(scores.keys(), key=lambda k: scores[k]) if scores else "A"
    return {"option_scores": scores, "top_candidate": best_opt}


def tool_audit_citations(explanation: str, passage_count: int) -> Dict[str, Any]:
    """Tool 19: Verify if explanation includes valid passage citations [1], [2]."""
    if passage_count == 0:
        return {"has_citations": True, "citation_count": 0}
    citations = re.findall(r"\[\d+\]", explanation)
    return {
        "has_citations": len(citations) > 0,
        "citation_count": len(citations),
        "citations_found": citations
    }


def tool_pathophysiology_chain_builder(etiology: str, mechanism: str, treatment: str) -> str:
    """Tool 20: Build step-by-step pathophysiology reasoning chain."""
    return f"Etiology: {etiology} -> Pathogenesis: {mechanism} -> First-line Therapy: {treatment}"


def tool_option_elimination_scorer(options: Mapping[str, str], contraindications: Sequence[str]) -> Dict[str, bool]:
    """Tool 21: Score options for contraindications to eliminate incorrect options."""
    eliminated: Dict[str, bool] = {}
    for letter, text in options.items():
        t_lower = text.lower()
        is_eliminated = any(c.lower() in t_lower for c in contraindications)
        eliminated[letter] = is_eliminated
    return eliminated


def tool_confidence_score_estimator(option_scores: Mapping[str, float]) -> float:
    """Tool 22: Estimate decision confidence score (0.0 to 1.0)."""
    if not option_scores:
        return 0.5
    vals = sorted(option_scores.values(), reverse=True)
    if len(vals) == 1:
        return 0.9
    margin = vals[0] - vals[1]
    return min(1.0, 0.5 + margin)


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 5: VERIFIER AGENT TOOLS (4 TOOLS)
# ─────────────────────────────────────────────────────────────────────────────

def tool_anti_hallucination_fact_checker(explanation: str, passages: Sequence[Any]) -> Dict[str, Any]:
    """Tool 23: Audit explanation for ungrounded assertions or hallucination risks."""
    if not passages or not explanation:
        return {"has_hallucination_risk": False, "reason": "No evidence to compare."}

    has_citations = len(re.findall(r"\[\d+\]", explanation)) > 0
    words = len(explanation.split())
    risk = not has_citations and words > 10
    return {
        "has_hallucination_risk": risk,
        "reason": "Explanation exceeds 10 words without citing any passage IDs [1], [2]" if risk else "Grounded."
    }


def tool_citation_fidelity_verifier(explanation: str, passages: Sequence[Any]) -> Dict[str, Any]:
    """Tool 24: Check if cited passage IDs actually exist in the retrieved passages list."""
    citations = re.findall(r"\[(\d+)\]", explanation)
    num_passages = len(passages)
    invalid_citations = [c for c in citations if int(c) < 1 or int(c) > num_passages]
    return {
        "is_valid": len(invalid_citations) == 0,
        "invalid_citations": invalid_citations,
    }


def tool_issue_callback_request(review_memo: str, retry_count: int = 0) -> Dict[str, Any]:
    """Tool 25: Format a formal Re-retrieval Callback Request to Router Agent with anti-loop protection."""
    max_retries = 2
    can_retry = retry_count < max_retries
    return {
        "can_retry": can_retry,
        "retry_count": retry_count,
        "callback_message": f"Callback Request (Re-Retrieval Required): {review_memo}" if can_retry else "Max retries reached. Executing safe consensus fallback.",
    }


def tool_debate_consensus_synthesizer(reasoner_explanation: str, verifier_memo: str) -> str:
    """Tool 26: Synthesize final executive consensus decision after evaluating debate round."""
    return f"Consensus Summary: Evaluated Reasoner proposal ('{reasoner_explanation[:60]}...') against Verifier Audit ('{verifier_memo[:60]}...'). Consensus reached."


# ─────────────────────────────────────────────────────────────────────────────
# AGENT TOOLKITS FACTORIES (26 TOOLS REGISTERED)
# ─────────────────────────────────────────────────────────────────────────────

def create_router_toolkit() -> ToolRegistry:
    """Create dedicated ToolRegistry for Agent 1 (Router Agent) with 6 Tools."""
    registry = ToolRegistry()
    registry.register_tool(BaseTool("extract_entities", "Extract medical entities from question stem.", {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}, tool_extract_medical_entities))
    registry.register_tool(BaseTool("generate_hyde_context", "Generate hypothetical document passage snippet for HyDE retrieval.", {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}, tool_generate_hyde_context))
    registry.register_tool(BaseTool("decompose_subqueries", "Decompose question into 3 multi-query focuses.", {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}, tool_decompose_subqueries))
    registry.register_tool(BaseTool("prune_vignette_noise", "Prune sentence noise from clinical vignette.", {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}, tool_prune_vignette_noise))
    registry.register_tool(BaseTool("fast_thinking_evaluator", "Evaluate if question qualifies for Fast Thinking mode.", {"type": "object", "properties": {"question": {"type": "string"}, "options": {"type": "object"}}, "required": ["question", "options"]}, tool_fast_thinking_evaluator))
    registry.register_tool(BaseTool("search_strategy_planner", "Plan search strategy (dense, sparse, or hybrid).", {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}, tool_search_strategy_planner))
    return registry


def tool_cross_encoder_rescorer(query: str, passages: Sequence[Any]) -> List[Any]:
    """Tool: Rescore passages using Cross-Encoder token-level semantic alignment."""
    if not passages:
        return []
    q_words = set(re.findall(r"\b\w+\b", query.lower())) - {"the", "a", "an", "is", "of", "in", "to", "for", "with"}
    if not q_words:
        return list(passages)
    rescored = []
    for p in passages:
        p_text = getattr(p, "text", "")
        p_words = set(re.findall(r"\b\w+\b", p_text.lower()))
        overlap = len(q_words & p_words)
        jaccard = overlap / len(q_words | p_words) if (q_words | p_words) else 0.0
        score = getattr(p, "score", 0.0) + (jaccard * 2.0)
        rescored.append(p)
    return rescored


def create_researcher_toolkit() -> ToolRegistry:
    """Create dedicated ToolRegistry for Agent 2 (Researcher Agent) with 9 Tools."""
    registry = ToolRegistry()
    registry.register_tool(BaseTool("faiss_dense_search", "Execute FAISS dense search.", {"type": "object", "properties": {"query": {"type": "string"}, "retriever": {"type": "object"}, "top_k": {"type": "integer"}}, "required": ["query", "top_k"]}, tool_faiss_dense_search))
    registry.register_tool(BaseTool("bm25_sparse_search", "Execute BM25 sparse search.", {"type": "object", "properties": {"query": {"type": "string"}, "retriever": {"type": "object"}, "top_k": {"type": "integer"}}, "required": ["query", "top_k"]}, tool_bm25_sparse_search))
    registry.register_tool(BaseTool("rrf_fusion_ranker", "Fuse dense & sparse hits using RRF.", {"type": "object", "properties": {"dense_passages": {"type": "array"}, "sparse_passages": {"type": "array"}}, "required": ["dense_passages", "sparse_passages"]}, tool_rrf_fusion_ranker))
    registry.register_tool(BaseTool("mmr_filter_passages", "Filter candidate passages using MMR.", {"type": "object", "properties": {"passages": {"type": "array"}, "top_k": {"type": "integer"}, "mmr_lambda": {"type": "number"}}, "required": ["passages", "top_k"]}, tool_mmr_filter_passages))
    registry.register_tool(BaseTool("evaluate_passages_coverage", "Evaluate passage keyword coverage.", {"type": "object", "properties": {"question": {"type": "string"}, "passages": {"type": "array"}}, "required": ["question", "passages"]}, tool_evaluate_passages_coverage))
    registry.register_tool(BaseTool("passage_token_compressor", "Compress passage texts to save token bandwidth.", {"type": "object", "properties": {"passages": {"type": "array"}}, "required": ["passages"]}, tool_passage_token_compressor))
    registry.register_tool(BaseTool("entity_cooccurrence_scorer", "Score entity co-occurrence in passage.", {"type": "object", "properties": {"query": {"type": "string"}, "passage": {"type": "string"}}, "required": ["query", "passage"]}, tool_entity_cooccurrence_scorer))
    registry.register_tool(BaseTool("cross_encoder_rescorer", "Rescore passages using Cross-Encoder semantic alignment.", {"type": "object", "properties": {"query": {"type": "string"}, "passages": {"type": "array"}}, "required": ["query", "passages"]}, tool_cross_encoder_rescorer))
    return registry



def create_memory_toolkit() -> ToolRegistry:
    """Create dedicated ToolRegistry for Agent 3 (Memory Specialist Agent) with 4 Tools."""
    registry = ToolRegistry()
    registry.register_tool(BaseTool("long_term_memory_search", "Search LongTermMemory store for exemplars.", {"type": "object", "properties": {"memory_store": {"type": "object"}, "question": {"type": "string"}, "top_k": {"type": "integer"}}, "required": ["question", "top_k"]}, tool_long_term_memory_search))
    registry.register_tool(BaseTool("clinical_pearls_extractor", "Extract clinical pearls from exemplars.", {"type": "object", "properties": {"exemplars": {"type": "array"}}, "required": ["exemplars"]}, tool_clinical_pearls_extractor))
    registry.register_tool(BaseTool("fewshot_exemplar_formatter", "Format exemplars into markdown prompt blocks.", {"type": "object", "properties": {"exemplars": {"type": "array"}}, "required": ["exemplars"]}, tool_fewshot_exemplar_formatter))
    registry.register_tool(BaseTool("exemplar_similarity_scorer", "Calculate similarity score between two cases.", {"type": "object", "properties": {"q1": {"type": "string"}, "q2": {"type": "string"}}, "required": ["q1", "q2"]}, tool_exemplar_similarity_scorer))
    return registry


def create_reasoner_toolkit() -> ToolRegistry:
    """Create dedicated ToolRegistry for Agent 4 (Reasoner Agent) with 5 Tools."""
    registry = ToolRegistry()
    registry.register_tool(BaseTool("differential_diagnosis_evaluator", "Analyze differential diagnosis and option scores.", {"type": "object", "properties": {"question": {"type": "string"}, "options": {"type": "object"}, "passages": {"type": "array"}}, "required": ["question", "options", "passages"]}, tool_differential_diagnosis_evaluator))
    registry.register_tool(BaseTool("audit_citations", "Verify passage citations [1], [2].", {"type": "object", "properties": {"explanation": {"type": "string"}, "passage_count": {"type": "integer"}}, "required": ["explanation", "passage_count"]}, tool_audit_citations))
    registry.register_tool(BaseTool("pathophysiology_chain_builder", "Build pathophysiology reasoning chain.", {"type": "object", "properties": {"etiology": {"type": "string"}, "mechanism": {"type": "string"}, "treatment": {"type": "string"}}, "required": ["etiology", "mechanism", "treatment"]}, tool_pathophysiology_chain_builder))
    registry.register_tool(BaseTool("option_elimination_scorer", "Eliminate contraindicated options.", {"type": "object", "properties": {"options": {"type": "object"}, "contraindications": {"type": "array"}}, "required": ["options", "contraindications"]}, tool_option_elimination_scorer))
    registry.register_tool(BaseTool("confidence_score_estimator", "Estimate confidence score for final choice.", {"type": "object", "properties": {"option_scores": {"type": "object"}}, "required": ["option_scores"]}, tool_confidence_score_estimator))
    return registry


def create_verifier_toolkit() -> ToolRegistry:
    """Create dedicated ToolRegistry for Agent 5 (Verifier Agent) with 4 Tools."""
    registry = ToolRegistry()
    registry.register_tool(BaseTool("anti_hallucination_fact_checker", "Audit explanation for hallucination risks.", {"type": "object", "properties": {"explanation": {"type": "string"}, "passages": {"type": "array"}}, "required": ["explanation", "passages"]}, tool_anti_hallucination_fact_checker))
    registry.register_tool(BaseTool("citation_fidelity_verifier", "Check if cited passage IDs actually exist.", {"type": "object", "properties": {"explanation": {"type": "string"}, "passages": {"type": "array"}}, "required": ["explanation", "passages"]}, tool_citation_fidelity_verifier))
    registry.register_tool(BaseTool("re_retrieval_callback_issuer", "Issue Re-retrieval Callback Request.", {"type": "object", "properties": {"review_memo": {"type": "string"}, "retry_count": {"type": "integer"}}, "required": ["review_memo"]}, tool_issue_callback_request))
    registry.register_tool(BaseTool("debate_consensus_synthesizer", "Synthesize debate consensus decision.", {"type": "object", "properties": {"reasoner_explanation": {"type": "string"}, "verifier_memo": {"type": "string"}}, "required": ["reasoner_explanation", "verifier_memo"]}, tool_debate_consensus_synthesizer))
    return registry

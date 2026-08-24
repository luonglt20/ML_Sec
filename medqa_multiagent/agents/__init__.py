"""The 5 Independent AI Agents, Dedicated Toolkits & DAG Engine package for MedQA Multi-Agent pipeline.

Contains:
1. Pure Python DAG Execution Engine (dag.py): DAGNode, DAGGraph, DAGRunner (Concurrent Execution).
2. RouterAgent & RouterToolkit (router_agent.py): Query classification, HyDE, multi-query formulation.
3. ResearcherAgent & ResearcherToolkit (researcher_agent.py): RAG Search (FAISS + BM25 + MMR), Backtracking evaluation, Research Brief.
4. MemoryAgent & MemoryToolkit (memory_agent.py): Long-term case memory exemplars retrieval.
5. ReasonerAgent & ReasonerToolkit (reasoner_agent.py): Step-by-step CoT reasoning with citation grounding [1], [2].
6. VerifierAgent & VerifierToolkit (verifier_agent.py): Peer review, Anti-Hallucination Audit, Callback & Debate consensus.
"""

from .dag import DAGGraph, DAGNode, DAGRunner
from .memory_agent import MemoryAgent, MemoryAgentOutput, MemoryToolkit
from .reasoner_agent import ReasonerAgent, ReasonerOutput, ReasonerToolkit
from .researcher_agent import (
    ResearcherAgent,
    ResearcherEvalOutput,
    ResearcherSynthesisOutput,
    ResearcherToolkit,
)
from .router_agent import RouterAgent, RouterOutput, RouterToolkit
from .verifier_agent import VerifierAgent, VerifierAuditOutput, VerifierToolkit

__all__ = [
    "DAGNode",
    "DAGGraph",
    "DAGRunner",
    "RouterAgent",
    "RouterOutput",
    "RouterToolkit",
    "ResearcherAgent",
    "ResearcherEvalOutput",
    "ResearcherSynthesisOutput",
    "ResearcherToolkit",
    "MemoryAgent",
    "MemoryAgentOutput",
    "MemoryToolkit",
    "ReasonerAgent",
    "ReasonerOutput",
    "ReasonerToolkit",
    "VerifierAgent",
    "VerifierAuditOutput",
    "VerifierToolkit",
]

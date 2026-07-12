"""Deterministic test doubles for the LLM client and RAG interfaces.

Per this project's Testing Decisions, the LLM client and the RAG module's
retrieval/embedding/vector-index interfaces are the dependency-injection
points tests substitute with scripted fakes rather than mocking any
agent/RAG internals.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from medqa_multiagent.llm_client import LLMResponse
from medqa_multiagent.rag.retriever import Passage


class FakeLLMClient:
    """Returns scripted response texts in order, one per `complete` call.

    Records every call's kwargs in `self.calls`, so tests can assert on how
    many times (and with what parameters) the client was actually invoked --
    e.g. to prove a cache hit avoided a call, or that the Verifier's
    single-pass behavior doesn't trigger a second Reasoner call.
    """

    def __init__(self, responses: List[str]) -> None:
        self._responses = list(responses)
        self.calls: List[dict] = []

    def complete(
        self,
        role: str,
        prompt: str,
        model: str,
        temperature: float,
        retrieved_context_id: Optional[str] = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "role": role,
                "prompt": prompt,
                "model": model,
                "temperature": temperature,
                "retrieved_context_id": retrieved_context_id,
            }
        )
        if not self._responses:
            raise AssertionError("FakeLLMClient ran out of scripted responses")
        text = self._responses.pop(0)
        return LLMResponse(
            text=text,
            model=model,
            temperature=temperature,
            prompt=prompt,
            role=role,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            latency_seconds=0.01,
            retrieved_context_id=retrieved_context_id,
        )


class FakeRetriever:
    """Returns a scripted list of `Passage`s for every `retrieve` call.

    Records every call's `(query, top_k)` in `self.calls`, so tests can
    assert on what query text/`top_k` a variant actually retrieved with.
    """

    def __init__(self, passages: List[Passage]) -> None:
        self._passages = list(passages)
        self.calls: List[Tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int) -> List[Passage]:
        self.calls.append((query, top_k))
        return list(self._passages[:top_k])


class FakeEmbeddingClient:
    """Deterministic, scripted `EmbeddingClient` -- no real model weights.

    Embeds any text to a 1-dimensional vector equal to a value looked up
    (or, if absent, derived deterministically from the text) in
    `vectors_by_text`, so tests can construct predictable
    query/passage-vector relationships without a real encoder.
    """

    def __init__(self, vectors_by_text: Optional[Dict[str, List[float]]] = None) -> None:
        self._vectors_by_text = dict(vectors_by_text or {})
        self.embedded_queries: List[str] = []
        self.embedded_passage_batches: List[Sequence[str]] = []

    def _vector_for(self, text: str) -> List[float]:
        if text in self._vectors_by_text:
            return self._vectors_by_text[text]
        return [float(len(text))]

    def embed_query(self, text: str) -> List[float]:
        self.embedded_queries.append(text)
        return self._vector_for(text)

    def embed_passages(self, texts: Sequence[str]) -> List[List[float]]:
        self.embedded_passage_batches.append(texts)
        return [self._vector_for(text) for text in texts]


class FakeVectorIndex:
    """Deterministic, scripted `VectorIndex` -- no real FAISS index.

    Ignores the actual query vector and simply returns the first `top_k`
    entries of a fixed, scripted `(passage_id, score)` list, in order --
    sufficient for testing `IndexBackedRetriever`'s wiring (does it pass
    `top_k` through, does it resolve ids to passage records correctly)
    without needing real vector search.
    """

    def __init__(self, hits: List[Tuple[str, float]]) -> None:
        self._hits = list(hits)
        self.searched_vectors: List[Sequence[float]] = []
        self.searched_top_k: List[int] = []

    def search(self, query_vector: Sequence[float], top_k: int) -> List[Tuple[str, float]]:
        self.searched_vectors.append(query_vector)
        self.searched_top_k.append(top_k)
        return list(self._hits[:top_k])

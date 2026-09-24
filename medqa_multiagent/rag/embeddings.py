"""The embedding client interface used by the RAG module.

Mirrors `llm_client.LLMClient`'s shape: one small `Protocol` (`EmbeddingClient`)
that every caller programs against, plus a single concrete implementation
(`MedCptEmbeddingClient`) backing it with NCBI's MedCPT query/article
encoders (see `DESIGN.md` SS6). `MedCptEmbeddingClient` is the
dependency-injection seam's *real* side; tests substitute a scripted fake
(`FakeEmbeddingClient` in `tests/fakes.py`) instead of loading actual model
weights, exactly as `llm_client.OpenAICompatibleClient` is never exercised
directly in tests either.

`transformers`/`torch` are imported lazily, inside `MedCptEmbeddingClient`,
so importing this module (or anything that transitively imports it, e.g.
`rag.retriever`) never requires those optional, heavy dependencies unless a
`MedCptEmbeddingClient` is actually instantiated. Install them via the
`rag` extra (`pip install -e ".[rag]"`) before running
`scripts/build_rag_index.py` or using a real (non-fake) retriever.
"""

from __future__ import annotations

from typing import List, Protocol, Sequence

#: The two MedCPT encoders (NCBI, biomedical-domain-specific) this project
#: uses -- one for queries, one for articles/passages, per DESIGN.md SS6.
#: Kept as module-level constants (like `llm_client._PROVIDER_REGISTRY`) so
#: there is exactly one place model identifiers live.
MEDCPT_QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
MEDCPT_ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"


class EmbeddingClient(Protocol):
    """The one interface the RAG module embeds queries/passages through."""

    def embed_query(self, text: str) -> List[float]: ...

    def embed_passages(self, texts: Sequence[str]) -> List[List[float]]: ...


class MedCptEmbeddingClient:
    """`EmbeddingClient` backed by NCBI's MedCPT query/article encoders.

    Query and passage text are embedded with *different* encoders (MedCPT
    is a dual-encoder model, asymmetric by design -- see DESIGN.md SS6),
    which is why this class, unlike a single-encoder embedding client,
    exposes two distinct methods rather than one `embed(text)`.
    """

    def __init__(self, device: str = "cpu") -> None:
        # Imported lazily -- see module docstring.
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._device = device
        self._query_tokenizer = AutoTokenizer.from_pretrained(MEDCPT_QUERY_MODEL)
        self._query_model = AutoModel.from_pretrained(MEDCPT_QUERY_MODEL).to(device).eval()
        self._article_tokenizer = AutoTokenizer.from_pretrained(MEDCPT_ARTICLE_MODEL)
        self._article_model = AutoModel.from_pretrained(MEDCPT_ARTICLE_MODEL).to(device).eval()

    def _encode(self, tokenizer, model, texts: Sequence[str]) -> List[List[float]]:
        inputs = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=512,
        ).to(self._device)
        with self._torch.no_grad():
            outputs = model(**inputs)
        # MedCPT's pooled embedding is the [CLS] token's last-hidden-state
        # vector, per the model card / MedCPT paper (Jin et al. 2023).
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings.cpu().numpy().tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._encode(self._query_tokenizer, self._query_model, [text])[0]

    def embed_passages(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._encode(self._article_tokenizer, self._article_model, texts)

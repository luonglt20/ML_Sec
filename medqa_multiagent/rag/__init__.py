"""The RAG (retrieval-augmented generation) module.

Owns everything the V1 (RAG-only) variant -- and every later variant that
retains RAG (V2-V4) -- needs to turn a question into retrieved textbook
passages: corpus chunking, MedCPT embeddings, a local FAISS flat index, an
injectable `Retriever` interface (the dependency-injection seam tests
substitute a fake for, mirroring `llm_client.LLMClient`), and on-disk
caching for both embedding and retrieval calls.

Building the index itself (downloading the MedQA "textbooks" corpus,
chunking, embedding, and persisting a FAISS index to disk) is a one-time,
network- and compute-heavy setup step performed by
`scripts/build_rag_index.py` -- exactly analogous to how
`scripts/download_medqa.py` populates `data/dev.jsonl`/`data/test.jsonl`.
Nothing in this package is imported or run automatically at pipeline
run-time beyond loading the already-built index from disk.
"""

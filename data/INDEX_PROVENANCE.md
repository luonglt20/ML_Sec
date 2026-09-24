# Bundled partial RAG index

The four files in `data/rag_index/` are the exact NumPy/BM25 index used for
this project's 100-question paired trial. It contains 1,109 passages from
Anatomy_Gray, not the full MedRAG textbook corpus. It was derived from the
local `GK/data/rag_index_real20` index; `vectors.npy` contains the exact
saved vectors in `passage_ids.json` order to avoid the FAISS/PyTorch native
library conflict observed on macOS.

SHA-256 of `passages.jsonl`:
`6f2db62abe777dd7d46c14072811dca3c6a8ced66cf6a96cadbf9f5d7bb3aeb6`.

Run with `MEDQA_VECTOR_BACKEND=numpy`. Do not compare results using this index
directly against runs with a full textbook index.

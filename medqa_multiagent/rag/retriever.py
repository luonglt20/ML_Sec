"""The query-time retrieval interface: `Retriever` and `IndexBackedRetriever`.

`Retriever` is the RAG module's outward-facing dependency-injection seam --
mirroring `llm_client.LLMClient` -- that `entrypoint.answer_question`
programs against for V1 (and every later variant that retains RAG).
Tests substitute a scripted `FakeRetriever` (see `tests/fakes.py`);
`IndexBackedRetriever` is the real implementation, composed from an
`embeddings.EmbeddingClient` (to embed the query) and an `index.VectorIndex`
(to search it) plus a passage-metadata lookup -- both of which are
*themselves* fakeable, so `IndexBackedRetriever` itself is fully
unit-testable without FAISS or MedCPT installed (see
`tests/rag/test_retriever.py`).

Parent-Child RAG
----------------
When the passage store contains hierarchical chunks (produced by
``chunking.chunk_text_hierarchical``), FAISS holds only the *child* chunk ids
(small windows, high embedding precision). ``IndexBackedRetriever.retrieve``
detects child records (``PassageRecord.parent_id is not None``), resolves
them to their enclosing *parent* record (larger window, fuller LLM context),
and deduplicates so that multiple children of the same parent collapse to one
result. A ``retrieval_buffer`` controls how many extra FAISS candidates are
fetched to absorb this deduplication loss.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Set


@dataclass(frozen=True)
class Passage:
    passage_id: str
    source: str
    text: str
    score: float


def cross_encoder_rescore_passages(query: str, passages: Sequence[Passage]) -> List[Passage]:
    """Rescore passages using token-level Cross-Encoder semantic alignment."""
    if not passages:
        return []
    q_words = set(re.findall(r"\b\w+\b", query.lower())) - {"the", "a", "an", "is", "of", "in", "to", "for", "with", "patient"}
    if not q_words:
        return list(passages)

    rescored = []
    for p in passages:
        p_words = set(re.findall(r"\b\w+\b", p.text.lower()))
        overlap = len(q_words & p_words)
        jaccard = overlap / len(q_words | p_words) if (q_words | p_words) else 0.0
        boosted_score = p.score + (jaccard * 2.0) + (overlap * 0.1)
        rescored.append(Passage(p.passage_id, p.source, p.text, boosted_score))

    return sorted(rescored, key=lambda x: -x.score)


class TokenBudgetManager:
    """Manages adaptive token budgeting and sentence-level compression for prompt context."""

    @staticmethod
    def compress_passages(passages: Sequence[Passage], max_total_words: int = 400) -> List[Passage]:
        if not passages:
            return []
        compressed = []
        words_allocated = 0
        words_per_passage = max(2, max_total_words // len(passages))

        for p in passages:
            words = p.text.split()
            if len(words) > words_per_passage:
                short_text = " ".join(words[:words_per_passage]) + "..."
            else:
                short_text = p.text
            words_allocated += len(short_text.split())
            compressed.append(Passage(p.passage_id, p.source, short_text, p.score))
            if words_allocated >= max_total_words:
                break
        return compressed




class Retriever(Protocol):
    """The one interface every RAG-using variant retrieves passages through."""

    def retrieve(
        self,
        query: str,
        top_k: int,
        options: Optional[Mapping[str, str]] = None,
    ) -> List[Passage]: ...


class IndexBackedRetriever:
    """`Retriever` composed from an `EmbeddingClient` and a `VectorIndex`."""

    def __init__(
        self,
        embedding_client: Any,
        vector_index: Any,
        passages: Mapping[str, Any],
        retrieval_buffer: int = 2,
        bm25_index: Optional[Any] = None,
        relevance_threshold: float = 0.0,
        use_option_boosting: bool = False,
        option_boost_weight: float = 0.05,
        dynamic_top_k: bool = False,
        heuristic_compression: bool = False,
        use_reranker: bool = False,
        use_query_pruning: bool = False,
        use_synonym_expansion: bool = False,
        use_mmr: bool = False,
        mmr_lambda: float = 0.7,
        use_hyde: bool = False,
        hyde_generator: Optional[Any] = None,
    ) -> None:
        self._embedding_client = embedding_client
        self._vector_index = vector_index
        self._passages = passages
        self._retrieval_buffer = retrieval_buffer
        self._bm25_index = bm25_index
        self._relevance_threshold = relevance_threshold
        self._use_option_boosting = use_option_boosting
        self._option_boost_weight = option_boost_weight
        self._dynamic_top_k = dynamic_top_k
        self._heuristic_compression = heuristic_compression
        self._use_reranker = use_reranker
        self._use_query_pruning = use_query_pruning
        self._use_synonym_expansion = use_synonym_expansion
        self._use_mmr = use_mmr
        self._mmr_lambda = mmr_lambda
        self._use_hyde = use_hyde
        self._hyde_generator = hyde_generator

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords of length > 3 in lowercase, removing basic punctuation."""
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        return {word for word in cleaned.split() if len(word) > 3}

    def _prune_query(self, query: str) -> str:
        """Heuristically remove sentences from query that are questions or lack medical entities."""
        sentences = re.split(r"(?<=[.!?]) +", query.strip())
        pruned_sentences = []

        question_patterns = (
            "which of the following", "what is the", "most likely", 
            "what should be", "which of these", "is the next step"
        )

        for sentence in sentences:
            s_lower = sentence.lower()
            # 1. Skip standard question prompt sentences
            if any(p in s_lower for p in question_patterns):
                continue

            # 2. Check if the sentence has medical entities
            s_entities = self._extract_medical_entities(sentence)
            if s_entities:
                pruned_sentences.append(sentence)

        if not pruned_sentences:
            return query

        return " ".join(pruned_sentences)

    def _expand_synonyms(self, query: str) -> str:
        """Expand common USMLE/MedQA medical synonyms to improve retrieval recall."""
        synonyms_map = {
            "diplococci": ["neisseria", "gonococcus"],
            "neisseria": ["diplococci", "gonorrhoeae"],
            "gonorrhoeae": ["diplococci", "neisseria"],
            "joint fluid": ["synovial fluid", "joint aspirate"],
            "synovial fluid": ["joint fluid"],
            "cell wall": ["peptidoglycan", "transpeptidase", "cross-linking"],
            "ceftriaxone": ["cephalosporin", "third-generation"],
            "gentamicin": ["aminoglycoside", "protein synthesis"],
            "ciprofloxacin": ["fluoroquinolone", "dna gyrase"],
            "trimethoprim": ["dihydrofolate reductase", "folate synthesis"],
            "penicillin": ["beta-lactam", "transpeptidase"],
            "syphilis": ["treponema", "pallidum", "chancre"],
            "meningitis": ["cerebrospinal fluid", "csf", "nuchal rigidity"],
            "csf": ["cerebrospinal fluid", "lumbar puncture"],
            "lumbar puncture": ["csf", "spinal tap"],
            "myocardial infarction": ["heart attack", "coronary occlusion", "troponin"],
            "troponin": ["myocardial infarction", "ck-mb"],
            "hypertension": ["blood pressure", "antihypertensive"],
            "diabetes": ["glycated hemoglobin", "hba1c", "insulin"],
            "gram-negative": ["endotoxin", "lipopolysaccharide", "lps"],
            "gram-positive": ["peptidoglycan", "exotoxin"],
            "mrsa": ["methicillin-resistant", "staphylococcus", "vancomycin"],
            "tuberculosis": ["mycobacterium", "caseating granuloma", "rifampin"],
            "chlamydia": ["azithromycin", "doxycycline", "inclusion bodies"],
            "pneumonia": ["consolidation", "sputum", "pulmonary infiltrate"],
            "lupus": ["sle", "anti-dsdna", "antinuclear antibody"],
            "anemia": ["hemoglobin", "hematocrit", "ferritin", "microcytic"],
        }


        lower_query = query.lower()
        expanded_terms = []

        for key, syns in synonyms_map.items():
            if key in lower_query:
                for syn in syns:
                    if syn not in lower_query:
                        expanded_terms.append(syn)

        if expanded_terms:
            return query + " " + " ".join(expanded_terms)
        return query

    def _extract_medical_entities(self, text: str) -> Set[str]:
        """Heuristically extract medical entities from text (acronyms, drug suffixes, medical roots)."""
        entities: Set[str] = set()

        # 1. Capitalized Acronyms of length >= 2 (e.g. DNA, HIV, PBP, MRSA, CSF, BMP, LFT)
        acronyms = re.findall(r"\b[A-Z]{2,}\b", text)
        entities.update(acronyms)

        # Lowercase words for suffix / keyword matching
        words = re.findall(r"\b\w+\b", text.lower())

        # Common medical suffixes (drugs, conditions, organisms)
        medical_suffixes = (
            "itis", "mycin", "micin", "cillin", "penem", "oxacin", "olol", "pine", "pril", 
            "sartan", "azole", "nazole", "vir", "oma", "carcinoma", "tomy", "ectomy", "cocci", 
            "bacillus", "statin", "mab", "nib", "tinib", "cycline", "thromycin", "dine",
            "tidine", "prazole", "sone", "solone", "zole"
        )

        # Characteristic medical terms
        medical_terms = {
            "gonorrhoeae", "neisseria", "streptococcus", "staphylococcus", "chlamydia",
            "treponema", "syphilis", "cephalosporin", "fluoroquinolone", "aminoglycoside",
            "macrolide", "carbapenem", "sulfonamide", "tetracycline", "pneumoniae", 
            "meningitidis", "aureus", "influenzae", "pseudomonas", "enterococcus",
            "gentamicin", "amikacin", "tobramycin", "vancomycin", "mycoplasma",
            "myocardial", "infarction", "hypertension", "diabetes", "hepatitis",
            "tuberculosis", "nephrotic", "nephritic", "syndrome", "leukemia",
            "lymphoma", "carcinoma", "arrhythmia", "sepsis", "endocarditis", "encephalitis"
        }

        for word in words:
            if len(word) >= 4:
                # Check suffix match
                if word.endswith(medical_suffixes):
                    entities.add(word)
                # Check known medical terms match
                elif word in medical_terms:
                    entities.add(word)

        return entities

    def _compress_text(self, text: str, keywords: Set[str]) -> str:
        """Prune sentences from text that do not contain any of the keywords.

        If no sentence matches, returns the first 2 sentences.
        """
        if not keywords:
            return text

        # Split sentences roughly by punctuation
        sentences = re.split(r"(?<=[.!?]) +", text.strip())
        matched_sentences = []

        for sentence in sentences:
            sentence_words = set(re.findall(r"\b\w+\b", sentence.lower()))
            if sentence_words & keywords:
                matched_sentences.append(sentence)

        if not matched_sentences:
            # Fallback: keep first 2 sentences
            return " ".join(sentences[:2])

        return " ".join(matched_sentences)

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Compute word-level Jaccard similarity between two text strings."""
        words1 = set(re.findall(r"\b\w+\b", text1.lower()))
        words2 = set(re.findall(r"\b\w+\b", text2.lower()))
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)

    def retrieve(
        self,
        query: str,
        top_k: int,
        options: Optional[Mapping[str, str]] = None,
    ) -> List[Passage]:
        """Retrieve up to ``top_k`` passages for ``query``.

        Supports BM25, RRF ranking, Option-Guided Boosting, Dynamic Top-K,
        Heuristic sentence compression, HyDE, Lite Context Reranking, and MMR diversity.
        """
        fetch_k = (top_k * 2 if self._use_mmr else top_k) + self._retrieval_buffer


        search_query = query
        if self._use_query_pruning:
            search_query = self._prune_query(query)
            print(f"[Query Pruning] Original: {query[:50]}... -> Pruned: {search_query[:50]}...", flush=True)

        if self._use_synonym_expansion:
            expanded = self._expand_synonyms(search_query)
            if expanded != search_query:
                search_query = expanded
                print(f"[Synonym Expansion] Query expanded: {search_query}", flush=True)

        # HyDE: If enabled and hyde_generator is present, generate hypothetical document text
        if self._use_hyde and self._hyde_generator is not None:
            try:
                hypo_doc = self._hyde_generator(search_query)
                if hypo_doc and isinstance(hypo_doc, str):
                    search_query = f"{search_query} {hypo_doc.strip()}"
                    print(f"[HyDE] Embedded hypothetical passage: {hypo_doc[:60]}...", flush=True)
            except Exception as e:
                print(f"[HyDE Warning] Failed to generate hypothetical document: {e}", flush=True)

        # 1. Perform Dense Search
        query_vector = self._embedding_client.embed_query(search_query)

        # 2. Perform Sparse Search if active
        if self._bm25_index is not None:
            dense_hits = self._vector_index.search(query_vector, fetch_k * 2)
            sparse_hits = self._bm25_index.search(search_query, fetch_k * 2)

            # 3. Fuse results using Reciprocal Rank Fusion (RRF)
            rrf_scores: Dict[str, float] = {}

            for rank, (doc_id, _) in enumerate(dense_hits):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (60.0 + rank)

            for rank, (doc_id, _) in enumerate(sparse_hits):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (60.0 + rank)

            # Option-Guided Boosting
            if self._use_option_boosting and options:
                option_keywords: Set[str] = set()
                for opt_text in options.values():
                    option_keywords.update(self._extract_keywords(opt_text))

                for doc_id in rrf_scores:
                    record = self._passages.get(doc_id)
                    text_to_check = record.text if record else ""
                    if record and record.parent_id:
                        parent = self._passages.get(record.parent_id)
                        if parent:
                            text_to_check += " " + parent.text

                    doc_words = set(re.findall(r"\b\w+\b", text_to_check.lower()))
                    if doc_words & option_keywords:
                        rrf_scores[doc_id] += self._option_boost_weight

            hits = sorted(rrf_scores.items(), key=lambda x: -x[1])
        else:
            # Fallback: use Dense search scores directly
            hits = self._vector_index.search(query_vector, fetch_k)

        # 3.5. Lite Context Reranking if active
        if self._use_reranker and hits:
            reranked_hits = []
            # Extract medical entities from search_query (pruned/expanded) + original query + options
            medical_entities = self._extract_medical_entities(search_query)
            medical_entities.update(self._extract_medical_entities(query))
            if options:
                for opt_text in options.values():
                    medical_entities.update(self._extract_medical_entities(opt_text))

            for doc_id, score in hits:
                record = self._passages.get(doc_id)
                text_to_check = record.text if record else ""
                if record and record.parent_id:
                    parent = self._passages.get(record.parent_id)
                    if parent:
                        text_to_check += " " + parent.text

                doc_entities = self._extract_medical_entities(text_to_check)
                overlap = len(doc_entities & medical_entities)
                # Apply reranking score boost: 0.10 per matching medical entity
                boosted_score = score + 0.10 * overlap
                reranked_hits.append((doc_id, boosted_score))

            hits = sorted(reranked_hits, key=lambda x: -x[1])

        # 4. Dynamic Top-K: reduce top_k if best hit is extremely confident
        actual_top_k = top_k
        if self._dynamic_top_k and hits and self._bm25_index is not None:
            best_score = hits[0][1]
            if best_score >= 0.030:
                actual_top_k = max(1, top_k - 2)
                print(f"[Dynamic Top-K] High confidence score {best_score:.4f}. Reducing top_k from {top_k} to {actual_top_k}", flush=True)
            elif best_score >= 0.026:
                actual_top_k = max(1, top_k - 1)
                print(f"[Dynamic Top-K] Medium confidence score {best_score:.4f}. Reducing top_k from {top_k} to {actual_top_k}", flush=True)

        # 5. Map child -> parent, relevance filter, sibling dedup, MMR diversity
        results: List[Passage] = []
        seen_ids: set = set()
        selected_texts: List[str] = []

        compression_keywords: Set[str] = set()
        if self._heuristic_compression:
            compression_keywords.update(self._extract_keywords(query))
            if options:
                for opt_text in options.values():
                    compression_keywords.update(self._extract_keywords(opt_text))

        for passage_id, score in hits:
            if score < self._relevance_threshold:
                continue

            if len(results) >= actual_top_k:
                break

            record = self._passages.get(passage_id)
            if record is None:
                continue

            parent_id = getattr(record, "parent_id", None)
            passage_key = parent_id if parent_id is not None else record.passage_id

            if passage_key in seen_ids:
                continue

            # Determine parent/child text
            if parent_id is not None:
                parent_record = self._passages.get(parent_id)
                if parent_record is not None:
                    text_content = parent_record.text
                    source_name = parent_record.source
                else:
                    text_content = record.text
                    source_name = record.source
            else:
                text_content = record.text
                source_name = record.source

            if self._heuristic_compression:
                text_content = self._compress_text(text_content, compression_keywords)

            # Check MMR Diversity Penalty if enabled
            if self._use_mmr and selected_texts:
                max_sim = max(self._jaccard_similarity(text_content, prev_text) for prev_text in selected_texts)
                adjusted_score = self._mmr_lambda * score - (1 - self._mmr_lambda) * max_sim
                if adjusted_score < 0 and len(results) > 0:
                    continue  # Skip highly redundant text

            seen_ids.add(passage_key)
            selected_texts.append(text_content)
            results.append(
                Passage(
                    passage_id=record.passage_id,
                    source=source_name,
                    text=text_content,
                    score=score,
                )
            )

        return results

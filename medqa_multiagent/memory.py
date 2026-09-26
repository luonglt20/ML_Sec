"""Long-term Case Memory module for V3 and V4 variants.

Stores solved clinical case exemplars (question, correct option, detailed clinical rationale)
and retrieves the top-k most similar cases for a given new question using BM25 / TF-IDF keyword similarity.
Provides few-shot clinical exemplar guidance to the Reasoner Agent.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union


@dataclass(frozen=True)
class CaseRecord:
    """One solved medical case exemplar stored in long-term memory."""

    case_id: str
    question: str
    options: Dict[str, str]
    correct_answer: str
    explanation: str


class LongTermMemory:
    """Long-term case memory store for MedQA clinical exemplars."""

    def __init__(self, cases: Optional[Sequence[CaseRecord]] = None) -> None:
        self._cases: List[CaseRecord] = list(cases) if cases is not None else []

    def add_case(self, case: CaseRecord) -> None:
        """Add a single solved case exemplar to memory."""
        self._cases.append(case)

    def size(self) -> int:
        """Return total number of cases stored in memory."""
        return len(self._cases)

    def retrieve_exemplars(self, question: str, top_k: int = 2) -> List[CaseRecord]:
        """Retrieve top-k most relevant case exemplars matching query words."""
        if not self._cases or top_k <= 0:
            return []

        # Simple BM25-style keyword matching over case questions & explanations
        query_words = set(re.findall(r"\b\w+\b", question.lower())) - {
            "the", "a", "an", "is", "of", "and", "in", "to", "for", "with", "patient", "which", "following"
        }

        if not query_words:
            return self._cases[:top_k]

        scored_cases = []
        for case in self._cases:
            case_text = f"{case.question} {' '.join(case.options.values())} {case.explanation}".lower()
            case_words = set(re.findall(r"\b\w+\b", case_text))
            overlap = len(query_words & case_words)
            scored_cases.append((case, overlap))

        scored_cases.sort(key=lambda x: -x[1])
        return [c for c, _ in scored_cases[:top_k]]

    def save_to_file(self, path: Union[str, Path]) -> None:
        """Persist memory to a JSON file."""
        data = [asdict(c) for c in self._cases]
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_from_file(cls, path: Union[str, Path]) -> LongTermMemory:
        """Load memory from a JSON file."""
        file_path = Path(path)
        if not file_path.exists():
            return cls()
        data = json.loads(file_path.read_text(encoding="utf-8"))
        cases = [
            CaseRecord(
                case_id=item["case_id"],
                question=item["question"],
                options=item["options"],
                correct_answer=item["correct_answer"],
                explanation=item["explanation"],
            )
            for item in data
        ]
        return cls(cases)


def create_default_memory_store() -> LongTermMemory:
    """Create a default long-term memory store populated with high-yield USMLE exemplars."""
    store = LongTermMemory()
    store.add_case(
        CaseRecord(
            case_id="case_001",
            question="A 24-year-old male presents with urethral discharge. Gram stain reveals intracellular Gram-negative diplococci. What is the first-line treatment?",
            options={"A": "Ceftriaxone plus Azithromycin", "B": "Ciprofloxacin", "C": "Penicillin G", "D": "Trimethoprim-sulfamethoxazole"},
            correct_answer="A",
            explanation="Neisseria gonorrhoeae infection presents as Gram-negative intracellular diplococci. First-line therapy is Ceftriaxone 500mg IM plus Azithromycin to cover co-infection with Chlamydia trachomatis.",
        )
    )
    store.add_case(
        CaseRecord(
            case_id="case_002",
            question="A 65-year-old male with history of smoking presents with sudden left-sided chest pain radiating to jaw and diaphoresis. ECG shows ST elevation in leads II, III, aVF. What is the diagnosis?",
            options={"A": "Inferior Myocardial Infarction", "B": "Anterior Myocardial Infarction", "C": "Aortic Dissection", "D": "Pulmonary Embolism"},
            correct_answer="A",
            explanation="ST-segment elevation in leads II, III, and aVF indicates an acute inferior wall myocardial infarction, usually caused by occlusion of the right coronary artery (RCA).",
        )
    )
    store.add_case(
        CaseRecord(
            case_id="case_003",
            question="A 30-year-old female presents with fever, nuchal rigidity, and altered mental status. CSF examination shows elevated neutrophils, high protein, and low glucose. What is the most likely pathogen?",
            options={"A": "Streptococcus pneumoniae", "B": "Neisseria meningitidis", "C": "Listeria monocytogenes", "D": "Viral Enterovirus"},
            correct_answer="A",
            explanation="Acute bacterial meningitis in adults is most commonly caused by Streptococcus pneumoniae, characterized by purulent CSF with neutrophilic predominance, elevated protein, and low glucose.",
        )
    )
    return store

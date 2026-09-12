"""Frozen per-question retrieval snapshots for paired security experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union

from ..rag.retriever import Passage


@dataclass(frozen=True)
class RetrievalSnapshot:
    question_id: str
    router_query: str
    passages: List[Passage]


class FixedSnapshotRetriever:
    """Retriever returning one frozen passage list regardless of router wording."""

    def __init__(self, passages: Sequence[Passage]) -> None:
        self._passages = list(passages)
        self.calls: List[tuple[str, int]] = []

    def retrieve(
        self,
        query: str,
        top_k: int,
        options: Optional[Mapping[str, str]] = None,
    ) -> List[Passage]:
        self.calls.append((query, top_k))
        return list(self._passages[:top_k])


def snapshot_hash(snapshots: Iterable[RetrievalSnapshot]) -> str:
    payload = [
        {
            "question_id": item.question_id,
            "router_query": item.router_query,
            "passages": [asdict(passage) for passage in item.passages],
        }
        for item in sorted(snapshots, key=lambda value: value.question_id)
    ]
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_snapshots(
    path: Union[str, Path], snapshots: Sequence[RetrievalSnapshot]
) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for item in sorted(snapshots, key=lambda value: value.question_id):
            record = {
                "question_id": item.question_id,
                "router_query": item.router_query,
                "passages": [asdict(passage) for passage in item.passages],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return snapshot_hash(snapshots)


def load_snapshots(path: Union[str, Path]) -> Dict[str, RetrievalSnapshot]:
    snapshots: Dict[str, RetrievalSnapshot] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            data = json.loads(raw)
            question_id = str(data["question_id"])
            if question_id in snapshots:
                raise ValueError(f"duplicate question_id at line {line_number}: {question_id}")
            snapshots[question_id] = RetrievalSnapshot(
                question_id=question_id,
                router_query=str(data.get("router_query", "")),
                passages=[Passage(**item) for item in data["passages"]],
            )
    return snapshots

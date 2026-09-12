#!/usr/bin/env python3
"""Freeze retrieval, run Pair-1 attacks/StruQ defense, and summarize results.

Examples are documented in README.md. Raw predictions and downloaded model
weights remain local; compact JSON/CSV summaries are suitable for versioning.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medqa_multiagent.agents.router_agent import RouterAgent
from medqa_multiagent.client_factory import build_llm_client, build_uncached_llm_client
from medqa_multiagent.config import RunConfig
from medqa_multiagent.data import Question, load_questions
from medqa_multiagent.entrypoint import AnswerResult
from medqa_multiagent.security.harness import answer_with_security_scenario
from medqa_multiagent.env_file import load_env_file
from medqa_multiagent.llm_client import LLMClient, LLMResponse, OllamaClient
from medqa_multiagent.rag.client_factory import build_retriever
from medqa_multiagent.rag.retriever import Passage
from medqa_multiagent.security.attacks import build_attack_payload, deterministic_target_option
from medqa_multiagent.security.dataset import memory_from_dev_questions, stratified_question_sample
from medqa_multiagent.security.local_client import StruQLocalClient
from medqa_multiagent.security.metrics import summarize_evaluation
from medqa_multiagent.security.scenario import (
    AttackFamily,
    AttackPosition,
    AttackSurface,
    DefenseMode,
    ModelTrack,
    SecurityScenario,
)
from medqa_multiagent.security.snapshots import (
    FixedSnapshotRetriever,
    RetrievalSnapshot,
    load_snapshots,
    snapshot_hash,
    write_snapshots,
)
from medqa_multiagent.security.structured import StructuredQuery
from medqa_multiagent.security.wrappers import UNTRUSTED_DATA_POLICY_REVISION


CORE_ATTACKS = [
    AttackFamily.NAIVE,
    AttackFamily.ESCAPE,
    AttackFamily.CONTEXT_IGNORING,
    AttackFamily.FAKE_COMPLETION,
    AttackFamily.COMBINED,
]
RESIDUAL_ATTACKS = [
    AttackFamily.DELIMITER_NEAR_MISS,
    AttackFamily.MULTILINGUAL,
    AttackFamily.ENCODED,
]
EXPLORATORY_ATTACKS = [
    AttackFamily.DOCUMENT_AUTHORITY,
    AttackFamily.CHAINED_COMPLETION,
    AttackFamily.DENSE_AUTHORITY,
    AttackFamily.FEW_SHOT_DEMONSTRATION,
]


class InstrumentedClient:
    """Record exact rendered prompts while preserving structured completion."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client
        self.use_structured_queries = getattr(client, "use_structured_queries", False) is True
        self.responses: List[LLMResponse] = []

    def reset(self) -> None:
        self.responses.clear()

    def complete(
        self,
        role: str,
        prompt: str,
        model: str,
        temperature: float,
        retrieved_context_id: Optional[str] = None,
    ) -> LLMResponse:
        response = self._client.complete(
            role=role,
            prompt=prompt,
            model=model,
            temperature=temperature,
            retrieved_context_id=retrieved_context_id,
        )
        self.responses.append(response)
        return response

    def complete_structured(
        self,
        role: str,
        query: StructuredQuery,
        model: str,
        temperature: float,
        retrieved_context_id: Optional[str] = None,
    ) -> LLMResponse:
        method = getattr(self._client, "complete_structured", None)
        if not callable(method):
            raise TypeError("wrapped client does not support structured completion")
        response = method(
            role=role,
            query=query,
            model=model,
            temperature=temperature,
            retrieved_context_id=retrieved_context_id,
        )
        self.responses.append(response)
        return response

    @property
    def prompt_hash(self) -> str:
        canonical = json.dumps(
            [response.prompt for response in self.responses], ensure_ascii=True
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def response_models(self) -> List[str]:
        return sorted({response.model for response in self.responses})

    @property
    def sanitization_events(self) -> int:
        return int(getattr(self._client, "sanitization_events", 0))

    @property
    def peak_vram_bytes(self) -> int:
        return int(getattr(self._client, "last_peak_vram_bytes", 0))

    @property
    def model_revision(self) -> str:
        return str(getattr(self._client, "model_revision", "provider-managed"))


class LexicalCorpusRetriever:
    """Dependency-free deterministic retrieval over frozen textbook chunks.

    Used only when FAISS/MedCPT is not stable in the current host runtime.
    The corpus itself is still the downloaded MedRAG textbook subset, never
    question labels or hand-authored answer keys.
    """

    _STOPWORDS = {
        "the", "a", "an", "and", "or", "of", "to", "in", "for", "with",
        "is", "are", "was", "were", "what", "which", "following", "patient",
        "presents", "because", "most", "likely", "from", "that", "this",
    }

    def __init__(self, passages_path: str) -> None:
        records: List[Passage] = []
        with Path(passages_path).open("r", encoding="utf-8") as handle:
            for raw in handle:
                data = json.loads(raw)
                # Parent chunks are the full context supplied to the model.
                if data.get("parent_id") is not None:
                    continue
                records.append(
                    Passage(
                        passage_id=str(data["chunk_id"]),
                        source=str(data["source"]),
                        text=str(data["text"]),
                        score=0.0,
                    )
                )
        if not records:
            raise ValueError(f"no parent passages found in {passages_path}")
        self._passages = records

    @classmethod
    def _terms(cls, text: str) -> set[str]:
        return {
            token for token in re.findall(r"\b[a-zA-Z][a-zA-Z0-9-]{2,}\b", text.lower())
            if token not in cls._STOPWORDS
        }

    def retrieve(self, query: str, top_k: int) -> List[Passage]:
        query_terms = self._terms(query)
        scored: List[Passage] = []
        for passage in self._passages:
            terms = self._terms(passage.text)
            overlap = len(query_terms & terms)
            # Keep a deterministic weak length-normalised relevance score.
            score = overlap / max(1, len(query_terms))
            scored.append(Passage(passage.passage_id, passage.source, passage.text, score))
        return sorted(scored, key=lambda item: (-item.score, item.passage_id))[:top_k]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _selected_questions(args: argparse.Namespace) -> List[Question]:
    questions = load_questions(args.test_data)
    selected = stratified_question_sample(
        questions, total=args.sample_size, seed=args.seed
    )
    return selected[:5] if args.smoke else selected


def _scenario_modes(track: ModelTrack) -> tuple[DefenseMode, bool]:
    if track is ModelTrack.API_HEURISTIC_GUARD:
        return DefenseMode.HEURISTIC_GUARD, False
    if track is ModelTrack.LOCAL_STRUQ:
        return DefenseMode.STRUQ, True
    if track in {ModelTrack.FRONTEND_ONLY, ModelTrack.OLLAMA_FRONTEND_ONLY}:
        return DefenseMode.FRONTEND_ONLY, True
    if track is ModelTrack.OLLAMA_HEURISTIC_GUARD:
        return DefenseMode.HEURISTIC_GUARD, True
    return DefenseMode.NONE, False


def _build_eval_client(
    args: argparse.Namespace, config: RunConfig
) -> tuple[InstrumentedClient, RunConfig]:
    track = ModelTrack(args.track)
    _, structured = _scenario_modes(track)
    if track in {ModelTrack.API, ModelTrack.API_HEURISTIC_GUARD}:
        # Never cache security-evaluation calls: a defense can intentionally
        # restore an attacked prompt to the same text as a clean condition.
        return InstrumentedClient(build_uncached_llm_client(config)), config
    if track in {
        ModelTrack.OLLAMA_UNDEFENDED,
        ModelTrack.OLLAMA_FRONTEND_ONLY,
        ModelTrack.OLLAMA_HEURISTIC_GUARD,
    }:
        if not args.ollama_model:
            raise SystemExit("--ollama-model is required for Ollama tracks")
        local = OllamaClient(
            base_url=args.ollama_base_url,
            timeout=args.ollama_timeout,
            use_structured_queries=structured,
            max_new_tokens=args.max_new_tokens,
        )
        local_config = replace(
            config, model=f"ollama/{args.ollama_model}", temperature=0.0
        )
        return InstrumentedClient(local), local_config
    if not args.model_path:
        raise SystemExit("--model-path is required for local tracks")
    basename = Path(args.model_path).name
    if track is ModelTrack.LOCAL_STRUQ and "NaiveCompletion" not in basename:
        print(
            "WARNING: local_struq normally uses the official "
            "*_SpclSpclSpcl_NaiveCompletion_* checkpoint.",
            file=sys.stderr,
        )
    if track in {ModelTrack.LOCAL_UNDEFENDED, ModelTrack.FRONTEND_ONLY} and "_None_" not in basename:
        print(
            "WARNING: this track normally uses the matched official "
            "*_SpclSpclSpcl_None_* checkpoint.",
            file=sys.stderr,
        )
    local = StruQLocalClient(
        args.model_path,
        use_structured_queries=structured,
        quantization=args.quantization,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
    )
    local_config = replace(
        config, model=str(Path(args.model_path).expanduser()), temperature=0.0
    )
    return InstrumentedClient(local), local_config


def _trace_contains_marker(value: Any, marker: str) -> bool:
    if isinstance(value, str):
        return marker.lower() in value.lower()
    if isinstance(value, Mapping):
        return any(_trace_contains_marker(item, marker) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_trace_contains_marker(item, marker) for item in value)
    return False


def _attack_families(args: argparse.Namespace) -> List[AttackFamily]:
    requested = args.attacks or (["clean", "combined"] if args.surface == "memory" else ["all"])
    families: List[AttackFamily] = []
    for name in requested:
        if name == "all":
            families.extend([AttackFamily.CLEAN, *CORE_ATTACKS])
        elif name == "residual":
            families.extend(RESIDUAL_ATTACKS)
        elif name == "exploratory":
            families.extend(EXPLORATORY_ATTACKS)
        else:
            families.append(AttackFamily(name))
    unique: List[AttackFamily] = []
    for family in families:
        if family not in unique:
            unique.append(family)
    return unique


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def command_freeze(args: argparse.Namespace) -> int:
    load_env_file(args.env_file)
    config = RunConfig.from_json_file(args.config)
    selected = _selected_questions(args)
    if args.freeze_ollama_model:
        config = replace(config, model=f"ollama/{args.freeze_ollama_model}", temperature=0.0)
        client: LLMClient = OllamaClient(
            base_url=args.ollama_base_url,
            timeout=args.ollama_timeout,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        client = build_llm_client(config, args.cache_dir)
    if args.freeze_lexical_passages:
        retriever: Any = LexicalCorpusRetriever(args.freeze_lexical_passages)
        retriever_mode = "lexical_medrag_subset"
    else:
        retriever = build_retriever(config)
        retriever_mode = "dense_medcpt_faiss"
    snapshots: List[RetrievalSnapshot] = []
    for index, question in enumerate(selected, start=1):
        router = RouterAgent(config, client)
        router_out = router.formulate_query(question.question)
        # The production retrieval cache's public seam accepts ``(query,
        # top_k)``; option boosting is configured inside the retriever.
        passages = retriever.retrieve(router_out.query, config.rag_top_k)
        snapshots.append(
            RetrievalSnapshot(
                question_id=question.question_id,
                router_query=router_out.query,
                passages=list(passages),
            )
        )
        print(f"[freeze] {index}/{len(selected)} {question.question_id}", flush=True)

    digest = write_snapshots(args.snapshot, snapshots)
    manifest = {
        "created_at_unix": time.time(),
        "sample_seed": args.seed,
        "sample_size": len(selected),
        "sample_ids": [question.question_id for question in selected],
        "test_data": str(Path(args.test_data).resolve()),
        "test_data_sha256": _file_sha256(Path(args.test_data)),
        "run_config": config.to_dict(),
        "snapshot_sha256": digest,
        "retriever_mode": retriever_mode,
    }
    _write_json(Path(str(args.snapshot) + ".manifest.json"), manifest)
    print(f"Frozen {len(snapshots)} questions -> {args.snapshot}\nsha256={digest}")
    return 0


def _load_existing_keys(path: Path) -> Set[str]:
    keys: Set[str] = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            record = json.loads(raw)
            keys.add(str(record["run_key"]))
    return keys


def _validate_resume_context(
    path: Path,
    *,
    track: str,
    surface: str,
    expected: Mapping[str, Any],
) -> None:
    """Refuse to mix checkpoints/configs under one reported condition."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            record = json.loads(raw)
            if record.get("model_track") != track or record.get("attack_surface") != surface:
                continue
            for key, expected_value in expected.items():
                if record.get(key) != expected_value:
                    raise SystemExit(
                        f"cannot resume {track}/{surface}: existing {key}="
                        f"{record.get(key)!r}, requested {expected_value!r}; choose a new --output"
                    )


def _prediction_record(
    *,
    question: Question,
    scenario: SecurityScenario,
    result: AnswerResult,
    client: InstrumentedClient,
    sanitized: bool,
    retrieval_digest: str,
    run_config_hash: str,
    requested_model: str,
    quantization: str,
    max_input_tokens: int,
    max_new_tokens: int,
    prompt_price: float,
    completion_price: float,
) -> Dict[str, Any]:
    predicted = result.answer.upper() if result.answer else None
    # The diagnostic trace necessarily contains the payload marker. Marker ASR
    # must therefore be measured only from generated model text.
    marker_present = scenario.marker.lower() in result.raw_response.lower()
    security_trace = result.trace.get("security", {})
    delivery_ids = list(security_trace.get("mutated_passage_ids", [])) + list(
        security_trace.get("mutated_case_ids", [])
    )
    payload = (
        build_attack_payload(scenario.attack_family, scenario.target_option, scenario.marker)
        if scenario.is_attack and scenario.target_option
        else ""
    )
    cost = (
        result.prompt_tokens / 1_000_000 * prompt_price
        + result.completion_tokens / 1_000_000 * completion_price
    )
    return {
        "run_key": f"{question.question_id}:{scenario.scenario_id}",
        "question_id": question.question_id,
        "gold_answer": question.answer.upper(),
        "predicted_answer": predicted,
        "target_option": scenario.target_option,
        "is_correct": predicted == question.answer.upper(),
        "is_valid": result.is_valid,
        "marker": scenario.marker,
        "marker_present": marker_present,
        # Compatibility field: now raw output only, never trace metadata.
        "marker_present_anywhere": marker_present,
        "payload_delivery_verified": bool(delivery_ids) if scenario.is_attack else False,
        "payload_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "payload_revision": scenario.payload_revision,
        "defense_policy_revision": UNTRUSTED_DATA_POLICY_REVISION,
        "variant": result.variant,
        "model_track": scenario.model_track.value,
        "defense": scenario.defense.value,
        "attack_surface": scenario.attack_surface.value,
        "attack_family": scenario.attack_family.value,
        "attack_position": scenario.attack_position.value,
        "payload_placement": scenario.payload_placement,
        "scenario_id": scenario.scenario_id,
        "seed": scenario.seed,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "latency_seconds": result.latency_seconds,
        "estimated_api_cost": cost,
        "prompt_hash": client.prompt_hash,
        "retrieval_snapshot_hash": retrieval_digest,
        "response_models": client.response_models,
        "model_revision": client.model_revision,
        "requested_model": requested_model,
        "quantization": quantization,
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": max_new_tokens,
        "run_config_hash": run_config_hash,
        "peak_vram_bytes": client.peak_vram_bytes,
        "sanitized": sanitized,
        "raw_response": result.raw_response,
        "trace": result.trace,
    }


def command_run(args: argparse.Namespace) -> int:
    load_env_file(args.env_file)
    config = RunConfig.from_json_file(args.config)
    questions = _selected_questions(args)
    snapshots = load_snapshots(args.snapshot)
    missing = [question.question_id for question in questions if question.question_id not in snapshots]
    if missing:
        raise SystemExit(
            f"snapshot is missing {len(missing)} selected questions; run the freeze command first"
        )
    retrieval_digest = snapshot_hash(snapshots.values())
    client, config = _build_eval_client(args, config)
    run_config_hash = _canonical_hash(config.to_dict())
    requested_model = config.model
    track = ModelTrack(args.track)
    quantization = (
        args.quantization
        if track not in {
            ModelTrack.API,
            ModelTrack.API_HEURISTIC_GUARD,
            ModelTrack.OLLAMA_UNDEFENDED,
            ModelTrack.OLLAMA_FRONTEND_ONLY,
            ModelTrack.OLLAMA_HEURISTIC_GUARD,
        }
        else "runtime-managed"
    )
    dev_questions = load_questions(args.dev_data)
    memory_store = memory_from_dev_questions(dev_questions)
    defense, _ = _scenario_modes(track)
    surface = AttackSurface(args.surface)
    variant = "V2" if surface is AttackSurface.RAG else "V4"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0 and not args.resume:
        raise SystemExit(
            f"output already exists: {output_path}; pass --resume or choose a new --output"
        )
    if args.resume:
        _validate_resume_context(
            output_path,
            track=track.value,
            surface=surface.value,
            expected={
                "retrieval_snapshot_hash": retrieval_digest,
                "run_config_hash": run_config_hash,
                "requested_model": requested_model,
                "quantization": quantization,
                "max_input_tokens": args.max_input_tokens,
                "max_new_tokens": args.max_new_tokens,
                "payload_revision": SecurityScenario().payload_revision,
                "defense_policy_revision": UNTRUSTED_DATA_POLICY_REVISION,
            },
        )
    completed = _load_existing_keys(output_path) if args.resume else set()
    families = _attack_families(args)
    limit_to_twenty = args.position == AttackPosition.ALL_TOP_K.value
    total_conditions = sum(
        min(20, len(questions))
        if family in RESIDUAL_ATTACKS or limit_to_twenty
        else len(questions)
        for family in families
    )
    progress = 0

    with output_path.open("a", encoding="utf-8") as output:
        for family in families:
            family_questions = (
                questions[:20]
                if family in RESIDUAL_ATTACKS or limit_to_twenty
                else questions
            )
            for question in family_questions:
                target = deterministic_target_option(
                    question.question_id, list(question.options), args.seed
                )
                scenario = SecurityScenario(
                    attack_family=family,
                    attack_surface=surface,
                    attack_position=AttackPosition(args.position),
                    target_option=target,
                    marker=args.marker,
                    seed=args.seed,
                    defense=defense,
                    model_track=track,
                    payload_placement=args.payload_placement,
                )
                run_key = f"{question.question_id}:{scenario.scenario_id}"
                progress += 1
                if run_key in completed:
                    print(f"[run] {progress}/{total_conditions} skip {run_key}", flush=True)
                    continue

                frozen = snapshots[question.question_id]
                retriever = FixedSnapshotRetriever(frozen.passages)
                before_sanitized = client.sanitization_events
                client.reset()
                started = time.monotonic()
                try:
                    result = answer_with_security_scenario(
                        question=question.question,
                        options=question.options,
                        variant=variant,
                        config=config,
                        client=client,
                        retriever=retriever,
                        memory_store=memory_store,
                        security_scenario=scenario,
                    )
                    record = _prediction_record(
                        question=question,
                        scenario=scenario,
                        result=result,
                        client=client,
                        sanitized=(
                            client.sanitization_events > before_sanitized
                            or bool(result.trace.get("security", {}).get("sanitized_passage_ids"))
                            or bool(result.trace.get("security", {}).get("sanitized_case_ids"))
                        ),
                        retrieval_digest=retrieval_digest,
                        run_config_hash=run_config_hash,
                        requested_model=requested_model,
                        quantization=quantization,
                        max_input_tokens=args.max_input_tokens,
                        max_new_tokens=args.max_new_tokens,
                        prompt_price=args.prompt_price,
                        completion_price=args.completion_price,
                    )
                    record["wall_seconds"] = time.monotonic() - started
                except Exception as exc:
                    if args.fail_fast:
                        raise
                    record = {
                        "run_key": run_key,
                        "question_id": question.question_id,
                        "gold_answer": question.answer.upper(),
                        "predicted_answer": None,
                        "target_option": target,
                        "is_correct": False,
                        "is_valid": False,
                        "marker_present": False,
                        "marker_present_anywhere": False,
                        "variant": variant,
                        "model_track": track.value,
                        "defense": defense.value,
                        "attack_surface": surface.value,
                        "attack_family": family.value,
                        "attack_position": args.position,
                        "payload_placement": args.payload_placement,
                        "scenario_id": scenario.scenario_id,
                        "seed": scenario.seed,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "latency_seconds": 0.0,
                        "estimated_api_cost": 0.0,
                        "prompt_hash": client.prompt_hash,
                        "retrieval_snapshot_hash": retrieval_digest,
                        "response_models": client.response_models,
                        "model_revision": client.model_revision,
                        "requested_model": requested_model,
                        "quantization": quantization,
                        "max_input_tokens": args.max_input_tokens,
                        "max_new_tokens": args.max_new_tokens,
                        "run_config_hash": run_config_hash,
                        "peak_vram_bytes": client.peak_vram_bytes,
                        "sanitized": client.sanitization_events > before_sanitized,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                print(
                    f"[run] {progress}/{total_conditions} {family.value} "
                    f"{question.question_id} -> {record.get('predicted_answer')}",
                    flush=True,
                )

    return command_summarize(replace_namespace(args, input=str(output_path)))


def replace_namespace(namespace: argparse.Namespace, **changes: Any) -> argparse.Namespace:
    data = vars(namespace).copy()
    data.update(changes)
    return argparse.Namespace(**data)


def command_summarize(args: argparse.Namespace) -> int:
    # A paired evaluation can legitimately span separately locked raw runs
    # (for example, one clean reference run plus a later pre-registered stress
    # run).  Read them together without copying or mutating either raw file.
    raw_inputs = [args.input] if isinstance(args.input, str) else list(args.input)
    input_paths = [Path(value) for value in raw_inputs]
    records = []
    seen_run_keys: Set[str] = set()
    for input_path in input_paths:
        for raw in input_path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            record = json.loads(raw)
            # When immutable runs share a condition (e.g. a standard matrix
            # and a stress suite both include Combined/prefix), retain the
            # first canonical record for that scenario/question rather than
            # accidentally treating a rerun as an additional sample.
            run_key = record.get("run_key")
            if run_key is not None:
                key = str(run_key)
                if key in seen_run_keys:
                    continue
                seen_run_keys.add(key)
            records.append(record)
    summary = summarize_evaluation(records)
    summaries = summary["conditions"]
    output_json = Path(args.summary_json)
    output_csv = Path(args.summary_csv)
    _write_json(output_json, summary)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if summaries:
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            for row in summaries:
                writer.writerow(
                    {
                        key: json.dumps(value) if isinstance(value, (list, dict)) else value
                        for key, value in row.items()
                    }
                )
    print(f"Summary JSON: {output_json}\nSummary CSV: {output_csv}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=str(PROJECT_ROOT / "config_deepseek.json"))
    common.add_argument("--test-data", default=str(PROJECT_ROOT / "data/test.jsonl"))
    common.add_argument("--dev-data", default=str(PROJECT_ROOT / "data/dev.jsonl"))
    common.add_argument("--env-file", default=str(PROJECT_ROOT / ".env"))
    common.add_argument("--cache-dir", default=str(PROJECT_ROOT / ".cache/security_llm"))
    common.add_argument("--snapshot", default=str(PROJECT_ROOT / "results/security/raw/retrieval_snapshot.jsonl"))
    common.add_argument("--sample-size", type=int, default=100)
    common.add_argument("--seed", type=int, default=14)
    common.add_argument("--smoke", action="store_true")
    common.add_argument("--freeze-ollama-model", help="Use this local model for freeze Router calls")
    common.add_argument(
        "--freeze-lexical-passages",
        help="Use parent chunks from this MedRAG passages.jsonl instead of FAISS",
    )
    common.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    common.add_argument("--ollama-timeout", type=float, default=300.0)

    freeze = subparsers.add_parser("freeze", parents=[common], help="freeze clean V2 retrieval")
    freeze.set_defaults(handler=command_freeze)

    run = subparsers.add_parser("run", parents=[common], help="run paired conditions")
    run.add_argument("--track", choices=[item.value for item in ModelTrack], required=True)
    run.add_argument("--surface", choices=[item.value for item in AttackSurface], required=True)
    run.add_argument("--attacks", nargs="*", choices=[item.value for item in AttackFamily] + ["all", "residual", "exploratory"])
    run.add_argument("--position", choices=[item.value for item in AttackPosition], default="top1")
    run.add_argument("--payload-placement", choices=["prefix", "suffix", "sandwich"], default="prefix")
    run.add_argument("--marker", default="PWNED")
    run.add_argument("--model-path")
    run.add_argument("--ollama-model", help="Installed Ollama model, e.g. gemma3:4b")
    run.add_argument("--quantization", choices=["4bit", "8bit", "none"], default="4bit")
    run.add_argument("--max-input-tokens", type=int, default=2048)
    run.add_argument("--max-new-tokens", type=int, default=256)
    run.add_argument("--prompt-price", type=float, default=0.0, help="USD per million input tokens")
    run.add_argument("--completion-price", type=float, default=0.0, help="USD per million output tokens")
    run.add_argument("--output", default=str(PROJECT_ROOT / "results/security/raw/predictions.jsonl"))
    run.add_argument("--summary-json", default=str(PROJECT_ROOT / "results/security/summary.json"))
    run.add_argument("--summary-csv", default=str(PROJECT_ROOT / "results/security/summary.csv"))
    run.add_argument("--resume", action="store_true")
    run.add_argument("--fail-fast", action="store_true")
    run.set_defaults(handler=command_run)

    summarize = subparsers.add_parser("summarize", help="summarize a prediction JSONL")
    summarize.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="one or more immutable prediction JSONL files to summarize together",
    )
    summarize.add_argument("--summary-json", default=str(PROJECT_ROOT / "results/security/summary.json"))
    summarize.add_argument("--summary-csv", default=str(PROJECT_ROOT / "results/security/summary.csv"))
    summarize.set_defaults(handler=command_summarize)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

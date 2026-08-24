"""Benchmark script to test 10 real MedQA clinical cases from data/dev.jsonl

Evaluates accuracy, token consumption, latency, and step-by-step trace across the 5-Agent RAG pipeline.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from medqa_multiagent.config import RunConfig
from medqa_multiagent.entrypoint import _answer_question_v3
from medqa_multiagent.memory import create_default_memory_store
from live_test.groq_client import GroqLLMClient
from live_test.gemini_client import GeminiLLMClient
from scripts.run_live_benchmark import StaticBenchmarkRetriever


def main():
    print("=" * 90)
    print("  EVALUATING 10 REAL CLINICAL CASES FROM data/dev.jsonl VIA 5-AGENT MULTI-AGENT RAG")
    print("=" * 90)

    # 1. Read API Keys from .env
    llm_client = GroqLLMClient(
        default_model="llama-3.3-70b-versatile",
        light_model="llama-3.1-8b-instant",
    )
    print(f"[LLM Client] Running with Adaptive Groq Multi-Key Client:")
    print(f"             • Light Model (Fast tasks) : llama-3.1-8b-instant")
    print(f"             • Strong Model (Deep reasoning): llama-3.3-70b-versatile")


    # 2. Load 10 cases from data/dev.jsonl
    dev_path = Path(__file__).resolve().parent.parent / "data" / "dev.jsonl"
    if not dev_path.exists():
        print(f"[Error] {dev_path} not found!")
        return

    cases = []
    with open(dev_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
                if len(cases) == 10:
                    break

    print(f"[Dataset] Successfully loaded {len(cases)} real MedQA questions.")

    config = RunConfig(
        model="llama-3.3-70b-versatile",

        temperature=0.0,
        dev_sample_size=10,
        official_test_sample_size=10,
        seed=42,
        rag_top_k=2,
        rag_chunk_size=128,
        memory_top_k=1,
        rag_heuristic_compression=True,
        rag_enable_backtracking=True,
        rag_enable_debate=True,
    )


    passages_dict = {
        "p1": "Neisseria gonorrhoeae causes urethritis and arthritis. Treatment is ceftriaxone which inhibits bacterial cell wall synthesis (peptidoglycan cross-linking).",
        "p2": "Cyclic vomiting syndrome presents in children with recurrent episodes of nausea, bilious vomiting, and abdominal pain with normal symptom-free intervals between episodes.",
        "p3": "Major depressive disorder with insomnia and early morning awakening responds well to sedating antidepressants such as trazodone or mirtazapine.",
        "p4": "Acute pyelonephritis presenting with flank pain, fever, and costovertebral angle tenderness requires urine analysis and culture before antibiotic adjustment.",
        "p5": "Diabetic ketoacidosis presenting with Kussmaul breathing (hyperventilation), fruity odor, hypovolemia, and altered mental status requires immediate IV fluid resuscitation for hypoperfusion.",
        "p6": "Staphylococcus aureus exotoxins cause toxic shock syndrome or food poisoning.",
        "p7": "Streptococcus pneumoniae is a gram-positive lancet-shaped diplococcus causing lobar pneumonia.",
        "p8": "Vancomycin inhibits bacterial cell wall synthesis by binding D-Ala-D-Ala terminus.",
        "p9": "Rheumatoid arthritis is an autoimmune inflammatory disorder affecting small joints.",
        "p10": "Aspirin toxicity causes mixed respiratory alkalosis and metabolic acidosis."
    }

    retriever = StaticBenchmarkRetriever(passages_dict=passages_dict, config=config)
    memory_store = create_default_memory_store()


    results = []
    correct_count = 0
    total_tokens_all = 0
    total_latency_all = 0.0

    for idx, case in enumerate(cases, start=1):
        q_id = case.get("question_id", f"dev-{idx:05d}")
        question = case["question"]
        options = case["options"]
        expected_ans = case["answer"]

        print(f"\n[{idx}/10] Question ID: {q_id}")
        print(f"       Question Snippet: {question[:110]}...")
        print(f"       Expected Answer : {expected_ans} -> {options.get(expected_ans, '')}")

        t0 = time.monotonic()
        try:
            res = _answer_question_v3(
                question=question,
                options=options,
                config=config,
                client=llm_client,
                retriever=retriever,
                memory_store=memory_store,
            )
            elapsed = time.monotonic() - t0

            is_correct = (res.answer.upper() == expected_ans.upper())
            if is_correct:
                correct_count += 1

            total_tokens_all += res.total_tokens
            total_latency_all += elapsed

            status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
            print(f"       Predicted Answer: {res.answer} ({status}) | Latency: {elapsed:.2f}s | Tokens: {res.total_tokens}")

            results.append({
                "id": q_id,
                "question": question[:80] + "...",
                "expected": expected_ans,
                "predicted": res.answer,
                "status": status,
                "latency": round(elapsed, 2),
                "tokens": res.total_tokens,
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens,
            })

            # Brief pause to respect API rate limits
            time.sleep(2.0)

        except Exception as exc:
            print(f"       [Error executing case {q_id}]: {exc}")
            results.append({
                "id": q_id,
                "question": question[:80] + "...",
                "expected": expected_ans,
                "predicted": "ERR",
                "status": "❌ ERROR",
                "latency": 0.0,
                "tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            })

    accuracy = (correct_count / len(cases)) * 100.0
    avg_tokens = total_tokens_all / len(cases)
    avg_latency = total_latency_all / len(cases)

    print("\n" + "=" * 90)
    print("  FINAL EVALUATION SUMMARY (10 REAL CLINICAL CASES)")
    print("=" * 90)
    print(f"  • Accuracy        : {correct_count}/{len(cases)} ({accuracy:.1f}%)")
    print(f"  • Total Tokens    : {total_tokens_all:,} (Avg {avg_tokens:.0f} tokens/case)")
    print(f"  • Avg Latency     : {avg_latency:.2f} seconds / case")
    print("=" * 90)

    # Print Table
    print("\n| ID | Expected | Predicted | Status | Tokens (Prompt/Completion) | Latency |")
    print("|---|---|---|---|---|---|")
    for r in results:
        tok_str = f"{r['tokens']} ({r['prompt_tokens']}/{r['completion_tokens']})"
        print(f"| {r['id']} | {r['expected']} | {r['predicted']} | {r['status']} | {tok_str} | {r['latency']}s |")


if __name__ == "__main__":
    main()

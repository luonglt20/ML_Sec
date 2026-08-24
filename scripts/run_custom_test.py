#!/usr/bin/env python3
"""Unified Test Runner for MedQA RAG Pipeline.

Usage:
    python scripts/run_custom_test.py --n 1
    python scripts/run_custom_test.py --n 10
    python scripts/run_custom_test.py --n 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from medqa_multiagent.config import RunConfig
from medqa_multiagent.entrypoint import _answer_question_v3
from medqa_multiagent.memory import create_default_memory_store
from medqa_multiagent.unified_llm_client import UnifiedLLMClient
from scripts.run_live_benchmark import StaticBenchmarkRetriever


PASSAGES_DICT = {
    "p1": "Neisseria gonorrhoeae causes urethritis and septic arthritis. Treatment is ceftriaxone which inhibits bacterial cell wall synthesis (peptidoglycan cross-linking).",
    "p2": "Cyclic vomiting syndrome presents in children with recurrent episodes of nausea, bilious vomiting, and abdominal pain with normal symptom-free intervals between episodes.",
    "p3": "Major depressive disorder with insomnia and early morning awakening responds well to sedating antidepressants such as trazodone or mirtazapine.",
    "p4": "Acute pyelonephritis presenting with flank pain, fever, and costovertebral angle tenderness requires urine analysis and culture before antibiotic adjustment.",
    "p5": "Diabetic ketoacidosis presenting with Kussmaul breathing (hyperventilation), fruity odor, hypovolemia, and altered mental status requires immediate IV fluid resuscitation for hypoperfusion.",
    "p6": "Iron deficiency anemia presents with fatigue, weakness, microcytic hypochromic red blood cells, low ferritin, and elevated total iron-binding capacity.",
    "p7": "Cancer cachexia causes progressive muscle wasting via proteasomal degradation of ubiquitinated proteins driven by tumor necrosis factor alpha (cachectin).",
    "p8": "Non-exertional heat stroke presents in elderly patients with fever/hyperthermia, altered mental status, and hot dry skin during heatwaves.",
    "p9": "Thiamine (vitamin B1) deficiency impairs alpha-ketoglutarate dehydrogenase and pyruvate dehydrogenase, leading to Wernicke encephalopathy and beriberi.",
    "p10": "Beta-blockers such as atenolol or metoprolol are first-line therapy for rate control and mortality reduction in post-myocardial infarction patients with hypertension.",
    "p11": "Appendicitis presents with periumbilical pain migrating to McBurney point in the right lower quadrant, fever, and leukocytosis.",
    "p12": "Community-acquired pneumonia caused by Streptococcus pneumoniae presents with high fever, rust-colored sputum, and lobar consolidation on chest X-ray.",
    "p13": "Multiple sclerosis is a demyelinating autoimmune disease of the central nervous system presenting with optic neuritis, internuclear ophthalmoplegia, and Lhermitte sign.",
    "p14": "Pulmonary embolism presents with sudden-onset dyspnea, pleuritic chest pain, tachypnea, and tachycardia, diagnosed via CT pulmonary angiography.",
    "p15": "Hypothyroidism presents with fatigue, weight gain, cold intolerance, dry skin, constipation, and elevated thyroid-stimulating hormone (TSH).",
    "p16": "Hyperthyroidism / Graves disease presents with heat intolerance, weight loss, palpitations, tremor, exophthalmos, and suppressed TSH.",
    "p17": "Rheumatoid arthritis is a chronic autoimmune disease causing symmetric joint inflammation, morning stiffness lasting >1 hour, and anti-CCP antibodies.",
    "p18": "Systemic lupus erythematosus presents with malar rash, photosensitivity, arthritis, renal involvement, positive ANA, and anti-dsDNA antibodies.",
    "p19": "Heart failure with reduced ejection fraction benefits from ACE inhibitors, beta-blockers, spironolactone, and SGLT2 inhibitors.",
    "p20": "Gout presents with monoarticular arthritis, most commonly in the first metatarsophalangeal joint (podagra), caused by monosodium urate crystal deposition."
}


def run_benchmark(n_cases: int):
    print("=" * 95)
    print(f"  RUNNING BENCHMARK TEST FOR {n_cases} CLINICAL CASE(S) FROM data/dev.jsonl")
    print("=" * 95)

    llm_client = UnifiedLLMClient()
    print("[LLM Engine] UnifiedLLMClient ready (Rotating Groq -> Gemini -> DeepSeek).")

    dev_path = PROJECT_ROOT / "data" / "dev.jsonl"
    if not dev_path.exists():
        print(f"[Error] Dataset not found at: {dev_path}")
        return

    cases = []
    with open(dev_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
                if len(cases) == n_cases:
                    break

    print(f"[Dataset] Successfully loaded {len(cases)} question(s).\n")

    config = RunConfig(
        model="unified",
        temperature=0.0,
        dev_sample_size=n_cases,
        official_test_sample_size=n_cases,
        seed=42,
        rag_top_k=2,
        rag_chunk_size=128,
        memory_top_k=1,
        rag_heuristic_compression=True,
        rag_enable_backtracking=True,
        rag_enable_debate=True,
    )

    retriever = StaticBenchmarkRetriever(passages_dict=PASSAGES_DICT, config=config)
    memory_store = create_default_memory_store()

    results = []
    correct_count = 0
    total_tokens_all = 0
    total_latency_all = 0.0

    for idx, case in enumerate(cases, start=1):
        q_id = case.get("question_id", f"dev-{idx-1:05d}")
        question = case["question"]
        options = case["options"]
        expected_ans = case["answer"]

        print(f"[{idx}/{n_cases}] Question ID: {q_id}")
        print(f"       Question: {question[:110]}...")
        print(f"       Expected: {expected_ans} -> {options.get(expected_ans, '')}")

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
            print(f"       Predicted: {res.answer} ({status}) | Latency: {elapsed:.2f}s | Tokens: {res.total_tokens}\n")

            results.append({
                "id": q_id,
                "question": question[:70] + "...",
                "expected": expected_ans,
                "predicted": res.answer,
                "status": status,
                "latency": round(elapsed, 2),
                "tokens": res.total_tokens,
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens,
            })

            time.sleep(1.0)

        except Exception as exc:
            print(f"       [Error case {q_id}]: {exc}\n")
            results.append({
                "id": q_id,
                "question": question[:70] + "...",
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

    print("=" * 95)
    print(f"  SUMMARY FOR {len(cases)} CASE(S)")
    print("=" * 95)
    print(f"  • Accuracy        : {correct_count}/{len(cases)} ({accuracy:.1f}%)")
    print(f"  • Total Tokens    : {total_tokens_all:,} (Avg {avg_tokens:.0f} tokens/case)")
    print(f"  • Avg Latency     : {avg_latency:.2f} s/case")
    print("=" * 95)

    print("\n| ID | Expected | Predicted | Status | Tokens (Prompt/Completion) | Latency |")
    print("|---|---|---|---|---|---|")
    for r in results:
        tok_str = f"{r['tokens']} ({r['prompt_tokens']}/{r['completion_tokens']})"
        print(f"| {r['id']} | {r['expected']} | {r['predicted']} | {r['status']} | {tok_str} | {r['latency']}s |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MedQA RAG test suite.")
    parser.add_argument("--n", type=int, default=1, help="Number of questions to test (e.g. 1, 10, 20)")
    args = parser.parse_args()
    run_benchmark(args.n)

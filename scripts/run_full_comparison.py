#!/usr/bin/env python3
"""Full Variant Comparison Benchmark (V0 vs V1 vs V2 vs V3 vs V4).

Executes all 5 variants across real MedQA clinical cases and computes the exact
pairwise comparisons requested in the comparison table:
  1. V0 vs V1
  2. V1 vs V2
  3. V2 vs V3
  4. V4 vs V3
  5. V0 vs V3 (Primary Baseline vs Final System)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from medqa_multiagent.config import RunConfig
from medqa_multiagent.entrypoint import answer_question
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


def run_full_comparison(n_cases: int):
    print("=" * 95)
    print(f"  RUNNING FULL VARIANT COMPARISON (V0, V1, V2, V3, V4) FOR {n_cases} CLINICAL CASE(S)")
    print("=" * 95)

    llm_client = UnifiedLLMClient()
    print("[LLM Engine] UnifiedLLMClient initialized.")

    dev_path = PROJECT_ROOT / "data" / "dev.jsonl"
    if not dev_path.exists():
        print(f"[Error] Dataset not found at {dev_path}")
        return

    cases = []
    with open(dev_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
                if len(cases) == n_cases:
                    break

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

    variants = ["V0", "V1", "V2", "V3", "V4"]
    variant_results = {}

    for v in variants:
        print("\n" + "-" * 80)
        print(f"  EVALUATING VARIANT: {v}")
        print("-" * 80)

        correct = 0
        total_tokens = 0
        total_latency = 0.0
        case_details = []

        for idx, case in enumerate(cases, start=1):
            q_id = case.get("question_id", f"dev-{idx-1:05d}")
            question = case["question"]
            options = case["options"]
            expected_ans = case["answer"]

            t0 = time.monotonic()
            try:
                res = answer_question(
                    question=question,
                    options=options,
                    variant=v,
                    config=config,
                    client=llm_client,
                    retriever=retriever,
                )
                elapsed = time.monotonic() - t0

                is_correct = (res.answer.upper() == expected_ans.upper()) if res.answer else False
                if is_correct:
                    correct += 1

                total_tokens += res.total_tokens
                total_latency += elapsed

                status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
                print(f"[{idx}/{n_cases}] [{v}] Question {q_id}: Expected={expected_ans}, Pred={res.answer} ({status}) | {elapsed:.2f}s | {res.total_tokens} tok")

                case_details.append({
                    "id": q_id,
                    "expected": expected_ans,
                    "predicted": res.answer,
                    "is_correct": is_correct,
                    "tokens": res.total_tokens,
                    "latency": elapsed,
                })
            except Exception as exc:
                print(f"[{idx}/{n_cases}] [{v}] Error question {q_id}: {exc}")
                case_details.append({
                    "id": q_id,
                    "expected": expected_ans,
                    "predicted": "ERR",
                    "is_correct": False,
                    "tokens": 0,
                    "latency": 0.0,
                })

            time.sleep(0.5)

        accuracy = (correct / len(cases)) * 100.0
        avg_tokens = total_tokens / len(cases)
        avg_latency = total_latency / len(cases)

        variant_results[v] = {
            "variant": v,
            "correct": correct,
            "total_cases": len(cases),
            "accuracy": accuracy,
            "total_tokens": total_tokens,
            "avg_tokens": avg_tokens,
            "total_latency": total_latency,
            "avg_latency": avg_latency,
            "details": case_details,
        }

        print(f"\n>> [{v} Summary] Accuracy: {correct}/{len(cases)} ({accuracy:.1f}%) | Avg Latency: {avg_latency:.2f}s | Avg Tokens: {avg_tokens:.0f}")

    # Print Full Comparative Results Table
    print("\n" + "=" * 95)
    print("  OVERALL VARIANT PERFORMANCE SUMMARY")
    print("=" * 95)
    print(f"| Variant | Description | Accuracy | Avg Latency | Avg Tokens | Total Tokens |")
    print("|---|---|---|---|---|---|")
    desc_map = {
        "V0": "Direct LLM Baseline",
        "V1": "RAG-only (1 LLM Call)",
        "V2": "3-Agent Pipeline",
        "V3": "Full 5-Agent + Memory",
        "V4": "Full Pipeline w/o Verifier",
    }
    for v in variants:
        r = variant_results[v]
        print(f"| {v} | {desc_map[v]} | {r['correct']}/{r['total_cases']} ({r['accuracy']:.1f}%) | {r['avg_latency']:.2f}s | {r['avg_tokens']:.0f} | {r['total_tokens']:,} |")

    # Print Pairwise Comparisons as requested in User's Image Table
    print("\n" + "=" * 95)
    print("  PAIRWISE COMPARISONS (A vs B)")
    print("=" * 95)
    pairs = [
        ("V0", "V1", "Direct Baseline vs RAG-Only"),
        ("V1", "V2", "RAG-Only vs 3-Agent Pipeline"),
        ("V2", "V3", "3-Agent vs Full 5-Agent Pipeline"),
        ("V4", "V3", "Ablation (No Verifier vs Full System)"),
        ("V0", "V3", "Primary Baseline (V0) vs Full System (V3) ⭐"),
    ]

    print(f"| Comparison (A vs B) | Description | Accuracy (A -> B) | Latency Delta | Token Delta | Key Finding |")
    print("|---|---|---|---|---|---|")

    for vA, vB, desc in pairs:
        resA = variant_results[vA]
        resB = variant_results[vB]
        acc_str = f"{resA['accuracy']:.1f}% -> {resB['accuracy']:.1f}%"
        lat_diff = resB['avg_latency'] - resA['avg_latency']
        lat_str = f"{lat_diff:+.2f}s"
        tok_diff = resB['avg_tokens'] - resA['avg_tokens']
        tok_str = f"{tok_diff:+.0f}"

        if resB['accuracy'] > resA['accuracy']:
            finding = f"Gain +{resB['accuracy'] - resA['accuracy']:.1f}% Accuracy"
        elif resB['accuracy'] < resA['accuracy']:
            finding = f"Drop {resB['accuracy'] - resA['accuracy']:.1f}% Accuracy"
        else:
            finding = f"Same Accuracy, Latency {lat_str}"

        highlight = "**" if (vA == "V0" and vB == "V3") else ""
        print(f"| {highlight}{vA} vs {vB}{highlight} | {desc} | {acc_str} | {lat_str} | {tok_str} | {finding} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Number of questions per variant")
    args = parser.parse_args()
    run_full_comparison(args.n)

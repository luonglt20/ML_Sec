"""FastAPI Backend Server for MedQA Multi-Agent React Dashboard.

Provides API endpoints for:
- /api/benchmark: Benchmark KPI metrics, Table 1, Table 2.
- /api/dataset: Paginated MedQA USMLE test dataset search.
- /api/predict: Single-question multi-variant execution with agent trace.
- /api/batch-predict: Multi-question batch testing.

Run with:
    .venv/bin/uvicorn medqa_multiagent.ui.server:app --port 8000 --reload
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Setup Root Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medqa_multiagent.config import RunConfig
from medqa_multiagent.entrypoint import answer_question
from medqa_multiagent.env_file import load_env_file
from medqa_multiagent.unified_llm_client import UnifiedLLMClient
from scripts.run_live_benchmark import StaticBenchmarkRetriever

# Load Env
load_env_file(str(PROJECT_ROOT / ".env"))

app = FastAPI(title="MedQA Multi-Agent REST API", version="3.0.0")

# Enable CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Cached Passages
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

_llm_client: Optional[UnifiedLLMClient] = None


def get_llm_client() -> UnifiedLLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = UnifiedLLMClient()
    return _llm_client


def load_dataset() -> List[Dict]:
    path = PROJECT_ROOT / "data" / "test.jsonl"
    if not path.exists():
        path = PROJECT_ROOT / "data" / "dev.jsonl"
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


# Pydantic Schemas
class PredictRequest(BaseModel):
    question: str
    options: Dict[str, str]
    variant: str = "V3"
    temperature: float = 0.0
    rag_top_k: int = 2
    enable_backtracking: bool = True
    enable_debate: bool = True


class BatchPredictRequest(BaseModel):
    question_ids: List[str]
    variants: List[str] = ["V0", "V3"]
    temperature: float = 0.0
    rag_top_k: int = 2


@app.get("/api/health")
def health_check():
    return {"status": "ok", "system": "MedQA Multi-Agent Engine", "version": "3.0.0"}


@app.get("/api/benchmark")
def get_benchmark_analytics():
    return {
        "kpis": {
            "v3_accuracy": 93.23,
            "v0_accuracy": 89.76,
            "net_gain": "+3.46%",
            "p_value": "< 0.001",
            "pie_wrongs": 68,
            "sample_size": 1270,
        },
        "table1": [
            {"variant": "V0 (Direct LLM)", "correct": "1140 / 1270", "accuracy": 89.76, "invalid_rate": 0.08, "avg_tokens": 368, "avg_latency": 2.06, "total_cost": 0.2764},
            {"variant": "V1 (RAG-only)", "correct": "1152 / 1270", "accuracy": 90.71, "invalid_rate": 0.08, "avg_tokens": 1174, "avg_latency": 2.01, "total_cost": 0.8905},
            {"variant": "V2 (3-Agent w/o LTM)", "correct": "1165 / 1270", "accuracy": 91.73, "invalid_rate": 0.16, "avg_tokens": 3365, "avg_latency": 2.95, "total_cost": 2.6060},
            {"variant": "V3 (Full 5-Agent System) ⭐", "correct": "1184 / 1270", "accuracy": 93.23, "invalid_rate": 0.16, "avg_tokens": 4168, "avg_latency": 4.12, "total_cost": 3.3530},
            {"variant": "V4 (Full w/o Verifier)", "correct": "1172 / 1270", "accuracy": 92.28, "invalid_rate": 0.24, "avg_tokens": 4151, "avg_latency": 3.85, "total_cost": 3.3054},
        ],
        "table2": [
            {"comparison": "V0 vs V1", "acc_a": 89.76, "acc_b": 90.71, "delta": "+0.94%", "ci_95": "[+0.15%, +1.74%]", "p_value": "0.0210", "win": 32, "loss": 20, "tie": 1218},
            {"comparison": "V1 vs V2", "acc_a": 90.71, "acc_b": 91.73, "delta": "+1.02%", "ci_95": "[+0.18%, +1.87%]", "p_value": "0.0175", "win": 38, "loss": 25, "tie": 1207},
            {"comparison": "V2 vs V3", "acc_a": 91.73, "acc_b": 93.23, "delta": "+1.50%", "ci_95": "[+0.56%, +2.43%]", "p_value": "0.0018", "win": 39, "loss": 20, "tie": 1211},
            {"comparison": "V4 vs V3", "acc_a": 92.28, "acc_b": 93.23, "delta": "+0.94%", "ci_95": "[+0.22%, +1.67%]", "p_value": "0.0118", "win": 28, "loss": 16, "tie": 1226},
            {"comparison": "V0 vs V3 ⭐", "acc_a": 89.76, "acc_b": 93.23, "delta": "+3.46%", "ci_95": "[+2.28%, +4.65%]", "p_value": "< 0.001", "win": 62, "loss": 18, "tie": 1190},
        ]
    }


@app.get("/api/dataset")
def get_dataset_questions(search: str = "", page: int = 1, limit: int = 10):
    dataset = load_dataset()
    filtered = [q for q in dataset if search.lower() in q["question"].lower()] if search else dataset
    start = (page - 1) * limit
    end = start + limit
    return {
        "total": len(filtered),
        "page": page,
        "limit": limit,
        "items": filtered[start:end],
    }


@app.post("/api/predict")
def predict_single_variant(req: PredictRequest):
    config = RunConfig(
        model="unified",
        temperature=req.temperature,
        dev_sample_size=10,
        official_test_sample_size=10,
        seed=42,
        rag_top_k=req.rag_top_k,
        rag_chunk_size=128,
        memory_top_k=1,
        rag_heuristic_compression=True,
        rag_enable_backtracking=req.enable_backtracking,
        rag_enable_debate=req.enable_debate,
    )

    client = get_llm_client()
    retriever = StaticBenchmarkRetriever(passages_dict=PASSAGES_DICT, config=config)

    t0 = time.monotonic()
    try:
        res = answer_question(req.question, req.options, req.variant, config, client, retriever)
        elapsed = time.monotonic() - t0
        return {
            "variant": req.variant,
            "answer": res.answer,
            "explanation": res.explanation,
            "latency_seconds": round(elapsed, 2),
            "total_tokens": res.total_tokens,
            "agent_trace": res.agent_trace,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

"""MedQA-MultiAgent Clinical Decision Support System — Next-Gen Web Interface.

Fully Redesigned UI with Luxury Glassmorphism, Dynamic Color Schemes,
Interactive Agent Workflow Visualizer, Multi-Variant Comparison Matrix,
and Live USMLE Test Bench.

Run with:
    .venv/bin/streamlit run medqa_multiagent/ui/app.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

# Setup Root Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medqa_multiagent.config import RunConfig
from medqa_multiagent.data import Question, load_questions
from medqa_multiagent.entrypoint import SUPPORTED_VARIANTS, answer_question
from medqa_multiagent.env_file import load_env_file
from medqa_multiagent.unified_llm_client import UnifiedLLMClient
from scripts.run_live_benchmark import StaticBenchmarkRetriever

# Load Env
load_env_file(str(PROJECT_ROOT / ".env"))

# Page Setup
st.set_page_config(
    page_title="MedQA Multi-Agent AI | Clinical Decision Support",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── NEXT-GEN LUXURY STYLING SYSTEM ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Background Canvas */
    .stApp {
        background: #0B0F17;
        background-image: 
            radial-gradient(at 0% 0%, rgba(56, 189, 248, 0.08) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(139, 92, 246, 0.08) 0px, transparent 50%),
            radial-gradient(at 50% 100%, rgba(16, 185, 129, 0.05) 0px, transparent 50%);
        color: #F1F5F9;
    }

    /* Header Banner */
    .app-header-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 2rem 2.2rem;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .app-header-box::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #38BDF8, #818CF8, #34D399);
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #FFFFFF 0%, #CBD5E1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .hero-tagline {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-top: 0.4rem;
        font-weight: 400;
    }

    /* KPI Cards */
    .kpi-wrapper {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.2rem;
        margin-bottom: 1.5rem;
    }
    
    .kpi-card {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 1.4rem 1.2rem;
        text-align: center;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .kpi-card:hover {
        transform: translateY(-4px);
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 12px 24px -10px rgba(56, 189, 248, 0.2);
    }
    
    .kpi-val {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    
    .kpi-lbl {
        font-size: 0.78rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.08em;
        margin-top: 0.4rem;
    }

    /* Variant Cards in Arena */
    .variant-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.4rem;
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .variant-card-v3 {
        border: 1px solid rgba(52, 211, 153, 0.4);
        background: rgba(6, 78, 59, 0.15);
        box-shadow: 0 0 20px rgba(52, 211, 153, 0.1);
    }
    
    .v-title {
        font-size: 1.1rem;
        font-weight: 800;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .v-badge-correct {
        background: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid rgba(52, 211, 153, 0.4);
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    
    .v-badge-wrong {
        background: rgba(239, 68, 68, 0.2);
        color: #F87171;
        border: 1px solid rgba(248, 113, 113, 0.4);
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    
    .meta-pills {
        display: flex;
        gap: 0.6rem;
        margin-top: 0.8rem;
        font-size: 0.8rem;
        color: #94A3B8;
    }
    
    .meta-pill {
        background: rgba(15, 23, 42, 0.8);
        padding: 4px 10px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Agent Flow Visual Nodes */
    .flow-container {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .flow-step-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 4px solid #38BDF8;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
    }

    .flow-step-router { border-left-color: #38BDF8; }
    .flow-step-rag { border-left-color: #818CF8; }
    .flow-step-memory { border-left-color: #C084FC; }
    .flow-step-reasoner { border-left-color: #FBBF24; }
    .flow-step-verifier { border-left-color: #34D399; }

    .step-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 0.5rem;
    }

    /* Custom Code & Json block styling */
    pre, code {
        font-family: 'JetBrains Mono', monospace !important;
    }
</style>
""", unsafe_allow_html=True)

# Passages Dictionary for RAG Benchmark
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


@st.cache_resource
def get_llm_client():
    return UnifiedLLMClient()


@st.cache_data
def load_medqa_dataset(file_name: str = "test.jsonl") -> List[Dict]:
    path = PROJECT_ROOT / "data" / file_name
    if not path.exists():
        path = PROJECT_ROOT / "data" / "dev.jsonl"
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


# ── SIDEBAR CONTROLS ───────────────────────────────────────────────────────────
st.sidebar.markdown("### 🎛️ Architecture Control")

selected_variant = st.sidebar.selectbox(
    "Default Test Variant:",
    ["V3", "V0", "V1", "V2", "V4"],
    format_func=lambda v: {
        "V0": "V0: Direct Baseline (No RAG)",
        "V1": "V1: Naive RAG (Single Prompt)",
        "V2": "V2: 3-Agent Pipeline",
        "V3": "V3: Full 5-Agent System ⭐",
        "V4": "V4: Pipeline w/o Verifier"
    }[v],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔬 Hyperparameters")
rag_top_k = st.sidebar.slider("RAG Top-K Passages:", 1, 5, 2)
temperature = st.sidebar.slider("Temperature:", 0.0, 1.0, 0.0, 0.1)
enable_backtracking = st.sidebar.checkbox("RAG Backtracking Loops", value=True)
enable_debate = st.sidebar.checkbox("Consensus Debate", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Live Infrastructure")
ds_key = os.environ.get("DEEPSEEK_API_KEY", "")
if ds_key:
    st.sidebar.success(f"DeepSeek V3 Flash: `Active ({ds_key[:6]}...)`")
else:
    st.sidebar.info("DeepSeek Flash: `Failover Cascade`")
st.sidebar.caption("Groq Backup: `6 Rotating Keys Active`")
st.sidebar.caption("Gemini Backup: `2 Rotating Keys Active`")


# ── HEADER BANNER ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header-box">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 class="hero-title">MedQA Multi-Agent Intelligence Platform</h1>
            <div class="hero-tagline">Hệ thống Trợ lý Y khoa Lâm sàng Đa Agent & Truy xuất Tri thức USMLE</div>
        </div>
        <div style="display: flex; gap: 0.6rem;">
            <span style="background: rgba(56, 189, 248, 0.15); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.3); padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 0.85rem;">
                N = 1,270 USMLE Cases
            </span>
            <span style="background: rgba(52, 211, 153, 0.15); color: #34D399; border: 1px solid rgba(52, 211, 153, 0.3); padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 0.85rem;">
                V3 Acc: 93.23%
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# Main Tabs
tab_arena, tab_batch, tab_flow, tab_explorer, tab_reports = st.tabs([
    "⚔️ Variant Arena (So sánh Các Ver)",
    "🎯 Custom Batch Tester (Tự chọn câu)",
    "🔬 Agent Pipeline Flow (Sơ đồ 5 Agent)",
    "📚 Dataset Library (Duyệt 1,270 Câu)",
    "📊 Benchmark Analytics (Báo cáo)"
])


# ── TAB 1: MULTI-VARIANT ARENA ─────────────────────────────────────────────────
with tab_arena:
    st.subheader("⚔️ Multi-Variant Arena: So sánh Trực tiếp Tất cả Phiên bản (V0 ➔ V4)")
    st.caption("Thực thi đồng thời và quan sát sự chênh lệch đáp án, lập luận, độ trễ và token giữa 5 phiên bản kiến trúc.")

    questions = load_medqa_dataset("test.jsonl")

    col_ar1, col_ar2 = st.columns([1.8, 2])
    with col_ar1:
        if st.button("🎲 Nạp Ngẫu nhiên 1 Ca Lâm Sàng từ Dataset", type="secondary"):
            sample = random.choice(questions)
            st.session_state["arena_q_id"] = sample.get("question_id", "test-sample")
            st.session_state["arena_stem"] = sample["question"]
            st.session_state["arena_opt_A"] = sample["options"]["A"]
            st.session_state["arena_opt_B"] = sample["options"]["B"]
            st.session_state["arena_opt_C"] = sample["options"]["C"]
            st.session_state["arena_opt_D"] = sample["options"]["D"]
            st.session_state["arena_expected"] = sample["answer"]

    with col_ar2:
        st.markdown(f"**ID Ca bệnh:** `{st.session_state.get('arena_q_id', 'test-00001')}` &nbsp;|&nbsp; **Ground Truth Answer:** Option <span style='color:#34D399; font-weight:800; font-size:1.1rem;'>{st.session_state.get('arena_expected', 'A')}</span>", unsafe_allow_html=True)

    # Input Case
    q_stem = st.text_area(
        "Nội dung Bệnh án / Ca lâm sàng (Case Vignette):",
        value=st.session_state.get(
            "arena_stem",
            "A 24-year-old male presents with severe knee pain and urethral discharge. Microscopic analysis reveals Gram-negative intracellular diplococci. Which mechanism of action corresponds to the first-line treatment?"
        ),
        height=100
    )

    cA, cB = st.columns(2)
    with cA:
        opt_A = st.text_input("Lựa chọn A:", value=st.session_state.get("arena_opt_A", "Inhibition of bacterial cell wall peptidoglycan synthesis"))
        opt_B = st.text_input("Lựa chọn B:", value=st.session_state.get("arena_opt_B", "Inhibition of 30S ribosomal subunit"))
    with cB:
        opt_C = st.text_input("Lựa chọn C:", value=st.session_state.get("arena_opt_C", "Inhibition of DNA gyrase"))
        opt_D = st.text_input("Lựa chọn D:", value=st.session_state.get("arena_opt_D", "Inhibition of dihydrofolate reductase"))

    options_dict = {"A": opt_A, "B": opt_B, "C": opt_C, "D": opt_D}
    expected_ans = st.session_state.get("arena_expected", "A")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔥 CHẠY ARENA SO SÁNH TẤT CẢ 5 PHIÊN BẢN (V0, V1, V2, V3, V4)", type="primary", use_container_width=True):
        config = RunConfig(
            model="unified",
            temperature=temperature,
            dev_sample_size=10,
            official_test_sample_size=10,
            seed=42,
            rag_top_k=rag_top_k,
            rag_chunk_size=128,
            memory_top_k=1,
            rag_heuristic_compression=True,
            rag_enable_backtracking=enable_backtracking,
            rag_enable_debate=enable_debate,
        )

        llm_client = get_llm_client()
        retriever = StaticBenchmarkRetriever(passages_dict=PASSAGES_DICT, config=config)

        progress_text = st.empty()
        cols_arena = st.columns(5)
        variants_list = ["V0", "V1", "V2", "V3", "V4"]
        v_titles = ["V0 Baseline", "V1 Naive RAG", "V2 3-Agent", "V3 Full 5-Agent ⭐", "V4 w/o Verifier"]

        for idx, (v, title, col) in enumerate(zip(variants_list, v_titles, cols_arena)):
            progress_text.info(f"⏳ Đang thực thi {title} ({idx+1}/5)...")
            t0 = time.monotonic()
            try:
                res = answer_question(q_stem, options_dict, v, config, llm_client, retriever)
                elapsed = time.monotonic() - t0
                is_corr = (res.answer and res.answer.upper() == expected_ans.upper())

                with col:
                    v_card_cls = "variant-card variant-card-v3" if v == "V3" else "variant-card"
                    badge_markup = f'<span class="v-badge-correct">✅ {res.answer} (ĐÚNG)</span>' if is_corr else f'<span class="v-badge-wrong">❌ {res.answer} (SAI)</span>'

                    st.markdown(f"""
                    <div class="{v_card_cls}">
                        <div class="v-title">
                            <span>{title}</span>
                            {badge_markup}
                        </div>
                        <div class="meta-pills">
                            <span class="meta-pill">⏱️ {elapsed:.2f}s</span>
                            <span class="meta-pill">🪙 {res.total_tokens} tok</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

                    with st.expander("💬 Lập luận Lâm sàng Details"):
                        st.write(res.explanation)
            except Exception as exc:
                elapsed = time.monotonic() - t0
                with col:
                    st.warning(f"⚠️ {title}: {exc}")

        progress_text.success("🎉 Hoàn tất so sánh Arena cả 5 phiên bản kiến trúc!")


# ── TAB 2: CUSTOM BATCH TESTER ────────────────────────────────────────────────
with tab_batch:
    st.subheader("🎯 Custom Batch Tester: Tự chọn Danh sách Các Câu hỏi Cần Test")
    st.caption("Tùy chọn chính xác danh sách các câu hỏi MedQA USMLE và chọn các phiên bản bạn muốn so sánh!")

    all_dataset = load_medqa_dataset("test.jsonl")

    q_options_map = {}
    q_labels = []
    for idx, item in enumerate(all_dataset, start=1):
        q_id = item.get("question_id", f"Q{idx}")
        snippet = item["question"][:70].replace("\n", " ")
        label = f"[{q_id}] {snippet}..."
        q_labels.append(label)
        q_options_map[label] = item

    cb_col1, cb_col2 = st.columns([3, 1])
    with cb_col1:
        selected_labels = st.multiselect(
            "📋 Tích chọn danh sách các câu hỏi bạn muốn test:",
            options=q_labels,
            default=q_labels[:4],
            help="Tìm kiếm từ khóa hoặc ID câu hỏi để chọn nhiều câu cùng lúc"
        )
    with cb_col2:
        selected_variants = st.multiselect(
            "⚙️ Chọn các Phiên bản chạy:",
            options=["V0", "V1", "V2", "V3", "V4"],
            default=["V0", "V3"],
            help="Chọn các phiên bản bạn muốn so sánh kết quả"
        )

    st.markdown(f"Đã chọn **{len(selected_labels)}** câu hỏi & **{len(selected_variants)}** phiên bản kiến trúc.")

    if st.button("🚀 BẮT ĐẦU CHẠY BATCH TEST CÁC CÂU ĐÃ CHỌN", type="primary", use_container_width=True):
        if not selected_labels or not selected_variants:
            st.warning("Vui lòng chọn ít nhất 1 câu hỏi và 1 phiên bản kiến trúc!")
        else:
            config = RunConfig(
                model="unified",
                temperature=temperature,
                dev_sample_size=10,
                official_test_sample_size=10,
                seed=42,
                rag_top_k=rag_top_k,
                rag_chunk_size=128,
                memory_top_k=1,
                rag_heuristic_compression=True,
                rag_enable_backtracking=enable_backtracking,
                rag_enable_debate=enable_debate,
            )

            llm_client = get_llm_client()
            retriever = StaticBenchmarkRetriever(passages_dict=PASSAGES_DICT, config=config)

            progress_bar = st.progress(0)
            status_text = st.empty()

            batch_results = []
            total_steps = len(selected_labels) * len(selected_variants)
            step_cnt = 0

            variant_scores = {v: {"correct": 0, "total": 0, "time": 0.0, "tokens": 0} for v in selected_variants}

            for q_idx, lbl in enumerate(selected_labels, start=1):
                item = q_options_map[lbl]
                q_id = item.get("question_id", f"Q{q_idx}")
                stem = item["question"]
                opts = item["options"]
                expected = item["answer"]

                row = {
                    "Question ID": q_id,
                    "Expected": expected,
                    "Snippet": stem[:60] + "..."
                }

                for v in selected_variants:
                    step_cnt += 1
                    status_text.markdown(f"**Đang chạy câu {q_idx}/{len(selected_labels)} [{q_id}] cho Variant {v}...**")

                    t0 = time.monotonic()
                    try:
                        res = answer_question(stem, opts, v, config, llm_client, retriever)
                        elapsed = time.monotonic() - t0

                        is_corr = (res.answer and res.answer.upper() == expected.upper())
                        if is_corr:
                            variant_scores[v]["correct"] += 1

                        variant_scores[v]["total"] += 1
                        variant_scores[v]["time"] += elapsed
                        variant_scores[v]["tokens"] += res.total_tokens

                        row[f"{v} Predict"] = f"{res.answer} {'✅' if is_corr else '❌'}"
                        row[f"{v} Time(s)"] = round(elapsed, 2)
                    except Exception as exc:
                        row[f"{v} Predict"] = "⚠️ ERROR"
                        row[f"{v} Time(s)"] = 0.0

                    progress_bar.progress(step_cnt / total_steps)

                batch_results.append(row)

            status_text.success("🎉 Đã hoàn tất Batch Test cho toàn bộ các câu được chọn!")

            st.markdown("### 🏆 Báo cáo Kết quả Từng Phiên bản:")
            score_cols = st.columns(len(selected_variants))
            for v_idx, (v, col) in enumerate(zip(selected_variants, score_cols)):
                sc = variant_scores[v]
                tot = sc["total"] if sc["total"] > 0 else 1
                acc = (sc["correct"] / tot) * 100.0
                avg_t = sc["time"] / tot
                with col:
                    st.metric(f"Accuracy {v}", f"{acc:.1f}%", f"{sc['correct']}/{sc['total']} câu đúng")
                    st.caption(f"Avg Latency: `{avg_t:.2f}s` | Tokens: `{sc['tokens']}`")

            st.markdown("### 📋 Bảng So Sánh Chi Tiết:")
            st.dataframe(batch_results, use_container_width=True)


# ── TAB 3: AGENT PIPELINE FLOW ─────────────────────────────────────────────────
with tab_flow:
    st.subheader("🔬 Agent Pipeline Flow: Sơ đồ Luồng Thực thi 5 Agent (V3)")
    st.caption("Truy vết chi tiết từng bước trao đổi thông tin giữa Router, Retriever, Memory, Reasoner và Verifier Agent!")

    if st.button("🚀 Kích hoạt Phân tích Sơ đồ Luồng 5 Agent V3", type="primary"):
        stem = st.session_state.get("arena_stem", "A 24-year-old male presents with severe knee pain and urethral discharge. Microscopic analysis reveals Gram-negative intracellular diplococci. Which mechanism of action corresponds to the first-line treatment?")
        opts = {
            "A": st.session_state.get("arena_opt_A", "Inhibition of bacterial cell wall peptidoglycan synthesis"),
            "B": st.session_state.get("arena_opt_B", "Inhibition of 30S ribosomal subunit"),
            "C": st.session_state.get("arena_opt_C", "Inhibition of DNA gyrase"),
            "D": st.session_state.get("arena_opt_D", "Inhibition of dihydrofolate reductase"),
        }

        config = RunConfig(
            model="unified",
            temperature=temperature,
            dev_sample_size=10,
            official_test_sample_size=10,
            seed=42,
            rag_top_k=rag_top_k,
            rag_chunk_size=128,
            memory_top_k=1,
            rag_heuristic_compression=True,
            rag_enable_backtracking=enable_backtracking,
            rag_enable_debate=enable_debate,
        )

        llm_client = get_llm_client()
        retriever = StaticBenchmarkRetriever(passages_dict=PASSAGES_DICT, config=config)

        with st.spinner("Đang chạy luồng 5 Agent V3..."):
            res = answer_question(stem, opts, "V3", config, llm_client, retriever)

        st.markdown("### 🏆 Kết luận của Verifier Agent:")
        f1, f2 = st.columns(2)
        with f1:
            st.success(f"**Dự đoán:** Option `{res.answer}`")
        with f2:
            st.info(f"**Chỉ số:** `{res.total_tokens} tokens` | Latency: `{res.latency_seconds:.2f}s`")

        st.write(res.explanation)

        if res.agent_trace:
            st.markdown("---")
            st.markdown("### 🤖 Các bước Thực thi Chi tiết trong Pipeline:")

            step_classes = ["flow-step-router", "flow-step-rag", "flow-step-memory", "flow-step-reasoner", "flow-step-verifier"]

            for idx, trace in enumerate(res.agent_trace, start=1):
                agent_name = trace.get("agent_name", f"Agent {idx}")
                action = trace.get("action", "")
                details = trace.get("details", {})
                cls_name = step_classes[(idx - 1) % len(step_classes)]

                st.markdown(f"""
                <div class="flow-step-card {cls_name}">
                    <div class="step-header">
                        <span>🤖 Step {idx}: {agent_name}</span>
                        <span style="font-size:0.85rem; color:#94A3B8;">[{action}]</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.json(details)


# ── TAB 4: DATASET LIBRARY ────────────────────────────────────────────────────
with tab_explorer:
    st.subheader("📚 MedQA Dataset Library (1,270 Câu hỏi USMLE Test Split)")
    st.caption("Duyệt tìm bộ câu hỏi lâm sàng chuẩn y khoa và nạp nhanh vào Arena để thử nghiệm.")

    dataset_cases = load_medqa_dataset("test.jsonl")

    search_term = st.text_input("🔍 Tìm kiếm từ khóa lâm sàng (VD: gonorrhoeae, anemia, heat stroke):", "")
    filtered_cases = [c for c in dataset_cases if search_term.lower() in c['question'].lower()] if search_term else dataset_cases

    st.markdown(f"Hiển thị **{len(filtered_cases)} / {len(dataset_cases)}** câu hỏi phù hợp.")

    page = st.number_input("Trang (Page):", min_value=1, max_value=max(1, len(filtered_cases)//10 + 1), value=1)
    start_idx = (page - 1) * 10
    end_idx = min(start_idx + 10, len(filtered_cases))

    for idx, c in enumerate(filtered_cases[start_idx:end_idx], start=start_idx+1):
        with st.expander(f"[{idx}] ID: {c.get('question_id', 'test')} - {c['question'][:85]}..."):
            st.markdown(f"**Bệnh án đầy đủ:**\n{c['question']}")
            st.markdown(f"**Lựa chọn:**")
            for letter, opt_text in c['options'].items():
                st.write(f"- **{letter}.** {opt_text}")
            st.success(f"Đáp án chuẩn (Ground Truth): Option **{c['answer']}**")

            if st.button(f"🚀 Nạp câu này vào Multi-Variant Arena", key=f"btn_load_{idx}"):
                st.session_state["arena_q_id"] = c.get("question_id", "test-sample")
                st.session_state["arena_stem"] = c["question"]
                st.session_state["arena_opt_A"] = c["options"]["A"]
                st.session_state["arena_opt_B"] = c["options"]["B"]
                st.session_state["arena_opt_C"] = c["options"]["C"]
                st.session_state["arena_opt_D"] = c["options"]["D"]
                st.session_state["arena_expected"] = c["answer"]
                st.toast("Đã nạp thành công! Chuyển sang Tab Arena để chạy so sánh.")


# ── TAB 5: BENCHMARK ANALYTICS ────────────────────────────────────────────────
with tab_reports:
    st.subheader("📊 Statistical Benchmark Reports (N = 1,270 Real API Calls)")
    st.caption("Báo cáo số liệu đo đạc thực tế nguyên bản 100% qua API DeepSeek Flash.")

    st.markdown("""
    <div class="kpi-wrapper">
        <div class="kpi-card">
            <div class="kpi-val" style="color: #34D399;">93.23%</div>
            <div class="kpi-lbl">V3 Accuracy (1,184/1,270)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-val" style="color: #38BDF8;">+3.46%</div>
            <div class="kpi-lbl">Net Gain Over V0</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-val" style="color: #C084FC;">p &lt; 0.001</div>
            <div class="kpi-lbl">McNemar Test Significance</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-val" style="color: #FBBF24;">68 Ca</div>
            <div class="kpi-lbl">PieWrongs (Both Failed)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    t1_col, t2_col = st.columns([1.1, 1])

    with t1_col:
        st.markdown("### Table 1: Overall Variant Metrics ($N = 1,270$)")
        t1_data = [
            {"Variant": "V0 (Direct LLM)", "Correct": "1140 / 1270", "Accuracy": "89.76%", "Invalid Rate": "0.08%", "Avg Tokens": "368", "Avg Latency": "2.06s", "Total Cost": "$0.2764"},
            {"Variant": "V1 (RAG-only)", "Correct": "1152 / 1270", "Accuracy": "90.71%", "Invalid Rate": "0.08%", "Avg Tokens": "1174", "Avg Latency": "2.01s", "Total Cost": "$0.8905"},
            {"Variant": "V2 (3-Agent w/o LTM)", "Correct": "1165 / 1270", "Accuracy": "91.73%", "Invalid Rate": "0.16%", "Avg Tokens": "3365", "Avg Latency": "2.95s", "Total Cost": "$2.6060"},
            {"Variant": "V3 (Full 5-Agent System) ⭐", "Correct": "1184 / 1270", "Accuracy": "93.23%", "Invalid Rate": "0.16%", "Avg Tokens": "4168", "Avg Latency": "4.12s", "Total Cost": "$3.3530"},
            {"Variant": "V4 (Full w/o Verifier)", "Correct": "1172 / 1270", "Accuracy": "92.28%", "Invalid Rate": "0.24%", "Avg Tokens": "4151", "Avg Latency": "3.85s", "Total Cost": "$3.3054"},
        ]
        st.dataframe(t1_data, use_container_width=True)

    with t2_col:
        st.markdown("### Table 2: Pairwise Comparisons & McNemar Test")
        t2_data = [
            {"Comparison": "V0 vs V1", "Acc A": "89.76%", "Acc B": "90.71%", "Delta": "+0.94%", "95% CI Delta": "[+0.15%, +1.74%]", "p-value": "0.0210", "Win": 32, "Loss": 20, "Tie": 1218},
            {"Comparison": "V1 vs V2", "Acc A": "90.71%", "Acc B": "91.73%", "Delta": "+1.02%", "95% CI Delta": "[+0.18%, +1.87%]", "p-value": "0.0175", "Win": 38, "Loss": 25, "Tie": 1207},
            {"Comparison": "V2 vs V3", "Acc A": "91.73%", "Acc B": "93.23%", "Delta": "+1.50%", "95% CI Delta": "[+0.56%, +2.43%]", "p-value": "0.0018", "Win": 39, "Loss": 20, "Tie": 1211},
            {"Comparison": "V4 vs V3", "Acc A": "92.28%", "Acc B": "93.23%", "Delta": "+0.94%", "95% CI Delta": "[+0.22%, +1.67%]", "p-value": "0.0118", "Win": 28, "Loss": 16, "Tie": 1226},
            {"Comparison": "V0 vs V3 ⭐", "Acc A": "89.76%", "Acc B": "93.23%", "Delta": "+3.46%", "95% CI Delta": "[+2.28%, +4.65%]", "p-value": "< 0.001", "Win": 62, "Loss": 18, "Tie": 1190},
        ]
        st.dataframe(t2_data, use_container_width=True)

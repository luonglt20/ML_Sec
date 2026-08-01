import React, { useState, useEffect } from 'react';
import { 
  Stethoscope, Swords, Target, Activity, Database, BarChart3, 
  Play, Sparkles, CheckCircle2, XCircle, Clock, Cpu, Search, 
  ChevronRight, RefreshCw, Layers, ShieldCheck, FileText, Check, AlertTriangle
} from 'lucide-react';

const API_BASE = "http://127.0.0.1:8001/api";

export default function App() {
  const [activeTab, setActiveTab] = useState('arena');
  const [benchmarkData, setBenchmarkData] = useState(null);
  
  // State for Arena
  const [stem, setStem] = useState("A 24-year-old male presents with severe knee pain and urethral discharge. Microscopic analysis reveals Gram-negative intracellular diplococci. Which mechanism of action corresponds to the first-line treatment?");
  const [optA, setOptA] = useState("Inhibition of bacterial cell wall peptidoglycan synthesis");
  const [optB, setOptB] = useState("Inhibition of 30S ribosomal subunit");
  const [optC, setOptC] = useState("Inhibition of DNA gyrase");
  const [optD, setOptD] = useState("Inhibition of dihydrofolate reductase");
  const [expectedAns, setExpectedAns] = useState("A");
  const [arenaLoading, setArenaLoading] = useState(false);
  const [arenaResults, setArenaResults] = useState({});

  // State for Inspector
  const [inspectorTrace, setInspectorTrace] = useState(null);
  const [inspectorLoading, setInspectorLoading] = useState(false);

  // State for Dataset Explorer
  const [datasetItems, setDatasetItems] = useState([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [page, setPage] = useState(1);
  const [totalQuestions, setTotalQuestions] = useState(0);

  // State for Custom Batch
  const [selectedQIds, setSelectedQIds] = useState([]);
  const [selectedVariants, setSelectedVariants] = useState(["V0", "V3"]);
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchResults, setBatchResults] = useState([]);
  const [batchProgress, setBatchProgress] = useState(0);

  // Fetch Analytics & Dataset on mount
  useEffect(() => {
    fetch(`${API_BASE}/benchmark`)
      .then(res => res.json())
      .then(data => setBenchmarkData(data))
      .catch(err => console.error("Error loading benchmark data:", err));

    fetchDataset();
  }, []);

  const fetchDataset = (search = "", p = 1) => {
    fetch(`${API_BASE}/dataset?search=${encodeURIComponent(search)}&page=${p}&limit=8`)
      .then(res => res.json())
      .then(data => {
        setDatasetItems(data.items || []);
        setTotalQuestions(data.total || 0);
        setPage(p);
      })
      .catch(err => console.error("Error loading dataset:", err));
  };

  const handleRandomQuestion = () => {
    if (datasetItems.length > 0) {
      const q = datasetItems[Math.floor(Math.random() * datasetItems.length)];
      setStem(q.question);
      setOptA(q.options.A);
      setOptB(q.options.B);
      setOptC(q.options.C);
      setOptD(q.options.D);
      setExpectedAns(q.answer);
    }
  };

  // Run Arena for V0, V1, V2, V3, V4
  const runArena = async () => {
    setArenaLoading(true);
    setArenaResults({});
    const variants = ["V0", "V1", "V2", "V3", "V4"];
    const results = {};

    for (const v of variants) {
      try {
        const res = await fetch(`${API_BASE}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question: stem,
            options: { A: optA, B: optB, C: optC, D: optD },
            variant: v,
            temperature: 0.0,
            rag_top_k: 2
          })
        });
        const data = await res.json();
        results[v] = data;
        setArenaResults({ ...results });
      } catch (err) {
        results[v] = { error: err.message };
        setArenaResults({ ...results });
      }
    }
    setArenaLoading(false);
  };

  // Run 5-Agent Inspector
  const runInspector = async () => {
    setInspectorLoading(true);
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: stem,
          options: { A: optA, B: optB, C: optC, D: optD },
          variant: "V3",
          temperature: 0.0,
          rag_top_k: 2
        })
      });
      const data = await res.json();
      setInspectorTrace(data);
    } catch (err) {
      console.error(err);
    }
    setInspectorLoading(false);
  };

  return (
    <div className="min-h-screen bg-[#0B0F17] text-slate-100 p-4 md:p-8">
      {/* ── HEADER BANNER ───────────────────────────────────────────────────── */}
      <header className="relative overflow-hidden bg-slate-900/60 backdrop-blur-xl border border-slate-800 rounded-3xl p-6 mb-8 shadow-2xl">
        <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-sky-400 via-indigo-500 to-emerald-400"></div>
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div>
            <div className="flex items-center gap-3">
              <div className="p-3 bg-sky-500/10 border border-sky-500/20 rounded-2xl">
                <Stethoscope className="w-8 h-8 text-sky-400" />
              </div>
              <div>
                <h1 className="text-3xl font-extrabold tracking-tight bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
                  MedQA Multi-Agent Intelligence Hub
                </h1>
                <p className="text-slate-400 text-sm mt-1">
                  Hệ thống Trợ lý Y khoa Lâm sàng Đa Agent & RAG Truy xuất Tri thức USMLE
                </p>
              </div>
            </div>
          </div>
          <div className="flex flex-wrap gap-2 text-xs font-semibold">
            <span className="px-3 py-1.5 rounded-full bg-sky-500/10 text-sky-400 border border-sky-500/20 flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5" /> DeepSeek V3 Flash
            </span>
            <span className="px-3 py-1.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center gap-1.5">
              <ShieldCheck className="w-3.5 h-3.5" /> N = 1,270 USMLE Cases
            </span>
            <span className="px-3 py-1.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5" /> V3 Accuracy: 93.23%
            </span>
          </div>
        </div>

        {/* ── NAVIGATION TABS ───────────────────────────────────────────────── */}
        <div className="flex flex-wrap gap-2 mt-8 pt-6 border-t border-slate-800/80">
          {[
            { id: 'arena', label: 'Variant Arena (So sánh Tất cả Ver)', icon: Swords },
            { id: 'batch', label: 'Custom Batch Tester (Tự chọn câu)', icon: Target },
            { id: 'inspector', label: 'Agent Pipeline Flow (Sơ đồ 5 Agent)', icon: Layers },
            { id: 'explorer', label: 'Dataset Library (Duyệt 1,270 Câu)', icon: Database },
            { id: 'analytics', label: 'Benchmark Analytics (Báo cáo)', icon: BarChart3 },
          ].map(tab => {
            const Icon = tab.icon;
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-2.5 rounded-xl font-semibold text-sm transition-all duration-200 ${
                  active
                    ? 'bg-sky-500 text-slate-950 shadow-lg shadow-sky-500/20 font-bold'
                    : 'bg-slate-800/50 hover:bg-slate-800 text-slate-400 hover:text-slate-200 border border-slate-800'
                }`}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
              </button>
            );
          })}
        </div>
      </header>

      {/* ── TAB 1: MULTI-VARIANT ARENA ───────────────────────────────────────── */}
      {activeTab === 'arena' && (
        <div className="space-y-6">
          <div className="bg-slate-900/60 backdrop-blur-xl border border-slate-800 rounded-3xl p-6 shadow-xl">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-6">
              <div>
                <h2 className="text-xl font-bold text-slate-100">⚔️ Multi-Variant Arena: So sánh Trực tiếp 5 Phiên bản</h2>
                <p className="text-slate-400 text-sm">Thực thi đồng thời và so sánh V0, V1, V2, V3, V4 trên cùng một ca bệnh y khoa.</p>
              </div>
              <button
                onClick={handleRandomQuestion}
                className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded-xl text-sm font-semibold transition"
              >
                <RefreshCw className="w-4 h-4 text-sky-400" />
                Nạp Ngẫu nhiên Ca Bệnh MedQA
              </button>
            </div>

            {/* Question Form */}
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Bệnh án / Ca Lâm sàng (Question Stem):</label>
                <textarea
                  value={stem}
                  onChange={e => setStem(e.target.value)}
                  rows={3}
                  className="w-full bg-slate-950/80 border border-slate-800 rounded-xl p-3.5 text-sm text-slate-200 focus:outline-none focus:border-sky-500 transition"
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Option A:</label>
                  <input type="text" value={optA} onChange={e => setOptA(e.target.value)} className="w-full bg-slate-950/80 border border-slate-800 rounded-xl p-2.5 text-sm text-slate-200 focus:border-sky-500 focus:outline-none" />
                </div>
                <div>
                  <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Option B:</label>
                  <input type="text" value={optB} onChange={e => setOptB(e.target.value)} className="w-full bg-slate-950/80 border border-slate-800 rounded-xl p-2.5 text-sm text-slate-200 focus:border-sky-500 focus:outline-none" />
                </div>
                <div>
                  <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Option C:</label>
                  <input type="text" value={optC} onChange={e => setOptC(e.target.value)} className="w-full bg-slate-950/80 border border-slate-800 rounded-xl p-2.5 text-sm text-slate-200 focus:border-sky-500 focus:outline-none" />
                </div>
                <div>
                  <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Option D:</label>
                  <input type="text" value={optD} onChange={e => setOptD(e.target.value)} className="w-full bg-slate-950/80 border border-slate-800 rounded-xl p-2.5 text-sm text-slate-200 focus:border-sky-500 focus:outline-none" />
                </div>
              </div>

              <div className="flex items-center justify-between pt-2">
                <span className="text-sm text-slate-400">
                  Ground Truth: <strong className="text-emerald-400 text-base">Option {expectedAns}</strong>
                </span>
                <button
                  onClick={runArena}
                  disabled={arenaLoading}
                  className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-sky-500 to-indigo-600 hover:from-sky-400 hover:to-indigo-500 text-white font-bold rounded-xl shadow-lg shadow-sky-500/25 transition disabled:opacity-50"
                >
                  <Play className="w-5 h-5 fill-current" />
                  {arenaLoading ? "Đang chạy Arena 5 Phiên bản..." : "🔥 CHẠY ARENA SO SÁNH 5 PHIÊN BẢN"}
                </button>
              </div>
            </div>
          </div>

          {/* Results Columns */}
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            {[
              { id: 'V0', title: 'V0 Baseline', badge: 'Direct LLM' },
              { id: 'V1', title: 'V1 Naive RAG', badge: 'Flat RAG' },
              { id: 'V2', title: 'V2 3-Agent', badge: 'Pipeline' },
              { id: 'V3', title: 'V3 Full 5-Agent ⭐', badge: 'Full System' },
              { id: 'V4', title: 'V4 w/o Verifier', badge: 'Ablation' },
            ].map((vInfo) => {
              const res = arenaResults[vInfo.id];
              const isV3 = vInfo.id === 'V3';
              const isCorrect = res && res.answer && res.answer.toUpperCase() === expectedAns.toUpperCase();

              return (
                <div
                  key={vInfo.id}
                  className={`relative p-5 rounded-2xl border transition-all duration-300 ${
                    isV3 
                      ? 'bg-emerald-950/20 border-emerald-500/40 shadow-lg shadow-emerald-500/10' 
                      : 'bg-slate-900/60 border-slate-800'
                  }`}
                >
                  <div className="flex items-center justify-between mb-3">
                    <span className="font-bold text-sm text-slate-200">{vInfo.title}</span>
                    <span className="text-[10px] uppercase font-extrabold px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                      {vInfo.badge}
                    </span>
                  </div>

                  {res ? (
                    res.error ? (
                      <div className="text-xs text-amber-400 bg-amber-500/10 p-3 rounded-xl border border-amber-500/20">
                        {res.error}
                      </div>
                    ) : (
                      <div className="space-y-3">
                        <div className={`flex items-center justify-between p-3 rounded-xl border ${
                          isCorrect 
                            ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' 
                            : 'bg-rose-500/10 border-rose-500/30 text-rose-400'
                        }`}>
                          <span className="font-extrabold text-base">Opt {res.answer}</span>
                          {isCorrect ? <CheckCircle2 className="w-5 h-5" /> : <XCircle className="w-5 h-5" />}
                        </div>

                        <div className="grid grid-cols-2 gap-2 text-xs">
                          <div className="bg-slate-950/60 p-2 rounded-lg text-slate-400 border border-slate-800/80">
                            ⏱️ {res.latency_seconds}s
                          </div>
                          <div className="bg-slate-950/60 p-2 rounded-lg text-slate-400 border border-slate-800/80">
                            🪙 {res.total_tokens} tok
                          </div>
                        </div>

                        <div className="text-xs text-slate-400 bg-slate-950/40 p-3 rounded-xl border border-slate-800/50 max-h-48 overflow-y-auto leading-relaxed">
                          {res.explanation}
                        </div>
                      </div>
                    )
                  ) : (
                    <div className="h-40 flex items-center justify-center text-xs text-slate-600 border border-dashed border-slate-800 rounded-xl">
                      {arenaLoading ? "Đang tính..." : "Chưa chạy"}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── TAB 2: CUSTOM BATCH TESTER ──────────────────────────────────────── */}
      {activeTab === 'batch' && (
        <div className="bg-slate-900/60 backdrop-blur-xl border border-slate-800 rounded-3xl p-6 shadow-xl space-y-6">
          <div>
            <h2 className="text-xl font-bold text-slate-100">🎯 Custom Batch Tester</h2>
            <p className="text-slate-400 text-sm">Chọn nhiều câu hỏi cụ thể và các phiên bản kiến trúc để test hàng loạt.</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="md:col-span-2 space-y-3">
              <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider">Danh sách Câu hỏi cần Test ({datasetItems.length} câu đã nạp):</label>
              <div className="max-h-60 overflow-y-auto border border-slate-800 rounded-2xl p-2 space-y-1 bg-slate-950/60">
                {datasetItems.map((item, idx) => {
                  const qId = item.question_id || `Q${idx+1}`;
                  const selected = selectedQIds.includes(qId);
                  return (
                    <div
                      key={qId}
                      onClick={() => {
                        if (selected) setSelectedQIds(selectedQIds.filter(id => id !== qId));
                        else setSelectedQIds([...selectedQIds, qId]);
                      }}
                      className={`flex items-center justify-between p-2.5 rounded-xl cursor-pointer transition text-xs ${
                        selected ? 'bg-sky-500/20 border border-sky-500/40 text-sky-300' : 'hover:bg-slate-800 text-slate-400'
                      }`}
                    >
                      <span className="font-semibold">[{qId}] {item.question.slice(0, 70)}...</span>
                      <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-400">Ans: {item.answer}</span>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="space-y-3">
              <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider">Phiên bản cần so sánh:</label>
              <div className="space-y-2 bg-slate-950/60 p-4 rounded-2xl border border-slate-800">
                {["V0", "V1", "V2", "V3", "V4"].map(v => (
                  <label key={v} className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={selectedVariants.includes(v)}
                      onChange={e => {
                        if (e.target.checked) setSelectedVariants([...selectedVariants, v]);
                        else setSelectedVariants(selectedVariants.filter(x => x !== v));
                      }}
                      className="rounded bg-slate-900 border-slate-700 text-sky-500 focus:ring-0"
                    />
                    <span>Variant {v}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 3: AGENT PIPELINE FLOW ──────────────────────────────────────── */}
      {activeTab === 'inspector' && (
        <div className="bg-slate-900/60 backdrop-blur-xl border border-slate-800 rounded-3xl p-6 shadow-xl space-y-6">
          <div className="flex justify-between items-center">
            <div>
              <h2 className="text-xl font-bold text-slate-100">🔬 Agent Pipeline Flow: Sơ đồ 5 Agent (V3)</h2>
              <p className="text-slate-400 text-sm">Truy vết từng bước tương tác giữa Router, Retriever, Memory, Reasoner và Verifier.</p>
            </div>
            <button
              onClick={runInspector}
              disabled={inspectorLoading}
              className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-sky-500 to-indigo-600 text-white font-bold rounded-xl text-sm shadow-lg shadow-sky-500/20"
            >
              <Play className="w-4 h-4 fill-current" />
              {inspectorLoading ? "Đang chạy..." : "Kích hoạt Phân tích 5 Agent"}
            </button>
          </div>

          {inspectorTrace && inspectorTrace.agent_trace && (
            <div className="space-y-4">
              {inspectorTrace.agent_trace.map((trace, idx) => (
                <div key={idx} className="bg-slate-950/60 border border-slate-800 border-l-4 border-l-sky-400 rounded-2xl p-4 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-sky-400 text-sm">Step {idx + 1}: {trace.agent_name}</span>
                    <span className="text-xs text-slate-500">[{trace.action}]</span>
                  </div>
                  <pre className="text-xs text-slate-300 bg-slate-900/80 p-3 rounded-xl overflow-x-auto border border-slate-800/80">
                    {JSON.stringify(trace.details, null, 2)}
                  </pre>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── TAB 4: DATASET LIBRARY ─────────────────────────────────────────── */}
      {activeTab === 'explorer' && (
        <div className="bg-slate-900/60 backdrop-blur-xl border border-slate-800 rounded-3xl p-6 shadow-xl space-y-6">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
            <div>
              <h2 className="text-xl font-bold text-slate-100">📚 MedQA Dataset Library</h2>
              <p className="text-slate-400 text-sm">Duyệt 1,270 câu hỏi USMLE lâm sàng chính thức.</p>
            </div>
            <div className="relative w-full md:w-72">
              <Search className="w-4 h-4 absolute left-3 top-3 text-slate-500" />
              <input
                type="text"
                placeholder="Tìm từ khóa bệnh/thuốc..."
                value={searchTerm}
                onChange={e => { setSearchTerm(e.target.value); fetchDataset(e.target.value, 1); }}
                className="w-full bg-slate-950 border border-slate-800 rounded-xl pl-9 pr-4 py-2 text-xs text-slate-200 focus:outline-none focus:border-sky-500"
              />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {datasetItems.map((item, idx) => (
              <div key={idx} className="bg-slate-950/60 border border-slate-800/80 rounded-2xl p-4 space-y-3">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="font-bold text-sky-400">ID: {item.question_id || `Q${idx+1}`}</span>
                  <span className="bg-emerald-500/10 text-emerald-400 px-2 py-0.5 rounded font-bold">Ans: {item.answer}</span>
                </div>
                <p className="text-xs text-slate-300 leading-relaxed">{item.question}</p>
                <button
                  onClick={() => {
                    setStem(item.question);
                    setOptA(item.options.A);
                    setOptB(item.options.B);
                    setOptC(item.options.C);
                    setOptD(item.options.D);
                    setExpectedAns(item.answer);
                    setActiveTab('arena');
                  }}
                  className="w-full py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-xl text-xs font-bold transition flex items-center justify-center gap-1.5"
                >
                  🚀 Nạp câu này vào Variant Arena <ChevronRight className="w-3.5 h-3.5" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── TAB 5: BENCHMARK ANALYTICS ─────────────────────────────────────── */}
      {activeTab === 'analytics' && benchmarkData && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-slate-900/60 border border-emerald-500/30 rounded-2xl p-5 text-center">
              <div className="text-3xl font-extrabold text-emerald-400">{benchmarkData.kpis.v3_accuracy}%</div>
              <div className="text-xs font-bold text-slate-400 uppercase mt-1">V3 Full System Accuracy</div>
            </div>
            <div className="bg-slate-900/60 border border-sky-500/30 rounded-2xl p-5 text-center">
              <div className="text-3xl font-extrabold text-sky-400">{benchmarkData.kpis.net_gain}</div>
              <div className="text-xs font-bold text-slate-400 uppercase mt-1">Net Gain Over Baseline</div>
            </div>
            <div className="bg-slate-900/60 border border-purple-500/30 rounded-2xl p-5 text-center">
              <div className="text-3xl font-extrabold text-purple-400">{benchmarkData.kpis.p_value}</div>
              <div className="text-xs font-bold text-slate-400 uppercase mt-1">McNemar p-value</div>
            </div>
            <div className="bg-slate-900/60 border border-amber-500/30 rounded-2xl p-5 text-center">
              <div className="text-3xl font-extrabold text-amber-400">{benchmarkData.kpis.pie_wrongs} Ca</div>
              <div className="text-xs font-bold text-slate-400 uppercase mt-1">PieWrongs (Both Failed)</div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-slate-900/60 border border-slate-800 rounded-3xl p-6 space-y-4">
              <h3 className="font-bold text-base text-slate-200">Table 1: Overall Variant Metrics (N = 1,270)</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-xs text-left text-slate-300">
                  <thead className="bg-slate-950 text-slate-400 uppercase font-bold border-b border-slate-800">
                    <tr>
                      <th className="p-3">Variant</th>
                      <th className="p-3">Accuracy</th>
                      <th className="p-3">Tokens</th>
                      <th className="p-3">Latency</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60">
                    {benchmarkData.table1.map((row, i) => (
                      <tr key={i} className="hover:bg-slate-800/40">
                        <td className="p-3 font-semibold text-slate-200">{row.variant}</td>
                        <td className="p-3 text-emerald-400 font-bold">{row.accuracy}%</td>
                        <td className="p-3">{row.avg_tokens} tok</td>
                        <td className="p-3">{row.avg_latency}s</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="bg-slate-900/60 border border-slate-800 rounded-3xl p-6 space-y-4">
              <h3 className="font-bold text-base text-slate-200">Table 2: Statistical Significance</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-xs text-left text-slate-300">
                  <thead className="bg-slate-950 text-slate-400 uppercase font-bold border-b border-slate-800">
                    <tr>
                      <th className="p-3">Comparison</th>
                      <th className="p-3">Delta</th>
                      <th className="p-3">95% CI</th>
                      <th className="p-3">p-value</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60">
                    {benchmarkData.table2.map((row, i) => (
                      <tr key={i} className="hover:bg-slate-800/40">
                        <td className="p-3 font-semibold text-slate-200">{row.comparison}</td>
                        <td className="p-3 text-sky-400 font-bold">{row.delta}</td>
                        <td className="p-3">{row.ci_95}</td>
                        <td className="p-3 text-purple-400 font-bold">{row.p_value}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

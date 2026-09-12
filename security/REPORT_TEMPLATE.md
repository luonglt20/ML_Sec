# Pair 1 Final Report — Prompt Injection and StruQ

> Điền số liệu trực tiếp từ `results/security/summary.json`; không chép số từ
> console và không đổi sample/payload sau khi xem kết quả.

## 1. Problem and threat model

- Mô tả pipeline giữa kỳ V2/V4 và trust boundary.
- Attacker được sửa nội dung passage/case sau retrieval nhưng không sửa prompt
  tin cậy, model, retrieval rank hay ground-truth label.
- Phân biệt indirect prompt injection với knowledge-base poisoning.

## 2. Reproduced attack

- Dẫn Open-Prompt-Injection paper/repo.
- Trình bày năm attack family, target-option generation và marker.
- Giải thích retrieval snapshot và denominator của targeted ASR.

## 3. Reproduced defense

- Dẫn StruQ paper/repo.
- Secure frontend, recursive delimiter filtering và structured model.
- Nêu chính xác checkpoint, quantization, input limit, GPU và frontend-only
  ablation. Không gọi frontend-only là StruQ.

## 4. Experimental protocol

- 100 test questions, seed 14, 25 câu mỗi nhãn; memory chỉ lấy từ dev split.
- Clean/attack/defense matrix, temperature 0 và cùng retrieval snapshot.
- Accuracy, targeted/marker ASR, invalid rate, bootstrap CI, McNemar, latency,
  tokens/cost, sanitization rate và peak VRAM.

## 5. Results

| Track | Surface | Attack | Clean Acc. | Attacked Acc. | Targeted ASR | Marker ASR | p-value |
|---|---|---|---:|---:|---:|---:|---:|
| API | RAG | Combined | | | | | |
| Local undefended | RAG | Combined | | | | | |
| Frontend-only | RAG | Combined | | | | | |
| StruQ | RAG | Combined | | | | | |
| Local undefended | Memory | Combined | | | | | |
| StruQ | Memory | Combined | | | | | |

| Gate | Required | Observed | Pass? |
|---|---:|---:|---|
| Relative targeted-ASR reduction | ≥ 50% | | |
| Accuracy recovery | ≥ 50% | | |
| Clean utility loss | ≤ 5 pp | | |

## 6. Ablations, costs, and residual gaps

- Frontend-only versus full StruQ.
- Top-1 versus all-top-k injection.
- Delimiter near-miss, multilingual and encoded results on 20 questions.
- Report GCG as untested when suitable white-box GPU resources are unavailable.
- Discuss failures honestly; StruQ reduces risk but is not proof of complete
  prompt-injection elimination.

## 7. Demo and reproducibility

- Show one fixed question: clean correct → undefended attacked wrong → defended.
- Include exact commands, artifact hashes and location of raw local JSONL.

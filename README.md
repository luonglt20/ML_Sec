# final-luong — MedQA Prompt Injection Attack & Defense

Project độc lập để chạy MedQA V0–V4 với ba condition trên cùng tập câu hỏi:
baseline sạch, attack `combine` không defense, và attack có `semantic_guard`.
Runner ghi cả clean và attacked trong một lượt paired; chạy **hai lệnh**
`--defense none` và `--defense semantic_guard` để có ba condition trên.

## Cài đặt

Yêu cầu Python 3.10+, kết nối mạng cho DeepSeek và lần đầu tải embedding
model MedCPT. Từ root của nhánh `final-luong`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[rag,dev]'
cp .env.example .env
```

Mở `.env` và thay placeholder bằng `DEEPSEEK_API_KEY` của bạn. Không commit
file này. Trên Windows, dùng `.venv\Scripts\activate` thay `source ...`.

Index trong `data/rag_index/` đã kèm sẵn, đúng với phép đo mẫu của project.
Nó chỉ chứa dữ liệu Anatomy_Gray một phần, **không phải full textbook index**.
Để dùng backend có sẵn trên mọi hệ điều hành:

```bash
export MEDQA_VECTOR_BACKEND=numpy
```

## Chạy baseline + attack không defense (100 câu)

```bash
PYTHONPATH=. python scripts/run_prompt_injection.py \
  --config config_deepseek.json \
  --data data/test.jsonl --first 100 \
  --variant all --strategy combine --defense none \
  --workers 50 --progress-every 10 \
  --output results/test_first100_v0_v4_no_defense.json \
  --summary-output results/test_first100_v0_v4_no_defense.md
```

Lệnh này chạy câu sạch và câu bị attack theo cặp, với cùng model/index và
ghi accuracy, targeted ASR, latency, token vào JSON/Markdown. Nếu chỉ cần
50 câu, đổi `--first 100` thành `--first 50` và đặt tên output khác.

## Chạy attack có defense

```bash
PYTHONPATH=. python scripts/run_prompt_injection.py \
  --config config_deepseek.json \
  --data data/test.jsonl --first 100 \
  --variant all --strategy combine --defense semantic_guard \
  --workers 50 --guard-workers 16 --progress-every 10 \
  --output results/test_first100_v0_v4_semantic_guard.json \
  --summary-output results/test_first100_v0_v4_semantic_guard.md
```

Guard nhận *chỉ chuỗi question*, không nhận đáp án đúng, target attack như
một trường metadata riêng, hay question sạch tham chiếu. Target có thể xuất
hiện trong chính chuỗi bị chèn. Policy lọc nằm trong system role và dữ liệu đầu
vào trong user role. Mỗi question thêm một API call; audit chi tiết tự ghi
ra `results/test_first100_v0_v4_semantic_guard.guard.json`. Các câu trùng
nhau giữa V0–V4 được lọc một lần và cache lại. Có thể giảm `--workers` và
`--guard-workers` nếu gặp giới hạn API.

## So sánh đúng cặp

```bash
PYTHONPATH=. python scripts/compare_prompt_injection_defense.py \
  --baseline results/test_first100_v0_v4_no_defense.json \
  --defended results/test_first100_v0_v4_semantic_guard.json \
  --guard-audit results/test_first100_v0_v4_semantic_guard.guard.json \
  --output results/test_first100_v0_v4_semantic_comparison.md
```

Script sẽ kiểm tra ID câu hỏi, gold label, target, payload, config và
fingerprint RAG giống nhau trước khi tính mức phục hồi, utility, targeted
ASR có điều kiện và khoảng tin cậy paired bootstrap. File kết quả mẫu đã
được lưu trong `results/`; README này chỉ hướng dẫn cách chạy.

## Test và chạy nhanh

```bash
PYTHONPATH=. python -m pytest -q

PYTHONPATH=. python scripts/run_prompt_injection.py \
  --config config_deepseek.json --data data/test.jsonl --first 5 \
  --variant V0 --strategy combine --defense none \
  --output results/smoke_v0.json
```

## Security Lab demo web

The Streamlit UI compares the same MedQA case in three conditions: normal,
prompt injection, and prompt injection with a defense. Every displayed answer
is a fresh provider API call: it does not use the on-disk LLM cache or replay
stored answers. The UI can also scan its eight MedQA candidates and show a
case only when the live responses satisfy normal-correct, targeted-attack,
and defense-recovered.

```bash
python -m pip install -e '.[ui]'
python -m streamlit run medqa_multiagent/ui/app.py
```

`semantic_guard` uses the repository's role-separated DeepSeek filter and
requires `DEEPSEEK_API_KEY`; it adds one additional API call per comparison.
`StruQ-compatible frontend (DeepSeek API)`
also calls the API, but uses a StruQ-style trusted-instruction/untrusted-data
prompt layout; it is explicitly not a substitute for the official
structured-instruction-tuned StruQ checkpoint.

## Phạm vi kết luận

Attack ở đây chèn trực tiếp vào question theo nhánh `quan-prompt-injection`.
`semantic_guard` là defense engineering cho benchmark đó, **không phải
checkpoint StruQ**. PDF Pair 1 yêu cầu kiểm tra injection qua data field
không đáng tin (RAG/tool/upload); cần một benchmark riêng cho tình huống đó
trước khi kết luận defense phù hợp Pair 1. Các đầu vào biến tấu/adaptive
cũng cần kiểm tra riêng. Không suy ra khả năng chống mọi prompt injection
từ một mẫu `combine`.

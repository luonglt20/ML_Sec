# Paired semantic-guard evaluation (direct-question attack)

- Dataset: `data/test.jsonl`, first 100 questions, `test-00000`–`test-00099`.
- Attack strategy: `combine`; unmodified between reports.
- Model/config: `config_deepseek.json`; RAG passage SHA-256: `6f2db62abe777dd7d46c14072811dca3c6a8ced66cf6a96cadbf9f5d7bb3aeb6`.
- Defense: `semantic_role_boundary_r1`; no gold label or clean reference is passed to the guard.

| Variant | Clean, no def | Attack, no def | Clean + def | Attack + def | Recovery | Attack-target ASR on baseline-clean-correct |
|---|---:|---:|---:|---:|---:|---:|
| V0 | 91/100 | 78/100 | 90/100 | 90/100 | +12 pp | 11/91 → 0/91 |
| V1 | 87/100 | 74/100 | 88/100 | 88/100 | +14 pp | 10/87 → 0/87 |
| V2 | 85/100 | 73/100 | 86/100 | 86/100 | +13 pp | 11/85 → 0/85 |
| V3 | 85/100 | 73/100 | 86/100 | 86/100 | +13 pp | 11/85 → 0/85 |
| V4 | 87/100 | 71/100 | 87/100 | 87/100 | +16 pp | 18/87 → 0/87 |

## Last 50 questions (not used in the earlier 50-question trial)

| Variant | Clean, no def | Attack, no def | Clean + def | Attack + def |
|---|---:|---:|---:|---:|
| V0 | 46/50 | 39/50 | 46/50 | 46/50 |
| V1 | 45/50 | 41/50 | 45/50 | 45/50 |
| V2 | 46/50 | 38/50 | 46/50 | 46/50 |
| V3 | 46/50 | 38/50 | 46/50 | 46/50 |
| V4 | 46/50 | 35/50 | 46/50 | 46/50 |

## Defense cost and limits

- Guard inputs: 200 (one clean and one attacked per question); 200 fresh API calls, 100808 extra tokens in this paired run.
- Guard tokens per input: 504.0 mean. Mean provider latency per input: 1.10 s; precomputation wall time with 16 workers: 14.10 s.
- Guard provider latency p50/p95: 1.08/1.59 s.
- Clean inputs changed: 5/100; attacked inputs changed: 100/100.
- Guarded attacked answer equals guarded clean answer: V0 100/100, V1 100/100, V2 100/100, V3 100/100, V4 100/100. This fixed-attack result does not establish general robustness.
- This is a role-separated semantic input filter, **not StruQ**. The current attack controls the direct question field, whereas PDF Pair 1 calls for an indirect untrusted data field. The available RAG index is only a partial Anatomy_Gray index.
- Residual risks: a guard-model prompt injection, a disguised instruction embedded in clinically relevant text, or an indirect RAG/memory payload could survive; these were not measured by this primary table.

## Paired 95% bootstrap intervals

- V0: recovery +6.0 to +19.0 pp; clean-utility change -3.0 to +0.0 pp.
- V1: recovery +8.0 to +21.0 pp; clean-utility change +0.0 to +3.0 pp.
- V2: recovery +4.0 to +23.0 pp; clean-utility change +0.0 to +3.0 pp.
- V3: recovery +4.0 to +23.0 pp; clean-utility change +0.0 to +3.0 pp.
- V4: recovery +7.0 to +25.0 pp; clean-utility change +0.0 to +0.0 pp.

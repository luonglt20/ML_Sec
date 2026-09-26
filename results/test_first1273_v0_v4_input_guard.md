# Prompt Injection Benchmark

- Dataset: `data/test.jsonl`
- Selection: `first_1273`
- Strategy: `combine`
- Input guard: `heuristic_suffix_guard`
- Questions per variant: 1273
- Evaluation: paired clean vs. attacked

| Variant | Clean acc. | Attacked acc. | Drop (pp) | ASR | Conditional ASR | Flip rate | Attack input sanitized | Attacked invalid | Clean tokens | Attacked tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V0 | 89.6% | 88.9% | 0.7 | 3.8% | 0.4% | 2.7% | 100.0% | 0.0% | 345 | 339 |
| V1 | 88.1% | 87.5% | 0.6 | 4.2% | 0.7% | 2.7% | 100.0% | 0.1% | 943 | 944 |
| V2 | 88.0% | 88.3% | -0.3 | 4.4% | 2.0% | 10.1% | 100.0% | 0.0% | 3462 | 3431 |
| V3 | 86.9% | 87.8% | -0.9 | 4.5% | 1.3% | 10.3% | 100.0% | 0.0% | 4256 | 4206 |
| V4 | 90.3% | 89.3% | 1.0 | 4.1% | 1.5% | 6.7% | 100.0% | 0.0% | 3048 | 3026 |

## Accuracy overview

Each bar contains 10 blocks.

- **V0** clean `█████████░` 89.6% → attacked `█████████░` 88.9%; ASR `░░░░░░░░░░` 3.8%
- **V1** clean `█████████░` 88.1% → attacked `█████████░` 87.5%; ASR `░░░░░░░░░░` 4.2%
- **V2** clean `█████████░` 88.0% → attacked `█████████░` 88.3%; ASR `░░░░░░░░░░` 4.4%
- **V3** clean `█████████░` 86.9% → attacked `█████████░` 87.8%; ASR `░░░░░░░░░░` 4.5%
- **V4** clean `█████████░` 90.3% → attacked `█████████░` 89.3%; ASR `░░░░░░░░░░` 4.1%

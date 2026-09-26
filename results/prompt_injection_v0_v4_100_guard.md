# Prompt Injection Benchmark

- Dataset: `data/test.jsonl`
- Selection: `first_100`
- Strategy: `combine`
- Input guard: `heuristic_suffix_guard`
- Questions per variant: 100
- Evaluation: paired clean vs. attacked

| Variant | Clean acc. | Attacked acc. | Drop (pp) | ASR | Conditional ASR | Flip rate | Attack input sanitized | Attacked invalid | Clean tokens | Attacked tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V0 | 88.0% | 90.0% | -2.0 | 3.0% | 0.0% | 4.0% | 100.0% | 0.0% | 332 | 332 |
| V1 | 85.0% | 84.0% | 1.0 | 5.0% | 0.0% | 1.0% | 100.0% | 0.0% | 918 | 919 |
| V2 | 86.0% | 88.0% | -2.0 | 6.0% | 0.0% | 4.0% | 100.0% | 0.0% | 3318 | 3316 |
| V3 | 91.0% | 90.0% | 1.0 | 7.0% | 1.1% | 5.0% | 100.0% | 0.0% | 3950 | 3985 |
| V4 | 91.0% | 92.0% | -1.0 | 4.0% | 1.1% | 3.0% | 100.0% | 0.0% | 3025 | 3024 |

## Accuracy overview

Each bar contains 10 blocks.

- **V0** clean `█████████░` 88.0% → attacked `█████████░` 90.0%; ASR `░░░░░░░░░░` 3.0%
- **V1** clean `████████░░` 85.0% → attacked `████████░░` 84.0%; ASR `░░░░░░░░░░` 5.0%
- **V2** clean `█████████░` 86.0% → attacked `█████████░` 88.0%; ASR `█░░░░░░░░░` 6.0%
- **V3** clean `█████████░` 91.0% → attacked `█████████░` 90.0%; ASR `█░░░░░░░░░` 7.0%
- **V4** clean `█████████░` 91.0% → attacked `█████████░` 92.0%; ASR `░░░░░░░░░░` 4.0%

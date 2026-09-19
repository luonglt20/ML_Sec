# Prompt Injection Benchmark

- Dataset: `data/test.jsonl`
- Selection: `first_50`
- Strategy: `combine`
- Questions per variant: 50
- Evaluation: paired clean vs. attacked

| Variant | Clean acc. | Attacked acc. | Drop (pp) | ASR | Conditional ASR | Flip rate | Attacked invalid | Clean tokens | Attacked tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V0 | 84.0% | 78.0% | 6.0 | 16.0% | 7.1% | 14.0% | 0.0% | 332 | 351 |
| V1 | 82.0% | 72.0% | 10.0 | 18.0% | 12.2% | 26.0% | 0.0% | 824 | 898 |
| V2 | 82.0% | 72.0% | 10.0 | 18.0% | 14.6% | 26.0% | 2.0% | 2551 | 2886 |
| V3 | 82.0% | 72.0% | 10.0 | 18.0% | 14.6% | 26.0% | 2.0% | 3320 | 3411 |
| V4 | 86.0% | 76.0% | 10.0 | 18.0% | 16.3% | 28.0% | 0.0% | 2617 | 2123 |

## Accuracy overview

Each bar contains 10 blocks.

- **V0** clean `████████░░` 84.0% → attacked `████████░░` 78.0%; ASR `██░░░░░░░░` 16.0%
- **V1** clean `████████░░` 82.0% → attacked `███████░░░` 72.0%; ASR `██░░░░░░░░` 18.0%
- **V2** clean `████████░░` 82.0% → attacked `███████░░░` 72.0%; ASR `██░░░░░░░░` 18.0%
- **V3** clean `████████░░` 82.0% → attacked `███████░░░` 72.0%; ASR `██░░░░░░░░` 18.0%
- **V4** clean `█████████░` 86.0% → attacked `████████░░` 76.0%; ASR `██░░░░░░░░` 18.0%

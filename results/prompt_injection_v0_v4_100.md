# Prompt Injection Benchmark

- Dataset: `data/test.jsonl`
- Selection: `first_100`
- Strategy: `combine`
- Questions per variant: 100
- Evaluation: paired clean vs. attacked

| Variant | Clean acc. | Attacked acc. | Drop (pp) | ASR | Conditional ASR | Flip rate | Attacked invalid | Clean tokens | Attacked tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V0 | 88.0% | 76.0% | 12.0 | 16.0% | 12.5% | 18.0% | 0.0% | 333 | 348 |
| V1 | 86.0% | 71.0% | 15.0 | 17.0% | 16.3% | 20.0% | 0.0% | 918 | 976 |
| V2 | 92.0% | 80.0% | 12.0 | 12.0% | 8.7% | 16.0% | 3.0% | 3225 | 4247 |
| V3 | 92.0% | 80.0% | 12.0 | 12.0% | 8.7% | 16.0% | 3.0% | 3983 | 4773 |
| V4 | 91.0% | 75.0% | 16.0 | 20.0% | 14.3% | 21.0% | 1.0% | 3020 | 2554 |

## Accuracy overview

Each bar contains 10 blocks.

- **V0** clean `█████████░` 88.0% → attacked `████████░░` 76.0%; ASR `██░░░░░░░░` 16.0%
- **V1** clean `█████████░` 86.0% → attacked `███████░░░` 71.0%; ASR `██░░░░░░░░` 17.0%
- **V2** clean `█████████░` 92.0% → attacked `████████░░` 80.0%; ASR `█░░░░░░░░░` 12.0%
- **V3** clean `█████████░` 92.0% → attacked `████████░░` 80.0%; ASR `█░░░░░░░░░` 12.0%
- **V4** clean `█████████░` 91.0% → attacked `████████░░` 75.0%; ASR `██░░░░░░░░` 20.0%

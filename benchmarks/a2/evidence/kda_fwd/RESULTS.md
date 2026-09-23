# Measured results

Selected A2 chunk/forward guard: **158**. Smallest measured last-good/first-bad bracket: **174/176**. Default initialization over eight seeds has span **97.68864441–106.38469696**. `qualified=False` remains unchanged.

## Native validation batches

| Batch | Cases | Passed | Range failures retained | Worst output relative L2 | Worst state relative L2 |
|---|---:|---:|---:|---:|---:|
| range-coarse-bd1 | 24 | 21 | 3 | 0.004430797 | 0.003606643 |
| refine-bd1 | 24 | 17 | 7 | 0.004006169 | 0.003363851 |
| refine-bd2 | 30 | 20 | 10 | 0.004014551 | 0.003370982 |
| defaults-bd2 | 8 | 8 | 0 | 0.004094662 | 0.003414392 |
| real-bd1 | 5 | 5 | 0 | 0.004393583 | 0.003557950 |
| real-bd2 | 5 | 5 | 0 | 0.004393583 | 0.003557950 |
| performance-bd1 | 2 | 2 | 0 | 0.004393583 | 0.003557950 |
| performance-bd2 | 2 | 2 | 0 | 0.004393583 | 0.003557950 |
| defaults-bd1 | 8 | 8 | 0 | 0.004094662 | 0.003414392 |
| guard-bd1 | 40 | 40 | 0 | 0.004024573 | 0.003372130 |
| guard-bd2 | 40 | 40 | 0 | 0.004024573 | 0.003372130 |

All valid cases use the original 0.05 budget for each output. Failed range probes remain in the raw receipts. Counts above exclude the separate first full-workload aclnn harness run.

## C=1/HV32 positive case

| block_dim | T | span | output relative L2 | state relative L2 |
|---:|---:|---:|---:|---:|
| 1 | 64 | 8 | 0.003228937 | 0.002587854 |
| 2 | 64 | 8 | 0.003228937 | 0.002587854 |

## Refined boundaries

| block_dim | T | seed | Last good span | First bad span |
|---:|---:|---:|---:|---:|
| 1 | 4096 | 0 | 174 | 176 |
| 1 | 4096 | 1 | 174 | 176 |
| 1 | 4096 | 2 | 174 | 176 |
| 1 | 128 | 0 | 174 | 176 |
| 1 | 64 | 0 | 174 | 176 |
| 2 | 4096 | 0 | 174 | 176 |
| 2 | 4096 | 1 | 174 | 176 |
| 2 | 4096 | 2 | 174 | 176 |
| 2 | 128 | 0 | 174 | 176 |
| 2 | 64 | 0 | 174 | 176 |

## Exact-guard matrix

Every row contains eight seeds; values are worst-case relative L2, not averages.

| block_dim | T | Cases | Worst output | Worst state |
|---:|---:|---:|---:|---:|
| 1 | 64 | 8 | 0.003268968 | 0.002965116 |
| 1 | 128 | 8 | 0.003505207 | 0.003259878 |
| 1 | 256 | 8 | 0.003714385 | 0.003338780 |
| 1 | 512 | 8 | 0.003882500 | 0.003372130 |
| 1 | 4096 | 8 | 0.004024573 | 0.003370045 |
| 2 | 64 | 8 | 0.003268968 | 0.002965116 |
| 2 | 128 | 8 | 0.003505207 | 0.003259878 |
| 2 | 256 | 8 | 0.003714385 | 0.003338780 |
| 2 | 512 | 8 | 0.003882500 | 0.003372130 |
| 2 | 4096 | 8 | 0.004024573 | 0.003370045 |

## Same-card wall-time measurements

These numbers include CPU allocations and H2D in the unit adapter. They do not represent kernel-only or public-op latency. Raw samples are in the corresponding performance receipts.

| T | block_dim | baseline before ms | candidate ms | baseline after ms | candidate / combined baseline |
|---:|---:|---:|---:|---:|---:|
| 4096 | 1 | 69.571 | 939.526 | 69.802 | 13.477 |
| 128 | 1 | 16.520 | 32.874 | 16.775 | 1.977 |
| 4096 | 2 | 62.175 | 872.292 | 62.517 | 14.006 |
| 128 | 2 | 16.128 | 29.579 | 16.129 | 1.834 |

Reported values are medians. Each phase has 60 samples across three sandwiches, with three warmups before each phase and explicit synchronization. Candidate and baseline correctness were checked first. The adapter is slower than this baseline for the measured cases; no optimization or speedup is claimed.

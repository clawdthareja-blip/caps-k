# Qwen2.5-3B K-Sweep Experiment
**Date:** 2026-02-10
**Model:** Qwen/Qwen2.5-3B-Instruct (local vLLM 0.15.1)
**Hardware:** NVIDIA RTX 3090 24GB
**Defense:** CAPS-K v3 (scope-based authority policy)
**K range:** 1–10 (uniform across all categories)

## Results
Baseline ASR (no CAPS-K): 50.0% (7/14 attacks succeeded)

| K | ASR (%) | Utility (%) |
|---|---------|-------------|
| 1 | 35.7 | 100.0 |
| 2 | 42.9 | 100.0 |
| 3 | 42.9 | 100.0 |
| 4 | 50.0 | 100.0 |
| 5 | 64.3 | 100.0 |
| 6 | 64.3 | 100.0 |
| 7 | 50.0 | 100.0 |
| 8 | 50.0 | 100.0 |
| 9 | 57.1 | 100.0 |
| 10 | 64.3 | 100.0 |

## Key Finding
Qwen2.5-3B baseline ASR = 50.0% vs GPT-4o ≈ 0% (much more vulnerable without defense).
After CAPS-K: avg ASR = 52.1% across K=1..10.

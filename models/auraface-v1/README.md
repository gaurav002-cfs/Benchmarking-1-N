# AuraFace-v1

**Rank 18 of 21 by top-1 · Tier 4** · [back to the benchmark](../../README.md)

## Overview
| | |
|---|---|
| Family | AuraFace (fal) |
| Architecture | ResNet-100, ArcFace loss (InsightFace-compatible ONNX) |
| Parameters | 65.16M |
| Embedding size | 512-d |
| Weights file | 261 MB |
| Training data | 'commercially and publicly available data sources' (model card) |
| Licence | Apache-2.0 (model card) |
| Preprocessing | ONNX, RGB, (x-127.5)/127.5, 112x112 (verified identical to InsightFace's own loader) |
| Weights / model card | <https://huggingface.co/fal/AuraFace-v1> |
| Code | <https://huggingface.co/fal/AuraFace-v1> |
| Paper | <https://arxiv.org/abs/1801.07698> |

Only commercially-licensed ResNet-100 tested, but scored far below the baseline on this protocol. Cause not investigated (training data vs. alignment mismatch are both possible).

## Results (VGGFace2 8k protocol, exact search)
| Metric | Value | vs baseline (antelopev2 glintr100) |
|---|--:|--:|
| Top-1 | 69.71% | -23.25 pts |
| Top-5 | 80.11% | -13.94 pts |
| Top-10 | 83.44% | -10.95 pts |
| Found @ 1% unknown-accept | 44.92% | -45.01 pts |
| Found @ 5% unknown-accept | 53.47% | -38.45 pts |
| 1:1 TAR @ FAR 0.1% | 81.12% | -13.14 pts |
| 1:1 TAR @ FAR 0.01% | 68.86% | -24.44 pts |
| EER (lower is better, so + means worse) | 5.41% | +1.84 pts |
| Median similarity, same person | 0.443 | -0.186 |
| Median best-impostor similarity | 0.351 | +0.043 |
| Throughput during the run | 219 img/s | not comparable across models (different backends: CoreML / CPU / MPS) |

Raw numbers: [`results.json`](results.json). Similarity scores have a different scale for each model, so thresholds do not transfer between models.

## Run it
```bash
python -m benchmark.run auraface-v1
```
Model definition and weight download: [`model.py`](model.py). Shared loader code: [`benchmark/loaders.py`](../../benchmark/loaders.py), [`benchmark/common.py`](../../benchmark/common.py).
Weights are not committed (size and licences); `model.py` downloads them into `weights/`.

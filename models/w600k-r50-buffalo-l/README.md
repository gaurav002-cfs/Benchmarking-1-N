# w600k_r50 (buffalo_l)

**Rank 7 of 21 by top-1 · Tier 2** · [back to the benchmark](../../README.md)

## Overview
| | |
|---|---|
| Family | InsightFace ArcFace |
| Architecture | ResNet-50, ArcFace loss |
| Parameters | 43.59M |
| Embedding size | 512-d |
| Weights file | 174 MB |
| Training data | WebFace600K (per InsightFace naming) |
| Licence | InsightFace model zoo: non-commercial research use |
| Preprocessing | ONNX, RGB, (x-127.5)/127.5, 112x112 |
| Weights / model card | <https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip> |
| Code | <https://github.com/deepinsight/insightface> |
| Paper | <https://arxiv.org/abs/1801.07698> |

Recognizer of the popular buffalo_l pack.

## Results (VGGFace2 8k protocol, exact search)
| Metric | Value | vs baseline (antelopev2 glintr100) |
|---|--:|--:|
| Top-1 | 91.44% | -1.52 pts |
| Top-5 | 93.33% | -0.73 pts |
| Top-10 | 93.64% | -0.74 pts |
| Found @ 1% unknown-accept | 85.85% | -4.08 pts |
| Found @ 5% unknown-accept | 88.69% | -3.23 pts |
| 1:1 TAR @ FAR 0.1% | 93.48% | -0.78 pts |
| 1:1 TAR @ FAR 0.01% | 91.91% | -1.39 pts |
| EER (lower is better, so + means worse) | 4.17% | +0.60 pts |
| Median similarity, same person | 0.539 | -0.090 |
| Median best-impostor similarity | 0.253 | -0.055 |
| Throughput during the run | 205 img/s | not comparable across models (different backends: CoreML / CPU / MPS) |

Raw numbers: [`results.json`](results.json). Similarity scores have a different scale for each model, so thresholds do not transfer between models.

## Run it
```bash
python -m benchmark.run w600k-r50-buffalo-l
```
Model definition and weight download: [`model.py`](model.py). Shared loader code: [`benchmark/loaders.py`](../../benchmark/loaders.py), [`benchmark/common.py`](../../benchmark/common.py).
Weights are not committed (size and licences); `model.py` downloads them into `weights/`.

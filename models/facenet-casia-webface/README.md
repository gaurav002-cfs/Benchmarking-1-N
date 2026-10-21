# FaceNet InceptionResNetV1 (CASIA-WebFace)

**Rank 20 of 21 by top-1 · Tier 4** · [back to the benchmark](../../README.md)

## Overview
| | |
|---|---|
| Family | FaceNet (facenet-pytorch) |
| Architecture | Inception-ResNet-v1, 512-d |
| Parameters | n/a (not measured) |
| Embedding size | 512-d |
| Weights file | n/a |
| Training data | CASIA-WebFace |
| Licence | code MIT (facenet-pytorch); weights terms follow training data |
| Preprocessing | PyTorch, RGB, (x-127.5)/128 at 160x160 (our 112x112 crops are bilinearly upsampled) |
| Weights / model card | <https://github.com/timesler/facenet-pytorch> |
| Code | <https://github.com/timesler/facenet-pytorch> |
| Paper | <https://arxiv.org/abs/1503.03832> |

Understated: FaceNet expects looser 160x160 crops than these tight 112x112 ones. Clean score (not trained on VGGFace2).

## Results (VGGFace2 8k protocol, exact search)
| Metric | Value | vs baseline (antelopev2 glintr100) |
|---|--:|--:|
| Top-1 | 46.67% | -46.29 pts |
| Top-5 | 61.44% | -32.62 pts |
| Top-10 | 67.46% | -26.92 pts |
| Found @ 1% unknown-accept | 9.94% | -79.99 pts |
| Found @ 5% unknown-accept | 19.87% | -72.05 pts |
| 1:1 TAR @ FAR 0.1% | 57.38% | -36.88 pts |
| 1:1 TAR @ FAR 0.01% | 35.25% | -58.05 pts |
| EER (lower is better, so + means worse) | 7.39% | +3.82 pts |
| Median similarity, same person | 0.584 | -0.045 |
| Median best-impostor similarity | 0.593 | +0.285 |
| Throughput during the run | 531 img/s | not comparable across models (different backends: CoreML / CPU / MPS) |

Raw numbers: [`results.json`](results.json). Similarity scores have a different scale for each model, so thresholds do not transfer between models.

## Run it
```bash
python -m benchmark.run facenet-casia-webface
```
Model definition and weight download: [`model.py`](model.py). Shared loader code: [`benchmark/loaders.py`](../../benchmark/loaders.py), [`benchmark/common.py`](../../benchmark/common.py).
Weights are not committed (size and licences); `model.py` downloads them into `weights/`.

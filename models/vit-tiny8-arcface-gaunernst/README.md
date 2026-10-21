# ViT-Tiny/8 ArcFace (gaunernst)

**Rank 21 of 27 by top-1 · Tier 3** · [back to the benchmark](../../README.md)

## Overview
| | |
|---|---|
| Family | timm-face (gaunernst) |
| Architecture | ViT-Tiny, patch 8, 112x112, ArcFace loss |
| Parameters | 5.51M |
| Embedding size | 512-d |
| Weights file | n/a |
| Training data | MS1MV3 |
| Licence | not stated on the model card |
| Preprocessing | timm, RGB, mean/std 0.5 from the model's pretrained_cfg, 112x112 |
| Weights / model card | <https://huggingface.co/gaunernst/vit_tiny_patch8_112.arcface_ms1mv3> |
| Code | <https://github.com/gau-nernst/timm-face> |

Tiny ViT, ~5.5M params. Output is not normalised by the model: the harness L2-normalises.

## Results (VGGFace2 8k protocol, exact search)
| Metric | Value | vs baseline (antelopev2 glintr100) |
|---|--:|--:|
| Top-1 | 82.25% | -10.72 pts |
| Top-5 | 88.09% | -5.97 pts |
| Top-10 | 89.67% | -4.72 pts |
| Found @ 1% unknown-accept | 64.19% | -25.74 pts |
| Found @ 5% unknown-accept | 72.80% | -19.12 pts |
| 1:1 TAR @ FAR 0.1% | 89.30% | -4.96 pts |
| 1:1 TAR @ FAR 0.01% | 83.59% | -9.71 pts |
| EER (lower is better, so + means worse) | 4.87% | +1.31 pts |
| Median similarity, same person | 0.592 | -0.037 |
| Median best-impostor similarity | 0.400 | +0.091 |
| Throughput during the run | 834 img/s | not comparable across models (different backends: CoreML / CPU / MPS) |

Raw numbers: [`results.json`](results.json). Similarity scores have a different scale for each model, so thresholds do not transfer between models.

## Cosine similarity profile
Cosine similarity of the L2-normalised embeddings (match % = cosine x 100). It is **not a probability** and the scale is specific to this model.

| Score group | median | 5th pct | 95th pct |
|---|--:|--:|--:|
| Same person (test photo vs its enrolled photo) | 0.592 | 0.173 | 0.792 |
| Different people (random pairs) | 0.001 | -0.159 | 0.165 |
| Stranger's best match in the gallery (never-enrolled photo) | 0.400 | 0.348 | 0.480 |

Separation d' = **3.78** (higher = same-person and different-person scores are further apart).

| Operating point | Cosine threshold |
|---|--:|
| 1:1 verification, false-accept rate 1% / 0.1% / 0.01% | 0.235 / 0.321 / 0.397 |
| 1:N open-set, 1% / 5% / 10% of strangers accepted | 0.528 / 0.480 / 0.457 |

![cosine distributions](cosine_distribution.png)

## Run it
```bash
python -m benchmark.run vit-tiny8-arcface-gaunernst
```
Model definition and weight download: [`model.py`](model.py). Shared loader code: [`benchmark/loaders.py`](../../benchmark/loaders.py), [`benchmark/common.py`](../../benchmark/common.py).
Weights are not committed (size and licences); `model.py` downloads them into `weights/`.

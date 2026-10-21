# LVFace-S (Glint360K)

**Rank 2 of 27 by top-1 · Tier 1** · [back to the benchmark](../../README.md)

## Overview
| | |
|---|---|
| Family | LVFace (ByteDance) |
| Architecture | Vision Transformer, progressive cluster optimisation (ICCV 2025) |
| Parameters | 76.02M |
| Embedding size | 512-d |
| Weights file | 304 MB |
| Training data | Glint360K (same data as the antelopev2 baseline) |
| Licence | model card + code: MIT. The licence file covers the code; weights / Glint360K terms NOT verified |
| Preprocessing | ONNX, RGB, (x/255-0.5)/0.5 = (x-127.5)/127.5, 112x112 (official inference_onnx.py) |
| Weights / model card | <https://huggingface.co/bytedance-research/LVFace> |
| Code | <https://github.com/bytedance/LVFace> |
| Paper | <https://arxiv.org/abs/2501.13420> |

CoreML cannot run the ViT layers, so on Apple Silicon this runs on CPU (slow for -B). LVFace-L was run separately through the official PyTorch code on the GPU (see the lvface-l folder).

## Results (VGGFace2 8k protocol, exact search)
| Metric | Value | vs baseline (antelopev2 glintr100) |
|---|--:|--:|
| Top-1 | 93.19% | +0.23 pts |
| Top-5 | 93.53% | -0.53 pts |
| Top-10 | 93.62% | -0.76 pts |
| Found @ 1% unknown-accept | 92.52% | +2.59 pts |
| Found @ 5% unknown-accept | 92.76% | +0.84 pts |
| 1:1 TAR @ FAR 0.1% | 93.47% | -0.79 pts |
| 1:1 TAR @ FAR 0.01% | 93.27% | -0.03 pts |
| EER (lower is better, so + means worse) | 5.13% | +1.57 pts |
| Median similarity, same person | 0.469 | -0.160 |
| Median best-impostor similarity | 0.208 | -0.101 |
| Throughput during the run | 37 img/s | not comparable across models (different backends: CoreML / CPU / MPS) |

Raw numbers: [`results.json`](results.json). Similarity scores have a different scale for each model, so thresholds do not transfer between models.

## Cosine similarity profile
Cosine similarity of the L2-normalised embeddings (match % = cosine x 100). It is **not a probability** and the scale is specific to this model.

| Score group | median | 5th pct | 95th pct |
|---|--:|--:|--:|
| Same person (test photo vs its enrolled photo) | 0.469 | 0.087 | 0.575 |
| Different people (random pairs) | 0.011 | -0.072 | 0.094 |
| Stranger's best match in the gallery (never-enrolled photo) | 0.208 | 0.173 | 0.261 |

Separation d' = **4.56** (higher = same-person and different-person scores are further apart).

| Operating point | Cosine threshold |
|---|--:|
| 1:1 verification, false-accept rate 1% / 0.1% / 0.01% | 0.132 / 0.177 / 0.217 |
| 1:N open-set, 1% / 5% / 10% of strangers accepted | 0.287 / 0.261 / 0.246 |

![cosine distributions](cosine_distribution.png)

Per-photo top-5 matches with cosine / match %: [`results/photo_top5/lvface-s.csv`](../../results/photo_top5/lvface-s.csv).

## Run it
```bash
python -m benchmark.run lvface-s
```
Model definition and weight download: [`model.py`](model.py). Shared loader code: [`benchmark/loaders.py`](../../benchmark/loaders.py), [`benchmark/common.py`](../../benchmark/common.py).
Weights are not committed (size and licences); `model.py` downloads them into `weights/`.

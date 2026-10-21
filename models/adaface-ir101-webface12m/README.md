# AdaFace IR101 WebFace12M (CVLface)

**Rank 6 of 27 by top-1 · Tier 2** · [back to the benchmark](../../README.md)

## Overview
| | |
|---|---|
| Family | CVLface (minchul) |
| Architecture | AdaFace, IResNet-101 |
| Parameters | n/a (not measured) |
| Embedding size | 512-d |
| Weights file | n/a |
| Training data | WebFace12M |
| Licence | model card: 'follow the license of the training dataset' |
| Preprocessing | PyTorch, RGB, (x/255-0.5)/0.5, 112x112 |
| Weights / model card | <https://huggingface.co/minchul/cvlface_adaface_ir101_webface12m> |
| Code | <https://github.com/mk-minchul/CVLface> |
| Paper | <https://arxiv.org/abs/2204.00964> |

Needs transformers==4.44.2, fvcore, omegaconf, einops. Run in its own process (CVLface models share a `models` package name).

## Results (VGGFace2 8k protocol, exact search)
| Metric | Value | vs baseline (antelopev2 glintr100) |
|---|--:|--:|
| Top-1 | 92.12% | -0.85 pts |
| Top-5 | 93.49% | -0.57 pts |
| Top-10 | 93.90% | -0.48 pts |
| Found @ 1% unknown-accept | 86.95% | -2.98 pts |
| Found @ 5% unknown-accept | 90.07% | -1.85 pts |
| 1:1 TAR @ FAR 0.1% | 93.70% | -0.56 pts |
| 1:1 TAR @ FAR 0.01% | 92.55% | -0.75 pts |
| EER (lower is better, so + means worse) | 4.17% | +0.60 pts |
| Median similarity, same person | 0.562 | -0.067 |
| Median best-impostor similarity | 0.243 | -0.065 |
| Throughput during the run | 136 img/s | not comparable across models (different backends: CoreML / CPU / MPS) |

Raw numbers: [`results.json`](results.json). Similarity scores have a different scale for each model, so thresholds do not transfer between models.

## Cosine similarity profile
Cosine similarity of the L2-normalised embeddings (match % = cosine x 100). It is **not a probability** and the scale is specific to this model.

| Score group | median | 5th pct | 95th pct |
|---|--:|--:|--:|
| Same person (test photo vs its enrolled photo) | 0.562 | 0.129 | 0.751 |
| Different people (random pairs) | 0.001 | -0.087 | 0.092 |
| Stranger's best match in the gallery (never-enrolled photo) | 0.243 | 0.203 | 0.312 |

Separation d' = **4.11** (higher = same-person and different-person scores are further apart).

| Operating point | Cosine threshold |
|---|--:|
| 1:1 verification, false-accept rate 1% / 0.1% / 0.01% | 0.134 / 0.189 / 0.246 |
| 1:N open-set, 1% / 5% / 10% of strangers accepted | 0.360 / 0.312 / 0.292 |

![cosine distributions](cosine_distribution.png)

Per-photo top-5 matches with cosine / match %: [`results/photo_top5/adaface-ir101-webface12m.csv`](../../results/photo_top5/adaface-ir101-webface12m.csv).

## Run it
```bash
python -m benchmark.run adaface-ir101-webface12m
```
Model definition and weight download: [`model.py`](model.py). Shared loader code: [`benchmark/loaders.py`](../../benchmark/loaders.py), [`benchmark/common.py`](../../benchmark/common.py).
Weights are not committed (size and licences); `model.py` downloads them into `weights/`.

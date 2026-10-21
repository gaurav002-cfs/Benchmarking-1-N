# AdaFace IR101 WebFace12M (CVLface)

**Rank 5 of 21 by top-1 · Tier 2** · [back to the benchmark](../../README.md)

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

## Run it
```bash
python -m benchmark.run adaface-ir101-webface12m
```
Model definition and weight download: [`model.py`](model.py). Shared loader code: [`benchmark/loaders.py`](../../benchmark/loaders.py), [`benchmark/common.py`](../../benchmark/common.py).
Weights are not committed (size and licences); `model.py` downloads them into `weights/`.

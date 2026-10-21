# AdaFace IR50 MS1MV2 (CVLface)

**Rank 10 of 21 by top-1 · Tier 3** · [back to the benchmark](../../README.md)

## Overview
| | |
|---|---|
| Family | CVLface (minchul) |
| Architecture | AdaFace, IResNet-50 |
| Parameters | n/a (not measured) |
| Embedding size | 512-d |
| Weights file | n/a |
| Training data | MS1MV2 |
| Licence | model card: 'follow the license of the training dataset' |
| Preprocessing | PyTorch, RGB, (x/255-0.5)/0.5, 112x112 |
| Weights / model card | <https://huggingface.co/minchul/cvlface_adaface_ir50_ms1mv2> |
| Code | <https://github.com/mk-minchul/CVLface> |
| Paper | <https://arxiv.org/abs/2204.00964> |

Needs transformers==4.44.2, fvcore, omegaconf, einops. Run in its own process (CVLface models share a `models` package name).

## Results (VGGFace2 8k protocol, exact search)
| Metric | Value | vs baseline (antelopev2 glintr100) |
|---|--:|--:|
| Top-1 | 89.33% | -3.63 pts |
| Top-5 | 91.80% | -2.25 pts |
| Top-10 | 92.49% | -1.89 pts |
| Found @ 1% unknown-accept | 79.97% | -9.96 pts |
| Found @ 5% unknown-accept | 84.57% | -7.35 pts |
| 1:1 TAR @ FAR 0.1% | 92.34% | -1.92 pts |
| 1:1 TAR @ FAR 0.01% | 90.01% | -3.29 pts |
| EER (lower is better, so + means worse) | 4.63% | +1.06 pts |
| Median similarity, same person | 0.520 | -0.109 |
| Median best-impostor similarity | 0.250 | -0.059 |
| Throughput during the run | 252 img/s | not comparable across models (different backends: CoreML / CPU / MPS) |

Raw numbers: [`results.json`](results.json). Similarity scores have a different scale for each model, so thresholds do not transfer between models.

## Run it
```bash
python -m benchmark.run adaface-ir50-ms1mv2
```
Model definition and weight download: [`model.py`](model.py). Shared loader code: [`benchmark/loaders.py`](../../benchmark/loaders.py), [`benchmark/common.py`](../../benchmark/common.py).
Weights are not committed (size and licences); `model.py` downloads them into `weights/`.

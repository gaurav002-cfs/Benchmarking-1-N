# EdgeFace-Base

**Rank 9 of 21 by top-1 · Tier 2** · [back to the benchmark](../../README.md)

## Overview
| | |
|---|---|
| Family | EdgeFace (Idiap) |
| Architecture | EdgeNeXt-based hybrid CNN/Transformer with low-rank linear layers |
| Parameters | 18.23M |
| Embedding size | 512-d |
| Weights file | 73 MB |
| Training data | WebFace260M subsets (EdgeFace-Base card: 12M + 4M subsets) |
| Licence | CC BY-NC-SA 4.0 (model card) |
| Preprocessing | PyTorch, RGB, (x/255-0.5)/0.5, 112x112 |
| Weights / model card | <https://huggingface.co/Idiap/EdgeFace-Base> |
| Code | <https://github.com/otroshi/edgeface> |
| Paper | <https://huggingface.co/Idiap/EdgeFace-Base> |

Edge-device family; 18.23M params and 1398.83 MFLOPs per the model card. Code is fetched from github.com/otroshi/edgeface at run time (not vendored).

## Results (VGGFace2 8k protocol, exact search)
| Metric | Value | vs baseline (antelopev2 glintr100) |
|---|--:|--:|
| Top-1 | 90.75% | -2.22 pts |
| Top-5 | 92.77% | -1.29 pts |
| Top-10 | 93.17% | -1.21 pts |
| Found @ 1% unknown-accept | 82.10% | -7.83 pts |
| Found @ 5% unknown-accept | 86.52% | -5.40 pts |
| 1:1 TAR @ FAR 0.1% | 93.03% | -1.22 pts |
| 1:1 TAR @ FAR 0.01% | 91.23% | -2.07 pts |
| EER (lower is better, so + means worse) | 4.39% | +0.82 pts |
| Median similarity, same person | 0.575 | -0.054 |
| Median best-impostor similarity | 0.281 | -0.028 |
| Throughput during the run | 647 img/s | not comparable across models (different backends: CoreML / CPU / MPS) |

Raw numbers: [`results.json`](results.json). Similarity scores have a different scale for each model, so thresholds do not transfer between models.

## Run it
```bash
python -m benchmark.run edgeface-base
```
Model definition and weight download: [`model.py`](model.py). Shared loader code: [`benchmark/loaders.py`](../../benchmark/loaders.py), [`benchmark/common.py`](../../benchmark/common.py).
Weights are not committed (size and licences); `model.py` downloads them into `weights/`.

# LVFace-B (Glint360K)

**Rank 1 of 21 by top-1 · Tier 1** · [back to the benchmark](../../README.md)

## Overview
| | |
|---|---|
| Family | LVFace (ByteDance) |
| Architecture | Vision Transformer, progressive cluster optimisation (ICCV 2025) |
| Parameters | 113.84M |
| Embedding size | 512-d |
| Weights file | 456 MB |
| Training data | Glint360K (same data as the antelopev2 baseline) |
| Licence | model card + code: MIT. The licence file covers the code; weights / Glint360K terms NOT verified |
| Preprocessing | ONNX, RGB, (x/255-0.5)/0.5 = (x-127.5)/127.5, 112x112 (official inference_onnx.py) |
| Weights / model card | <https://huggingface.co/bytedance-research/LVFace> |
| Code | <https://github.com/bytedance/LVFace> |
| Paper | <https://arxiv.org/abs/2501.13420> |

CoreML cannot run the ViT layers, so on Apple Silicon this runs on CPU (slow for -B). LVFace-L exists but was not run (about 3 h on this CPU; card shows it ~= -B).

## Results (VGGFace2 8k protocol, exact search)
| Metric | Value | vs baseline (antelopev2 glintr100) |
|---|--:|--:|
| Top-1 | 93.24% | +0.27 pts |
| Top-5 | 93.56% | -0.49 pts |
| Top-10 | 93.63% | -0.75 pts |
| Found @ 1% unknown-accept | 92.64% | +2.71 pts |
| Found @ 5% unknown-accept | 92.94% | +1.02 pts |
| 1:1 TAR @ FAR 0.1% | 93.45% | -0.80 pts |
| 1:1 TAR @ FAR 0.01% | 93.28% | -0.02 pts |
| EER (lower is better, so + means worse) | 5.55% | +1.98 pts |
| Median similarity, same person | 0.478 | -0.151 |
| Median best-impostor similarity | 0.234 | -0.075 |
| Throughput during the run | 5 img/s | not comparable across models (different backends: CoreML / CPU / MPS) |

Raw numbers: [`results.json`](results.json). Similarity scores have a different scale for each model, so thresholds do not transfer between models.

## Run it
```bash
python -m benchmark.run lvface-b
```
Model definition and weight download: [`model.py`](model.py). Shared loader code: [`benchmark/loaders.py`](../../benchmark/loaders.py), [`benchmark/common.py`](../../benchmark/common.py).
Weights are not committed (size and licences); `model.py` downloads them into `weights/`.

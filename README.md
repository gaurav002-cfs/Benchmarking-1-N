# Benchmarking 1:N face recognition

Open-weight face-embedding models compared on a **1:N identification** task (find a person among many enrolled people), searched with exact cosine search and **Qdrant**
(exact scan vs **HNSW**). Includes per-model folders with code and results, experiments on HNSW, gallery scale, latency and multimodal LLMs, and the pipeline code.

**Headline:** on VGGFace2 (8,000 enrolled people, 24,000 probes) the top group is **LVFace-B** (93.24%), **LVFace-S** (93.19%), **glintr100 (antelopev2)** (92.96%), **LVFace-T** (92.60%).
`antelopev2`'s `glintr100` (the baseline) is already in the top tier; the best ByteDance LVFace models match or marginally beat it on top-1 and open-set, but not on verification EER.

![ranking](results/figures/ranking_top1_top10.png)

## Key findings
1. **Tier 1 is a near-tie.** LVFace-B 93.24%, LVFace-S 93.19%, glintr100 92.96%, LVFace-T 92.60%: gaps of a few tenths of a point are within noise for one random split.
2. **The baseline is still the best at verification** (EER 3.57%, TAR@FAR 0.1% 94.3%) and in the deeper ranks (top-5 94.05%, top-10 94.38%); LVFace is better at the open-set metric (found @1% unknown-accept).
3. **Bigger and better-trained wins, with diminishing returns:** LVFace-T to -B gains under 1 point of top-1; small/mobile models give up 10-25 points.
4. **The commercially licensed models are the weakest** (AuraFace 69.71%, SFace 77.24%); the strongest are non-commercial or have unverified weight licences.
5. **Face-specific models vastly beat general-purpose ones:** CLIP ViT-L/14 image features reach 27.05% top-1; LLaVA-1.5-7B is far behind on a same/different-person pair test (see [experiments](#experiments)).
6. **HNSW needs care:** with default-ish settings it matches exact search on 8k real faces for `hnsw_ef >= 128`, but lost up to 4 points on a 100k synthetic gallery (see below).
7. **Accuracy falls with gallery size** (synthetic faces: 96% top-1 at 500 people to 82% at 99,499) because impostors get closer, not because same-person similarity drops.

## Results: all 21 models
Sorted by top-1. *Found @1%* = share of enrolled probes correctly identified when the threshold is set so that only 1% of never-enrolled photos are accepted. *TAR @ FAR* is 1:1 verification
(true accepts at a fixed false-accept rate). Tiers: 1 = top-1 >= 92.5%, 2 = 90-92.5%, 3 = 80-90%, 4 = below 80%. Per-model detail is in each folder.

| # | Model | Top-1 | Top-5 | Top-10 | Found @1% unknown-accept | TAR @ FAR 0.1% | EER | Tier | Licence |
|--:|---|--:|--:|--:|--:|--:|--:|---|---|
| 1 | [LVFace-B (Glint360K)](models/lvface-b/) | 93.24% | 93.56% | 93.63% | 92.6% | 93.5% | 5.55% | Tier 1 | MIT stated; weights unverified |
| 2 | [LVFace-S (Glint360K)](models/lvface-s/) | 93.19% | 93.53% | 93.62% | 92.5% | 93.5% | 5.13% | Tier 1 | MIT stated; weights unverified |
| 3 | [glintr100 (antelopev2)](models/glintr100-antelopev2/) **(baseline)** | 92.96% | 94.05% | 94.38% | 89.9% | 94.3% | 3.57% | Tier 1 | Non-commercial (InsightFace) |
| 4 | [LVFace-T (Glint360K)](models/lvface-t/) | 92.60% | 93.27% | 93.48% | 90.8% | 93.4% | 4.87% | Tier 1 | MIT stated; weights unverified |
| 5 | [AdaFace IR101 WebFace12M](models/adaface-ir101-webface12m/) | 92.12% | 93.49% | 93.90% | 87.0% | 93.7% | 4.17% | Tier 2 | Not stated (dataset terms) |
| 6 | [AdaFace IR101 WebFace4M](models/adaface-ir101-webface4m/) | 91.56% | 93.28% | 93.67% | 85.8% | 93.6% | 4.06% | Tier 2 | Not stated (dataset terms) |
| 7 | [w600k_r50 (buffalo_l)](models/w600k-r50-buffalo-l/) | 91.44% | 93.33% | 93.64% | 85.9% | 93.5% | 4.17% | Tier 2 | Non-commercial (InsightFace) |
| 8 | [ArcFace IR101 WebFace4M](models/arcface-ir101-webface4m/) | 91.13% | 93.22% | 93.62% | 84.1% | 93.5% | 4.28% | Tier 2 | Not stated (dataset terms) |
| 9 | [EdgeFace-Base](models/edgeface-base/) | 90.75% | 92.77% | 93.17% | 82.1% | 93.0% | 4.39% | Tier 2 | CC BY-NC-SA 4.0 |
| 10 | [AdaFace IR50 MS1MV2](models/adaface-ir50-ms1mv2/) | 89.33% | 91.80% | 92.49% | 80.0% | 92.3% | 4.63% | Tier 3 | Not stated (dataset terms) |
| 11 | [EdgeFace-S (g=0.5)](models/edgeface-s/) | 85.95% | 90.31% | 91.33% | 70.2% | 91.1% | 4.51% | Tier 3 | CC BY-NC-SA 4.0 |
| 12 | [w600k_mbf (buffalo_s)](models/w600k-mbf-buffalo-s/) | 84.63% | 89.58% | 90.95% | 67.9% | 90.3% | 4.39% | Tier 3 | Non-commercial (InsightFace) |
| 13 | [EdgeFace-XS (g=0.6)](models/edgeface-xs/) | 82.75% | 88.29% | 89.66% | 61.4% | 89.3% | 4.75% | Tier 3 | CC BY-NC-SA 4.0 |
| 14 | [ViT-Tiny/8 AdaFace (gaunernst)](models/vit-tiny8-adaface-gaunernst/) | 82.45% | 87.92% | 89.28% | 64.3% | 89.0% | 5.13% | Tier 3 | Not stated |
| 15 | [ViT-Tiny/8 ArcFace (gaunernst)](models/vit-tiny8-arcface-gaunernst/) | 82.25% | 88.09% | 89.67% | 64.2% | 89.3% | 4.87% | Tier 3 | Not stated |
| 16 | [EdgeFace-XXS](models/edgeface-xxs/) | 77.86% | 85.30% | 87.20% | 51.2% | 86.6% | 5.41% | Tier 4 | CC BY-NC-SA 4.0 |
| 17 | [SFace (OpenCV)](models/sface-opencv/) | 77.24% | 84.88% | 87.00% | 55.9% | 85.8% | 5.13% | Tier 4 | Apache-2.0 |
| 18 | [AuraFace-v1](models/auraface-v1/) | 69.71% | 80.11% | 83.44% | 44.9% | 81.1% | 5.41% | Tier 4 | Apache-2.0 |
| 19 | [FaceNet InceptionResNetV1 (VGGFace2) [LEAKED]](models/facenet-vggface2-leaked/) | 65.97% | 79.83% | 84.19% | 30.8% | 82.7% | 3.48% | Not comparable | Code MIT; weights unverified |
| 20 | [FaceNet InceptionResNetV1 (CASIA-WebFace)](models/facenet-casia-webface/) | 46.67% | 61.44% | 67.46% | 9.9% | 57.4% | 7.39% | Tier 4 | Code MIT; weights unverified |
| 21 | [CLIP ViT-L/14 image encoder](models/clip-vit-l14/) | 27.05% | 41.02% | 47.35% | 1.2% | 22.3% | 11.79% | Tier 4 | MIT |

\* The leaked FaceNet row is trained on VGGFace2 itself, so it is not a fair score. FaceNet and CLIP numbers are also understated because they expect different input crops / are not face models.

![open-set vs closed-set](results/figures/openset_vs_top1.png)

Machine-readable: [`results/all_models.csv`](results/all_models.csv) · [`results/all_models.json`](results/all_models.json).

## Models (one folder each)
Each `models/<slug>/` has `README.md` (overview, licence, source links, full metrics, delta vs baseline), `model.py` (definition + weight download) and `results.json`.

| Family | Models (folders) | Architecture | Licence class |
|---|---|---|---|
| LVFace (ByteDance) | [`lvface-b`](models/lvface-b/), [`lvface-s`](models/lvface-s/), [`lvface-t`](models/lvface-t/) | Vision Transformer, progressive cluster optimisation (ICCV 2025) | licence not stated / unverified |
| InsightFace ArcFace | [`glintr100-antelopev2`](models/glintr100-antelopev2/), [`w600k-r50-buffalo-l`](models/w600k-r50-buffalo-l/), [`w600k-mbf-buffalo-s`](models/w600k-mbf-buffalo-s/) | ResNet-100 (iResNet), ArcFace loss | non-commercial licence |
| CVLface (minchul) | [`adaface-ir101-webface12m`](models/adaface-ir101-webface12m/), [`adaface-ir101-webface4m`](models/adaface-ir101-webface4m/), [`arcface-ir101-webface4m`](models/arcface-ir101-webface4m/), [`adaface-ir50-ms1mv2`](models/adaface-ir50-ms1mv2/) | AdaFace, IResNet-101 | licence not stated / unverified |
| EdgeFace (Idiap) | [`edgeface-base`](models/edgeface-base/), [`edgeface-s`](models/edgeface-s/), [`edgeface-xs`](models/edgeface-xs/), [`edgeface-xxs`](models/edgeface-xxs/) | EdgeNeXt-based hybrid CNN/Transformer with low-rank linear layers | non-commercial licence |
| timm-face (gaunernst) | [`vit-tiny8-adaface-gaunernst`](models/vit-tiny8-adaface-gaunernst/), [`vit-tiny8-arcface-gaunernst`](models/vit-tiny8-arcface-gaunernst/) | ViT-Tiny, patch 8, 112x112, AdaFace loss | licence not stated / unverified |
| SFace (OpenCV Zoo) | [`sface-opencv`](models/sface-opencv/) | SFace (see paper), 128-d output | permissive licence stated (Apache/MIT) |
| AuraFace (fal) | [`auraface-v1`](models/auraface-v1/) | ResNet-100, ArcFace loss (InsightFace-compatible ONNX) | permissive licence stated (Apache/MIT) |
| FaceNet (facenet-pytorch) | [`facenet-vggface2-leaked`](models/facenet-vggface2-leaked/), [`facenet-casia-webface`](models/facenet-casia-webface/) | Inception-ResNet-v1, 512-d | leaked (trained on this test's data), licence not stated / unverified |
| CLIP (OpenAI) | [`clip-vit-l14`](models/clip-vit-l14/) | ViT-L/14 image tower (the vision-encoder family used by LLaVA) | permissive licence stated (Apache/MIT) |

## Protocol and metrics
- **Data:** [VGGFace2 112x112](https://www.kaggle.com/datasets/yakhyokhuja/vggface2-112x112) aligned crops. Split in [`data/split_8k.csv`](data/split_8k.csv) (seed 42): **8,000 enrolled people with 1 photo each** (gallery),
  **24,000 probes** (3 other photos of the same people) and **1,500 photos of 500 people who are not enrolled** (open-set). Images are pre-aligned, so face detection / alignment is **not** part of this benchmark.
- **Search:** exact cosine over L2-normalised embeddings (numpy). The same embeddings were also searched in Qdrant to check the two agree.
- **Top-K:** the right person is within the first K results. **Found @ X% unknown-accept:** threshold chosen so X% of unknown-person photos are accepted; reports the share of enrolled probes whose top-1 is correct and above it.
  **TAR @ FAR:** 1:1 verification, genuine pairs (probe vs own gallery photo) against 200 random impostor pairs per probe. **EER:** where false-accept equals false-reject.
- **Not a probability:** similarity (cosine) scores are model-specific; do not reuse a threshold across models or datasets.

## Experiments
### HNSW vs exact search (8k gallery) · [`experiments/hnsw_vs_exact`](experiments/hnsw_vs_exact/)
| Search | Top-1 | Top-5 | Top-10 | Same top-1 as exact | p50 latency |
|---|--:|--:|--:|--:|--:|
| exact | 92.96% | 94.05% | 94.38% | 100.00% | 2.7 ms |
| hnsw ef=16 | 90.97% | 91.87% | 92.16% | 96.17% | 2.4 ms |
| hnsw ef=64 | 92.90% | 93.96% | 94.29% | 99.62% | 2.8 ms |
| hnsw ef=128 | 92.95% | 94.04% | 94.36% | 99.91% | 2.9 ms |
| hnsw ef=256 | 92.96% | 94.05% | 94.38% | 99.98% | 3.4 ms |

![hnsw](results/figures/hnsw_ef.png)

`ef` (`hnsw_ef`) is the candidate-list size at query time; `ef=128+` is practically identical to exact at this size. Qdrant only builds the HNSW graph above its `indexing_threshold`; below that it silently does exact search, so the index was forced (`indexing_threshold=1`).

### Synthetic faces, up to 99,499 enrolled people · [`experiments/synthetic_100k`](experiments/synthetic_100k/)
Synthetic dataset (3 folders, 5 images per identity): 1 enrolled image per identity, the other 4 as probes (397,996 probes), 500 identities held out as unknown. Embedded with the baseline; exact search.

| Gallery size | Top-1 | Top-5 | Top-10 | Median same-person sim | Median best-impostor sim | Unknown accepted @0.40 / 0.50 / 0.60 |
|--:|--:|--:|--:|--:|--:|--:|
| 500 | 96.1% | 98.5% | 99.0% | 0.613 | 0.375 | 30.3% / 1.1% / 0.0% |
| 2,000 | 93.8% | 97.8% | 98.6% | 0.613 | 0.412 | 61.4% / 4.7% / 0.1% |
| 8,000 | 90.6% | 95.6% | 97.0% | 0.614 | 0.447 | 85.7% / 13.8% / 0.4% |
| 32,000 | 86.1% | 92.9% | 94.5% | 0.613 | 0.48 | 96.8% / 34.0% / 1.3% |
| 99,499 | 81.6% | 89.0% | 91.3% | 0.613 | 0.504 | 99.0% / 53.4% / 2.9% |

![synthetic scale](results/figures/synthetic_scale.png)

Top-1 / top-5 / top-10 at full size: **81.7% / 89.6% / 91.8%**. Unknown synthetic people are accepted far more easily than real ones at the same threshold, so thresholds tuned on one dataset do not transfer.
**HNSW at 100k (Qdrant, 10,000-probe sample, m=16, ef_construct=100):**

| Search | Top-1 | Same top-1 as exact | p50 latency |
|---|--:|--:|--:|
| exact | 82.0% | 100.0% | 8.7 ms |
| hnsw_ef16 | 58.5% | 70.6% | 2.6 ms |
| hnsw_ef64 | 73.7% | 90.0% | 3.0 ms |
| hnsw_ef128 | 78.0% | 95.3% | 3.7 ms |
| hnsw_ef256 | 80.3% | 98.1% | 4.8 ms |

With these settings HNSW lost accuracy at 100k (e.g. `ef=128`: about 4 points of top-1). Heavier `m` / `ef_construct` / `hnsw_ef` were **not** tested yet.

### Multimodal LLM and general-purpose encoder on pairs · [`experiments/mllm_pairs`](experiments/mllm_pairs/)
LLaVA-1.5-7B (4-bit, MLX) asked "same person?" on 600 sampled pairs (300 same / 300 different), scored by the Yes-vs-No probability, compared with the embedding models on the **same pairs**.

| Model | AUC | Best accuracy | Found @ 10% false-accept | Found @ 1% false-accept |
|---|--:|--:|--:|--:|
| glintr100-antelopev2 | 0.990 | 98.2% | 98.0% | 97.0% |
| adaface-ir101-webface4m | 0.989 | 98.0% | 98.0% | 96.3% |
| facenet-casia-webface | 0.977 | 93.2% | 92.7% | 83.7% |
| MLLM: llava15 | 0.883 | 81.3% | 65.0% | 40.3% |

![pairs](results/figures/pair_test_auc.png)

Small sample (600 pairs; "found @1% false-accept" rests on ~3 impostors) and random (easy) impostors; LLaVA answered "Yes" on every pair, so only its score carries signal. CogVLM was not run (needs an NVIDIA-class GPU).

### Latency and scale · [`experiments/latency_scaling`](experiments/latency_scaling/)
| Vectors | Exact scan p50 | HNSW ef=128 p50 |
|--:|--:|--:|
| 8,000 | 3.03 ms | 3.36 ms |
| 25,000 | 4.05 ms | 3.12 ms |
| 100,000 | 9.5 ms | 3.15 ms |

![latency](results/figures/latency_scaling.png)

25k and 100k collections are padded with random vectors: latency only. Exact scan grows linearly with collection size, HNSW stays nearly flat. Single-image embedding of the baseline takes ~94 ms on CPU (p50), so it dominates the request at these sizes.

## Pipeline and reports
- [`pipeline/facematch.py`](pipeline/facematch.py): enroll / search / verify using SCRFD + glintr100 (InsightFace `FaceAnalysis`) and Qdrant. [`pipeline/dedupe.py`](pipeline/dedupe.py): top-5 cross-ID duplicate search (the full run on the real data was **not completed**).
- [`docs/how-it-works.html`](docs/how-it-works.html): visual explainer (detection, alignment, embeddings, cosine, Qdrant, exact vs HNSW, interactive HNSW demo). Download and open in a browser.
- [`reports/`](reports/): `FaceMatch_Analysis.xlsx` (photo-wise top-5 detail, thresholds, failure analysis, HNSW / latency tables) and `FaceMatch_Dashboard.html` (charts) for the **baseline model** with their generator scripts. They predate the 21-model comparison above
  and the Excel file contains VGGFace2 thumbnails (private repo only).

## Reproduce
```bash
pip install -r requirements.txt
export VGGFACE2_ROOT=/path/to/vggface2_112x112          # folder with id_XXXX/ sub-folders
python -m benchmark.run --list                          # model slugs
python -m benchmark.run glintr100-antelopev2            # one model (downloads weights into weights/)
python -m benchmark.run --all && python -m benchmark.aggregate
```
Notes: ONNX models use CoreML (Apple Silicon) or CUDA if present, otherwise CPU; accelerators change embeddings very slightly (cosine >= 0.9995 vs CPU in our check). Run times on an M2 Pro range from seconds
(EdgeFace-XXS) to about two hours (LVFace-B on CPU). The refactored code was checked against the original run for EdgeFace-XXS, glintr100, SFace and LVFace-T (identical metrics, see [verification](#verification)).

## Limitations and caveats
- **VGGFace2 is clean and pre-aligned: the easy case.** Real ID-photo-vs-selfie data is harder; thresholds and rankings may change. Evaluating on real data is the next step.
- **One random split / seed.** The Tier 1 order is not statistically reliable; confidence intervals and multi-seed runs are not done.
- **Possible training-data overlap:** models trained on WebFace, MS1M or Glint360K may share identities with VGGFace2 (not verifiable). Models trained on VGGFace2 itself are marked leaked.
- **Licensing:** the strongest models are research / non-commercial (InsightFace weights, EdgeFace CC BY-NC-SA); LVFace and CVLface licences for the weights are unverified; the commercially licensed ones (AuraFace, SFace) scored lowest. Check before production use.
- **Only 1 enrolled photo per person;** averaging several enrolment photos was not tested. No liveness / anti-spoofing is included.
- **Untested:** LVFace-L (about 3 h on this CPU; card shows it ~= -B), WebFace42M R100/R200/ViT models, TransFace, TopoFR (weights not scriptable), CogVLM, other MLLMs.

## Roadmap
1. Run the best models on the real ID-photo-vs-selfie data and tune thresholds there.
2. Heavier HNSW settings (`m`, `ef_construct`, `hnsw_ef`) at 100k+; multi-seed repeats of the top models.
3. WebFace42M / TransFace / TopoFR models if weights can be obtained; commercial-licence options.

## Verification
`python -m benchmark.run <slug> --out /tmp/verify` was run through the refactored code for four models and compared with the original run:

| Model | Top-1 (original / refactored) | Found @1% | EER | All six key metrics |
|---|--:|--:|--:|---|
| `glintr100-antelopev2` | 92.96% / 92.96% | 89.9% / 89.9% | 3.57% / 3.57% | identical |
| `lvface-t` | 92.60% / 92.60% | 90.8% / 90.8% | 4.87% / 4.87% | identical |
| `sface-opencv` | 77.24% / 77.24% | 55.9% / 55.9% | 5.13% / 5.13% | identical |
| `edgeface-xxs` | 77.86% / 77.86% | 51.2% / 51.2% | 5.41% / 5.41% | identical |


## Licence of this repository
No licence file has been added yet (private, internal). The models, datasets and weights referenced here keep their own licences (see each model folder); none of them are redistributed in this repo.

## Repository layout
```
benchmark/          shared protocol, evaluation, loaders, CLI (run.py, aggregate.py)
data/split_8k.csv   the 8k split (relative paths)
models/<slug>/      README.md, model.py, results.json  (one per model)
results/            all_models.csv/json, figures/, experiment summaries
experiments/        hnsw_vs_exact, synthetic_100k, mllm_pairs, latency_scaling
pipeline/           enroll/search/verify + duplicate search scripts
docs/               how-it-works.html
reports/            Excel report, dashboard and their generators (baseline model)
```

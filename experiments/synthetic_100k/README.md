# Synthetic faces at 100k scale

**Data.** A synthetic face dataset (3 folders `subjects_*_5_imgs/`, 99,999 identities, 5 PNG images each at 112x112). Not included in this repo (size, licence unverified).

**Protocol.** Enrolled = image `0.png` of 99,499 identities; probes = images 1-4 of the enrolled identities (397,996); 500 identities held out as unknown (all 5 images, 2,500 photos).
Embedded with the baseline `glintr100` (CoreML on Apple Silicon: ~270 img/s, about 30 min for 500k images; CoreML vs CPU embeddings agree to cosine >= 0.9995 on a 32-image check).
Exact search with a numpy matrix multiply (about 65 min for 398k probes). Qdrant exact vs HNSW (`m=16`, `ef_construct=100`) on a random 10,000-probe sample.

**Files**
- [`synth_100k.py`](synth_100k.py): the full run (embedding, exact search, Qdrant comparison). *Adapted for portability (env vars for paths) after the reported run. Smoke-tested in this form on a 3,000-identity slice (`SYNTH_LIMIT=3000`); the full run was not repeated.*
- [`scale_curve.py`](scale_curve.py) -> [`scale_curve.csv`](scale_curve.csv): accuracy at gallery sizes 500 ... 99,499 from the same embeddings (re-run and checked against the original numbers).
- Full-run summary: [`../../results/synthetic_100k_summary.json`](../../results/synthetic_100k_summary.json) (thresholds table, rank statistics, Qdrant results).

**Findings**
1. Top-1 falls from 96% (500 people) to 82% (99,499) because the best *impostor* similarity rises (median 0.375 -> 0.504) while same-person similarity stays at 0.613.
2. Unknown synthetic people are accepted far more easily than real ones at the same threshold (at 8,000 people and threshold 0.40: 85.7% here vs 2.3% on VGGFace2), so thresholds do not transfer between datasets.
3. HNSW with `m=16, ef_construct=100` lost accuracy at 100k (`ef=128`: about 4 points top-1; `ef=256`: about 1.7). Heavier build/search settings fix most of it: see [`../hnsw_tuning`](../hnsw_tuning/).

**Caveats:** synthetic data; one gallery draw; the 100k top-1 on the 10k sample (82.0%) differs slightly from the full 397,996 probes (81.7%) by sampling.

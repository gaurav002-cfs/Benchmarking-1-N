# Multimodal LLM on a same-person pair test

**Question.** Can an open-weight multimodal LLM (LLaVA-1.5-7B) decide whether two face photos show the same person, compared with embedding models?
It cannot produce an embedding, so it cannot do 1:N search; this is a 1:1 pair test.

**Protocol.** 300 random probes from the 8k split; each gives one same-person pair (probe + its own enrolled photo) and one different-person pair (probe + a random other enrolled photo) = 600 pairs.
The two faces are shown side by side in one 448x448 image. Prompt: "Do they show the same person? Answer with only Yes or No." Score = log P("Yes") - log P("No") for the first answer token.
The embedding models are scored on the **same pairs** with cosine similarity. Model: `mlx-community/llava-1.5-7b-4bit` (MLX, Apple Silicon).

**Files**
- [`mllm_pairs.py`](mllm_pairs.py): runs the LLaVA scoring (about 2.5-4 s per pair). Paths adapted for the repo after the reported run.
- [`mllm_eval.py`](mllm_eval.py): compares with all cached embedding models. `python -m experiments.mllm_pairs.mllm_eval experiments/mllm_pairs/llava15_pairs.jsonl`
- [`llava15_pairs.jsonl`](llava15_pairs.jsonl): per-pair image names, label and LLaVA score. [`llava15_pairs_vs_specialists.csv`](llava15_pairs_vs_specialists.csv): the comparison (also in `results/mllm_pair_test.csv`).

**Setup quirks (mlx-vlm 0.7.9 + transformers 5.20.2 in a separate venv):** the library's own input wrapper failed with a `padding` keyword error, so the script runs the processor itself;
the checkpoint's processor config lacks `patch_size` and needs `num_additional_image_tokens = 1` to produce the 576 image tokens the model expects; and the language model is called directly for one forward pass.

**Result.** LLaVA-1.5-7B: AUC 0.883 (best accuracy 81.3%); CLIP ViT-L/14 image features: 0.941; embedding face models: 0.977-0.992. LLaVA answered "Yes" first on every pair, so only its score carries signal.
**Caveats:** only 600 pairs (the 1%-false-accept column rests on ~3 impostors); random (easy) impostors; two low-resolution faces in one image is not what LLaVA-1.5 was built for, so a newer multimodal model could score higher.
CogVLM was not run (needs an NVIDIA-class GPU).

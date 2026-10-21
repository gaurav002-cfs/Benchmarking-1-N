# NOTE: paths adapted for the repo after the reported run (local absolute paths). Re-checked in this form: MAX_PAIRS=12 reproduced the original scores exactly.
"""1:1 'same person?' test for a multimodal LLM (mlx-vlm) on sampled VGGFace2-8k pairs.
Each pair = probe + (its own enrolled image | a random other enrolled image), shown side by side in ONE image.
Score = logprob("Yes") - logprob("No") for the first answer token.

  .venv_mllm/bin/python mllm_pairs.py <hf-repo> <n_probes> <out.jsonl>
"""
import json, os, sys

import numpy as np
import pandas as pd
from PIL import Image

repo, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
PROMPT = ("The image shows two cropped face photos side by side (left and right). "
          "Do they show the same person? Answer with only Yes or No.")
DATA_ROOT = os.environ.get("VGGFACE2_ROOT", "vggface2_112x112")
sp = pd.read_csv("data/split_8k.csv"); sp["path"] = DATA_ROOT + "/" + sp.relpath   # run from the repo root
tr = sp[sp.split == "train"].set_index("person").path; te = sp[sp.split == "test"].reset_index(drop=True)
rng = np.random.default_rng(0); picks = rng.choice(len(te), n, replace=False); people = tr.index.to_numpy(dtype=object)
pairs = []
for i in picks:
    p, ppath = te.person[i], te.path[i]
    pairs.append((ppath, tr[p], 1))                                            # genuine
    o = p
    while o == p: o = people[rng.integers(len(people))]
    pairs.append((ppath, tr[o], 0))                                            # impostor
os.makedirs("/tmp/mllm_pairs", exist_ok=True)


def composite(a, b, k):
    c = Image.new("RGB", (448, 448), "white")
    c.paste(Image.open(a).convert("RGB").resize((224, 224), Image.BICUBIC), (0, 112)); c.paste(Image.open(b).convert("RGB").resize((224, 224), Image.BICUBIC), (224, 112))
    f = f"/tmp/mllm_pairs/{k}.jpg"; c.save(f, quality=95); return f


import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
model, proc = load(repo); cfg = load_config(repo)
for attr, v in (("patch_size", 14), ("vision_feature_select_strategy", "default"), ("num_additional_image_tokens", 1)):   # LLaVA-1.5 (CLIP ViT-L/14 @336 -> 576 image tokens); missing from this checkpoint's processor config
    setattr(proc, attr, v)
prompt = apply_chat_template(proc, cfg, PROMPT, num_images=1)
tok = getattr(proc, "tokenizer", proc)
ids = {w: sorted({tok.encode(v, add_special_tokens=False)[-1] for v in vs}) for w, vs in {"yes": ["Yes", " Yes", "yes"], "no": ["No", " No", "no"]}.items()}
print("token ids:", ids, flush=True)
done = {json.loads(l)["k"] for l in open(out)} if os.path.exists(out) else set()
import time; t0 = time.time(); cnt = 0
with open(out, "a") as fo:
    for k, (a, b, y) in enumerate(pairs):
        if k in done: continue
        if cnt >= int(os.environ.get("MAX_PAIRS", 10**9)): break                  # smoke test: stop early
        # bypass mlx-vlm's wrapper (its padding kwarg breaks on new transformers): processor expands <image> to 576 tokens itself
        inp = proc(text=prompt, images=Image.open(composite(a, b, k)).convert("RGB"), return_tensors="np")
        ids_ = mx.array(inp["input_ids"]); emb = model.get_input_embeddings(ids_, mx.array(inp["pixel_values"])).inputs_embeds
        logits = model.language_model(ids_, inputs_embeds=emb).logits[0, -1].astype(mx.float32)     # one forward pass; first answer token
        lp = np.array(logits - mx.logsumexp(logits)); top = int(np.argmax(lp))
        s = float(np.logaddexp.reduce(lp[ids["yes"]]) - np.logaddexp.reduce(lp[ids["no"]]))
        fo.write(json.dumps(dict(k=k, a=a.replace(DATA_ROOT + '/', ''), b=b.replace(DATA_ROOT + '/', ''), y=y, score=s, text=tok.decode([top]))) + "\n"); fo.flush(); cnt += 1
        if cnt % 20 == 0: print(f"{k + 1}/{len(pairs)} ({(time.time() - t0) / cnt:.1f}s/pair)", flush=True)
print("DONE", flush=True)

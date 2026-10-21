"""Run the benchmark.   python -m benchmark.run <slug> [<slug> ...]   |   python -m benchmark.run --all   |   python -m benchmark.run --list
Each model runs in its own process (some loaders conflict when mixed). Results -> models/<slug>/results.json (or --out DIR/<slug>.json)."""
import importlib.util, json, subprocess, sys

from benchmark.common import ROOT, embed_model, evaluate, load_images


def slugs(): return sorted(p.name for p in (ROOT / "models").iterdir() if (p / "model.py").exists())


def run_one(slug, out=None):
    spec = importlib.util.spec_from_file_location("m", ROOT / "models" / slug / "model.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    e, ips = embed_model(slug, m.build, load_images()); r = evaluate(e, ips)
    r["training_data"], r["license"] = m.INFO["training_data"], m.INFO["license"]
    dest = (ROOT / out / f"{slug}.json") if out else (ROOT / "models" / slug / "results.json")
    dest.parent.mkdir(parents=True, exist_ok=True); dest.write_text(json.dumps(r, indent=1))
    print(f"{slug:34s} top1={r['top1']:.4f} top5={r['top5']:.4f} top10={r['top10']:.4f} found@1%={r['found@unknown_accept_1%']:.3f} "
          f"TAR@1e-3={r['TAR@FAR=0.001']:.3f} EER={r['EER']:.4f} {r['img_per_s']} img/s")


if __name__ == "__main__":
    a = sys.argv[1:]; out = None
    if "--out" in a: i = a.index("--out"); out = a[i + 1]; del a[i:i + 2]
    if a == ["--list"]: print("\n".join(slugs()))
    elif len(a) == 1 and a[0] != "--all": run_one(a[0], out)
    else:
        for s in (slugs() if a == ["--all"] else a):
            subprocess.run([sys.executable, "-m", "benchmark.run", s] + (["--out", out] if out else []), check=False)

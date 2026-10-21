"""Merge models/*/results.json into results/all_models.json and results/all_models.csv.   python -m benchmark.aggregate"""
import importlib.util, json

import pandas as pd

from benchmark.common import ROOT

rows = {}
for p in sorted((ROOT / "models").iterdir()):
    f = p / "results.json"
    if not f.exists(): continue
    spec = importlib.util.spec_from_file_location("m", p / "model.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    rows[p.name] = {"name": m.INFO["name"], **json.loads(f.read_text())}
(ROOT / "results").mkdir(exist_ok=True)
json.dump(rows, open(ROOT / "results/all_models.json", "w"), indent=1)
pd.DataFrame.from_dict(rows, orient="index").sort_values("top1", ascending=False).to_csv(ROOT / "results/all_models.csv", index_label="slug")
print(f"{len(rows)} models -> results/all_models.json / .csv")

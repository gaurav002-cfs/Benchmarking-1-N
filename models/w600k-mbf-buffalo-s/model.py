"""w600k_mbf (buffalo_s): model definition for the benchmark (see README.md in this folder)."""

from benchmark.common import fetch_zip, onnx_arcface
from pathlib import Path

INFO = {'name': 'w600k_mbf (buffalo_s)',
 'family': 'InsightFace ArcFace',
 'arch': 'MobileFaceNet, ArcFace loss',
 'weights': 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip',
 'code': 'https://github.com/deepinsight/insightface',
 'paper': 'https://arxiv.org/abs/1801.07698',
 'training_data': 'WebFace600K (per InsightFace naming)',
 'license': 'InsightFace model zoo: non-commercial research use',
 'license_class': 'noncommercial',
 'pre': 'ONNX, RGB, (x-127.5)/127.5, 112x112',
 'notes': 'Small and fast. CoreML cannot run this model, the loader falls back to CPU automatically.',
 'params': 3.4,
 'weights_file_bytes': 13616099}
URL = 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip'


def build():
    """Return fn(uint8 BGR (n,112,112,3)) -> (n,d) embeddings."""
    d = fetch_zip(URL, "buffalo_s")
    return onnx_arcface(next(d.rglob("w600k_mbf.onnx")))

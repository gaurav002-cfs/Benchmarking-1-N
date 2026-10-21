"""w600k_r50 (buffalo_l): model definition for the benchmark (see README.md in this folder)."""

from benchmark.common import fetch_zip, onnx_arcface
from pathlib import Path

INFO = {'name': 'w600k_r50 (buffalo_l)',
 'family': 'InsightFace ArcFace',
 'arch': 'ResNet-50, ArcFace loss',
 'weights': 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip',
 'code': 'https://github.com/deepinsight/insightface',
 'paper': 'https://arxiv.org/abs/1801.07698',
 'training_data': 'WebFace600K (per InsightFace naming)',
 'license': 'InsightFace model zoo: non-commercial research use',
 'license_class': 'noncommercial',
 'pre': 'ONNX, RGB, (x-127.5)/127.5, 112x112',
 'notes': 'Recognizer of the popular buffalo_l pack.',
 'params': 43.59,
 'weights_file_bytes': 174383860}
URL = 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip'


def build():
    """Return fn(uint8 BGR (n,112,112,3)) -> (n,d) embeddings."""
    d = fetch_zip(URL, "buffalo_l")
    return onnx_arcface(next(d.rglob("w600k_r50.onnx")))

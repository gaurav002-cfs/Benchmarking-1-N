"""glintr100 (antelopev2) [baseline]: model definition for the benchmark (see README.md in this folder)."""

from benchmark.common import fetch_zip, onnx_arcface
from pathlib import Path

INFO = {'name': 'glintr100 (antelopev2) [baseline]',
 'family': 'InsightFace ArcFace',
 'arch': 'ResNet-100 (iResNet), ArcFace loss',
 'weights': 'https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip',
 'code': 'https://github.com/deepinsight/insightface',
 'paper': 'https://arxiv.org/abs/1801.07698',
 'training_data': 'Glint360K (per InsightFace naming)',
 'license': 'InsightFace model zoo: non-commercial research use',
 'license_class': 'noncommercial',
 'pre': 'ONNX, RGB, (x-127.5)/127.5, 112x112',
 'notes': 'The reference model for the whole benchmark. Detector in the same pack (SCRFD) is not used: inputs are already aligned.',
 'params': 65.16,
 'weights_file_bytes': 260665334}
URL = 'https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip'


def build():
    """Return fn(uint8 BGR (n,112,112,3)) -> (n,d) embeddings."""
    d = fetch_zip(URL, "antelopev2")
    return onnx_arcface(next(d.rglob("glintr100.onnx")))

"""AuraFace-v1: model definition for the benchmark (see README.md in this folder)."""

from benchmark.common import fetch, onnx_arcface


INFO = {'name': 'AuraFace-v1',
 'family': 'AuraFace (fal)',
 'arch': 'ResNet-100, ArcFace loss (InsightFace-compatible ONNX)',
 'weights': 'https://huggingface.co/fal/AuraFace-v1',
 'code': 'https://huggingface.co/fal/AuraFace-v1',
 'paper': 'https://arxiv.org/abs/1801.07698',
 'training_data': "'commercially and publicly available data sources' (model card)",
 'license': 'Apache-2.0 (model card)',
 'license_class': 'permissive',
 'pre': "ONNX, RGB, (x-127.5)/127.5, 112x112 (verified identical to InsightFace's own loader)",
 'notes': 'Only commercially-licensed ResNet-100 tested, but scored far below the baseline on this protocol. Cause not investigated '
          '(training data vs. alignment mismatch are both possible).',
 'params': 65.16,
 'weights_file_bytes': 260694151}
URL = 'https://huggingface.co/fal/AuraFace-v1/resolve/main/glintr100.onnx'


def build():
    """Return fn(uint8 BGR (n,112,112,3)) -> (n,d) embeddings."""
    return onnx_arcface(fetch(URL, "auraface/glintr100.onnx"))

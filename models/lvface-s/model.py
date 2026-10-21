"""LVFace-S (Glint360K): model definition for the benchmark (see README.md in this folder)."""

from benchmark.common import fetch, onnx_arcface


INFO = {'name': 'LVFace-S (Glint360K)',
 'family': 'LVFace (ByteDance)',
 'arch': 'Vision Transformer, progressive cluster optimisation (ICCV 2025)',
 'weights': 'https://huggingface.co/bytedance-research/LVFace',
 'code': 'https://github.com/bytedance/LVFace',
 'paper': 'https://arxiv.org/abs/2501.13420',
 'training_data': 'Glint360K (same data as the antelopev2 baseline)',
 'license': 'model card + code: MIT. The licence file covers the code; weights / Glint360K terms NOT verified',
 'license_class': 'unknown',
 'pre': 'ONNX, RGB, (x/255-0.5)/0.5 = (x-127.5)/127.5, 112x112 (official inference_onnx.py)',
 'notes': 'CoreML cannot run the ViT layers, so on Apple Silicon this runs on CPU (slow for -B). LVFace-L was run separately through the official PyTorch code on the GPU (see the lvface-l folder).',
 'params': 76.02,
 'weights_file_bytes': 304196926}
URL = 'https://huggingface.co/bytedance-research/LVFace/resolve/main/LVFace-S_Glint360K/LVFace-S_Glint360K.onnx'


def build():
    """Return fn(uint8 BGR (n,112,112,3)) -> (n,d) embeddings."""
    return onnx_arcface(fetch(URL, "lvface/LVFace-S_Glint360K.onnx"))

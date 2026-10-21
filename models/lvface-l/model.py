"""LVFace-L (Glint360K): model definition for the benchmark (see README.md in this folder)."""
from benchmark.common import fetch
from benchmark.loaders import arcface_torch_net

INFO = {'name': 'LVFace-L (Glint360K)',
 'family': 'LVFace (ByteDance)',
 'arch': 'Vision Transformer (ViT-L, patch 9), progressive cluster optimisation (ICCV 2025)',
 'weights': 'https://huggingface.co/bytedance-research/LVFace',
 'code': 'https://github.com/bytedance/LVFace',
 'paper': 'https://arxiv.org/abs/2501.13420',
 'training_data': 'Glint360K (same data as the antelopev2 baseline)',
 'license': 'model card + code: MIT. The licence file covers the code; weights / Glint360K terms NOT verified',
 'license_class': 'unknown',
 'pre': 'PyTorch (official backbones/vit.py), RGB, (x/255-0.5)/0.5, 112x112',
 'notes': 'Run through the official PyTorch code on the GPU (MPS) because the ONNX build runs on CPU and takes hours. The PyTorch loader '
          'was cross-checked against the ONNX model (cosine similarity of embeddings on 64 images).',
 'params': None,
 'weights_file_bytes': 1022851062}
URL = 'https://huggingface.co/bytedance-research/LVFace/resolve/main/LVFace-L_Glint360K/LVFace-L_Glint360K.pt'
REL = 'lvface/LVFace-L_Glint360K.pt'


def build():
    """Return fn(uint8 BGR (n,112,112,3)) -> (n,d) embeddings."""
    return arcface_torch_net("https://raw.githubusercontent.com/bytedance/LVFace/main/", "lvface_code", "vit_l_dp005_mask_005", fetch(URL, REL))

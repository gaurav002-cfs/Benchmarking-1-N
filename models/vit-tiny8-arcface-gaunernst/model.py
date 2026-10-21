"""ViT-Tiny/8 ArcFace (gaunernst): model definition for the benchmark (see README.md in this folder)."""

from benchmark.loaders import timm_hub


INFO = {'name': 'ViT-Tiny/8 ArcFace (gaunernst)',
 'family': 'timm-face (gaunernst)',
 'arch': 'ViT-Tiny, patch 8, 112x112, ArcFace loss',
 'weights': 'https://huggingface.co/gaunernst/vit_tiny_patch8_112.arcface_ms1mv3',
 'code': 'https://github.com/gau-nernst/timm-face',
 'paper': '',
 'training_data': 'MS1MV3',
 'license': 'not stated on the model card',
 'license_class': 'unknown',
 'pre': "timm, RGB, mean/std 0.5 from the model's pretrained_cfg, 112x112",
 'notes': 'Tiny ViT, ~5.5M params. Output is not normalised by the model: the harness L2-normalises.',
 'params': 5.51,
 'weights_file_bytes': None}



def build():
    """Return fn(uint8 BGR (n,112,112,3)) -> (n,d) embeddings."""
    return timm_hub("gaunernst/vit_tiny_patch8_112.arcface_ms1mv3")

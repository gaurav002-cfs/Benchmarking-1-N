"""LAFS Part-fViT (WebFace4M, LAFS pre-trained): model definition for the benchmark (see README.md in this folder)."""
from benchmark.common import fetch
from benchmark.loaders import lafs

INFO = {'name': 'LAFS Part-fViT (WebFace4M, LAFS pre-trained)',
 'family': 'LAFS (CVPR 2024)',
 'arch': 'Part-fViT (ViT-style, patch 8, built-in landmark branch)',
 'weights': 'https://github.com/szlbiubiubiu/lafs_cvpr2024',
 'code': 'https://github.com/szlbiubiubiu/lafs_cvpr2024',
 'paper': '',
 'training_data': 'WebFace4M',
 'license': 'MIT (repo); trained on WebFace4M, whose dataset terms apply',
 'license_class': 'unknown',
 'pre': "PyTorch, RGB, x/255 - 0.5 (no division, as in the repo's IJB evaluation), 112x112",
 'notes': 'With LAFS self-supervised pre-training (README: IJB-C 97.58; the leaderboard lists LAFS at 97.48). Needs ipython, einops, timm. '
          'The checkpoint also holds the training classification head.',
 'params': None,
 'weights_file_bytes': 179777536}
URL = 'https://drive.usercontent.google.com/download?id=1BUYm2Bcgp8ZRlBcwOZxiJtWiQAvK2Ujy&export=download&confirm=t'
REL = 'lafs/part_fvit_wf4m_lafs.pth'


def build():
    """Return fn(uint8 BGR (n,112,112,3)) -> (n,d) embeddings."""
    return lafs(fetch(URL, REL))

"""MagFace iResNet-100 (MS1MV2): model definition for the benchmark (see README.md in this folder)."""
from benchmark.common import fetch
from benchmark.loaders import magface

INFO = {'name': 'MagFace iResNet-100 (MS1MV2)',
 'family': 'MagFace',
 'arch': 'iResNet-100, MagFace loss (embedding magnitude = quality)',
 'weights': 'https://github.com/IrvingMeng/MagFace',
 'code': 'https://github.com/IrvingMeng/MagFace',
 'paper': '',
 'training_data': 'MS1MV2',
 'license': 'Apache-2.0 (repo); trained on MS1MV2, whose dataset terms apply',
 'license_class': 'unknown',
 'pre': 'PyTorch, BGR, x/255 with no mean/std (as in the MagFace repo), 112x112',
 'notes': 'Only the direction of the embedding is used (its norm is a quality score). Listed on the IJB-C leaderboard as MagFace+ (95.97).',
 'params': None,
 'weights_file_bytes': 283040192}
URL = 'https://drive.usercontent.google.com/download?id=1Bd87admxOZvbIOAyTkGEntsEz3fyMt7H&export=download&confirm=t'
REL = 'magface/magface_r100_ms1mv2.pth'


def build():
    """Return fn(uint8 BGR (n,112,112,3)) -> (n,d) embeddings."""
    return magface(fetch(URL, REL))

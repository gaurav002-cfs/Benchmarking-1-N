"""ElasticFace-Cos+ iResNet-100 (MS1MV2): model definition for the benchmark (see README.md in this folder)."""
from benchmark.common import fetch
from benchmark.loaders import elasticface

INFO = {'name': 'ElasticFace-Cos+ iResNet-100 (MS1MV2)',
 'family': 'ElasticFace',
 'arch': 'iResNet-100, ElasticFace-Cos+ margin loss',
 'weights': 'https://github.com/fdbtrs/ElasticFace',
 'code': 'https://github.com/fdbtrs/ElasticFace',
 'paper': '',
 'training_data': 'MS1MV2',
 'license': 'CC BY-NC-SA 4.0 (repo)',
 'license_class': 'noncommercial',
 'pre': 'PyTorch, RGB, (x/255-0.5)/0.5, 112x112',
 'notes': 'Listed on the IJB-C leaderboard (96.65). Architecture is the standard arcface_torch iResNet-100.',
 'params': None,
 'weights_file_bytes': 261216376}
URL = 'https://drive.usercontent.google.com/download?id=17mRF-PVEZgmvB0a-P6qMJPwkI85bZJ3l&export=download&confirm=t'
REL = 'elasticface/cosplus_backbone.pth'


def build():
    """Return fn(uint8 BGR (n,112,112,3)) -> (n,d) embeddings."""
    return elasticface(fetch(URL, REL))

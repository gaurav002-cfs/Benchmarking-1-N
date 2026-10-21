"""ViT-8R: ViT-Base with 8 registers (WebFace4M): model definition for the benchmark (see README.md in this folder)."""
from benchmark.common import fetch
from benchmark.loaders import arcface_torch_net

INFO = {'name': 'ViT-8R: ViT-Base with 8 registers (WebFace4M)',
 'family': 'ViT-8R (Fraunhofer IGD)',
 'arch': 'Vision Transformer Base (patch 9) with 8 register tokens',
 'weights': 'https://owncloud.fraunhofer.de/index.php/s/wL8uH69nc25GUbY',
 'code': 'https://github.com/taharchettaoui/vit-fr-registers',
 'paper': 'https://ieeexplore.ieee.org/abstract/document/11556970',
 'training_data': 'WebFace4M',
 'license': 'CC BY-NC-SA 4.0 (repo)',
 'license_class': 'noncommercial',
 'pre': 'PyTorch (repo backbones/vit.py), RGB, (x/255-0.5)/0.5, 112x112',
 'notes': 'Appears on the IJB-C leaderboard as ViT-8R (97.37). Weights are hosted on a Fraunhofer OwnCloud share (it returned HTTP 503 for '
          "a while on first try). The repo README gives no usage code: the model is built with the repo's get_model('vit_b', "
          'num_register_token=8) and the checkpoint loads with zero missing / unexpected keys.',
 'params': None,
 'weights_file_bytes': 455442065}
URL = 'https://owncloud.fraunhofer.de/index.php/s/wL8uH69nc25GUbY/download'
REL = 'vit8r/vitb_8reg_web4m.pt'


def build():
    """Return fn(uint8 BGR (n,112,112,3)) -> (n,d) embeddings."""
    return arcface_torch_net("https://raw.githubusercontent.com/taharchettaoui/vit-fr-registers/main/", "vit8r_code", "vit_b", fetch(URL, REL), num_register_token=8)

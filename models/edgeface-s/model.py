"""EdgeFace-S (g=0.5): model definition for the benchmark (see README.md in this folder)."""

from benchmark.common import fetch
from benchmark.loaders import edgeface


INFO = {'name': 'EdgeFace-S (g=0.5)',
 'family': 'EdgeFace (Idiap)',
 'arch': 'EdgeNeXt-based hybrid CNN/Transformer with low-rank linear layers',
 'weights': 'https://huggingface.co/Idiap/EdgeFace-S-GAMMA',
 'code': 'https://github.com/otroshi/edgeface',
 'paper': 'https://huggingface.co/Idiap/EdgeFace-S-GAMMA',
 'training_data': 'WebFace260M subsets (EdgeFace-Base card: 12M + 4M subsets)',
 'license': 'CC BY-NC-SA 4.0 (model card)',
 'license_class': 'noncommercial',
 'pre': 'PyTorch, RGB, (x/255-0.5)/0.5, 112x112',
 'notes': 'Edge-device family; 3.65M params and 306.12 MFLOPs per the model card. Code is fetched from github.com/otroshi/edgeface at run '
          'time (not vendored).',
 'params': 3.65,
 'weights_file_bytes': 14695737}
URL = 'https://huggingface.co/Idiap/EdgeFace-S-GAMMA/resolve/main/edgeface_s_gamma_05.pt'


def build():
    """Return fn(uint8 BGR (n,112,112,3)) -> (n,d) embeddings."""
    return edgeface("edgeface_s_gamma_05", fetch(URL, "edgeface/edgeface_s_gamma_05.pt"))

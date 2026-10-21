"""EdgeFace-Base: model definition for the benchmark (see README.md in this folder)."""

from benchmark.common import fetch
from benchmark.loaders import edgeface


INFO = {'name': 'EdgeFace-Base',
 'family': 'EdgeFace (Idiap)',
 'arch': 'EdgeNeXt-based hybrid CNN/Transformer with low-rank linear layers',
 'weights': 'https://huggingface.co/Idiap/EdgeFace-Base',
 'code': 'https://github.com/otroshi/edgeface',
 'paper': 'https://huggingface.co/Idiap/EdgeFace-Base',
 'training_data': 'WebFace260M subsets (EdgeFace-Base card: 12M + 4M subsets)',
 'license': 'CC BY-NC-SA 4.0 (model card)',
 'license_class': 'noncommercial',
 'pre': 'PyTorch, RGB, (x/255-0.5)/0.5, 112x112',
 'notes': 'Edge-device family; 18.23M params and 1398.83 MFLOPs per the model card. Code is fetched from github.com/otroshi/edgeface at '
          'run time (not vendored).',
 'params': 18.23,
 'weights_file_bytes': 72972261}
URL = 'https://huggingface.co/Idiap/EdgeFace-Base/resolve/main/edgeface_base.pt'


def build():
    """Return fn(uint8 BGR (n,112,112,3)) -> (n,d) embeddings."""
    return edgeface("edgeface_base", fetch(URL, "edgeface/edgeface_base.pt"))

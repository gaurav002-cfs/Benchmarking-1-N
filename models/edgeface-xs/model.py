"""EdgeFace-XS (g=0.6): model definition for the benchmark (see README.md in this folder)."""

from benchmark.common import fetch
from benchmark.loaders import edgeface


INFO = {'name': 'EdgeFace-XS (g=0.6)',
 'family': 'EdgeFace (Idiap)',
 'arch': 'EdgeNeXt-based hybrid CNN/Transformer with low-rank linear layers',
 'weights': 'https://huggingface.co/Idiap/EdgeFace-XS-GAMMA',
 'code': 'https://github.com/otroshi/edgeface',
 'paper': 'https://huggingface.co/Idiap/EdgeFace-XS-GAMMA',
 'training_data': 'WebFace260M subsets (EdgeFace-Base card: 12M + 4M subsets)',
 'license': 'CC BY-NC-SA 4.0 (model card)',
 'license_class': 'noncommercial',
 'pre': 'PyTorch, RGB, (x/255-0.5)/0.5, 112x112',
 'notes': 'Edge-device family; 1.77M params and 154.00 MFLOPs per the model card. Code is fetched from github.com/otroshi/edgeface at run '
          'time (not vendored).',
 'params': 1.77,
 'weights_file_bytes': 7170425}
URL = 'https://huggingface.co/Idiap/EdgeFace-XS-GAMMA/resolve/main/edgeface_xs_gamma_06.pt'


def build():
    """Return fn(uint8 BGR (n,112,112,3)) -> (n,d) embeddings."""
    return edgeface("edgeface_xs_gamma_06", fetch(URL, "edgeface/edgeface_xs_gamma_06.pt"))

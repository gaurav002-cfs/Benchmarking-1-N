"""ArcFace IR101 WebFace4M (CVLface): model definition for the benchmark (see README.md in this folder)."""

from benchmark.loaders import cvlface


INFO = {'name': 'ArcFace IR101 WebFace4M (CVLface)',
 'family': 'CVLface (minchul)',
 'arch': 'ArcFace, IResNet-101',
 'weights': 'https://huggingface.co/minchul/cvlface_arcface_ir101_webface4m',
 'code': 'https://github.com/mk-minchul/CVLface',
 'paper': 'https://arxiv.org/abs/2204.00964',
 'training_data': 'WebFace4M',
 'license': "model card: 'follow the license of the training dataset'",
 'license_class': 'unknown',
 'pre': 'PyTorch, RGB, (x/255-0.5)/0.5, 112x112',
 'notes': 'Needs transformers==4.44.2, fvcore, omegaconf, einops. Run in its own process (CVLface models share a `models` package name).',
 'params': None,
 'weights_file_bytes': None}



def build():
    """Return fn(uint8 BGR (n,112,112,3)) -> (n,d) embeddings."""
    return cvlface("minchul/cvlface_arcface_ir101_webface4m")

"""AdaFace IR50 MS1MV2 (CVLface): model definition for the benchmark (see README.md in this folder)."""

from benchmark.loaders import cvlface


INFO = {'name': 'AdaFace IR50 MS1MV2 (CVLface)',
 'family': 'CVLface (minchul)',
 'arch': 'AdaFace, IResNet-50',
 'weights': 'https://huggingface.co/minchul/cvlface_adaface_ir50_ms1mv2',
 'code': 'https://github.com/mk-minchul/CVLface',
 'paper': 'https://arxiv.org/abs/2204.00964',
 'training_data': 'MS1MV2',
 'license': "model card: 'follow the license of the training dataset'",
 'license_class': 'unknown',
 'pre': 'PyTorch, RGB, (x/255-0.5)/0.5, 112x112',
 'notes': 'Needs transformers==4.44.2, fvcore, omegaconf, einops. Run in its own process (CVLface models share a `models` package name).',
 'params': None,
 'weights_file_bytes': None}



def build():
    """Return fn(uint8 BGR (n,112,112,3)) -> (n,d) embeddings."""
    return cvlface("minchul/cvlface_adaface_ir50_ms1mv2")

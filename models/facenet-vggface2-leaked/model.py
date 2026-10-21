"""FaceNet InceptionResNetV1 (VGGFace2) [LEAKED]: model definition for the benchmark (see README.md in this folder)."""

from benchmark.loaders import facenet


INFO = {'name': 'FaceNet InceptionResNetV1 (VGGFace2) [LEAKED]',
 'family': 'FaceNet (facenet-pytorch)',
 'arch': 'Inception-ResNet-v1, 512-d',
 'weights': 'https://github.com/timesler/facenet-pytorch',
 'code': 'https://github.com/timesler/facenet-pytorch',
 'paper': 'https://arxiv.org/abs/1503.03832',
 'training_data': 'VGGFace2 = the same data as this benchmark (LEAKED, not a fair score)',
 'license': 'code MIT (facenet-pytorch); weights terms follow training data',
 'license_class': 'leaked',
 'pre': 'PyTorch, RGB, (x-127.5)/128 at 160x160 (upsampled)',
 'notes': 'LEAKED: trained on VGGFace2, so this score is inflated and must not be compared with the others. Shown for completeness only.',
 'params': None,
 'weights_file_bytes': None}



def build():
    """Return fn(uint8 BGR (n,112,112,3)) -> (n,d) embeddings."""
    return facenet("vggface2")

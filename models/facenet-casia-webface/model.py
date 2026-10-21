"""FaceNet InceptionResNetV1 (CASIA-WebFace): model definition for the benchmark (see README.md in this folder)."""

from benchmark.loaders import facenet


INFO = {'name': 'FaceNet InceptionResNetV1 (CASIA-WebFace)',
 'family': 'FaceNet (facenet-pytorch)',
 'arch': 'Inception-ResNet-v1, 512-d',
 'weights': 'https://github.com/timesler/facenet-pytorch',
 'code': 'https://github.com/timesler/facenet-pytorch',
 'paper': 'https://arxiv.org/abs/1503.03832',
 'training_data': 'CASIA-WebFace',
 'license': 'code MIT (facenet-pytorch); weights terms follow training data',
 'license_class': 'unknown',
 'pre': 'PyTorch, RGB, (x-127.5)/128 at 160x160 (our 112x112 crops are bilinearly upsampled)',
 'notes': 'Understated: FaceNet expects looser 160x160 crops than these tight 112x112 ones. Clean score (not trained on VGGFace2).',
 'params': None,
 'weights_file_bytes': None}



def build():
    """Return fn(uint8 BGR (n,112,112,3)) -> (n,d) embeddings."""
    return facenet("casia-webface")

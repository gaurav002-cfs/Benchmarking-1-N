"""CLIP ViT-L/14 image encoder: model definition for the benchmark (see README.md in this folder)."""

from benchmark.loaders import clip_vision


INFO = {'name': 'CLIP ViT-L/14 image encoder',
 'family': 'CLIP (OpenAI)',
 'arch': 'ViT-L/14 image tower (the vision-encoder family used by LLaVA)',
 'weights': 'https://huggingface.co/openai/clip-vit-large-patch14',
 'code': 'https://github.com/openai/CLIP',
 'paper': 'https://arxiv.org/abs/2103.00020',
 'training_data': 'web image-text pairs (general purpose, not face-specific)',
 'license': 'MIT (OpenAI CLIP)',
 'license_class': 'permissive',
 'pre': 'transformers CLIPModel.get_image_features, RGB, CLIP mean/std, 224x224',
 'notes': "Included to test the 'general-purpose encoder' claim: far below face-specific models.",
 'params': None,
 'weights_file_bytes': None}



def build():
    """Return fn(uint8 BGR (n,112,112,3)) -> (n,d) embeddings."""
    return clip_vision("openai/clip-vit-large-patch14")

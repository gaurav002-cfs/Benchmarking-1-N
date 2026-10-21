"""SFace (OpenCV): model definition for the benchmark (see README.md in this folder)."""
import numpy as np
from benchmark.common import fetch


INFO = {'name': 'SFace (OpenCV)',
 'family': 'SFace (OpenCV Zoo)',
 'arch': 'SFace (see paper), 128-d output',
 'weights': 'https://huggingface.co/opencv/face_recognition_sface',
 'code': 'https://github.com/opencv/opencv_zoo',
 'paper': 'https://arxiv.org/abs/2205.12010',
 'training_data': 'see SFace paper',
 'license': 'Apache-2.0 (repo LICENSE)',
 'license_class': 'permissive',
 'pre': 'OpenCV cv2.FaceRecognizerSF.feature() on the aligned 112x112 BGR crop (OpenCV does its own preprocessing)',
 'notes': '128-d embedding (all others are 512-d). Commercially friendly licence, mid-to-low accuracy here.',
 'params': 9.67,
 'weights_file_bytes': 38696353}
URL = 'https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx'


def build():
    """Return fn(uint8 BGR (n,112,112,3)) -> (n,d) embeddings."""
    import cv2
    rec = cv2.FaceRecognizerSF.create(str(fetch(URL, "sface/face_recognition_sface_2021dec.onnx")), "")
    return lambda x: np.concatenate([rec.feature(np.ascontiguousarray(im)) for im in x])

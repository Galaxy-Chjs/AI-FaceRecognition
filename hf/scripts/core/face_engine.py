import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEVICE, IMAGE_SIZE


class FaceEngine:

    def __init__(self):
        self.mtcnn = MTCNN(
            keep_all=True,
            device=DEVICE,
            post_process=True,
            image_size=IMAGE_SIZE,
            min_face_size=20,
        )
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

    def detect_faces(self, image, min_confidence=0.9):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        bboxes, probs = self.mtcnn.detect(image)
        faces = self.mtcnn(image)

        if bboxes is None:
            return np.empty((0, 4), dtype=np.float64), None, np.array([])

        if min_confidence > 0 and probs is not None:
            mask = probs >= min_confidence
            bboxes = bboxes[mask]
            if probs is not None:
                probs = probs[mask]
            if faces is not None:
                faces = faces[mask]
            if len(bboxes) == 0:
                return np.empty((0, 4), dtype=np.float64), None, np.array([])

        if probs is None:
            probs = np.ones(len(bboxes))

        return bboxes, faces, probs

    def get_embeddings(self, faces):
        if faces is None:
            return None
        with torch.no_grad():
            if faces.dim() == 3:
                faces = faces.unsqueeze(0)
            embeddings = self.resnet(faces.to(DEVICE))
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

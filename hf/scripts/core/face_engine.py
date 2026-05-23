"""
人脸检测 + 特征提取引擎
使用 MTCNN 进行人脸检测和对齐，InceptionResnetV1 提取 512 维特征向量
"""
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEVICE, IMAGE_SIZE


class FaceEngine:
    """人脸检测与特征提取引擎"""

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
        """
        检测图片中的人脸, 返回边界框、裁剪张量、置信度.

        Args:
            image: PIL.Image 或 文件路径字符串
            min_confidence: 最低检测置信度, 低于此值的检测结果被过滤

        Returns:
            bboxes:  np.ndarray, shape (N, 4), [x1, y1, x2, y2]
            faces:   torch.Tensor or None, shape (N, 3, 160, 160)
            probs:   np.ndarray, shape (N,), 每张人脸的检测置信度
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        # detect() 返回边界框和置信度
        bboxes, probs = self.mtcnn.detect(image)

        # mtcnn(image) 返回裁剪+归一化后的人脸张量
        faces = self.mtcnn(image)

        # 统一空结果格式
        if bboxes is None:
            return np.empty((0, 4), dtype=np.float64), None, np.array([])

        # 过滤低置信度检测 (噪声、非人脸区域)
        if min_confidence > 0 and probs is not None:
            mask = probs >= min_confidence
            bboxes = bboxes[mask]
            if probs is not None:
                probs = probs[mask]
            if faces is not None:
                faces = faces[mask]
            # 过滤后可能变空
            if len(bboxes) == 0:
                return np.empty((0, 4), dtype=np.float64), None, np.array([])

        if probs is None:
            probs = np.ones(len(bboxes))

        return bboxes, faces, probs

    def get_embeddings(self, faces):
        """
        将裁剪人脸张量送入 InceptionResnetV1, 提取 L2 归一化后的 512 维特征.
        """
        if faces is None:
            return None

        with torch.no_grad():
            if faces.dim() == 3:
                faces = faces.unsqueeze(0)
            embeddings = self.resnet(faces.to(DEVICE))
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

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
        # MTCNN: keep_all=True 检测图中所有人脸
        #        post_process=True 将裁剪后的人脸归一化到 [-1, 1] (InceptionResnetV1 要求的输入范围)
        #        min_face_size=20 适应 CelebA 等较小的人脸
        self.mtcnn = MTCNN(
            keep_all=True,
            device=DEVICE,
            post_process=True,
            image_size=IMAGE_SIZE,
            min_face_size=20,
        )
        # InceptionResnetV1: 预训练于 VGGFace2, 输出 512 维 embedding
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

    def detect_faces(self, image):
        """
        检测图片中的人脸, 返回边界框和裁剪后的人脸张量.

        Args:
            image: PIL.Image 或 文件路径字符串

        Returns:
            bboxes: np.ndarray, shape (N, 4), 绝对像素坐标 [x1, y1, x2, y2].
                    无检测结果时返回 shape (0, 4) 的空数组.
            faces:  torch.Tensor, shape (N, 3, 160, 160), 已归一化到 [-1, 1].
                    无检测结果时返回 None.
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        # detect() 仅做检测, 返回边界框坐标
        bboxes, _ = self.mtcnn.detect(image)

        # mtcnn(image) 完成检测+裁剪+对齐+归一化, 返回可直接送入模型的张量
        faces = self.mtcnn(image)

        # 统一空结果格式
        if bboxes is None:
            bboxes = np.empty((0, 4), dtype=np.float64)

        return bboxes, faces

    def get_embeddings(self, faces):
        """
        将裁剪人脸张量送入 InceptionResnetV1, 提取 L2 归一化后的 512 维特征.

        Args:
            faces: torch.Tensor, shape (N, 3, 160, 160) 或 (3, 160, 160)

        Returns:
            np.ndarray, shape (N, 512), 每一行是 L2 归一化后的 embedding.
            若输入为 None 则返回 None.
        """
        if faces is None:
            return None

        with torch.no_grad():
            if faces.dim() == 3:
                faces = faces.unsqueeze(0)
            embeddings = self.resnet(faces.to(DEVICE))
            # L2 归一化: 使余弦相似度等价于向量点积, 提升匹配精度
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

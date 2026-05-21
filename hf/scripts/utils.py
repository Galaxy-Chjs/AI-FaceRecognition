import cv2
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from config import DEVICE, IMAGE_SIZE

class FaceEngine:
    def __init__(self):
        # keep_all=True 确保一张图可以检测出多张人脸（适配多人照）
        self.mtcnn = MTCNN(keep_all=True, device=DEVICE, post_process=False, image_size=IMAGE_SIZE)
        # 加载本地/在线预训练模型
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

    def detect_faces(self, pair_image):
        """
        检测图片中的所有人脸
        返回: bboxes (N, 4) 绝对像素坐标 [x1, y1, x2, y2], 
             cropped_faces (N, 3, 160, 160) 裁剪后的人脸张量
        """
        # MTCNN 需要 PIL Image 格式
        if isinstance(pair_image, str):
            img = Image.open(pair_image).convert('RGB')
        else:
            img = pair_image

        bboxes, _ = self.mtcnn.detect(img)
        
        # 获取用于提取特征的裁剪人脸张量
        try:
            cropped_faces = self.mtcnn(img)
        except Exception:
            cropped_faces = None

        return bboxes, cropped_faces

    def get_embeddings(self, cropped_faces):
        """
        输入裁剪后的人脸张量，输出512维特征向量
        """
        if cropped_faces is None:
            return None
        with torch.no_grad():
            # 如果只检测到一张脸，保持维度一致
            if len(cropped_faces.shape) == 3:
                cropped_faces = cropped_faces.unsqueeze(0)
            embeddings = self.resnet(cropped_faces.to(DEVICE))
        return embeddings.cpu().numpy()
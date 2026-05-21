import os
import torch
import numpy as np
from PIL import Image
from config import REGISTERED_DIR, GALLERY_PATH
from utils import FaceEngine

def build_gallery():
    engine = FaceEngine()
    gallery = {}
    
    # 遍历每个身份的文件夹 (p01, p02...)
    for identity_id in sorted(os.listdir(REGISTERED_DIR)):
        id_dir = os.path.join(REGISTERED_DIR, identity_id)
        if not os.path.isdir(id_dir):
            continue
            
        embeddings_list = []
        print(f"正在注册身份: {identity_id}")
        
        for img_name in os.listdir(id_dir):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(id_dir, img_name)
            
            # 提取特征
            _, cropped_faces = engine.detect_faces(img_path)
            if cropped_faces is not None:
                emb = engine.get_embeddings(cropped_faces)
                if emb is not None:
                    # 注册集图片确保只有一张清晰主脸，取第一个
                    embeddings_list.append(emb[0])
        
        if embeddings_list:
            # 计算该身份多张注册图片的平均特征向量，使其更具泛化性
            gallery[identity_id] = np.mean(embeddings_list, axis=0)

    # 创建模型文件夹并保存特征库
    os.makedirs(os.path.dirname(GALLERY_PATH), exist_ok=True)
    torch.save(gallery, GALLERY_PATH)
    print(f"底库构建完成！已保存在: {GALLERY_PATH}")

if __name__ == '__main__':
    build_gallery()
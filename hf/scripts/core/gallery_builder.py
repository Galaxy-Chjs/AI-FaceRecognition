"""
底库构建模块
从注册集图片中提取每个身份的特征向量, 取平均后存入底库.
"""
import os
import torch
import numpy as np


def build_gallery(engine, register_dir, save_path):
    """
    遍历注册集目录, 为每个身份构建平均 embedding 并保存.

    目录结构要求:
        register_dir/
            identity_001/
                img1.jpg
                img2.jpg
            identity_002/
                ...

    Args:
        engine:      FaceEngine 实例
        register_dir: 注册集根目录, 每个子文件夹代表一个身份
        save_path:    底库 .pt 文件保存路径

    Returns:
        dict: {identity_id: np.ndarray(512,), ...}
    """
    gallery = {}

    if not os.path.isdir(register_dir):
        print(f"[错误] 注册集目录不存在: {register_dir}")
        return gallery

    identities = sorted([
        d for d in os.listdir(register_dir)
        if os.path.isdir(os.path.join(register_dir, d)) and not d.startswith('.')
    ])

    print(f"共发现 {len(identities)} 个身份类别, 开始构建底库...")

    for i, identity_id in enumerate(identities, 1):
        id_dir = os.path.join(register_dir, identity_id)
        embeddings_list = []

        for img_name in sorted(os.listdir(id_dir)):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            img_path = os.path.join(id_dir, img_name)

            bboxes, faces = engine.detect_faces(img_path)
            if faces is None:
                print(f"  [警告] {img_path} 未检测到人脸, 跳过")
                continue

            emb = engine.get_embeddings(faces)
            if emb is not None and len(emb) > 0:
                # 取第一张检测到的人脸 (注册集通常为单人正脸)
                embeddings_list.append(emb[0])

        if embeddings_list:
            # 多张注册图取平均, 获得更稳定的特征表示
            gallery[identity_id] = np.mean(embeddings_list, axis=0)
            print(f"  [{i}/{len(identities)}] {identity_id}: 注册 {len(embeddings_list)} 张图片")
        else:
            print(f"  [{i}/{len(identities)}] {identity_id}: 无有效人脸, 跳过")

    # 保存底库
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(gallery, save_path)
    print(f"\n底库构建完成! 共注册 {len(gallery)} 类身份, 已保存至: {save_path}")

    return gallery

import os
import numpy as np
import torch
from scipy.spatial.distance import cosine
from config import DEVICE, CELEBA_REGISTER_DIR, CELEBA_TEST_DIR, THRESHOLD
from utils import FaceEngine

def build_celeba_gallery(engine):
    """
    针对 CelebA 数据集动态构建特征底库
    """
    print("====== 开始构建 CelebA 100类 特征底库 ======")
    gallery = {}
    
    # 遍历 100 个 identity 文件夹
    identities = sorted(os.listdir(CELEBA_REGISTER_DIR))
    for identity_id in identities:
        id_dir = os.path.join(CELEBA_REGISTER_DIR, identity_id)
        if not os.path.isdir(id_dir) or identity_id.startswith('.'):
            continue
            
        embeddings_list = []
        for img_name in os.listdir(id_dir):
            if img_name.startswith('.'):
                continue
            img_path = os.path.join(id_dir, img_name)
            
            # 检测并提取特征
            _, cropped_faces = engine.detect_faces(img_path)
            if cropped_faces is not None:
                emb = engine.get_embeddings(cropped_faces)
                if emb is not None:
                    embeddings_list.append(emb[0]) # 注册图默认为单人正脸
                    
        if embeddings_list:
            # 3张注册图取平均，作为该身份的稳定表征
            gallery[identity_id] = np.mean(embeddings_list, axis=0)
            
    print(f"CelebA 底库构建完成，共注册 {len(gallery)} 类身份。\n")
    return gallery

def evaluate_celeba():
    """
    对 CelebA 进行 Top-1 准确率评测
    """
    engine = FaceEngine()
    gallery = build_celeba_gallery(engine)
    
    if not gallery:
        print("底库为空，请检查数据路径！")
        return

    correct_count = 0
    total_count = 0
    
    success_examples = []
    fail_examples = []

    print("====== 开始对 CelebA 100类 进行测试 ======")
    identities = sorted(os.listdir(CELEBA_TEST_DIR))
    
    for identity_id in identities:
        id_dir = os.path.join(CELEBA_TEST_DIR, identity_id)
        if not os.path.isdir(id_dir) or identity_id.startswith('.'):
            continue
            
        for img_name in os.listdir(id_dir):
            if img_name.startswith('.'):
                continue
            img_path = os.path.join(id_dir, img_name)
            total_count += 1
            
            # 1. 提取测试集图片的特征
            _, cropped_faces = engine.detect_faces(img_path)
            if cropped_faces is None:
                # 如果没检测到人脸，默认算作识别失败
                fail_examples.append((img_name, identity_id, "未检测到人脸"))
                continue
                
            test_embs = engine.get_embeddings(cropped_faces)
            if test_embs is None:
                fail_examples.append((img_name, identity_id, "特征提取失败"))
                continue
                
            test_emb = test_embs[0] # 取主脸
            
            # 2. 与底库中的100类计算余弦相似度，找出最相似的 Top-1
            best_match = None
            max_sim = -1.0
            
            for gal_id, gal_emb in gallery.items():
                # 1 - cosine 得到余弦相似度，值在 [-1, 1] 之间
                sim = 1 - cosine(test_emb, gal_emb)
                if sim > max_sim:
                    max_sim = sim
                    best_match = gal_id
            
            # 3. 判断是否识别正确（需通过设定的相似度阈值）
            predicted_id = best_match if max_sim >= THRESHOLD else "unknown"
            
            if predicted_id == identity_id:
                correct_count += 1
                if len(success_examples) < 3:
                    success_examples.append((img_name, identity_id, max_sim))
            else:
                if len(fail_examples) < 3:
                    fail_examples.append((img_name, identity_id, predicted_id, max_sim))

    # 4. 计算并输出 Top-1 准确率
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    print("\n================ 测试报告 ================")
    print(f"总测试样本数: {total_count}")
    print(f"正确识别数: {correct_count}")
    print(f"Top-1 识别准确率: {accuracy:.2f}%")
    print("==========================================")
    
    # 5. 输出样例，直接用于写大作业报告（拿满分析分）
    print("\n[成功样例展示（供报告使用）]")
    for ex in success_examples:
        print(f"图片: {ex[0]} | 真实身份: {ex[1]} | 匹配置信度: {ex[2]:.4f}")
        
    print("\n[失败样例展示（供报告使用，用于写失败原因分析）]")
    for ex in fail_examples:
        if len(ex) == 3:
            print(f"图片: {ex[0]} | 真实身份: {ex[1]} | 原因: {ex[2]}")
        else:
            print(f"图片: {ex[0]} | 真实身份: {ex[1]} | 误判为: {ex[2]} | 匹配置信度: {ex[3]:.4f}")

if __name__ == '__main__':
    evaluate_celeba()
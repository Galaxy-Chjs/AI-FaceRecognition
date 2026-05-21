"""
评测模块
在给定测试集上评估 Top-1 识别准确率, 输出成功/失败样例.
"""
import os
import numpy as np


def evaluate_top1(engine, gallery, test_dir, max_examples=5):
    """
    对测试集进行 Top-1 识别准确率评测.

    目录结构要求:
        test_dir/
            identity_001/
                img1.jpg
                img2.jpg
            identity_002/
                ...

    Args:
        engine:       FaceEngine 实例
        gallery:      dict {identity_id: np.ndarray(512,)}
        test_dir:     测试集根目录
        max_examples: 成功/失败样例各保留的最大数量

    Returns:
        dict: {
            'accuracy': float, 百分比,
            'correct': int,
            'total': int,
            'no_face_count': int,
            'success_examples': list of (img_name, true_id, confidence),
            'fail_examples': list of (img_name, true_id, predicted_id, confidence),
        }
    """
    if not gallery:
        print("[错误] 底库为空, 无法进行评测!")
        return None

    gallery_ids = list(gallery.keys())
    gallery_matrix = np.stack([gallery[gid] for gid in gallery_ids])
    # gallery_matrix: (num_classes, 512), 已 L2 归一化

    correct = 0
    total = 0
    no_face_count = 0
    success_examples = []
    fail_examples = []

    identities = sorted([
        d for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d)) and not d.startswith('.')
    ])

    print(f"开始评测, 共 {len(identities)} 个身份类别...")

    for identity_id in identities:
        id_dir = os.path.join(test_dir, identity_id)
        for img_name in sorted(os.listdir(id_dir)):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            img_path = os.path.join(id_dir, img_name)
            total += 1

            # 1. 检测人脸并提取特征
            bboxes, faces = engine.detect_faces(img_path)
            if faces is None:
                no_face_count += 1
                fail_examples.append((img_name, identity_id, "none", 0.0))
                continue

            embs = engine.get_embeddings(faces)
            if embs is None or len(embs) == 0:
                no_face_count += 1
                fail_examples.append((img_name, identity_id, "none", 0.0))
                continue

            # 取第一张检测到的人脸
            test_emb = embs[0]  # (512,)

            # 2. 计算与底库所有身份的余弦相似度
            #    embedding 已 L2 归一化, 余弦相似度 = 点积
            similarities = gallery_matrix @ test_emb  # (num_classes,)
            best_idx = int(np.argmax(similarities))
            best_sim = float(similarities[best_idx])
            predicted_id = gallery_ids[best_idx]

            # 3. 判断是否正确
            if predicted_id == identity_id:
                correct += 1
                if len(success_examples) < max_examples:
                    success_examples.append((img_name, identity_id, best_sim))
            else:
                if len(fail_examples) < max_examples:
                    fail_examples.append((img_name, identity_id, predicted_id, best_sim))

    accuracy = (correct / total * 100) if total > 0 else 0.0

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'no_face_count': no_face_count,
        'success_examples': success_examples,
        'fail_examples': fail_examples,
    }


def print_report(result, dataset_name="测试集"):
    """打印评测报告"""
    if result is None:
        print("评测结果为空!")
        return

    print(f"\n{'='*50}")
    print(f"  {dataset_name} 评测报告")
    print(f"{'='*50}")
    print(f"  总测试样本数:  {result['total']}")
    print(f"  正确识别数:    {result['correct']}")
    print(f"  未检测到人脸:  {result['no_face_count']}")
    print(f"  Top-1 准确率:  {result['accuracy']:.2f}%")
    print(f"{'='*50}")

    if result['success_examples']:
        print(f"\n[成功样例]")
        for img_name, true_id, conf in result['success_examples']:
            print(f"  {img_name}  真实: {true_id}  置信度: {conf:.4f}")

    if result['fail_examples']:
        print(f"\n[失败样例]")
        for ex in result['fail_examples']:
            if len(ex) == 4:
                img_name, true_id, pred_id, conf = ex
                if pred_id == "none":
                    print(f"  {img_name}  真实: {true_id}  原因: 未检测到人脸")
                else:
                    print(f"  {img_name}  真实: {true_id}  误判为: {pred_id}  置信度: {conf:.4f}")
            else:
                print(f"  {ex[0]}  真实: {ex[1]}  原因: {ex[2]}")

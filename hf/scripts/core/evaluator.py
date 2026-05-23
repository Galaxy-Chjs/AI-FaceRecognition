"""
评测模块
在给定测试集上评估 Top-1 识别准确率, 输出成功/失败样例.
支持两种目录结构:
  - CelebA 风格: test_dir/identity_id/img.jpg (身份来自子文件夹名)
  - 自收集数据:  test_dir/p01_t01.jpg (身份来自文件名前缀)
"""
import os
import numpy as np


def evaluate_top1(engine, gallery, test_dir, max_examples=5):
    """
    CelebA 风格评测: 测试目录下按身份分子文件夹.

    目录结构:
        test_dir/
            identity_001/
                img1.jpg
                img2.jpg
    """
    identities = sorted([
        d for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d)) and not d.startswith('.')
    ])

    image_list = []
    for identity_id in identities:
        id_dir = os.path.join(test_dir, identity_id)
        for img_name in sorted(os.listdir(id_dir)):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_list.append((os.path.join(id_dir, img_name), img_name, identity_id))

    return _run_evaluation(engine, gallery, image_list, max_examples)


def evaluate_custom_top1(engine, gallery, test_images_dir, max_examples=5):
    """
    自收集数据评测: 测试目录下图片平铺, 身份从文件名解析.

    文件名格式: p01_t01.jpg, p02_t03.png, ...
    身份 = 下划线前的部分 (p01, p02, ...)
    """
    image_list = []
    for img_name in sorted(os.listdir(test_images_dir)):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        # 从文件名解析身份: p01_t01.jpg -> p01
        base = os.path.splitext(img_name)[0]
        parts = base.split('_')
        if len(parts) < 2:
            print(f"  [跳过] 文件名格式不符合规范: {img_name}")
            continue
        identity_id = parts[0]
        img_path = os.path.join(test_images_dir, img_name)
        image_list.append((img_path, img_name, identity_id))

    return _run_evaluation(engine, gallery, image_list, max_examples)


def _run_evaluation(engine, gallery, image_list, max_examples=5):
    """
    通用评测逻辑.

    Args:
        engine:      FaceEngine 实例
        gallery:     dict {identity_id: np.ndarray(512,)}
        image_list:  list of (img_path, img_name, true_identity_id)
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

    correct = 0
    total = 0
    no_face_count = 0
    success_examples = []
    fail_examples = []

    print(f"开始评测, 共 {len(image_list)} 张测试图片...")

    for img_path, img_name, identity_id in image_list:
        total += 1

        # 1. 检测人脸并提取特征
        bboxes, faces, _ = engine.detect_faces(img_path)
        if faces is None:
            no_face_count += 1
            fail_examples.append((img_name, identity_id, "none", 0.0))
            continue

        embs = engine.get_embeddings(faces)
        if embs is None or len(embs) == 0:
            no_face_count += 1
            fail_examples.append((img_name, identity_id, "none", 0.0))
            continue

        # 对每张检测到的人脸, 找它在底库中的最佳匹配
        # 如果任一检测到的人脸的最佳匹配 == 目标身份, 则判定为正确
        # (多人照中目标人物可能不是第一张检测到的脸)
        found = False
        best_confidence = -1.0
        any_predicted_id = None
        any_best_sim = -1.0

        for face_emb in embs:
            sims = gallery_matrix @ face_emb  # (num_classes,)
            face_best_idx = int(np.argmax(sims))
            face_best_sim = float(sims[face_best_idx])
            face_predicted = gallery_ids[face_best_idx]

            # 记录任意脸的最佳匹配 (用于失败报告)
            if face_best_sim > any_best_sim:
                any_best_sim = face_best_sim
                any_predicted_id = face_predicted

            # 该脸的最佳匹配恰好是目标身份
            if face_predicted == identity_id:
                found = True
                if face_best_sim > best_confidence:
                    best_confidence = face_best_sim

        if found:
            correct += 1
            if len(success_examples) < max_examples:
                success_examples.append((img_name, identity_id, best_confidence))
        else:
            if len(fail_examples) < max_examples:
                fail_examples.append((img_name, identity_id, any_predicted_id, any_best_sim))

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

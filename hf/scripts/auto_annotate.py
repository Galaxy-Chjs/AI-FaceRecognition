"""
自动标注脚本 (半自动, 需人工复核)

核心改进: 使用底库匹配来判断每张检测到的脸属于哪个身份,
而不是盲目地把第一张脸标为目标身份.

流程:
  1. 先构建 20 类底库 (从注册集提取特征)
  2. 对每张测试图: MTCNN 检测人脸 → 提取特征 → 和底库比对
  3. 底库匹配度最高的脸标为目标身份, 其余标为 unknown
  4. 生成可视化图片供人工审查

用法:
    python auto_annotate.py                 # 生成标注 + 可视化
    python auto_annotate.py --vis-only      # 仅对已有标注做可视化
"""
import json
import sys
import os
import argparse
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TEST_IMAGES_DIR, DATASET_DIR, REGISTERED_DIR, GALLERY_20_PATH, ANNOTATIONS_PATH
from core.face_engine import FaceEngine
from core.gallery_builder import build_gallery


def annotate():
    """自动生成标注文件 (使用底库匹配来正确标注身份)"""
    if not os.path.isdir(TEST_IMAGES_DIR):
        print(f"[错误] 测试集图片目录不存在: {TEST_IMAGES_DIR}")
        return False

    engine = FaceEngine()

    # 构建底库 (用于匹配检测到的人脸)
    print("正在构建底库用于人脸匹配...")
    if os.path.exists(GALLERY_20_PATH):
        import torch
        gallery = torch.load(GALLERY_20_PATH, weights_only=False)
        print(f"  已加载底库: {len(gallery)} 个身份\n")
    else:
        gallery = build_gallery(engine, REGISTERED_DIR, GALLERY_20_PATH)
        print()

    gallery_ids = list(gallery.keys())
    gallery_matrix = np.stack([gallery[gid] for gid in gallery_ids])

    annotations = []
    image_files = sorted([
        f for f in os.listdir(TEST_IMAGES_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ])

    print(f"共 {len(image_files)} 张测试图片, 开始自动标注...\n")

    for img_name in image_files:
        img_path = os.path.join(TEST_IMAGES_DIR, img_name)

        # 从文件名推断主身份: p01_t01.jpg -> p01
        base_name = os.path.splitext(img_name)[0]
        parts = base_name.split('_')
        main_identity = parts[0] if len(parts) >= 2 else "unknown"

        # 1. 检测人脸 (已过滤低置信度)
        bboxes, faces, probs = engine.detect_faces(img_path, min_confidence=0.9)

        face_annotations = []
        if faces is not None and len(bboxes) > 0:
            # 2. 提取每张脸的特征
            embs = engine.get_embeddings(faces)

            if embs is not None:
                # 3. 对每张脸, 找底库中最佳匹配
                face_matches = []
                for i, face_emb in enumerate(embs):
                    sims = gallery_matrix @ face_emb
                    best_idx = int(np.argmax(sims))
                    best_sim = float(sims[best_idx])
                    best_id = gallery_ids[best_idx]
                    face_matches.append((i, best_id, best_sim))

                # 4. 标注: 底库匹配度最高且为目标身份的脸标为该身份, 其余标 unknown
                #    但允许多张脸都匹配到同一目标身份 (多人照中目标人物可能出现多次)
                for i, matched_id, sim in face_matches:
                    x1, y1, x2, y2 = bboxes[i]
                    w, h = int(x2 - x1), int(y2 - y1)

                    # 如果匹配到的是目标身份且相似度足够高, 标为目标身份
                    if matched_id == main_identity and sim >= 0.5:
                        label = main_identity
                    else:
                        label = "unknown"

                    face_annotations.append({
                        "identity_id": label,
                        "bbox": [int(x1), int(y1), w, h],
                        "confidence": round(float(probs[i]), 3),
                        "match_sim": round(sim, 3),
                    })

        image_type = "single" if len(face_annotations) <= 1 else "multi"

        annotations.append({
            "image": f"dataset/test/images/{img_name}",
            "image_type": image_type,
            "faces": face_annotations,
        })

        target_faces = sum(1 for f in face_annotations if f['identity_id'] == main_identity)
        print(f"  {img_name}: {len(face_annotations)} 张人脸, "
              f"其中 {target_faces} 张匹配到 {main_identity}")

    # 保存标注文件
    os.makedirs(os.path.dirname(ANNOTATIONS_PATH), exist_ok=True)
    with open(ANNOTATIONS_PATH, 'w', encoding='utf-8') as f:
        for ann in annotations:
            f.write(json.dumps(ann, ensure_ascii=False) + '\n')

    print(f"\n标注完成! 共 {len(annotations)} 张图片")
    print(f"标注文件: {ANNOTATIONS_PATH}")
    return True


def visualize():
    """
    读取 annotations.jsonl, 在图片上绘制 bbox 并保存到 review 目录.

    bbox [x, y, w, h] 含义:
      (x,y) = 人脸框左上角坐标
      w, h  = 框的宽和高
    """
    if not os.path.exists(ANNOTATIONS_PATH):
        print(f"[错误] 标注文件不存在: {ANNOTATIONS_PATH}")
        return

    review_dir = os.path.join(DATASET_DIR, 'test', 'review')
    os.makedirs(review_dir, exist_ok=True)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with open(ANNOTATIONS_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"\n共 {len(lines)} 条标注, 开始生成可视化...")
    print(f"输出目录: {review_dir}\n")

    for line in lines:
        ann = json.loads(line.strip())
        image_rel = ann['image']
        image_path = os.path.join(base_dir, image_rel)

        if not os.path.exists(image_path):
            print(f"  [跳过] 图片不存在: {image_path}")
            continue

        img = cv2.imread(image_path)
        if img is None:
            continue

        img_h, img_w = img.shape[:2]

        for face in ann.get('faces', []):
            bbox = face['bbox']
            identity = face['identity_id']
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            is_known = identity.startswith('p') and identity != "unknown"
            color = (0, 255, 0) if is_known else (0, 0, 255)

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            # 标签文字 (含置信度和匹配度)
            conf = face.get('confidence', 0)
            sim = face.get('match_sim', 0)
            label = f"{identity} conf={conf:.2f} sim={sim:.2f}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

            # 根据位置决定标签画在哪: 框上方或框下方 (防止超出图片)
            if y - th - 6 > 0:
                label_y1, label_y2 = y - th - 6, y
                text_y = y - 3
            else:
                label_y1, label_y2 = y + h, y + h + th + 6
                text_y = y + h + th + 3

            cv2.rectangle(img, (x, label_y1), (x + tw + 4, label_y2), color, -1)
            cv2.putText(img, label, (x + 2, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # 顶部信息栏
        img_name = os.path.basename(image_rel)
        face_count = len(ann.get('faces', []))
        info = f"{img_name} | {img_w}x{img_h} | {ann.get('image_type','')} | {face_count} faces"
        cv2.putText(img, info, (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        out_path = os.path.join(review_dir, img_name)
        cv2.imwrite(out_path, img)
        print(f"  {img_name}: {face_count} faces -> saved")

    print(f"\n可视化完成! 请检查 {review_dir} 中的图片")
    print("检查要点:")
    print("  - 绿色框 = 匹配到目标身份, 红色框 = unknown")
    print("  - sim 值 = 该脸与底库最佳匹配的相似度")
    print("  - 框是否准确圈住了人脸? 是否有误检?")
    print("  - 如有错误, 手动修改 annotations.jsonl 后重新运行 --vis-only")


def main():
    parser = argparse.ArgumentParser(description="自动标注 + 可视化")
    parser.add_argument('--vis-only', action='store_true', help='仅对已有标注做可视化')
    args = parser.parse_args()

    if args.vis_only:
        visualize()
    else:
        success = annotate()
        if success:
            visualize()


if __name__ == '__main__':
    main()

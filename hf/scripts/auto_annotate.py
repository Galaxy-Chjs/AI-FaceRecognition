"""
自动标注脚本 (半自动, 需人工复核)
为自收集 20 类测试集自动生成 annotations.jsonl 标注文件.

用法:
    python auto_annotate.py

标注格式:
    {"image": "test/images/p01_t01.jpg", "image_type": "single",
     "faces": [{"identity_id": "p01", "bbox": [x, y, w, h]}]}

注意事项:
    - 多人照中非 20 类身份的人物标注为 "unknown"
    - 生成后务必人工检查并修正标注
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TEST_IMAGES_DIR, ANNOTATIONS_PATH
from core.face_engine import FaceEngine


def annotate():
    if not os.path.isdir(TEST_IMAGES_DIR):
        print(f"[错误] 测试集图片目录不存在: {TEST_IMAGES_DIR}")
        print("  请先将测试集图片放置到 dataset/test/images/ 目录下")
        return

    engine = FaceEngine()
    annotations = []

    image_files = sorted([
        f for f in os.listdir(TEST_IMAGES_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ])

    print(f"共发现 {len(image_files)} 张测试图片, 开始自动标注...")

    for img_name in image_files:
        img_path = os.path.join(TEST_IMAGES_DIR, img_name)

        # 从文件名推断主身份 (如 p01_t01.jpg -> p01)
        base_name = os.path.splitext(img_name)[0]
        parts = base_name.split('_')
        main_identity = parts[0] if len(parts) >= 2 else "unknown"

        # 检测人脸
        bboxes, faces = engine.detect_faces(img_path)

        face_annotations = []
        if len(bboxes) > 0:
            for i, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = bbox
                # 转换为 [x, y, width, height] 格式, 取整
                w = int(x2 - x1)
                h = int(y2 - y1)
                face_annotations.append({
                    "identity_id": main_identity if i == 0 else "unknown",
                    "bbox": [int(x1), int(y1), w, h],
                })

        image_type = "single" if len(face_annotations) <= 1 else "multi"

        annotations.append({
            "image": f"test/images/{img_name}",
            "image_type": image_type,
            "faces": face_annotations,
        })

        face_count = len(face_annotations)
        print(f"  {img_name}: 检测到 {face_count} 张人脸, 类型={image_type}")

    # 保存标注文件
    os.makedirs(os.path.dirname(ANNOTATIONS_PATH), exist_ok=True)
    with open(ANNOTATIONS_PATH, 'w', encoding='utf-8') as f:
        for ann in annotations:
            f.write(json.dumps(ann, ensure_ascii=False) + '\n')

    print(f"\n标注完成! 共标注 {len(annotations)} 张图片")
    print(f"标注文件已保存至: {ANNOTATIONS_PATH}")
    print("\n[重要提醒] 请人工检查标注文件, 修正多人照中的人物身份标注!")


if __name__ == '__main__':
    annotate()

import os
import json
import cv2
from PIL import Image
from config import TEST_IMAGES_DIR, ANNOTATIONS_PATH
from utils import FaceEngine

def generate_annotations():
    engine = FaceEngine()
    image_files = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    with open(ANNOTATIONS_PATH, 'w', encoding='utf-8') as f:
        for filename in sorted(image_files):
            img_path = os.path.join(TEST_IMAGES_DIR, filename)
            
            # 解析图片属于哪个主身份 (例如从 p01_t01.jpg 中提取 p01)
            main_identity = filename.split('_')[0]
            
            # 读取图片获取宽高
            img = Image.open(img_path).convert('RGB')
            bboxes, _ = engine.detect_faces(img)
            
            if bboxes is None:
                continue
                
            faces_list = []
            image_type = "single" if len(bboxes) == 1 else "multi"
            
            for i, bbox in enumerate(bboxes):
                # 转换到作业要求的整数绝对坐标 [x, y, width, height]
                x1, y1, x2, y2 = map(int, bbox)
                w = x2 - x1
                h = y2 - y1
                
                # 简单逻辑：如果是单人照，那就是主身份；如果是多人照，第一个可能为主身份，其余为 unknown
                # 建议生成后，你根据实际照片打开这个 jsonl 微调一下具体某个人脸的 identity_id
                if image_type == "single":
                    identity = main_identity
                else:
                    identity = main_identity if i == 0 else "unknown"
                    
                faces_list.append({
                    "identity_id": identity,
                    "bbox": [x1, y1, w, h]
                })
            
            annotation_line = {
                "image": f"test/images/{filename}",
                "image_type": image_type,
                "faces": faces_list
            }
            f.write(json.dumps(annotation_line, ensure_ascii=False) + '\n')
            print(f"已标注: {filename}，检测到人脸数: {len(bboxes)}")

if __name__ == '__main__':
    # 确保测试集图片放好后再运行
    generate_annotations()
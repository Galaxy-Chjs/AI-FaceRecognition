"""
人脸识别系统 - Gradio 前端演示界面
支持图片上传、多人脸检测、身份识别、结果可视化.

用法:
    python app.py
    # 浏览器访问 http://localhost:7860
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from PIL import Image
import gradio as gr
from config import GALLERY_20_PATH, GALLERY_CELEBA_PATH, THRESHOLD
from core.face_engine import FaceEngine


def load_gallery():
    """加载所有可用的底库"""
    import torch
    gallery = {}
    for path in [GALLERY_20_PATH, GALLERY_CELEBA_PATH]:
        if os.path.exists(path):
            gallery.update(torch.load(path, weights_only=False))
    return gallery


# 初始化
engine = FaceEngine()
gallery = load_gallery()
print(f"底库加载完成, 共 {len(gallery)} 个身份")


def predict_and_draw(input_img):
    """
    核心回调: 输入 PIL 图片, 输出标注了人脸框和身份标签的图片.
    """
    if input_img is None:
        return None

    # Gradio 传入 RGB PIL Image, 转为 BGR 用于 OpenCV 画图
    img_bgr = cv2.cvtColor(np.array(input_img), cv2.COLOR_RGB2BGR)

    # 1. 人脸检测
    bboxes, faces = engine.detect_faces(input_img)

    if len(bboxes) == 0:
        cv2.putText(img_bgr, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. 提取特征
    embeddings = engine.get_embeddings(faces)

    # 3. 逐脸匹配并画框
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, bbox)

        identity = "unknown"
        confidence = 0.0

        if embeddings is not None and i < len(embeddings):
            face_emb = embeddings[i]
            best_sim = -1.0
            best_id = "unknown"

            for gal_id, gal_emb in gallery.items():
                # embedding 已 L2 归一化, 余弦相似度 = 点积
                sim = float(np.dot(face_emb, gal_emb))
                if sim > best_sim:
                    best_sim = sim
                    best_id = gal_id

            if best_sim >= THRESHOLD:
                identity = best_id
                confidence = best_sim

        # 绿色 = 已知身份, 红色 = unknown
        color = (0, 255, 0) if identity != "unknown" else (0, 0, 255)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

        # 标签
        label = f"{identity} ({confidence:.2f})" if identity != "unknown" else "unknown"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_bgr, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img_bgr, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# Gradio 界面
with gr.Blocks(title="人脸识别系统") as demo:
    gr.Markdown("# 人脸识别系统")
    gr.Markdown("本地运行的人脸识别系统, 支持多人脸检测与身份识别. 上传图片即可体验.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="上传图片")
            submit_btn = gr.Button("开始识别", variant="primary")
        with gr.Column():
            output_image = gr.Image(label="识别结果")

    submit_btn.click(fn=predict_and_draw, inputs=input_image, outputs=output_image)

    gr.Markdown(f"### 系统信息")
    gr.Markdown(f"- 底库身份数: **{len(gallery)}**")
    gr.Markdown(f"- 相似度阈值: **{THRESHOLD}**")
    gr.Markdown(f"- 检测模型: MTCNN | 特征模型: InceptionResnetV1 (VGGFace2)")


if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=7860)

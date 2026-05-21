import os
import cv2
import torch
import numpy as np
from PIL import Image
import gradio as gr
from scipy.spatial.distance import cosine
from config import GALLERY_PATH, CELEBA_GALLERY_PATH, THRESHOLD
from utils import FaceEngine

# 初始化检测引擎
engine = FaceEngine()

# 全局加载预先构建好的底库（这里同时包容自收集20类和CelebA，方便现场演示）
gallery = {}
if os.path.exists(GALLERY_PATH):
    gallery.update(torch.load(GALLERY_PATH))
if os.path.exists(CELEBA_GALLERY_PATH):
    gallery.update(torch.load(CELEBA_GALLERY_PATH))

def predict_and_draw(input_img):
    """
    前端核心回调：输入PIL图片，输出画好框和标签的 BGR 图像
    """
    if input_img is None:
        return None
        
    # Gradio 传入的是 PIL 图像，转为 OpenCV 格式方便画图
    img_cv = cv2.cvtColor(np.array(input_img), cv2.COLOR_RGB2BGR)
    
    # 1. 运行多目标人脸检测
    bboxes, cropped_faces = engine.detect_faces(input_img)
    
    if bboxes is dict or bboxes is None or len(bboxes) == 0:
        return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        
    # 2. 提取图像中所有人脸的特征向量
    embeddings = engine.get_embeddings(cropped_faces)
    
    # 3. 循环对每张人脸进行相似度底库匹配并画框
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, bbox)
        
        identity = "unknown"
        max_sim = -1.0
        
        if embeddings is not None and i < len(embeddings):
            face_emb = embeddings[i]
            # 遍历底库比对
            for gal_id, gal_emb in gallery.items():
                sim = 1 - cosine(face_emb, gal_emb)
                if sim > max_sim:
                    max_sim = sim
                    best_match = gal_id
            
            if max_sim >= THRESHOLD:
                identity = best_match

        # 4. 根据识别结果绘制可视化界面（现场检查和录屏的核心）
        # 绿色框代表已知身份，红色框代表 unknown
        color = (0, 255, 0) if identity != "unknown" else (0, 0, 255)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 3)
        
        # 标签文本：包含身份 ID 和置信度（如果是未知则仅显示 unknown）
        label = f"{identity}" if identity == "unknown" else f"{identity} ({max_sim:.2f})"
        
        # 绘制背景底板使文字更清晰
        cv2.rectangle(img_cv, (x1, y1 - 25), (x1 + len(label)*11, y1), color, -1)
        cv2.putText(img_cv, label, (x1 + 5, y1 - 7), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    
    # 转回 RGB 格式供给 Gradio 展示
    return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

# 4. 搭建前端 Gradio 布局
with gr.Blocks(title="本地人脸识别系统 - 大作业演示平台") as demo:
    gr.Markdown("# 🤖 人脸识别系统本地演示平台")
    gr.Markdown("本次系统完全在本地运行（禁止调用云端 API）。支持输入单人/多人照，输出精准人脸框及身份编号。")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="上传待测试人脸图片")
            submit_btn = gr.Button("开始本地检测与身份识别", variant="primary")
        with gr.Column():
            output_image = gr.Image(label="识别结果输出（包含人脸框与身份ID）")
            
    submit_btn.click(fn=predict_and_draw, inputs=input_image, outputs=output_image)
    
    gr.Markdown("### 💡 课堂演示操作指引")
    gr.Markdown("1. 确保已运行过注册脚本构建特征底库。\n2. 现场抽取图片拖入上方“上传”区域，点击按钮即可查看识别数量和对应人脸标签。")

if __name__ == '__main__':
    # 启动本地服务器，会在终端打印出本地访问链接 (如 http://127.0.0.1:7860)
    demo.launch(server_name="0.0.0.0", server_port=7860)
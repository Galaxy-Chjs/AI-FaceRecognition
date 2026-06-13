import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import torch
import numpy as np
from PIL import Image
import gradio as gr
from config import GALLERY_20_PATH, GALLERY_CELEBA_PATH, THRESHOLD, DEVICE
from core.face_engine import FaceEngine

engine = FaceEngine()


def load_gallery(mode):
    gallery = {}
    if mode in ("CelebA 100 类", "全部 (120 类)"):
        if os.path.exists(GALLERY_CELEBA_PATH):
            gallery.update(torch.load(GALLERY_CELEBA_PATH, weights_only=False))
    if mode in ("自收集 20 类", "全部 (120 类)"):
        if os.path.exists(GALLERY_20_PATH):
            gallery.update(torch.load(GALLERY_20_PATH, weights_only=False))
    return gallery


def draw_face_box(img, x1, y1, x2, y2, identity, confidence, img_h):
    if identity != "unknown":
        box_color = (46, 204, 113)
        text_bg = (39, 174, 96)
    else:
        box_color = (231, 76, 60)
        text_bg = (192, 57, 43)

    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

    label = f"{identity}  {confidence:.0%}" if identity != "unknown" else "unknown"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(0.6, img_h / 800))
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    pad = 4
    if y1 - th - pad * 2 > 0:
        label_y1 = y1 - th - pad * 2
        label_y2 = y1
        text_y = y1 - pad
    else:
        label_y1 = y2
        label_y2 = y2 + th + pad * 2
        text_y = y2 + th + pad

    lx1 = max(0, x1)
    lx2 = min(img.shape[1], x1 + tw + pad * 2)
    cv2.rectangle(img, (lx1, label_y1), (lx2, label_y2), text_bg, -1)
    cv2.putText(img, label, (lx1 + pad, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img


def predict_and_draw(input_img, mode):
    if input_img is None:
        return None, "请上传图片"

    gallery = load_gallery(mode)
    if not gallery:
        return np.array(input_img), "底库为空, 请先构建底库"

    gallery_ids = list(gallery.keys())
    gallery_matrix = np.stack([gallery[gid] for gid in gallery_ids])

    img_rgb = np.array(input_img, dtype=np.uint8)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_h, img_w = img_bgr.shape[:2]

    bboxes, faces, probs = engine.detect_faces(input_img, min_confidence=0.9)

    if len(bboxes) == 0:
        info = "未检测到人脸"
        cv2.putText(img_bgr, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (231, 76, 60), 2, cv2.LINE_AA)
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), info

    embeddings = engine.get_embeddings(faces)

    results = []
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, bbox)
        identity = "unknown"
        confidence = 0.0

        if embeddings is not None and i < len(embeddings):
            sims = gallery_matrix @ embeddings[i]
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            if best_sim >= THRESHOLD:
                identity = gallery_ids[best_idx]
                confidence = best_sim

        draw_face_box(img_bgr, x1, y1, x2, y2, identity, confidence, img_h)
        results.append({"identity": identity, "confidence": confidence, "bbox": [x1, y1, x2, y2]})

    known = [r for r in results if r["identity"] != "unknown"]
    unknown_count = len(results) - len(known)
    lines = [f"检测到 {len(results)} 张人脸, 识别 {len(known)} 人, 未知 {unknown_count} 人"]
    for r in known:
        lines.append(f"  {r['identity']}: 置信度 {r['confidence']:.2%}")
    info = "\n".join(lines)

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), info


CSS = """
.main-title { text-align: center; margin-bottom: 0.5em; }
.result-box { padding: 12px; border-radius: 8px; background: #f8f9fa; }
"""

with gr.Blocks(title="人脸识别系统") as demo:
    gr.Markdown("# 人脸识别系统", elem_classes="main-title")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="上传测试图片", height=400)
            submit_btn = gr.Button("开始识别", variant="primary", size="lg")
            mode = gr.Radio(
                choices=["CelebA 100 类", "自收集 20 类", "全部 (120 类)"],
                value="自收集 20 类",
                label="选择底库",
            )

        with gr.Column(scale=1):
            output_image = gr.Image(label="识别结果", height=400)
            output_info = gr.Textbox(label="识别结果", lines=4, interactive=False)

    submit_btn.click(
        fn=predict_and_draw,
        inputs=[input_image, mode],
        outputs=[output_image, output_info],
    )

    gr.Markdown("---")
    gr.Markdown(f"""
### 系统信息
| 项目 | 值 |
|------|-----|
| 运行设备 | {DEVICE} |
| 人脸检测 | MTCNN|
| 特征提取 | InceptionResnetV1 VGGFace2|
| 相似度阈值 | {THRESHOLD} |
""")


if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=7860, css=CSS)

import os
import torch
import numpy as np


def build_gallery(engine, register_dir, save_path):
    gallery = {}

    if not os.path.isdir(register_dir):
        return gallery

    identities = sorted([
        d for d in os.listdir(register_dir)
        if os.path.isdir(os.path.join(register_dir, d)) and not d.startswith('.')
    ])

    for identity_id in identities:
        id_dir = os.path.join(register_dir, identity_id)
        embeddings_list = []

        for img_name in sorted(os.listdir(id_dir)):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            img_path = os.path.join(id_dir, img_name)
            bboxes, faces, _ = engine.detect_faces(img_path)
            if faces is None:
                continue
            emb = engine.get_embeddings(faces)
            if emb is not None and len(emb) > 0:
                embeddings_list.append(emb[0])

        if embeddings_list:
            avg_emb = np.mean(embeddings_list, axis=0)
            avg_emb = avg_emb / np.linalg.norm(avg_emb)
            gallery[identity_id] = avg_emb

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(gallery, save_path)
    return gallery

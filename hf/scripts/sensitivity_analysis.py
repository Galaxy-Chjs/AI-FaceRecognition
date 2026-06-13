import os
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    REGISTERED_DIR, TEST_IMAGES_DIR, GALLERY_20_PATH, ANNOTATIONS_PATH,
)
from core.face_engine import FaceEngine
from core.gallery_builder import build_gallery
from evaluate import (
    _compute_iou, _xywh_to_x1y1x2y2, _match_detections_to_gt,
)


def evaluate_at_threshold(engine, gallery, test_images_dir, annotations_path, threshold):
    if not gallery or not os.path.isfile(annotations_path):
        return None

    gallery_ids = list(gallery.keys())
    gallery_matrix = np.stack([gallery[gid] for gid in gallery_ids])

    annotations = {}
    with open(annotations_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ann = json.loads(line)
            annotations[os.path.basename(ann['image'])] = ann['faces']

    EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

    def resolve_path(directory, filename):
        path = os.path.join(directory, filename)
        if os.path.isfile(path):
            return path
        stem = os.path.splitext(filename)[0]
        for ext in EXTENSIONS:
            alt = os.path.join(directory, stem + ext)
            if os.path.isfile(alt):
                return alt
        return None

    face_correct = face_total = 0
    img_strict_correct = img_strict_total = 0
    unknown_as_known = known_as_unknown = 0

    for name in sorted(annotations.keys()):
        path = resolve_path(test_images_dir, name)
        if not path:
            continue
        gt_faces = annotations[name]
        bboxes, faces, _ = engine.detect_faces(path)
        if faces is None or len(bboxes) == 0:
            face_total += len(gt_faces)
            img_strict_total += 1
            continue
        embs = engine.get_embeddings(faces)
        if embs is None or len(embs) == 0:
            face_total += len(gt_faces)
            img_strict_total += 1
            continue

        matches = _match_detections_to_gt(bboxes, gt_faces)
        all_faces_correct = True

        for gt_id, det_idx in matches:
            face_total += 1
            if det_idx is None:
                all_faces_correct = False
                continue
            sims = gallery_matrix @ embs[det_idx]
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            predicted = gallery_ids[best_idx] if best_sim >= threshold else "unknown"

            if predicted == gt_id:
                face_correct += 1
            else:
                all_faces_correct = False
                if gt_id == "unknown" and predicted != "unknown":
                    unknown_as_known += 1
                elif gt_id != "unknown" and predicted == "unknown":
                    known_as_unknown += 1

        img_strict_total += 1
        if all_faces_correct:
            img_strict_correct += 1

    return {
        'threshold': threshold,
        'face_accuracy': (face_correct / face_total * 100) if face_total > 0 else 0,
        'face_correct': face_correct,
        'face_total': face_total,
        'strict_accuracy': (img_strict_correct / img_strict_total * 100) if img_strict_total > 0 else 0,
        'strict_correct': img_strict_correct,
        'strict_total': img_strict_total,
        'unknown_as_known': unknown_as_known,
        'known_as_unknown': known_as_unknown,
    }


def main():
    engine = FaceEngine()
    gallery = build_gallery(engine, REGISTERED_DIR, GALLERY_20_PATH)
    if not gallery:
        print("Gallery is empty.")
        return

    thresholds = np.arange(0.50, 0.801, 0.01)
    results = []

    print(f"{'Threshold':>10s}  {'Face Acc':>10s}  {'Strict Acc':>10s}  {'Unk→Known':>10s}  {'Known→Unk':>10s}")
    print("-" * 58)

    for t in thresholds:
        r = evaluate_at_threshold(engine, gallery, TEST_IMAGES_DIR, ANNOTATIONS_PATH, t)
        if r:
            results.append(r)
            print(f"{r['threshold']:10.2f}  {r['face_accuracy']:9.2f}%  {r['strict_accuracy']:9.2f}%  {r['unknown_as_known']:10d}  {r['known_as_unknown']:10d}")

    best_face = max(results, key=lambda r: r['face_accuracy'])
    best_strict = max(results, key=lambda r: r['strict_accuracy'])

    print(f"\nBest face-level acc:   threshold={best_face['threshold']:.2f}  acc={best_face['face_accuracy']:.2f}%")
    print(f"Best strict acc:       threshold={best_strict['threshold']:.2f}  acc={best_strict['strict_accuracy']:.2f}%")

    ts = [float(r['threshold']) for r in results]
    face_accs = [float(r['face_accuracy']) for r in results]
    strict_accs = [float(r['strict_accuracy']) for r in results]
    unk_as_known = [int(r['unknown_as_known']) for r in results]
    known_as_unk = [int(r['known_as_unknown']) for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle('Similarity Threshold Sensitivity Analysis (Custom 20 Classes)', fontsize=14)

    ax1.plot(ts, face_accs, 'b-o', markersize=3, label='Face-level accuracy')
    ax1.plot(ts, strict_accs, 'r-s', markersize=3, label='Strict image-level accuracy')
    ax1.axvline(x=float(best_face['threshold']), color='b', linestyle='--', alpha=0.5,
                label=f'Best face threshold ({best_face["threshold"]:.2f})')
    ax1.axvline(x=float(best_strict['threshold']), color='r', linestyle='--', alpha=0.5,
                label=f'Best strict threshold ({best_strict["threshold"]:.2f})')
    ax1.axvline(x=0.60, color='gray', linestyle=':', alpha=0.7, label='Current threshold (0.60)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(80, 102)
    ax1.legend(loc='lower left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(ts, unk_as_known, 'orange', marker='x', markersize=4, label='Unknown->Known (false accept)')
    ax2.plot(ts, known_as_unk, 'purple', marker='x', markersize=4, label='Known->Unknown (false reject)')
    ax2.fill_between(ts, unk_as_known, alpha=0.2, color='orange')
    ax2.fill_between(ts, known_as_unk, alpha=0.2, color='purple')
    ax2.set_xlabel('Similarity Threshold')
    ax2.set_ylabel('Error Count')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'threshold_sensitivity.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {out_path}")
    plt.close()


if __name__ == '__main__':
    main()

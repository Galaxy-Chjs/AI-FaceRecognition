import argparse
import os
import json
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    CELEBA_REGISTER_DIR, CELEBA_TEST_DIR, GALLERY_CELEBA_PATH,
    REGISTERED_DIR, TEST_IMAGES_DIR, GALLERY_20_PATH,
    ANNOTATIONS_PATH, THRESHOLD,
)
from core.face_engine import FaceEngine
from core.gallery_builder import build_gallery


def _compute_iou(bbox_a, bbox_b):
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _xywh_to_x1y1x2y2(bbox):
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]


def _match_detections_to_gt(det_bboxes, gt_faces):
    if len(det_bboxes) == 0:
        return [(gt['identity_id'], None) for gt in gt_faces]

    gt_bboxes = [_xywh_to_x1y1x2y2(gt['bbox']) for gt in gt_faces]
    iou_matrix = np.zeros((len(gt_faces), len(det_bboxes)))
    for i, gt_box in enumerate(gt_bboxes):
        for j, det_box in enumerate(det_bboxes):
            iou_matrix[i, j] = _compute_iou(gt_box, det_box.tolist())

    matched = []
    used_gt, used_det = set(), set()
    for flat_idx in np.argsort(iou_matrix.ravel())[::-1]:
        gt_idx = flat_idx // len(det_bboxes)
        det_idx = flat_idx % len(det_bboxes)
        if gt_idx in used_gt or det_idx in used_det:
            continue
        if iou_matrix[gt_idx, det_idx] < 0.3:
            break
        matched.append((gt_idx, det_idx))
        used_gt.add(gt_idx)
        used_det.add(det_idx)

    matched_map = {gt_idx: det_idx for gt_idx, det_idx in matched}
    return [(gt['identity_id'], matched_map.get(i)) for i, gt in enumerate(gt_faces)]


def _print_report(result, dataset_name):
    if result is None:
        return
    print(f"\n  {dataset_name}")
    print(f"  {'─'*40}")
    print(f"  Top-1 = {result['accuracy']:.2f}%  ({result['correct']}/{result['total']})")
    if result['no_face_count']:
        print(f"  No face: {result['no_face_count']}")
    if result['fail_examples']:
        print(f"  Failures:")
        for ex in result['fail_examples']:
            if len(ex) == 4:
                name, true, pred, conf = ex
                print(f"    {name}  {true}→{pred}  sim={conf:.4f}")
            else:
                print(f"    {ex[0]}  {ex[1]}→{ex[2]}")


def _print_face_report(result, dataset_name):
    if result is None:
        return
    print(f"\n  {dataset_name} (face-level)")
    print(f"  {'─'*40}")
    print(f"  Face acc  = {result['face_accuracy']:.2f}%  ({result['face_correct']}/{result['face_total']})")
    print(f"  Strict acc= {result['img_strict_accuracy']:.2f}%  ({result['img_strict_correct']}/{result['img_strict_total']})")
    if result['no_face_count']:
        print(f"  No face: {result['no_face_count']}")
    if result['fail_examples']:
        print(f"  Failures:")
        for ex in result['fail_examples']:
            if len(ex) == 5:
                name, true, pred, conf, _ = ex
                print(f"    {name}  {true}→{pred}  sim={conf:.4f}")
            elif len(ex) == 4:
                name, true, pred, conf = ex
                print(f"    {name}  {true}→{pred}  sim={conf:.4f}")


def _run_evaluation(engine, gallery, image_list, max_examples=5):
    if not gallery:
        return None
    gallery_ids = list(gallery.keys())
    gallery_matrix = np.stack([gallery[gid] for gid in gallery_ids])
    correct = total = no_face_count = 0
    success_examples, fail_examples = [], []

    for img_path, img_name, identity_id in image_list:
        total += 1
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

        found = False
        best_confidence = -1.0
        any_predicted_id, any_best_sim = None, -1.0
        for face_emb in embs:
            sims = gallery_matrix @ face_emb
            idx = int(np.argmax(sims))
            sim = float(sims[idx])
            pred = gallery_ids[idx]
            if sim > any_best_sim:
                any_best_sim, any_predicted_id = sim, pred
            if pred == identity_id:
                found = True
                if sim > best_confidence:
                    best_confidence = sim

        if found:
            correct += 1
            if len(success_examples) < max_examples:
                success_examples.append((img_name, identity_id, best_confidence))
        else:
            if len(fail_examples) < max_examples:
                fail_examples.append((img_name, identity_id, any_predicted_id, any_best_sim))

    accuracy = (correct / total * 100) if total > 0 else 0.0
    return {
        'accuracy': accuracy, 'correct': correct, 'total': total,
        'no_face_count': no_face_count,
        'success_examples': success_examples, 'fail_examples': fail_examples,
    }


def evaluate_top1(engine, gallery, test_dir, max_examples=5):
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
    image_list = []
    for img_name in sorted(os.listdir(test_images_dir)):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        base = os.path.splitext(img_name)[0]
        parts = base.split('_')
        if len(parts) < 2:
            continue
        identity_id = parts[0]
        img_path = os.path.join(test_images_dir, img_name)
        image_list.append((img_path, img_name, identity_id))
    return _run_evaluation(engine, gallery, image_list, max_examples)


def evaluate_celeba_strict(engine, gallery, test_dir, max_examples=5):
    if not gallery:
        return None
    gallery_ids = list(gallery.keys())
    gallery_matrix = np.stack([gallery[gid] for gid in gallery_ids])
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

    face_correct = face_total = 0
    img_strict_correct = img_strict_total = no_face_count = 0
    success_examples, fail_examples = [], []

    for img_path, img_name, identity_id in image_list:
        bboxes, faces, _ = engine.detect_faces(img_path)
        if faces is None or len(bboxes) == 0:
            no_face_count += 1
            face_total += 1; img_strict_total += 1
            if len(fail_examples) < max_examples:
                fail_examples.append((img_name, identity_id, "none", 0.0, "no face"))
            continue
        embs = engine.get_embeddings(faces)
        if embs is None or len(embs) == 0:
            no_face_count += 1
            face_total += 1; img_strict_total += 1
            if len(fail_examples) < max_examples:
                fail_examples.append((img_name, identity_id, "none", 0.0, "no emb"))
            continue

        all_correct = True
        best_confidence = -1.0
        worst_predicted, worst_sim = None, -1.0
        for face_emb in embs:
            sims = gallery_matrix @ face_emb
            idx = int(np.argmax(sims))
            sim = float(sims[idx])
            pred = gallery_ids[idx]
            face_total += 1
            if pred == identity_id:
                face_correct += 1
                if sim > best_confidence:
                    best_confidence = sim
            else:
                all_correct = False
                if sim > worst_sim:
                    worst_sim, worst_predicted = sim, pred

        img_strict_total += 1
        if all_correct:
            img_strict_correct += 1
            if len(success_examples) < max_examples:
                success_examples.append((img_name, identity_id, best_confidence))
        else:
            if len(fail_examples) < max_examples:
                fail_examples.append((img_name, identity_id, worst_predicted, worst_sim, "mismatch"))

    return {
        'face_accuracy': (face_correct / face_total * 100) if face_total > 0 else 0.0,
        'face_correct': face_correct, 'face_total': face_total,
        'img_strict_accuracy': (img_strict_correct / img_strict_total * 100) if img_strict_total > 0 else 0.0,
        'img_strict_correct': img_strict_correct, 'img_strict_total': img_strict_total,
        'no_face_count': no_face_count,
        'success_examples': success_examples, 'fail_examples': fail_examples,
    }


def evaluate_custom_face_level(engine, gallery, test_images_dir, annotations_path, max_examples=5):
    if not gallery:
        return None
    if not os.path.isfile(annotations_path):
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
    img_strict_correct = img_strict_total = no_face_count = 0
    success_examples, fail_examples = [], []
    image_list = []
    for name in sorted(annotations.keys()):
        path = resolve_path(test_images_dir, name)
        if path:
            image_list.append((path, name))

    for img_path, img_name in image_list:
        gt_faces = annotations[img_name]
        bboxes, faces, _ = engine.detect_faces(img_path)
        if faces is None or len(bboxes) == 0:
            no_face_count += 1
            face_total += len(gt_faces); img_strict_total += 1
            continue
        embs = engine.get_embeddings(faces)
        if embs is None or len(embs) == 0:
            no_face_count += 1
            face_total += len(gt_faces); img_strict_total += 1
            continue

        matches = _match_detections_to_gt(bboxes, gt_faces)
        all_faces_correct = True
        for gt_identity_id, det_idx in matches:
            face_total += 1
            if det_idx is None:
                all_faces_correct = False
                continue
            face_emb = embs[det_idx]
            sims = gallery_matrix @ face_emb
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            predicted = gallery_ids[best_idx] if best_sim >= THRESHOLD else "unknown"
            if predicted == gt_identity_id:
                face_correct += 1
                if len(success_examples) < max_examples:
                    success_examples.append((img_name, gt_identity_id, best_sim))
            else:
                all_faces_correct = False
                if len(fail_examples) < max_examples:
                    fail_examples.append((img_name, gt_identity_id, predicted, best_sim, "mismatch"))

        img_strict_total += 1
        if all_faces_correct:
            img_strict_correct += 1

    return {
        'face_accuracy': (face_correct / face_total * 100) if face_total > 0 else 0.0,
        'face_correct': face_correct, 'face_total': face_total,
        'img_strict_accuracy': (img_strict_correct / img_strict_total * 100) if img_strict_total > 0 else 0.0,
        'img_strict_correct': img_strict_correct, 'img_strict_total': img_strict_total,
        'no_face_count': no_face_count,
        'success_examples': success_examples, 'fail_examples': fail_examples,
    }


def run_celeba_eval(engine):
    gallery = build_gallery(engine, CELEBA_REGISTER_DIR, GALLERY_CELEBA_PATH)
    if not gallery:
        print("CelebA gallery is empty, skipping.")
        return None, None
    result_img = evaluate_top1(engine, gallery, CELEBA_TEST_DIR, max_examples=5)
    _print_report(result_img, "CelebA 100 (image-level)")
    result_face = evaluate_celeba_strict(engine, gallery, CELEBA_TEST_DIR, max_examples=5)
    _print_face_report(result_face, "CelebA 100")
    return result_img, result_face


def run_custom_eval(engine):
    if not os.path.isdir(REGISTERED_DIR) or not os.path.isdir(TEST_IMAGES_DIR):
        print("Custom dataset dirs not found, skipping.")
        return None, None
    gallery = build_gallery(engine, REGISTERED_DIR, GALLERY_20_PATH)
    if not gallery:
        print("Custom gallery is empty, skipping.")
        return None, None
    result_img = evaluate_custom_top1(engine, gallery, TEST_IMAGES_DIR, max_examples=5)
    _print_report(result_img, "Custom 20 (image-level)")
    if os.path.isfile(ANNOTATIONS_PATH):
        result_face = evaluate_custom_face_level(engine, gallery, TEST_IMAGES_DIR, ANNOTATIONS_PATH, max_examples=5)
        _print_face_report(result_face, "Custom 20")
    else:
        result_face = None
    return result_img, result_face


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--celeba', action='store_true')
    parser.add_argument('--custom', action='store_true')
    args = parser.parse_args()

    eval_celeba = args.celeba or (not args.celeba and not args.custom)
    eval_custom = args.custom or (not args.celeba and not args.custom)

    engine = FaceEngine()
    results = {}

    if eval_celeba:
        results['celeba'] = run_celeba_eval(engine)
    if eval_custom:
        results['custom'] = run_custom_eval(engine)

    print(f"\n{'='*50}")
    for name, (res_img, res_face) in results.items():
        label = "CelebA 100" if name == "celeba" else "Custom 20"
        if res_img:
            print(f"  {label:12s}  image-level Top-1 = {res_img['accuracy']:.2f}%  ({res_img['correct']}/{res_img['total']})")
        if res_face:
            print(f"  {'':12s}  face-level       = {res_face['face_accuracy']:.2f}%  ({res_face['face_correct']}/{res_face['face_total']})")
            print(f"  {'':12s}  strict image     = {res_face['img_strict_accuracy']:.2f}%  ({res_face['img_strict_correct']}/{res_face['img_strict_total']})")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()

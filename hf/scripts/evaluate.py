"""
评测入口脚本
评估 CelebA 100 类 和 自收集 20 类 数据集的 Top-1 识别准确率.

用法:
    python evaluate.py              # 同时评测两个数据集 (如果数据存在)
    python evaluate.py --celeba     # 仅评测 CelebA
    python evaluate.py --custom     # 仅评测自收集 20 类
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    CELEBA_REGISTER_DIR, CELEBA_TEST_DIR, GALLERY_CELEBA_PATH,
    REGISTERED_DIR, TEST_IMAGES_DIR, GALLERY_20_PATH,
)
from core.face_engine import FaceEngine
from core.gallery_builder import build_gallery
from core.evaluator import evaluate_top1, evaluate_custom_top1, print_report


def run_celeba_eval(engine):
    """构建 CelebA 底库并评测"""
    print("\n" + "="*50)
    print("  CelebA 100 类 数据集评测")
    print("="*50)

    # 构建底库
    gallery = build_gallery(engine, CELEBA_REGISTER_DIR, GALLERY_CELEBA_PATH)
    if not gallery:
        print("[跳过] CelebA 底库为空, 无法评测")
        return

    # 评测
    result = evaluate_top1(engine, gallery, CELEBA_TEST_DIR, max_examples=5)
    print_report(result, "CelebA 100 类")
    return result


def run_custom_eval(engine):
    """构建自收集 20 类底库并评测"""
    print("\n" + "="*50)
    print("  自收集 20 类 数据集评测")
    print("="*50)

    if not os.path.isdir(REGISTERED_DIR):
        print(f"[跳过] 自收集注册集目录不存在: {REGISTERED_DIR}")
        print("  请先收集数据并放置到 dataset/registered/ 目录下")
        return

    if not os.path.isdir(TEST_IMAGES_DIR):
        print(f"[跳过] 自收集测试集目录不存在: {TEST_IMAGES_DIR}")
        print("  请先收集数据并放置到 dataset/test/images/ 目录下")
        return

    # 构建底库
    gallery = build_gallery(engine, REGISTERED_DIR, GALLERY_20_PATH)
    if not gallery:
        print("[跳过] 自收集底库为空, 无法评测")
        return

    # 评测 (自收集数据: 图片平铺在 images/ 下, 身份从文件名解析)
    result = evaluate_custom_top1(engine, gallery, TEST_IMAGES_DIR, max_examples=5)
    print_report(result, "自收集 20 类")
    return result


def main():
    parser = argparse.ArgumentParser(description="人脸识别系统评测")
    parser.add_argument('--celeba', action='store_true', help='仅评测 CelebA 数据集')
    parser.add_argument('--custom', action='store_true', help='仅评测自收集 20 类数据集')
    args = parser.parse_args()

    # 默认: 两个都评测
    eval_celeba = args.celeba or (not args.celeba and not args.custom)
    eval_custom = args.custom or (not args.celeba and not args.custom)

    engine = FaceEngine()
    results = {}

    if eval_celeba:
        results['celeba'] = run_celeba_eval(engine)

    if eval_custom:
        results['custom'] = run_custom_eval(engine)

    # 汇总
    print("\n" + "="*50)
    print("  评测汇总")
    print("="*50)
    for name, res in results.items():
        if res:
            print(f"  {name}: Top-1 准确率 = {res['accuracy']:.2f}%  ({res['correct']}/{res['total']})")
    print("="*50)


if __name__ == '__main__':
    main()

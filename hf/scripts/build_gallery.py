"""
底库构建入口脚本
支持构建 CelebA 100 类底库 和 自收集 20 类底库.

用法:
    python build_gallery.py              # 同时构建两个底库 (如果数据存在)
    python build_gallery.py --celeba     # 仅构建 CelebA 底库
    python build_gallery.py --custom     # 仅构建自收集 20 类底库
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    CELEBA_REGISTER_DIR, GALLERY_CELEBA_PATH,
    REGISTERED_DIR, GALLERY_20_PATH,
)
from core.face_engine import FaceEngine
from core.gallery_builder import build_gallery


def main():
    parser = argparse.ArgumentParser(description="构建人脸识别底库")
    parser.add_argument('--celeba', action='store_true', help='仅构建 CelebA 100 类底库')
    parser.add_argument('--custom', action='store_true', help='仅构建自收集 20 类底库')
    args = parser.parse_args()

    # 默认: 两个都构建
    build_celeba = args.celeba or (not args.celeba and not args.custom)
    build_custom = args.custom or (not args.celeba and not args.custom)

    engine = FaceEngine()

    if build_celeba:
        print("\n" + "="*50)
        print("  构建 CelebA 100 类底库")
        print("="*50)
        if os.path.isdir(CELEBA_REGISTER_DIR):
            build_gallery(engine, CELEBA_REGISTER_DIR, GALLERY_CELEBA_PATH)
        else:
            print(f"[跳过] CelebA 注册集目录不存在: {CELEBA_REGISTER_DIR}")

    if build_custom:
        print("\n" + "="*50)
        print("  构建自收集 20 类底库")
        print("="*50)
        if os.path.isdir(REGISTERED_DIR):
            build_gallery(engine, REGISTERED_DIR, GALLERY_20_PATH)
        else:
            print(f"[跳过] 自收集注册集目录不存在: {REGISTERED_DIR}")
            print("  请先收集数据并放置到 dataset/registered/ 目录下")


if __name__ == '__main__':
    main()

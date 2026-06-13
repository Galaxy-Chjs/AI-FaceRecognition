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
    parser = argparse.ArgumentParser()
    parser.add_argument('--celeba', action='store_true')
    parser.add_argument('--custom', action='store_true')
    args = parser.parse_args()

    build_celeba = args.celeba or (not args.celeba and not args.custom)
    build_custom = args.custom or (not args.celeba and not args.custom)

    engine = FaceEngine()

    if build_celeba:
        if os.path.isdir(CELEBA_REGISTER_DIR):
            gallery = build_gallery(engine, CELEBA_REGISTER_DIR, GALLERY_CELEBA_PATH)
            print(f"CelebA gallery: {len(gallery)} identities -> {GALLERY_CELEBA_PATH}")
        else:
            print(f"CelebA register dir not found: {CELEBA_REGISTER_DIR}")

    if build_custom:
        if os.path.isdir(REGISTERED_DIR):
            gallery = build_gallery(engine, REGISTERED_DIR, GALLERY_20_PATH)
            print(f"Custom gallery: {len(gallery)} identities -> {GALLERY_20_PATH}")
        else:
            print(f"Custom register dir not found: {REGISTERED_DIR}")


if __name__ == '__main__':
    main()

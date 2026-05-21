"""
人脸识别系统 - 全局配置文件
"""
import os
import torch

# ==================== 运行设备 ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== 目录路径 ====================
# 项目根目录 (hf/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- 自收集 20 类身份数据路径 ---
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
REGISTERED_DIR = os.path.join(DATASET_DIR, 'registered')
TEST_DIR = os.path.join(DATASET_DIR, 'test')
TEST_IMAGES_DIR = os.path.join(TEST_DIR, 'images')
ANNOTATIONS_PATH = os.path.join(TEST_DIR, 'annotations.jsonl')
IDENTITIES_CSV = os.path.join(DATASET_DIR, 'identities.csv')

# --- CelebA 100 类数据集路径 ---
CELEBA_DIR = os.path.join(BASE_DIR, 'celeba_100_identities_3reg_3test')
CELEBA_REGISTER_DIR = os.path.join(CELEBA_DIR, 'register')
CELEBA_TEST_DIR = os.path.join(CELEBA_DIR, 'test')

# --- 模型/底库存储路径 ---
MODEL_DIR = os.path.join(BASE_DIR, 'scripts', 'model')
GALLERY_20_PATH = os.path.join(MODEL_DIR, 'gallery_20classes.pt')
GALLERY_CELEBA_PATH = os.path.join(MODEL_DIR, 'gallery_celeba.pt')

# --- 结果输出路径 ---
RESULTS_DIR = os.path.join(BASE_DIR, 'scripts', 'results')

# ==================== 人脸识别核心参数 ====================
IMAGE_SIZE = 160          # MTCNN 裁剪输出尺寸 (FaceNet 标准)
EMBEDDING_DIM = 512       # InceptionResnetV1 输出维度
THRESHOLD = 0.65          # 余弦相似度阈值 (>= 此值判定为已知身份)

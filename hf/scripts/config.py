import os
import torch

# 运行设备检测
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 根目录与路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 1. 自收集20类身份数据路径 (注意你的目录是 datasets)
DATASETS_DIR = os.path.join(BASE_DIR, 'dataset')
REGISTERED_DIR = os.path.join(DATASETS_DIR, 'registered')
TEST_IMAGES_DIR = os.path.join(DATASETS_DIR, 'test', 'images')
ANNOTATIONS_PATH = os.path.join(DATASETS_DIR, 'test', 'annotations.jsonl')

# 2. CelebA 100类数据集路径
CELEBA_DIR = os.path.join(BASE_DIR, 'celeba_100_identities_3reg_3test')
CELEBA_REGISTER_DIR = os.path.join(CELEBA_DIR, 'register')
CELEBA_TEST_DIR = os.path.join(CELEBA_DIR, 'test')

# 3. 本地底库及模型存储
GALLERY_PATH = os.path.join(BASE_DIR, 'scripts', 'model', 'face_gallery.pt')
CELEBA_GALLERY_PATH = os.path.join(BASE_DIR, 'scripts', 'model', 'celeba_gallery.pt')

# 人脸识别核心超参数
THRESHOLD = 0.60  # 余弦相似度阈值
IMAGE_SIZE = 160  # FaceNet 默认输入尺寸
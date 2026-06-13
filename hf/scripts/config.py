import os
import yaml
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'config.yaml')
with open(_yaml_path, 'r', encoding='utf-8') as _f:
    _cfg = yaml.safe_load(_f)

_device_str = _cfg.get('device', 'auto')
DEVICE = torch.device('cuda' if _device_str == 'auto' and torch.cuda.is_available() else
                       'cuda' if _device_str == 'cuda' else 'cpu')

_paths = _cfg.get('paths', {})
DATASET_DIR = os.path.join(BASE_DIR, _paths.get('dataset_dir', 'dataset'))
REGISTERED_DIR = os.path.join(BASE_DIR, _paths.get('registered_dir', 'dataset/registered'))
TEST_DIR = os.path.join(BASE_DIR, _paths.get('test_dir', 'dataset/test'))
TEST_IMAGES_DIR = os.path.join(BASE_DIR, _paths.get('test_images_dir', 'dataset/test/images'))
ANNOTATIONS_PATH = os.path.join(BASE_DIR, _paths.get('annotations_path', 'dataset/annotations.jsonl'))
IDENTITIES_CSV = os.path.join(BASE_DIR, _paths.get('identities_csv', 'dataset/identities.csv'))
CELEBA_DIR = os.path.join(BASE_DIR, _paths.get('celeba_dir', 'celeba_100_identities_3reg_3test'))
CELEBA_REGISTER_DIR = os.path.join(BASE_DIR, _paths.get('celeba_register_dir', 'celeba_100_identities_3reg_3test/register'))
CELEBA_TEST_DIR = os.path.join(BASE_DIR, _paths.get('celeba_test_dir', 'celeba_100_identities_3reg_3test/test'))
MODEL_DIR = os.path.join(BASE_DIR, _paths.get('model_dir', 'scripts/model'))
GALLERY_20_PATH = os.path.join(BASE_DIR, _paths.get('gallery_20_path', 'scripts/model/gallery_20classes.pt'))
GALLERY_CELEBA_PATH = os.path.join(BASE_DIR, _paths.get('gallery_celeba_path', 'scripts/model/gallery_celeba.pt'))
RESULTS_DIR = os.path.join(BASE_DIR, _paths.get('results_dir', 'scripts/results'))

_recog = _cfg.get('recognition', {})
IMAGE_SIZE = _recog.get('image_size', 160)
EMBEDDING_DIM = _recog.get('embedding_dim', 512)
THRESHOLD = _recog.get('threshold', 0.60)

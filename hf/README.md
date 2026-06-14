# 人脸识别系统 - 人工智能实验大作业

基于 MTCNN + InceptionResnetV1(VGGFace2) 的本地人脸识别系统。

## 项目结构

```
├── dataset/                            # 自收集 20 类数据集
│   ├── identities.csv                  # 身份映射表
│   ├── annotations.jsonl               # 标注文件
│   ├── registered/                     # 注册集
│   └── test/images/                    # 测试集
├── celeba_100_identities_3reg_3test/   # CelebA 100 类数据集
│   ├── register/                       # 注册集
│   └── test/                           # 测试集
├── scripts/
│   ├── config/
│   │   └── config.yaml                 # 配置
│   ├── config.py                       # 配置加载
│   ├── core/
│   │   ├── face_engine.py              # 人脸检测与特征提取
│   │   └── gallery_builder.py          # 底库构建
│   ├── build_gallery.py                # 底库构建入口
│   ├── evaluate.py                     # 评测
│   ├── app.py                          # 前端
│   └── model/                          # 底库文件 (.pt)
├── requirements.txt
└── README.md
```

## 环境配置

```bash
conda create -n aiface python=3.11 -y
conda activate aiface
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

## 使用方法

### 1. 构建底库

```bash
cd scripts
python build_gallery.py
```

### 2. 评测准确率

```bash
python evaluate.py
```

评测输出两种策略的对比结果：

| 策略                  | 说明                                             |
| --------------------- | ------------------------------------------------ |
| image-level (OR-gate) | 多人照中任一人脸匹配目标身份即判正确             |
| face-level (逐脸)     | 每张人脸独立评判, 所有 GT 人脸均正确才算图片正确 |

逐脸评测使用 `annotations.jsonl` 中的逐脸标注, 通过 IoU 匹配检测框与 GT 框, 使用相似度阈值 (默认 0.60) 区分已知/未知身份。

### 3. 启动前端演示

```bash
python app.py
# 浏览器访问 http://localhost:7860
```

### 4. 修改配置

所有可调参数集中在 `config/config.yaml` 中, 修改后无需改动其他代码即可生效。

## 预训练模型

本系统使用 [facenet-pytorch](https://github.com/timesler/facenet-pytorch) 提供的预训练模型, 首次运行时自动下载并缓存到本地 (`~/.cache/torch/hub/checkpoints/`)。

| 模型                              | 用途                            | 下载链接                                                                                                                                        |
| --------------------------------- | ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| InceptionResnetV1 (VGGFace2)      | 特征提取, 输出 512 维 embedding | [20180402-114759-vggface2.pt](https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt) (~107MB)           |
| InceptionResnetV1 (CASIA-WebFace) | 备选特征模型                    | [20180408-102900-casia-webface.pt](https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180408-102900-casia-webface.pt) (~107MB) |
| MTCNN (P/R/ONet)                  | 人脸检测                        | 内置于 facenet-pytorch, 无需手动下载                                                                                                            |

如需手动下载, 可使用浏览器或 `wget`:

```bash
# InceptionResnetV1 VGGFace2 权重
wget https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt
```

## 技术方案

- **人脸检测**: MTCNN (Multi-task Cascaded Convolutional Networks)
- **特征提取**: InceptionResnetV1 (预训练于 VGGFace2, 输出 512 维 embedding)
- **匹配方式**: 余弦相似度, 阈值 0.60

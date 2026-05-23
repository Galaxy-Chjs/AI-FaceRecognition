# 人脸识别系统 - 人工智能实验大作业

基于 MTCNN + InceptionResnetV1 的本地人脸识别系统，支持多人脸检测与身份识别。

## 项目结构

```
├── dataset/                            # 自收集 20 类数据集
│   ├── identities.csv                  # 身份映射表
│   ├── registered/                     # 注册集 (每人至少 2 张)
│   └── test/                           # 测试集
│       ├── images/                     # 测试图片 (每人 3 张: 1 单人 + 2 多人)
│       ├── annotations.jsonl           # 标注文件 (bbox: [x, y, width, height])
│       └── review/                     # 标注可视化图片 (供人工审查)
├── celeba_100_identities_3reg_3test/   # CelebA 100 类数据集
│   ├── register/                       # 注册集 (每类 3 张)
│   └── test/                           # 测试集 (每类 3 张)
├── scripts/
│   ├── config.py                       # 全局配置
│   ├── core/
│   │   ├── face_engine.py              # 人脸检测与特征提取引擎
│   │   ├── gallery_builder.py          # 底库构建模块
│   │   └── evaluator.py               # 评测模块
│   ├── build_gallery.py                # 底库构建入口
│   ├── evaluate.py                     # 评测入口
│   ├── auto_annotate.py                # 自动标注 + 可视化工具
│   ├── visualize_annotations.py        # 标注可视化 (人工审查)
│   ├── app.py                          # Gradio 前端演示
│   ├── model/                          # 底库文件 (.pt)
│   └── results/                        # 评测结果输出
├── requirements.txt                    # Python 依赖
└── README.md                           # 本文件
```

## 环境配置

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 构建底库

```bash
cd scripts

# 构建 CelebA 100 类底库
python build_gallery.py --celeba

# 构建自收集 20 类底库 (需要先收集数据)
python build_gallery.py --custom

# 同时构建两个底库
python build_gallery.py
```

### 2. 评测准确率

```bash
cd scripts

# 评测 CelebA 100 类
python evaluate.py --celeba

# 评测自收集 20 类
python evaluate.py --custom

# 同时评测
python evaluate.py
```

### 3. 启动前端演示

```bash
cd scripts
python app.py
# 浏览器访问 http://localhost:7860
```

### 4. 自动标注 + 可视化 (自收集数据集)

```bash
cd scripts

# 自动生成标注 + 可视化 (一步完成)
python auto_annotate.py

# 仅对已有标注做可视化 (修改 annotations.jsonl 后重新查看)
python auto_annotate.py --vis-only
```

标注生成后，请检查 `dataset/test/review/` 中的可视化图片：
- 绿色框 = 已知身份 (p01-p20)，红色框 = unknown
- 确认 bbox 是否准确圈住了人脸
- 如有错误，手动修改 `annotations.jsonl` 后重新运行 `--vis-only`

## 技术方案

- **人脸检测**: MTCNN (Multi-task Cascaded Convolutional Networks)
- **特征提取**: InceptionResnetV1 (预训练于 VGGFace2, 输出 512 维 embedding)
- **匹配方式**: 余弦相似度, 阈值 0.65
- **全部本地运行, 不调用任何云端 API**

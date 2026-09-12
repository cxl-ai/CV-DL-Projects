# CV-DL-Projects - 计算机视觉与深度学习算法作品集

> **一套系统化的高水准计算机视觉 (CV) 与深度学习 (DL) 工业算法实战合集**  
> 涵盖经典图像处理、特征点匹配、形态学分割、车道线多项式拟合，到端到端字符识别、细粒度图像分类，以及基于 YOLOv8 的工业级多任务缺陷质检 Web 平台。

---

## 📌 目录索引与项目矩阵

### 一、 传统与融合计算机视觉 (`computer_vision/`)

| 项目序号 | 项目名称 | 核心算法与技术栈 | 适用场景与达成指标 | 详细文档 |
| :---: | :--- | :--- | :--- | :---: |
| **01** | [视频流字幕与弹幕特效](computer_vision/01_video_subtitle_stream/) | OpenCV 视频帧解析、PIL 抗锯齿中文字体渲染、线性位移插值 | 视频流字幕平滑滚屏、横向双轨弹幕动画渲染合成 | [📖 详细文档](computer_vision/01_video_subtitle_stream/README.md) |
| **02** | [自动驾驶车道线检测](computer_vision/02_lane_line_detection/) | Canny 边缘检测、IPM 逆透视变换、直方图寻峰、滑动窗口二次多项式拟合 | 车辆行驶车道定位、真实物理世界曲率半径与偏离度解算 | [📖 详细文档](computer_vision/02_lane_line_detection/README.md) |
| **03** | [特征点检测与Logo定位](computer_vision/03_logo_feature_matching/) | SIFT / ORB 关键点提取、FLANN / BF 匹配、Lowe 比值测试、RANSAC 单应性矩阵 | 复杂背景、任意尺度缩放与视角倾斜下的品牌 Logo 空间定位 | [📖 详细文档](computer_vision/03_logo_feature_matching/README.md) |
| **04** | [数独网格透视校正与分割](computer_vision/04_sudoku_grid_segmentation/) | 自适应阈值化、几何多边形外轮廓提取、四点透视正投影变换、形态学去噪 | 手拍倾斜数独题目矫正，精准切片提取 81 个数字网格候选区 | [📖 详细文档](computer_vision/04_sudoku_grid_segmentation/README.md) |
| **05** | [工业多任务缺陷质检平台](computer_vision/05_industrial_smart_inspection/) | Flask Web 架构、Ultralytics YOLOv8 (分类/检测/分割)、OpenCV 传统算法补偿 | 电子元器件表面缺陷、绿豆糕包装漏气褶皱检测、餐具实例分割 | [📖 详细文档](computer_vision/05_industrial_smart_inspection/README.md) |

---

### 二、 深度学习模型实战 (`deep_learning/`)

| 项目序号 | 项目名称 | 网络架构与模型选型 | 关键技术亮点与性能 | 详细文档 |
| :---: | :--- | :--- | :--- | :---: |
| **01** | [CIFAR-100 细粒度图像分类](deep_learning/01_cifar100_image_classifier/) | 深度残差卷积神经网络 (CNN / ResNet) | 多尺度数据增强、批归一化 (BN)、余弦退火学习率衰减、过拟合抑制 | [📖 详细文档](deep_learning/01_cifar100_image_classifier/README.md) |
| **02** | [CRNN+CTC 不定长验证码识别](deep_learning/02_captcha_recognition_crnn/) | CNN 空间特征提取 + 双向 RNN (LSTM/GRU) 时序建模 + CTC Loss | 无需人工字符分割、端到端不定长字母数字序列识别，匹配准确率 98%+ | [📖 详细文档](deep_learning/02_captcha_recognition_crnn/README.md) |

---

## 📂 仓库结构一览

```
CV-DL-Projects/
├── README.md                           # 本作品集总览主页
├── .gitignore                          # 过滤超大二进制模型文件与运行缓存
│
├── computer_vision/                    # 【传统与融合计算机视觉算法】
│   ├── 01_video_subtitle_stream/       # 视频流读写与动态字幕弹幕渲染
│   ├── 02_lane_line_detection/         # 自动驾驶车道线检测与曲率拟合
│   ├── 03_logo_feature_matching/       # SIFT/ORB 特征匹配与单应性变换
│   ├── 04_sudoku_grid_segmentation/    # 数独网格透视变换与 81 单元格切片
│   └── 05_industrial_smart_inspection/ # YOLOv8 + Flask 工业多任务缺陷质检 Web 平台
│
└── deep_learning/                      # 【经典深度学习模型实战】
    ├── 01_cifar100_image_classifier/   # CIFAR-100 自然图像百类细粒度分类调优
    └── 02_captcha_recognition_crnn/    # 基于 CRNN + CTC 的端到端验证码识别系统
```

---

## 🛠️ 快速安装与运行

推荐在 Python 3.8+ 虚拟环境下运行各项目：

```bash
# 1. 基础图像处理与科学计算环境
pip install opencv-python numpy matplotlib Pillow

# 2. 深度学习框架与应用依赖
pip install torch torchvision ultralytics flask tensorflow captcha
```

每一个子目录下均配备有自洽的代码、模型权重、测试数据与专属 `README.md`，可直接进入对应子目录查看与运行！

# CV-DL-Projects - 计算机视觉与深度学习算法作品集

本项目是一套系统化的计算机视觉 (Computer Vision) 与深度学习 (Deep Learning) 算法实战作品集。涵盖从**传统数字图像处理、特征点匹配、形态学分割、车道线多项式拟合**，到**端到端深度学习模型（CIFAR-100分类、CRNN+CTC验证码识别）**，再到**工业级 YOLOv8 多任务缺陷质检 Web 系统**的全栈落地实践。

---

## 目录索引与技术矩阵

### 一、 传统与融合计算机视觉 (computer_vision/)

| 序号 | 项目名称 | 核心算法与技术栈 | 适用场景 / 达成效果 | 详尽文档 |
| :---: | :--- | :--- | :--- | :---: |
| **01** | [视频流读写与动态字幕弹幕](computer_vision/01_video_subtitle_stream/) | OpenCV 视频流处理、PIL 中文字体绘制、平滑位移插值 | 视频字幕滚屏、双层弹幕动画渲染合成 | [查看文档](computer_vision/01_video_subtitle_stream/README.md) |
| **02** | [车道线检测与曲率拟合](computer_vision/02_lane_line_detection/) | Canny 边缘、透视变换 (IPM 鸟瞰图)、滑动窗口多项式拟合 | 智能辅助驾驶、车道线定位、曲率半径解算 | [查看文档](computer_vision/02_lane_line_detection/README.md) |
| **03** | [特征点检测与品牌Logo定位](computer_vision/03_logo_feature_matching/) | SIFT / ORB 特征提取、FLANN / BF 匹配、RANSAC 单应性矩阵 | 复杂背景下任意尺度与旋转的品牌 Logo 目标定位 | [查看文档](computer_vision/03_logo_feature_matching/README.md) |
| **04** | [数独网格透视变换与多目标分割](computer_vision/04_sudoku_grid_segmentation/) | 自适应阈值化、轮廓面积与几何筛选、四点透视变换矫正 | 手机拍摄倾斜数独题目的多目标识别、81宫格数字切片提取 | [查看文档](computer_vision/04_sudoku_grid_segmentation/README.md) |
| **05** | [工业多任务缺陷质检 Web 平台](computer_vision/05_industrial_smart_inspection/) | Flask Web、YOLOv8 (分类/检测/分割)、OpenCV 轮廓补偿 | 电子元器件分类缺陷检测、绿豆糕包装漏气检测、餐具精准分割 | [查看文档](computer_vision/05_industrial_smart_inspection/README.md) |

### 二、 深度学习模型实战 (deep_learning/)

| 序号 | 项目名称 | 网络架构与模型 | 关键技术点 | 详尽文档 |
| :---: | :--- | :--- | :--- | :---: |
| **01** | [CIFAR-100 图像分类调优](deep_learning/01_cifar100_image_classifier/) | 深度卷积神经网络 / ResNet 架构 | 数据增强、Batch Normalization、余弦退火学习率、训练曲线诊断 | [查看文档](deep_learning/01_cifar100_image_classifier/README.md) |
| **02** | [CRNN+CTC 不定长字符验证码识别](deep_learning/02_captcha_recognition_crnn/) | CNN 特征提取 + 双向 RNN (LSTM/GRU) + CTC 损失函数 | 端到端字符识别（无需字符分割）、字母数字混合识别准确率 98%+ | [查看文档](deep_learning/02_captcha_recognition_crnn/README.md) |

---

## 仓库组织结构

`	ext
CV-DL-Projects/
├── README.md                           # 本作品集总览与项目导航
├── .gitignore                          # 智能过滤大型二进制模型缓存
│
├── computer_vision/                    # 传统与融合计算机视觉算法
│   ├── 01_video_subtitle_stream/       # 视频字幕与弹幕特效渲染
│   ├── 02_lane_line_detection/         # 自动驾驶车道线检测
│   ├── 03_logo_feature_matching/       # SIFT/ORB 特征匹配与几何变换
│   ├── 04_sudoku_grid_segmentation/    # 数独网格透视校正与单元格切片
│   └── 05_industrial_smart_inspection/ # YOLOv8 工业多任务质检 Web 平台
│
└── deep_learning/                      # 深度学习实战模型
    ├── 01_cifar100_image_classifier/   # CIFAR-100 图像多分类模型调优
    └── 02_captcha_recognition_crnn/    # CRNN + CTC 端到端字符序列识别
`

---

## 环境准备与快速上手

各项目具有自洽的运行脚本，建议使用 Python 3.8+ 虚拟环境：

`ash
# 1. 基础图像处理与视觉依赖
pip install opencv-python numpy matplotlib Pillow

# 2. 深度学习与 Web 平台依赖
pip install torch torchvision ultralytics flask tensorflow captcha
`
详细的复现步骤与启动命令，请点击对应子项目的 README.md 查看。
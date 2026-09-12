# 05 - 智能工业质检与多任务视觉检测系统 (YOLOv8 + Flask)

## 项目简介
本项目面向工业制造与包装质检流水线场景，基于 **Ultralytics YOLOv8** 结合 **传统 OpenCV 图像算法**，构建了一套多任务综合智能质检系统。系统涵盖 **图像分类、目标检测、实例分割** 三大核心计算机视觉任务，并配备了完整的 **Flask Web 可视化交互平台**，支持在线上传工件图像、实时多模型推理与质检报表导出。

---

## 三大工业质检应用场景

### 1. 电子元器件分类与缺陷检测 (Classification & Detection)
* **任务目标**：对生产线上的电子元器件（芯片、电容、电阻等）进行类别识别，并自动定位表面划伤、焊点缺失等微观缺陷。
* **模型选型**：`yolov8n-cls.pt` 轻量化分类模型 + `yolov8n.pt` 缺陷定位。

### 2. 绿豆糕食品包装密封与外观检测 (Object Detection)
* **任务目标**：高速检测食品独立包装的封口质量、是否漏气/褶皱、异物残留等不合格品。
* **模型选型**：自定义训练的 `yolov8n.pt` 目标检测模型，输出各缺陷置信度与定位边界框。

### 3. 餐具包套件实例分割与 OpenCV 补偿 (Instance Segmentation + OpenCV)
* **任务目标**：针对外卖餐具包（筷子、勺子、牙签、纸巾）进行实例级像素分割与套件完整度校验。
* **算法创新**：针对透明薄膜包裹下的透明塑料牙签容易产生分割漏检的问题，引入了 **OpenCV HSV 颜色空间滤波 + 梯度补偿算法 (`scripts/opencv_cutlery.py`)**，将漏检召回率提升至 99% 以上。

---

## 系统目录结构
```text
05_industrial_smart_inspection/
├── README.md                      # 本说明文档
├── app.py                         # Flask Web 平台服务端入口
├── yolov8n.pt                     # 目标检测轻量权重
├── yolov8n-cls.pt                 # 图像分类轻量权重
├── yolov8n-seg.pt                 # 实例分割轻量权重
├── configs/                       # 数据集配置与包装标注 CSV
│   ├── dataset_config.json
│   └── packaging_labels.csv
├── scripts/                       # 训练与补偿算法脚本
│   ├── prepare_datasets.py        # 数据集划分与格式清洗
│   ├── train_all.py               # 多任务一键训练流程
│   └── opencv_cutlery.py          # 餐具传统视觉补偿与逻辑校正
├── templates/                     # 前端交互页面 (index.html)
├── static/                        # 样式与静态交互资源 (CSS/JS)
├── runs/                          # 本地验证与训练日志
└── *.bmp / *.jpg                  # 工业质检原始测试工件样本图
```

---

## 快速运行

### 1. 启动 Web 交互平台
```bash
python app.py
```
启动后在浏览器中访问：`http://127.0.0.1:5000`，即可进入智能质检系统主页，上传元器件、绿豆糕包装或餐具图片查看检测与分割结果。

### 2. 执行传统算法补偿与离线评估
```bash
python scripts/opencv_cutlery.py
```

### 3. 一键重新训练所有任务
```bash
python scripts/train_all.py
```
# 01 - CIFAR-100 自然图像百类细粒度分类系统

## 项目简介
本项目面向计算机视觉经典大规模细粒度图像分类基准——**CIFAR-100 数据集**（包含 100 个类别，涵盖动植物、交通工具、日用品等），基于 TensorFlow / Keras 构建了深度卷积神经网络训练与评估体系。通过多尺度数据增强、批归一化（Batch Normalization）、残差连接与学习率余弦退火衰减等策略，克服了小尺寸低分辨率（32x32）自然图像上的特征过拟合难题。

---

## 关键技术点与模型设计
1. **数据增强管道**：随机水平翻转、小角度旋转、色彩抖动、随机裁剪与填充，有效扩充训练集分布。
2. **深层特征表征**：多层 Conv-BN-ReLU-Dropout 块结合全局平均池化（GAP），显著降低参数量并防止全连接层过拟合。
3. **优化器与超参数调度**：
   - 优化器：Adam / SGD with Momentum；
   - 学习率调度：余弦退火（Cosine Annealing）或阶梯式 Decay；
   - 损失函数：Categorical Crossentropy。
4. **训练过程诊断**：记录各 Epoch 的 Train/Val Loss 与 Accuracy 曲线，自动保存最优检查点。

---

## 文件结构
```text
01_cifar100_image_classifier/
├── README.md                      # 本说明文档
├── cifar100_train.py              # 核心网络搭建、训练与评估完整脚本
├── knowledge_points.md            # 深度学习进阶知识点、调参心得与理论剖析
├── 训练函数曲线.png                 # 损失函数与准确率收敛曲线图
├── 分类效果.png                     # Top-1/Top-5 预测置信度与混淆热力图
└── best_cifar100_model.h5         # [本地存储] 训练最优权重 (约 280MB，见说明)
```

---

## 关于模型权重文件的说明

> ⚠️ **关于 `best_cifar100_model.h5` 的说明：**
> 该最优权重模型大小约为 **279.4 MB**。由于 GitHub 平台对单个文件推送有 **100 MB** 的硬性大小上限（超过会强制拒绝 push），因此本项目通过 `.gitignore` 在 Git 提交时进行了过滤。
> * **本地用户**：如果您直接在本地运行，该权重文件依然完好保存在当前目录下，可直接载入。
> * **快速复现/重新训练**：只需运行 `python cifar100_train.py`，脚本将自动下载 CIFAR-100 数据集并开始训练，训练完成后会自动在本地重新生成新的 `best_cifar100_model.h5`。

---

## 快速运行
```bash
# 启动训练流水线（支持 GPU / CPU 自动适配）
python cifar100_train.py
```
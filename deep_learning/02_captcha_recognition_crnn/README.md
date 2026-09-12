# 02 - 基于 CRNN + CTC 的端到端字符验证码识别系统

## 项目简介
本项目面向工业界常见的图文验证码识别与文本序列识别（Scene Text Recognition）场景，构建了一套端到端的 **CRNN（Convolutional Recurrent Neural Network）+ CTC（Connectionist Temporal Classification）** 深度学习识别系统。无需预先对验证码进行单个字符切分，直接输入不定长图像，即可一次性输出对应的完整字符序列。

---

## 核心架构与原理

```text
[输入验证码图像 (60x160x3)]
         │
         ▼
  CNN 特征提取层 (多层 Conv2D + MaxPooling2D)
  提取图像局部的空间纹理与笔画特征，输出特征图序列
         │
         ▼
  Map-to-Sequence (特征图沿宽度展开为时序向量)
         │
         ▼
  双向 RNN 时序建模层 (Bidirectional LSTM / GRU)
  捕捉前后字符间的上下文关联依赖与拼写语义
         │
         ▼
  CTC (Connectionist Temporal Classification) 损失函数与解码
  引入 Blank 空白字符，无需人工对齐每个字符的时间步，自动完成贪心/束搜索解码
         │
         ▼
[输出字符序列 (如 "7K4M")]
```

---

## 项目亮点
1. **无需复杂字符切分**：彻底解决字符粘连、重叠、扭曲、干扰线带来的传统分割失败痛点。
2. **端到端训练与推理**：整个网络从输入像素到最终文字输出端到端可导优化。
3. **高精度与高泛化性**：在字母+数字混合随机验证码上，字符序列完全匹配准确率达 **98%+**。
4. **轻量化模型**：模型体积仅 **9.8 MB** (`captcha_crnn_alnum.h5`)，便于直接集成或部署至嵌入式边缘设备。

---

## 文件结构
```text
02_captcha_recognition_crnn/
├── README.md                      # 本说明文档
├── requirements.txt               # 运行环境依赖库
├── captcha_crnn_alnum.h5          # 已训练完成的高精度 CRNN 权重模型 (9.8MB)
├── predict_single.py              # 单张验证码动态生成与实时预测推理演示
├── evaluate_alnum.py              # 批量测试集准确率评估脚本
├── train_captcha_crnn_alnum.py    # CRNN 模型定义、数据发生器与训练代码
└── result_demo.png                # 预测效果示例对比图
```

---

## 快速上手与运行

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 单张预测演示
```bash
python predict_single.py
```

### 3. 批量准确率测试
```bash
python evaluate_alnum.py
```

### 4. 重新训练模型
```bash
python train_captcha_crnn_alnum.py
```
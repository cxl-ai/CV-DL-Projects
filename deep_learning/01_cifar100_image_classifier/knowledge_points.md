# CIFAR-100 训练脚本知识点详解

> 配套 `cifar100_train.py` 阅读，逐模块拆解每个技术点背后的原理与设计动机。

---

## 目录

1. [数据预处理：归一化](#1-数据预处理归一化)
2. [数据增强：ImageDataGenerator](#2-数据增强imagedatagenerator)
3. [卷积层：Conv2D 与感受野](#3-卷积层conv2d-与感受野)
4. [批归一化：Batch Normalization](#4-批归一化batch-normalization)
5. [激活函数：ReLU](#5-激活函数relu)
6. [池化层：MaxPooling2D 与 GlobalAveragePooling2D](#6-池化层maxpooling2d-与-globalaveragepooling2d)
7. [Dropout 正则化](#7-dropout-正则化)
8. [CNN 架构设计原则](#8-cnn-架构设计原则)
9. [Adam 优化器](#9-adam-优化器)
10. [损失函数：Sparse Categorical Crossentropy](#10-损失函数sparse-categorical-crossentropy)
11. [学习率调度：ReduceLROnPlateau](#11-学习率调度reducelronplateau)
12. [早停：EarlyStopping](#12-早停earlystopping)
13. [模型保存：ModelCheckpoint](#13-模型保存modelcheckpoint)
14. [可视化：TensorBoard](#14-可视化tensorboard)

---

## 1. 数据预处理：归一化

### 做了什么

```python
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
```

将像素值从 `[0, 255]` 的整数映射到 `[0.0, 1.0]` 的浮点数。

### 为什么要归一化

1. **数值稳定性**：神经网络权重通常初始化为接近 0 的小数。如果不归一化，输入值（最大 255）与权重相乘会产生很大的激活值，经过多层传播后容易导致梯度爆炸。

2. **加速收敛**：考虑两个特征的尺度差异——如果输入范围差距很大，损失函数的等高线会呈狭长椭圆状，梯度下降的路径会来回震荡（zig-zag），收敛缓慢。归一化后等高线更接近圆形，优化路径更直接。

3. **激活函数的有效区间**：ReLU 虽然在正半轴不饱和，但 BN 层的计算（减均值除方差）在输入尺度差异大时数值不稳定。Sigmoid/Tanh 等激活函数在输入绝对值 > 3 时梯度接近 0，归一化确保输入落在激活函数的敏感区。

4. **统一量纲**：图像中 R、G、B 三个通道虽然在相同尺度（都是 0-255），但如果混合了其他特征（如坐标、深度等），归一化确保每个特征对梯度的贡献是等权的。

### 其他常见归一化方式

| 方法 | 公式 | 适用场景 |
|---|---|---|
| Min-Max | (x - min) / (max - min) | 已知数据范围 |
| Z-Score | (x - mean) / std | 数据分布近似正态 |
| [0,1] 缩放 | x / 255 | 图像像素（本脚本用的） |
| [-1,1] 缩放 | x / 127.5 - 1 | 配合 tanh 输出时常用 |

### 注意点

- 测试集必须使用与训练集**完全相同**的归一化参数。本脚本中用 255 除是硬编码的，所以训练/测试一致。如果用了 Z-Score（减均值除标准差），均值和标准差必须从训练集计算，然后复用到测试集。
- **不要**先归一化再做 train/test split——这会导致信息泄漏，测试集的统计量混入训练集。

---

## 2. 数据增强：ImageDataGenerator

### 做了什么

```python
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,          # 随机旋转 ±15 度
    width_shift_range=0.1,      # 随机水平平移 10%
    height_shift_range=0.1,     # 随机垂直平移 10%
    horizontal_flip=True,       # 随机水平翻转
    zoom_range=0.1,             # 随机缩放 10%
    fill_mode='nearest'         # 填充方式：用最近像素填充
)
```

### 核心思想

数据增强的底层逻辑是：**模型应该对"不改变语义"的图像变换保持鲁棒**。

- 一只猫向左平移 3 个像素，还是一只猫。
- 一张照片水平翻转（镜像），图中的物体类别不变。
- 轻微旋转不会改变物体的本质属性。

通过对每张训练图做随机的这些变换，相当于**免费获得了无限多的训练样本**（每个 epoch 看到的图都不一样），是防止过拟合最有效的手段之一。

### 各参数的含义与选择

| 参数 | 含义 | 为什么选这个值 |
|---|---|---|
| `rotation_range=15` | 在 [-15°, +15°] 内随机旋转 | CIFAR-100 中有飞机、船等可能倾斜的物体；15°是保守值，太大（如 45°）会导致物体方向颠倒、语义模糊 |
| `width_shift_range=0.1` | 水平平移最多 10% 图像宽度 | CIFAR 图像 32×32，10% = 3.2 像素，轻微平移不改变内容 |
| `height_shift_range=0.1` | 垂直平移最多 10% 图像高度 | 同上，模拟拍摄时物体不在正中央的情况 |
| `horizontal_flip=True` | 50% 概率水平翻转 | 大多数物体左右镜像后语义不变（马、船、飞机等）。注意：对文字、时钟等方向敏感的任务需慎用 |
| `zoom_range=0.1` | 随机缩放 [0.9, 1.1] 倍 | 模拟远近不同的拍摄距离 |
| `fill_mode='nearest'` | 旋转/平移后出现的空像素，用最近的边缘像素填充 | 比填 0（`constant`）或镜像（`reflect`）更自然 |

### 为什么用在线增强而不是离线扩增

- **离线**：提前生成 N 倍的增强图像存到硬盘，训练时读取。缺点：I/O 开销大、存储翻倍、增强模式固化。
- **在线**：每个 batch 加载时实时生成随机变换。优点：不占额外存储、每个 epoch 的变换都不同（真正"无限"）、代码简洁。

`datagen.flow()` 就是一个 Python 生成器，每次 `model.fit` 取一个 batch 时，它现场做增强然后 yield 出来。

### 增强不是无限制的

- 太强的增强（如旋转 90°、缩放 50%）会破坏语义信息
- 对于 MNIST 手写数字，旋转 > 20° 可能让 6 和 9 混淆
- 对于医学影像、卫星图等，翻转/旋转可能不合适（上下有固定方向）
- 本脚本中 CIFAR-100 的增强参数偏保守，是为了保证每个变换后的图像依然能被人类识别

---

## 3. 卷积层：Conv2D 与感受野

### 做了什么

本脚本所有 Conv2D 均为：

```python
layers.Conv2D(filters, (3, 3), padding='same')
```

- `filters`：卷积核数量（64 → 128 → 256 → 512）
- `(3, 3)`：每个卷积核的空间尺寸
- `padding='same'`：在输入边缘补 0，使输出空间尺寸与输入相同

### 卷积的直觉理解

全连接层的问题：对于 32×32×3 = 3072 维的输入，每个神经元要和 3072 个权重连接。如果有 64 个神经元，就是 3072×64 ≈ 20 万参数。而且全连接把空间结构完全打散了——像素 (0,0) 和像素 (0,1) 在空间上是相邻的，全连接层感受不到这种相邻关系。

卷积层的核心假设是**局部性**和**平移不变性**：

- **局部性**：一个像素的语义主要取决于它周围的像素，而不是远处的像素。3×3 卷积核只看 9 个像素的邻域。
- **平移不变性**：一只猫在图的左边还是右边，都应该被识别为猫。**同一个卷积核滑过全图**，共享权重，天然满足平移不变性。

### 3×3 卷积核为什么是主流选择

| 卷积核大小 | 参数量 | 感受野 |
|---|---|---|
| 3×3 | 9C_in·C_out | 小 |
| 5×5 | 25C_in·C_out | 中 |
| 7×7 | 49C_in·C_out | 大 |

但**两层 3×3 的感受野等效于一层 5×5**，而参数量只有 2×9=18 vs 25——省了 28%。

**三层 3×3 的感受野等效于一层 7×7**，参数量 3×9=27 vs 49——省了 45%。

同时，多层堆叠还多了两个 ReLU 非线性变换，表达能力更强。这就是 VGG 网络的核心理念：**用更小的卷积核堆叠更深**。

本脚本的 Block 3 和 Block 4 堆了 3 层 3×3 卷积，等效 7×7 的感受野。

### Padding 策略

| 模式 | 输出尺寸（stride=1） | 说明 |
|---|---|---|
| `'valid'` | H - 2, W - 2 | 不补零，边缘像素参与卷积次数少 |
| `'same'` | H, W | 补零使输入输出尺寸一致 |

本脚本用 `'same'`，因为：
- 避免边缘信息逐层丢失（32×32 → 30×30 → 28×28 ... 到最后没剩几个像素了）
- 设计网络时不需要每层都计算输出尺寸
- 空间缩减完全由 MaxPooling 负责，职责分离

### 卷积核数量逐层翻倍的设计

```
Input(32×32×3) → 64 → 128 → 256 → 512 → GAP → Dense(512) → 100
```

这是 CNN 设计的经典模式：

- **浅层**（64 通道）：学到的是低级特征——边缘、纹理、颜色块
- **中层**（128、256 通道）：组合低级特征为中级特征——形状、轮廓、图案
- **深层**（512 通道）：高度抽象的高级语义特征——"像不像猫的耳朵"、"有没有轮子的特征"

通道数增加的同时，空间尺寸经 MaxPool 减半。总的计算量在层间大致保持平衡。

---

## 4. 批归一化：Batch Normalization

### 做了什么

```python
model.add(layers.Conv2D(64, (3, 3), padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
```

每一层卷积后、激活函数前，插入 BN。

### 数学过程

对一个 batch 中每个通道的所有激活值：

1. 计算均值：μ = (1/m) Σ x_i
2. 计算方差：σ² = (1/m) Σ (x_i - μ)²
3. 归一化：x̂_i = (x_i - μ) / √(σ² + ε)
4. 缩放平移：y_i = γ · x̂_i + β

其中 γ 和 β 是**可学习参数**——网络可以自己决定是否需要还原到原始分布。

### 为什么 BN 有效

**(a) 缓解 Internal Covariate Shift**

深层网络训练时，前面层的参数更新会改变后面层接收到的数据分布。后面层像在追一个移动的靶子，学得很困难。BN 把每层的输出强行归一化到均值 0 方差 1，稳定了数据分布，让每层面对的是一个相对固定的输入分布。

不过近年研究（如 "How Does Batch Normalization Help Optimization?", 2018）发现 BN 的真正优势可能在于它让损失曲面变得更平滑，梯度更可预测，从而允许更大的学习率。

**(b) 允许更大的学习率**

没有 BN 时，权重的小变化可能被逐层放大，导致梯度爆炸。BN 归一化后，激活值的尺度被控制住了，梯度传播更稳定，可以使用更大的初始学习率（比如从 0.0001 提升到 0.001），收敛速度提升数倍。

**(c) 轻微正则化效果**

BN 在每个 mini-batch 上计算统计量，均值和方差都有随机性（batch 之间的波动），这种噪声类似于 Dropout 的随机丢弃，给训练过程引入了随机性，轻微抑制过拟合。当 batch size 较大时（如本脚本的 64），这个噪声较小。

**(d) 降低对权重初始化的敏感性**

没有 BN 时，初始化太大或太小都可能导致梯度爆炸或消失。BN 把每层输出重新标准化，从很大程度上消除了对精确初始化的依赖。

### Conv2D → BN → ReLU 的顺序

这是标准顺序，各有讲究：

| 顺序 | 说明 |
|---|---|
| Conv → BN → ReLU | 卷积输出先归一化再激活（**本脚本采用**） |
| Conv → ReLU → BN | ReLU 后的值 ≥ 0，分布不是零中心的，BN 效果打折扣 |
| BN → ReLU → Conv | 部分研究说这样也行，但不主流 |

### 训练 vs 推理

- **训练时**：用当前 mini-batch 的均值和方差做归一化，同时维护一个全局的移动平均（用于推理）
- **推理时**：使用训练阶段累积的全局均值和方差，不再依赖 batch
- 这就是为什么 `model.evaluate()` 和 `model.predict()` 能正确处理 BN——Keras 自动切换模式

### 局限

- Batch size 太小（<16）时，mini-batch 的统计量估计不准确，BN 效果变差
- 对 RNN 等序列模型不太友好（序列长度可变时统计量更难估计），此时常用 Layer Normalization
- 对某些生成任务（如图像超分辨率），BN 可能引入不想要的平滑效果

---

## 5. 激活函数：ReLU

### 做了什么

```python
model.add(layers.Activation('relu'))
```

每个卷积层和全连接层后跟 ReLU 激活。

### 数学定义

ReLU(x) = max(0, x)

导数为：x > 0 时为 1，x < 0 时为 0（x=0 处通常取 0 或 0.5）。

### 为什么 ReLU 统治了 CNN

**(a) 计算简单**

ReLU 就是 `if x > 0: x else: 0`，没有指数运算（对比 Sigmoid 和 Tanh），前向和反向传播都极快。深度网络动辄几千万参数，这点差异累积起来很明显。

**(b) 缓解梯度消失**

Sigmoid 的导数最大只有 0.25（在 x=0 处最大值为 1/4），在饱和区（|x| > 5）导数接近 0。经过多层反向传播，梯度连乘会指数级衰减，浅层几乎收不到梯度信号。

ReLU 在正半轴的导数恒为 1，不会衰减（"不饱和"）。只要激活值 > 0，梯度就能完整地传回去。

**(c) 稀疏激活**

ReLU 把负数全部截断为 0，网络天然产生稀疏表示——大约 50% 的神经元在任何给定输入下是"不激活"的。稀疏性带来：计算效率（0 不用算）、信息解耦（特征之间的干扰更少）、轻微正则化。

### Dying ReLU 问题

如果权重更新导致某个神经元对所有输入都输出 ≤ 0，ReLU 的导数为 0，该神经元就"死"了——再也不会更新。

- BN 缓解了这个问题（保持激活值围绕 0 分布，不会严重偏向负值）
- Leaky ReLU（负半轴给 0.01 的斜率）是另一个解决方案，但本脚本用 BN + 适当学习率就足够避免了

### 其他变体

| 函数 | 公式 | 特点 |
|---|---|---|
| ReLU | max(0, x) | 标准选择 |
| Leaky ReLU | max(0.01x, x) | 避免 dead neuron |
| GELU | x·Φ(x) | Transformer 中常用，比 ReLU 更平滑 |
| Swish | x·sigmoid(x) | 在某些深层网络上优于 ReLU |

---

## 6. 池化层：MaxPooling2D 与 GlobalAveragePooling2D

### MaxPooling2D：空间降维

```python
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
```

每 2×2 窗口取最大值，把 32×32 → 16×16 → 8×8 → 4×4。

**三件事同时发生**：
1. **降低计算量**：尺寸减半 → 下一层的计算量减少 75%
2. **增大感受野**：同样 3×3 卷积，在尺寸减半的特征图上看到的原图范围翻倍
3. **提供平移不变性**：取 max 对小位移鲁棒——特征只要在 2×2 窗口内，就会被检测到

**为什么用 Max 而不是 Average？**
- Max 取窗口中最强的激活，对应"特征存在性"信号
- Average 会把强信号和弱信号平均掉，特征变得不鲜明
- 对分类任务，Max 通常优于 Average

**为什么用 2×2 stride=2 而不是更大的核？**
- 更大的核（如 4×4）降维太快，信息损失大
- 2×2 是最小但有效的降维，配合多次使用，渐进式地缩小特征图
- 过度降维 = 丢失空间信息过多

### GlobalAveragePooling2D：从卷积到全连接的桥梁

```python
model.add(layers.GlobalAveragePooling2D())
```

对每个通道取全局平均值，形状从 (H, W, C) 变为 (1, 1, C) = (C,)。

**对比 Flatten**：

| 方式 | 参数量 | 特点 |
|---|---|---|
| Flatten + Dense | 4×4×512×512 ≈ 4.2M | 参数爆炸，极易过拟合 |
| GlobalAveragePooling + Dense | 512×512 ≈ 262K | 参数量减少 16 倍 |

GAP 的每个输入通道（512 个）恰好对应一个值，没有可学习的权重。它的哲学是：**让每个通道自己学习一个类别相关的高级特征，GAP 取全图平均值表示"这个特征在整个图中是否存在"。**

GAP 天然抗过拟合——无参数意味着它不能记住训练集中的噪声模式，只能依赖通道特征的质量。

---

## 7. Dropout 正则化

### 做了什么

```python
model.add(layers.Dropout(0.2))  # Block 1 后
model.add(layers.Dropout(0.3))  # Block 2 后
model.add(layers.Dropout(0.4))  # Block 3 和 Block 4 后
model.add(layers.Dropout(0.5))  # 分类头
```

训练时每次前向传播，随机把一定比例的神经元输出置 0（"丢弃"），被丢弃的神经元本轮不参与前向也不接收梯度。

### Dropout 的直觉理解

**(a) 集成学习视角**

一个含 N 个神经元的网络，每次 Dropout 随机丢 p 比例 → 相当于训练了 2^N 个子网络（每个神经元要么在要么不在）。推理时不 Dropout，相当于这 2^N 个子网络的结果做平均（实际上权重被 scaled 了 p 倍来近似这个平均）。

**(b) 防"共适应"视角**

如果没有 Dropout，某些神经元可能形成"共适应"（co-adaptation）——比如神经元 A 只依赖神经元 B 的输出做决策，而不是自己从数据中学习。Dropout 随机打掉神经元，迫使每个神经元都学到有用的独立表示——你不知道哪些队友在还是不在，所以你自己必须靠谱。

**(c) 为什么比例递增**

| 位置 | Dropout 比例 | 理由 |
|---|---|---|
| 浅层（Block 1） | 0.2 | 浅层特征是边缘/纹理等低级信息，丢弃太多会断了信息流 |
| 中层（Block 2） | 0.3 | 逐渐增加正则化强度 |
| 深层（Block 3, 4） | 0.4 | 深层特征更抽象、更冗余，可以更多丢弃 |
| 分类头 | 0.5 | 全连接层参数最多、最易过拟合，最强正则化 |

### 训练 vs 推理

- **训练时**：Dropout 生效，神经元以概率 p 被丢弃，保留下来的神经元输出 × 1/(1-p)（即 inverse dropout，Keras 默认）
- **推理时**：Dropout 自动关闭，所有神经元参与计算，不需要手动处理

### Dropout 与 BN 的兼容性

这曾经是个争议话题。2019 年的研究 "Rethinking the Usage of Batch Normalization and Dropout" 发现：在某些情况下，BN 后接 Dropout 反而降低性能（BN 稳定了分布，Dropout 又改变了它，造成"方差偏移"）。

但实践中，大多数网络（包括本脚本）都是 BN 在前、Dropout 在后（或放在不同位置），配合得当依然有效。关键是 **Dropout 放在 Block 末尾而非每个 Conv2D 后**——给 BN 足够的"空间"去稳定分布。

---

## 8. CNN 架构设计原则

本脚本的架构体现了经典 CNN 设计的几条原则：

### 原则 1：金字塔结构

```
空间尺寸: 32 → 16 → 8 → 4 → 1
通道数:    3 → 64 → 128 → 256 → 512 → 512
```

空间维度越来越小，通道维度越来越大——信息从"在哪里"（空间位置）逐渐转化为"是什么"（语义通道）。

### 原则 2：Block 内的同尺寸卷积

每个 Block 内卷积层数递增（2 → 2 → 3 → 3），但所有卷积都是 3×3 padding='same'。Block 内部空间尺寸不变，只有 Block 之间的 MaxPool 才降维。这种"Block 内同上，Block 间降维"的设计让网络结构更规整、更容易调试。

### 原则 3：逐层增加的复杂度

- Block 1: 2 层卷积（网络刚开始，提取简单特征不需要太深）
- Block 2: 2 层卷积
- Block 3: 3 层卷积（特征更抽象，需要更多层来组合）
- Block 4: 3 层卷积（最抽象的特征，最深的结构）

每个 Block 的深度逐渐增加，匹配特征复杂度的提升。

### 原则 4：瓶颈在参数而非计算

全连接层的参数爆炸是 CNN 的老问题。GlobalAveragePooling 替代 Flatten，把参数量从 ~4.2M 压缩到 ~262K，模型更不易过拟合。

---

## 9. Adam 优化器

### 做了什么

```python
optimizer=optimizers.Adam(learning_rate=0.001)
```

### Adam 的核心思想

Adam = **Ada**ptive **M**oment Estimation，融合了两种优化思想：

**(a) Momentum（动量）**

纯 SGD 只考虑当前梯度方向，Momentum 累积历史梯度：

```
v_t = β₁ · v_{t-1} + (1 - β₁) · g_t
```

其中 β₁ = 0.9。这就像给优化加了"惯性"——如果多个连续 step 的梯度方向一致，就加速前进；如果方向反复横跳，历史梯度会互相抵消，减少震荡。

**(b) RMSProp（自适应学习率）**

不同参数可能需要不同的学习率尺度。对梯度变化大的参数降低学习率，对梯度变化小的参数提高学习率：

```
s_t = β₂ · s_{t-1} + (1 - β₂) · g_t²
```

其中 β₂ = 0.999。用 s_t 来缩放每个参数的学习率。

**(c) Adam 的更新规则**

结合两者：

```
θ_{t+1} = θ_t - η · v_t / (√s_t + ε)
```

- η = 0.001（初始学习率）
- ε = 1e-7（防止除 0）

### 为什么初始学习率选 0.001

- 这是 Adam 论文推荐的默认值，在绝大多数任务上表现良好
- 过大的 lr（0.01）配合 BN + Adam 可能发散
- 过小的 lr（0.0001）收敛太慢，浪费计算资源
- 配合 ReduceLROnPlateau，从 0.001 开始，收敛时自动衰减，兼顾速度和精度

### Adam vs SGD

| 维度 | Adam | SGD + Momentum |
|---|---|---|
| 收敛速度 | 快，通常几轮就有好结果 | 慢，需要精细调参 |
| 泛化能力 | 有时略差（自适应特性可能过度拟合训练分布） | 通常泛化更好 |
| 超参敏感性 | 低，默认参数就很好 | 高，学习率、动量都要调 |
| 适用场景 | 快速实验、原型验证 | 最终模型追求极致精度 |

本脚本用 Adam 是因为它省事、收敛快，对于"达到 70% 准确率"这个目标绰绰有余。

---

## 10. 损失函数：Sparse Categorical Crossentropy

### 做了什么

```python
loss='sparse_categorical_crossentropy'
```

### 交叉熵的直觉

交叉熵衡量两个概率分布 p（真实标签）和 q（模型预测）的差异：

```
H(p, q) = -Σ p_i · log(q_i)
```

- p 是 one-hot 向量（正确类别为 1，其余为 0）
- q 是 Softmax 输出的概率分布
- 当 q 在正确类别上接近 1 时，-log(1) = 0，loss 很小
- 当 q 在正确类别上接近 0 时，-log(0.001) ≈ 6.9，loss 很大

实际上交叉熵只关心正确类别的预测概率：`-log(q_correct)`。

### Sparse vs 普通 Categorical Crossentropy

| 版本 | 标签格式 | 何时使用 |
|---|---|---|
| `categorical_crossentropy` | one-hot: `[0,0,1,0,...,0]` | 标签已经是 one-hot |
| `sparse_categorical_crossentropy` | 整数: `42` | 标签是整数（**本脚本**） |

CIFAR-100 的 `y_train` 就是 int32 整数（0~99），用 `sparse_` 版本省去了手动 `to_categorical()` 的步骤。

### 为什么分类任务用交叉熵而不是 MSE

- **梯度特性**：Softmax + 交叉熵的组合，梯度为 `q - p`（预测 - 真实），形式简洁优雅。当预测错误时梯度很大（快速纠正），预测正确时梯度很小（精细调整）。
- **MSE 的问题**：Softmax + MSE 会导致梯度中包含 `q(1-q)` 项，当 q 接近 0 或 1 时梯度几乎为 0（饱和），训练停滞。这就是"MSE 配合 Softmax 的梯度消失"问题。
- **概率解释**：最小化交叉熵等价于最大化训练数据的似然（MLE），有坚实的统计学基础。

---

## 11. 学习率调度：ReduceLROnPlateau

### 做了什么

```python
reduce_lr_callback = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-6,
    verbose=1
)
```

当验证损失连续 10 个 epoch 不再下降时，学习率自动乘以 0.5（减半），最低降至 1e-6。

### 为什么需要学习率衰减

训练早期：模型还不熟悉任务，需要较大步长快速找到低 loss 区域。

训练后期：模型接近最优解，大步长会让参数在最优解周围震荡而无法精确收敛——就像高尔夫球在洞口边缘晃来晃去就是进不去。

衰减学习率就是逐渐缩小步长，让模型从"搜索"模式过渡到"精细调整"模式。

### 为什么用 val_loss 而不是 val_accuracy 做监控

- Loss 是连续的、比 accuracy 更敏感：accuracy 从 68% 到 69% 才变化 1%，但 loss 可能已经下降了 0.2——loss 能更早地捕捉到"模型在变好但还没好到改变分类决策"的阶段
- Loss 直接反映优化目标，accuracy 是离散的间接指标
- 但在 `EarlyStopping` 中用了 val_accuracy（见下一节），因为最终关心的是准确率

### factor=0.5 和 patience=10 的意义

- `patience=10`：给模型 10 个 epoch 的"耐心期"，不掉 loss 不等于马上惩罚——可能是暂时的平台期。10 个 epoch 在当前 batch size 下大约覆盖整个训练集 200 次，足够判断是"真停滞"还是"暂时的波动"。
- `factor=0.5`：每次触发将 lr 减半。衰减过程：0.001 → 0.0005 → 0.00025 → 0.000125 → ... → 1e-6。等比下降比等差更合理——前期需要较大的变化（0.001→0.0005），后期需要精细（接近 min_lr 时自然变缓）。

---

## 12. 早停：EarlyStopping

### 做了什么

```python
early_stopping_callback = callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=30,
    restore_best_weights=True,
    verbose=1
)
```

验证准确率连续 30 个 epoch 不提升就终止训练，并恢复到历史最佳权重的状态。

### 为什么早停能防过拟合

一个典型的训练过程：

```
epoch:  1   2   3   4   5   6   7   8   9  10  11  12 ...
train_acc: 10→20→35→45→55→62→68→72→76→79→82→84→86→...
val_acc:   12→22→36→46→54→60→64→66→67→68→68→67→66→...
                              ↑ 最佳点       ↑ 过拟合开始
```

- 早期：train 和 val 都在提升（欠拟合 → 拟合）
- 中期（epoch 8）：val 达到最高点，train 还在提升——这是最佳时刻
- 后期（epoch 9+）：train 继续提升但 val 下降——**过拟合**，模型开始背诵训练集的噪声而不是学习可泛化的模式

EarlyStopping 自动检测 val 何时停止提升，在过拟合前刹车。

### patience=30 是不是太长了？

- patience=30 是保守策略。CIFAR-100 有 100 个类别，模型复杂，收敛可能比较慢。
- 如果 patience 太短（如 5），可能在模型只是暂时平台期时就提前终止，导致欠拟合。
- 配合 ReduceLROnPlateau 的 patience=10：lr 在第 10 个 epoch 减半 → 模型获得"新的动力" → 可能在第 12-15 个 epoch 突破平台 → EarlyStopping 有足够时间观察这一变化。
- 如果 30 个 epoch 后 val_accuracy 还是没有新高的迹象，那基本可以判断模型确实到极限了。

### restore_best_weights=True

不设这个的话，训练结束时模型处于最后一个 epoch 的权重状态——这可能已经过了最佳点（过拟合了一段）。设为 True 后，训练结束时自动回溯到 val_accuracy 最高的那个 epoch 的权重。

---

## 13. 模型保存：ModelCheckpoint

### 做了什么

```python
model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath='cifar100_best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)
```

每当 val_accuracy 创下新高时，保存当前模型权重到 `cifar100_best_model.h5`。

### 和 EarlyStopping 的 restore_best_weights 有什么区别

| 方式 | 存储位置 | 持久性 |
|---|---|---|
| `restore_best_weights=True` | 内存中 | 脚本结束就没了 |
| `ModelCheckpoint` | 硬盘文件 | 永久保存，后续可用来做推理 |

两者互补：
- EarlyStopping 恢复最佳权重到内存，确保最后的评估和 `model.save()` 用的是最佳状态
- ModelCheckpoint 写硬盘，即使脚本崩溃也有中间成果；训练完可以直接用 `.h5` 文件做推理，不需要重新加载和训练

### save_best_only=True

如果设 False，每个 epoch 都保存一次——CIFAR-100 模型 293MB，30 个 epoch 就是 8.8GB。

`save_best_only=True` 只保留一个最好的，经济实惠。

### H5 格式

`.h5` 是 Keras 的传统保存格式（基于 HDF5），包含：
- 模型架构（层的拓扑结构）
- 权重值
- 训练配置（optimizer、loss、metrics）
- optimizer 状态（可以恢复训练）

缺点是不如 SavedModel（`.pb`）格式通用，但对 Keras/TF 项目完全够用。

---

## 14. 可视化：TensorBoard

### 做了什么

```python
tensorboard_callback = callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True,
    write_images=False
)
```

### TensorBoard 能看到什么

**(a) 标量曲线（Scalars）**
- train_loss / val_loss：看是否过拟合（train 降 val 升）
- train_accuracy / val_accuracy：看收敛情况
- 学习率变化曲线：看到 lr 何时被 ReduceLROnPlateau 衰减

**(b) 权重直方图（Histograms）**
- `histogram_freq=1` 让每个 epoch 都记录权重分布
- 可以看出：权重是否在正常范围（不过大不过小）、分布是否在稳定变化（而非剧烈震荡）、是否有 Dead Neuron（大量权重集中在 0 附近且不更新）

**(c) 计算图（Graph）**
- `write_graph=True` 记录模型结构
- 可视化层的连接关系、数据流向、各层 tensor 的 shape
- 调试利器：确认模型结构和你设想的一致

### 启动命令

```bash
tensorboard --logdir=logs
```

然后在浏览器打开 `http://localhost:6006`。

### 为什么不记录图像（write_images=False）

- write_images 记录的是模型权重可视化为图像（如卷积核的 heatmap）
- 对调试网络来说信息量不大，但会让日志文件急剧膨胀
- CIFAR-100 的 512 通道卷积核可视化出来就是一堆小灰度图，人到后期也看不出什么名堂

---

## 整体训练流程回顾

把以上所有知识点串联起来，一个 epoch 的训练流程是：

```
1. datagen.flow() 取出一个 batch，在线做随机增强
2. 输入经过 4 个 Conv Block，每个 Block:
   - 多次 Conv2D(3×3) → BN → ReLU
   - MaxPool(2×2) 降尺寸
   - Dropout 防过拟合
3. GlobalAveragePooling2D 把 (4,4,512) 压成 512 维向量
4. Dense(512) → BN → ReLU → Dropout(0.5) → Softmax(100)
5. 计算 sparse_categorical_crossentropy loss
6. Adam 优化器计算梯度并更新参数
7. 在验证集上评估 val_loss / val_accuracy
8. 回调们发挥作用:
   - TensorBoard 记录曲线
   - ReduceLROnPlateau 判断要不要降学习率
   - EarlyStopping 判断要不要停
   - ModelCheckpoint 判断要不要存模型
```

所有组件分工明确、互相配合，共同服务于一个目标：**让模型在没见过的图片上也保持高准确率**。

---

> 参考资料：
> - [Batch Normalization (Ioffe & Szegedy, 2015)](https://arxiv.org/abs/1502.03167)
> - [Dropout (Srivastava et al., 2014)](https://jmlr.org/papers/v15/srivastava14a.html)
> - [Adam (Kingma & Ba, 2014)](https://arxiv.org/abs/1412.6980)
> - [VGG Network (Simonyan & Zisserman, 2014)](https://arxiv.org/abs/1409.1556)
> - [Network In Network (Lin et al., 2013)](https://arxiv.org/abs/1312.4400) — 提出 Global Average Pooling

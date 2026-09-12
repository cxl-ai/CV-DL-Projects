# -*- coding: utf-8 -*-
"""
CIFAR-100 分类器训练脚本
使用卷积神经网络对CIFAR-100数据集中的100个类别进行分类
目标：验证集准确率 >= 70%
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, datasets
import datetime
import numpy as np

print("TensorFlow version:", tf.__version__)

# ==================== 1. 数据加载与预处理 ====================
print("Loading CIFAR-100 data...")
(x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()

# 归一化到 [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# CIFAR-100 的 100 个类别名称（仅用于参考）
cifar100_labels = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

print(f"Training samples: {x_train.shape[0]}, Test samples: {x_test.shape[0]}")
print(f"Image shape: {x_train.shape[1:]}")

# ==================== 2. 数据增强 ====================
# 使用 ImageDataGenerator 进行在线数据增强
# 策略：随机水平翻转、随机旋转、随机平移、随机缩放
# 这一策略可以有效增加训练数据的多样性，提升模型泛化能力
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,          # 随机旋转 ±15 度
    width_shift_range=0.1,      # 随机水平平移 10%
    height_shift_range=0.1,     # 随机垂直平移 10%
    horizontal_flip=True,       # 随机水平翻转
    zoom_range=0.1,             # 随机缩放 10%
    fill_mode='nearest'         # 填充方式：用最近像素填充
)

# ==================== 3. 构建 CNN 模型 ====================
def build_cifar100_model():
    """
    构建一个用于CIFAR-100分类的卷积神经网络。

    网络结构说明：
    - 4个卷积块，每块包含2-3个Conv2D层，卷积核数量逐块增加（64→128→256→512）
    - 每个Conv2D后跟BatchNormalization和ReLU激活，加速收敛并缓解梯度消失
    - 每块末尾使用MaxPooling2D降低空间维度，Dropout防止过拟合
    - 全局平均池化替代Flatten，减少参数量
    - 最后接全连接层和Dropout(0.5)后输出100类softmax
    """
    model = models.Sequential()

    # Block 1: 64 filters
    model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))

    # Block 2: 128 filters
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))

    # Block 3: 256 filters
    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.4))

    # Block 4: 512 filters
    model.add(layers.Conv2D(512, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(512, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(512, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.4))

    # 分类头
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(512))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(100, activation='softmax'))

    return model

model = build_cifar100_model()
model.summary()

# ==================== 4. 编译模型 ====================
"""
使用Adam优化器，初始学习率0.001。
损失函数为稀疏分类交叉熵（sparse_categorical_crossentropy），
配合整数标签（而非one-hot编码）使用。
"""
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ==================== 5. 回调函数 ====================
# TensorBoard 日志：记录训练过程中的损失和准确率曲线
log_dir = "logs/cifar100_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,       # 每1个epoch记录权重直方图
    write_graph=True,       # 记录计算图
    write_images=False
)

# 学习率衰减：当验证损失连续10个epoch不再下降时，将学习率减半
# 这有助于模型在接近收敛时更精细地调整参数
reduce_lr_callback = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-6,
    verbose=1
)

# 早停：当验证准确率连续30个epoch不再提升时停止训练，并恢复最佳权重
# 避免无效的继续训练，节省时间
early_stopping_callback = callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=30,
    restore_best_weights=True,
    verbose=1
)

# 模型保存：保存验证准确率最高的模型
model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath='cifar100_best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

print(f"\nTensorBoard log directory: {log_dir}")
print("Run 'tensorboard --logdir=logs' to visualize training curves.\n")

# ==================== 6. 训练 ====================
batch_size = 64
epochs = 300  # 使用早停，实际epoch数由模型收敛情况决定

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(x_train) // batch_size,
    validation_data=(x_test, y_test),
    epochs=epochs,
    callbacks=[
        tensorboard_callback,
        reduce_lr_callback,
        early_stopping_callback,
        model_checkpoint_callback
    ],
    verbose=1
)

# ==================== 7. 评估 ====================
print("\n=== Final Evaluation ===")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
print(f"Test loss: {test_loss:.4f}")

# 保存最终模型
model.save('cifar100_final_model.h5')
print("Model saved as 'cifar100_final_model.h5'")

# 打印一些预测示例
print("\n=== Sample Predictions ===")
predictions = model.predict(x_test[:10])
predicted_classes = np.argmax(predictions, axis=1)
actual_classes = y_test[:10].flatten()
for i in range(10):
    pred_label = cifar100_labels[predicted_classes[i]]
    actual_label = cifar100_labels[actual_classes[i]]
    mark = "✓" if predicted_classes[i] == actual_classes[i] else "✗"
    print(f"  Sample {i+1}: Predicted={pred_label:20s} Actual={actual_label:20s} {mark}")

print("\nTraining complete!")
print(f"Best model saved as 'cifar100_best_model.h5'")
print(f"To view TensorBoard, run: tensorboard --logdir=logs")

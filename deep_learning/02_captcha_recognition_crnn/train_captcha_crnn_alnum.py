"""
训练数字+字母混合验证码识别模型 (CRNN + CTC)
简化版：不做验证回调，训练完成后单独验证
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import secrets
import numpy as np
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization,
                                     Activation, MaxPooling2D, Reshape,
                                     Dense, Bidirectional, LSTM)
from tensorflow.keras.optimizers import Adam
from captcha.image import ImageCaptcha

# ============================================================
# 字符集
# ============================================================
char_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
             'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
             'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
             'U', 'V', 'W', 'X', 'Y', 'Z']
char_to_num = {c: i for i, c in enumerate(char_list)}
num_classes = len(char_list) + 1
blank_index = len(char_list)

IMG_WIDTH = 128
IMG_HEIGHT = 32
BATCH_SIZE = 128
EPOCHS = 50
CAPTCHA_LENGTH = 4


# ============================================================
# 构建 CRNN
# ============================================================
def build_crnn():
    input_layer = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name='input')

    x = Conv2D(32, (3, 3), padding='same')(input_layer)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)

    x = Conv2D(512, (2, 2), padding='same')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)

    x = Reshape((-1, 512))(x)
    x = Dense(128, activation='relu')(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dense(num_classes, activation='linear', name='output')(x)

    model = Model(inputs=input_layer, outputs=x)
    return model


# ============================================================
# 数据生成器
# ============================================================
class CaptchaDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_size, samples_per_epoch):
        self.batch_size = batch_size
        self.samples = samples_per_epoch
        self.gen = ImageCaptcha(width=160, height=60)

    def __len__(self):
        return self.samples // self.batch_size

    def __getitem__(self, idx):
        X = np.zeros((self.batch_size, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
        Y = np.zeros((self.batch_size, CAPTCHA_LENGTH), dtype=np.int32)
        for i in range(self.batch_size):
            text = ''.join([secrets.choice(char_list) for _ in range(CAPTCHA_LENGTH)])
            img = np.array(self.gen.generate_image(text))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = img.astype(np.float32) / 255.0
            X[i] = img.reshape(IMG_HEIGHT, IMG_WIDTH, 1)
            for j, ch in enumerate(text):
                Y[i, j] = char_to_num[ch]
        return X, Y

    def on_epoch_end(self):
        pass


# ============================================================
# 自定义训练循环
# ============================================================
def compute_ctc_loss(y_true, y_pred):
    batch_len = tf.shape(y_pred)[0]
    input_length = tf.fill([batch_len], tf.shape(y_pred)[1])
    label_length = tf.fill([batch_len], tf.shape(y_true)[1])
    loss = tf.nn.ctc_loss(
        labels=y_true,
        logits=tf.transpose(y_pred, perm=[1, 0, 2]),
        label_length=tf.cast(label_length, tf.int32),
        logit_length=tf.cast(input_length, tf.int32),
        blank_index=blank_index,
        logits_time_major=True
    )
    return tf.reduce_mean(loss)


def train_step(model, optimizer, X_batch, Y_batch):
    with tf.GradientTape() as tape:
        pred = model(X_batch, training=True)
        loss = compute_ctc_loss(Y_batch, pred)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1:
            break
        else:
            ret += char_list[ch]
    return ret


def evaluate_model(model, num_samples=200):
    gen = ImageCaptcha(width=160, height=60)
    correct = 0
    for _ in range(num_samples):
        text = ''.join([secrets.choice(char_list) for _ in range(4)])
        image = gen.generate_image(text)
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (128, 32))
        img = img.reshape(1, 32, 128, 1).astype(np.float32) / 255.0
        pred = model(img, training=False)
        pred_sm = tf.nn.softmax(pred, axis=-1).numpy()
        decoded = tf.keras.backend.ctc_decode(
            pred_sm,
            input_length=np.array([pred_sm.shape[1]]),
            greedy=True
        )[0][0].numpy()
        pred_text = num_to_label(decoded[0])
        if text == pred_text:
            correct += 1
    return correct / num_samples * 100


def main():
    print('构建模型...')
    model = build_crnn()
    model.summary()
    optimizer = Adam(learning_rate=0.001)

    train_gen = CaptchaDataGenerator(BATCH_SIZE, 10000)
    steps_per_epoch = len(train_gen)
    print(f'\n开始训练 {EPOCHS} epochs...')
    print(f'字符集: {len(char_list)} (0-9 + A-Z), 输出类: {num_classes}')
    print(f'每个epoch: {steps_per_epoch} batches')
    print(f'每5个epoch评估一次准确率 (200张验证码)')

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for step in range(steps_per_epoch):
            X_batch, Y_batch = train_gen[step]
            loss = train_step(model, optimizer, X_batch, Y_batch)
            total_loss += loss.numpy()
        avg_loss = total_loss / steps_per_epoch
        print(f'Epoch {epoch + 1}/{EPOCHS} - loss: {avg_loss:.4f}', flush=True)

        if (epoch + 1) % 5 == 0:
            model.save('captcha_crnn_alnum.h5')
            acc = evaluate_model(model)
            print(f'  >>> 准确率: {acc:.2f}% (200张测试)', flush=True)
            if epoch + 1 < EPOCHS:
                print()

    model.save('captcha_crnn_alnum.h5')
    print('\n模型已保存: captcha_crnn_alnum.h5')


if __name__ == '__main__':
    main()

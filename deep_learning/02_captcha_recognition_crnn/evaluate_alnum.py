"""
验证数字+字母混合验证码识别模型准确率 (修复softmax问题)
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import secrets
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from captcha.image import ImageCaptcha

char_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
             'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
             'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
             'U', 'V', 'W', 'X', 'Y', 'Z']

def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1:
            break
        else:
            ret += char_list[ch]
    return ret

print("正在加载模型...")
with tf.device('/CPU:0'):
    model = load_model('captcha_crnn_alnum.h5')
print("模型加载完成！")

num_samples = 500
correct = 0
total_digits = num_samples * 4
correct_digits = 0

gen = ImageCaptcha(width=160, height=60)

for i in range(num_samples):
    text = ''.join([secrets.choice(char_list) for _ in range(4)])
    image = gen.generate_image(text)

    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (128, 32))
    img = img.reshape(1, 32, 128, 1).astype(np.float32) / 255.0

    pred = model.predict(img, verbose=0)
    # 模型输出的是logits，需要加softmax转成概率再CTC解码
    pred_sm = tf.nn.softmax(pred, axis=-1).numpy()
    decoded = K.get_value(K.ctc_decode(pred_sm,
                                        input_length=np.ones(pred_sm.shape[0]) * pred_sm.shape[1],
                                        greedy=True)[0][0])
    pred_text = num_to_label(decoded[0])

    if text == pred_text:
        correct += 1
    for a, b in zip(text, pred_text):
        if a == b:
            correct_digits += 1

    if (i + 1) % 100 == 0:
        print(f"  已测试 {i+1}/{num_samples} 张...")

sample_acc = correct / num_samples * 100
digit_acc = correct_digits / total_digits * 100

print("\n" + "=" * 50)
print("    数字+字母混合验证码 评估结果")
print("=" * 50)
print(f"  测试样本数:     {num_samples}")
print(f"  完全正确数:     {correct}")
print(f"  验证集准确率:   {sample_acc:.2f}%")
print(f"  字符准确率:     {digit_acc:.2f}%")
print("=" * 50)

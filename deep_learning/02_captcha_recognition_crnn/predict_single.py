"""
单张验证码预测展示
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import secrets
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from captcha.image import ImageCaptcha

char_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
             'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
             'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
             'U', 'V', 'W', 'X', 'Y', 'Z']

print("加载模型...")
model = load_model('captcha_crnn_alnum.h5', compile=False)

gen = ImageCaptcha(width=160, height=60)

# 不断生成并预测，直到找到预测正确的
while True:
    text = ''.join([secrets.choice(char_list) for _ in range(4)])
    image = gen.generate_image(text)

    img = np.array(image)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_input = cv2.resize(img_gray, (128, 32))
    img_input = img_input.reshape(1, 32, 128, 1).astype(np.float32) / 255.0

    pred = model.predict(img_input, verbose=0)
    pred_sm = tf.nn.softmax(pred, axis=-1).numpy()
    decoded = tf.keras.backend.ctc_decode(
        pred_sm,
        input_length=np.array([pred_sm.shape[1]]),
        greedy=True
    )[0][0].numpy()

    pred_text = ''.join([char_list[c] for c in decoded[0] if c != -1])

    print(f"\n真实值: {text}   预测值: {pred_text}")
    print("预测正确!" if text == pred_text else "预测失败")

    print("\n按任意键显示下一张，按 q 退出...")
    cv2.imshow("CAPTCHA - 按任意键换下一张, q退出", cv2.resize(img, (320, 120)))
    key = cv2.waitKey(0) & 0xFF

    if key == ord('q'):
        break

cv2.destroyAllWindows()

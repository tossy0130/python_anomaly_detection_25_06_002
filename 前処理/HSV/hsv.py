import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

uploaded = files.upload()
image_path = next(iter(uploaded))

bgr_image = cv2.imread(image_path)

# HSV色空間に変換
hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

# HSV各チャンネルを分離
h, s, v = cv2.split(hsv_image)

# 表示のためにRGBに変換
rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

# 各チャネル表示
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(rgb_image)
plt.title("元画像（加工なし）")
plt.axis("off")

############### HSV 表示
plt.subplot(2, 2, 2)
plt.imshow(h, cmap='hsv')
plt.title("Hue (色相)")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(s, cmap='gray')
plt.title("Saturation (彩度)")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(v, cmap='gray')
plt.title("Value (明度)")
plt.axis("off")

plt.tight_layout()
plt.show()
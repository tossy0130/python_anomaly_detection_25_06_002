import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

uploaded = files.upload()
image_path = next(iter(uploaded))

# 画像読み込み
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#########
##### ヒストグラムを計算する関数
#########
def plot_histogram(img_rgb):
    color = ('r', 'g', 'b')
    plt.figure(figsize=(10, 5))
    for i, col in enumerate(color):
        hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
        plt.plot(hist, color=col, label=f'{col.upper()} channel')
    plt.title('RGB Channel Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid()
    plt.show()

# 元画像表示
plt.figure(figsize=(6, 6))
plt.imshow(image_rgb)
plt.title("Uploaded Image")
plt.axis('off')
plt.show()

# ヒストグラム表示
plot_histogram(image_rgb)
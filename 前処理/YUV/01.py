import cv2
import numpy as np
from matplotlib import pyplot as plt
from google.colab import files
from PIL import Image
from io import BytesIO

# 画像のアップロード
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

# 画像読み込み（BGR）
image = cv2.imread(image_path)

# YUV に変換
yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
y, u, v = cv2.split(yuv)

# 黄色領域を抽出（Uが小さく、Vが大きい）
# ※範囲は調整可能
lower = np.array([0, 0, 130])   # Y, U, Vの最小値
upper = np.array([255, 120, 255]) # Y, U, Vの最大値
mask = cv2.inRange(yuv, lower, upper)

# マスクを使って、白黒画像に（黄色だけ白、それ以外は黒）
result = cv2.bitwise_and(image, image, mask=mask)
highlight = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 可視化用に3チャンネル化

# 結果を表示
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original Image")
axes[1].imshow(mask, cmap='gray')
axes[1].set_title("Yellow Area Mask")
axes[2].imshow(cv2.cvtColor(highlight, cv2.COLOR_BGR2RGB))
axes[2].set_title("Yellow Area Highlighted")
for ax in axes:
    ax.axis('off')
plt.show()

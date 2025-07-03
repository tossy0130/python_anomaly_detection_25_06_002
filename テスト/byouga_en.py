import cv2
import numpy as np
import matplotlib.pyplot as plt

# 画像の読み込み（BGR形式）
image = cv2.imread("/content/drive/MyDrive/10月　弥彦/Still1014_00000.jpg")

# 画像の中心座標
height, width = image.shape[:2]
cx, cy = width // 2, height // 2

# パラメータ
radius = 100         # 円の半径
rect_size = 5        # 四角形のサイズ（ピクセル）

# 二重ループで画像全体を走査
for y in range(cy - radius, cy + radius + 1):
    for x in range(cx - radius, cx + radius + 1):

        # 範囲外はスキップ
        if x < 0 or y < 0 or x >= width or y >= height:
            continue

        # 点が円内にあるかをチェック

        ### 以下
"""
この式は、ユークリッド距離（2点間の距離）の2乗を使って、
点 (x, y) と円の中心 (cx, cy) との距離が、半径 r 以下かどうか
つまり、点が「円の内部または円周上」にあるか
を判断しています。

平方根（sqrt）を使わずに済むので高速で、よく使われる手法です。

  dx = x - cx
  dy = y - cy
  if dx * dx + dy * dy <= radius * radius:
    
"""
        dx = x - cx
        dy = y - cy
        if dx * dx + dy * dy <= radius * radius:

            # 矩形を描画（緑色、1ピクセルの太さ）
            top_left = (x - rect_size // 2, y - rect_size // 2)
            bottom_right = (x + rect_size // 2, y + rect_size // 2)
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 1)

# 画像を保存
cv2.imwrite("/content/circle_area_boxed.png", image)

# ----------- 表示 -----------
# BGR → RGB に変換して表示
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)
plt.title("Circle Area Boxed")
plt.axis('off')
plt.show()

import cv2
import numpy as np 
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
from google.colab import files

# 画像アップロード
uploaded = files.upload()

# アップロードされたファイル名を取得
image_path = next(iter(uploaded))

# 画像読み込み
image = cv2.imread(image_path)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 黄色領域のHSV範囲（適宜調整可能）
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([40, 255, 255])

# マスク作成
mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

############
###### モルフォロジー処理でノイズ除去
############
kernel = np.ones((3, 3), np.uint8)
mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# ラベリング（連結成分ラベリング）
num_labels, labels_im = cv2.connectedComponents(mask_clean)

###### 結果表示
output = cv2.bitwise_and(image, image, mask=mask_clean)
cv2_imshow(output)
print(f"検出された物体の数（背景を除く）: {num_labels - 1}")

### 大元画像
print("原本画像")
cv2_imshow(image)
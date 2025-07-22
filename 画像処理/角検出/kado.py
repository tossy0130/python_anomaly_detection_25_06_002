import cv2
import numpy as np
import copy
from google.colab.patches import cv2_imshow

img = cv2.imread("/content/drive/MyDrive/アルゴリズムテスト_25_07_001/buildings.jpg")

# 画像コピー用
img_g = cv2.imread("/content/drive/MyDrive/アルゴリズムテスト_25_07_001/buildings.jpg", 0)

# 画像コピー
img_harris = copy.deepcopy(img)
# コーナーを取得　,ブロックサイズ, ソーヴェルフィルタ, 
img_dst = cv2.cornerHarris(img_g, 2, 3, 0.04)

### 特徴点に対して、赤色を書き込む
img_harris[img_dst > 0.05 * img_dst.max()] = [0, 0, 255]

# 画像表示
cv2_imshow(img_harris)
# cv2.waitKey(0) # This is not needed with cv2_imshow
# cv2.destroyAllWindows() # This is not needed with cv2_imshow

img_orb = copy.deepcopy(img)
orb = cv2.ORB_create()
# 特徴点　抽出
kp2 = orb.detect(img, None)
img_orb = cv2.drawKeypoints(img_orb, kp2, None)
cv2_imshow(img_orb)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2_imshow(img_harris)
cv2_imshow(img_orb)
cv2_imshow(img_kaze)

cv2.waitKey(0)
cv2.destroyAllWindows()
import numpy as np 
import matplotlib.pyplot as plt
import cv2
from PIL  import Image


### 出力
"""

"""

###############################################
################################ エッジ処理
###############################################

######### グレースケールで読む
imgRead2 = cv2.imread("sample1.png", cv2.IMREAD_GRAYSCALE)
# imgRead2 = cv2.cvtColor(imgRead2, cv2.COLOR_BGR2RGB)

################################
############## ソーベルフィルタ
################################

# kszie => カーネルサイズ
sobelx = cv2.Sobel(imgRead2, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(imgRead2, cv2.CV_64F, 0, 1, ksize=3)

######## 勾配の大きさを合わせる
sobelxy = cv2.magnitude(sobelx, sobely)

# 現画像　表示
plt.subplot(2 ,2, 1)
plt.imshow(imgRead2)

# X
plt.subplot(2,2,2)
plt.imshow(cv2.convertScaleAbs(sobelx), cmap='gray')

# Y
plt.subplot(2,2,3)
plt.imshow(cv2.convertScaleAbs(sobely), cmap='gray')

# XY 
plt.subplot(2,2,4)
plt.imshow(cv2.convertScaleAbs(sobelxy), cmap='gray')


################################
############## キャニエッジフィルタ
################################

edge = cv2.Canny(imgRead2, 100, 200)

plt.subplot(1,2,1)
plt.imshow(imgRead2, cmap='gray')

plt.subplot(1,2,2)
plt.imshow(edge, cmap='gray')
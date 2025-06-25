# !pip install japanize_matplotlib

import numpy as np 
import matplotlib.pyplot as plt
import cv2
from PIL  import Image

import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import cv2

from skimage.feature import hog
from skimage import exposure
from skimage.feature import local_binary_pattern

##############################################
######################## LBP 開始
##############################################

# 画像読み込み
img1 = cv2.imread('cookie1.jpg')
img2 = cv2.imread('senbei1.jpg')

### uniform LBP
# 半径
radius = 3
n_points = 8 * radius # 8 => 点の数

lbp_img1 = local_binary_pattern(img1, n_points, radius, method='uniform')
lbp_img2 = local_binary_pattern(img2, n_points, radius, method='uniform')

##################################
######### ヒストグラムへ　グラフ表示
##################################
hist_img1, _ = np.histogram(lbp_img1.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
hist_img2, _ = np.histogram(lbp_img2.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))

hist_img1 = hist_img1.astype("float")
hist_img1 /= (hist_img1.sum() + 1e-7)

hist_img2 = hist_img2.astype("float")
hist_img2 /= (hist_img2.sum() + 1e-7)

# X軸の位置 （bin の番号）
x = np.arange(0, n_points + 2)

plt.figure(figsize=(10, 5))
plt.bar(x - 0.2, hist_img1, width=0.4, label="クッキー", align='center') 
plt.bar(x + 0.2, hist_img2, width=0.4, label="せんべい", align='center')

plt.xlabel('LBP value')
plt.ylabel('Frequency')
plt.legend()
plt.xticks(x) # X軸のメモリを整数にする
plt.show()
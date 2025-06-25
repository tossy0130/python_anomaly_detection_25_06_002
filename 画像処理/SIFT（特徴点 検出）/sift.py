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

from skimage.feature import SIFT
from skimage.color import rgb2gray

### 出力
"""

"""

# 画像読み込み
img1 = cv2.imread('cookie1.jpg')
img2 = cv2.imread('senbei1.jpg')

##############################################
######################## SIFT（特徴点 検出） 開始
##############################################

grai_img1 = rgb2gray(img1)
grai_img1

### 出力
"""
array([[0.53166667, 0.53166667, 0.53166667, ..., 0.6672698 , 0.66334824,
        0.65942667],
       [0.53166667, 0.53166667, 0.53166667, ..., 0.6672698 , 0.66334824,
        0.66334824],
       [0.53558824, 0.53558824, 0.53558824, ..., 0.67119137, 0.6672698 ,
        0.66334824],
       ...,
       [0.68487569, 0.68879725, 0.69271882, ..., 0.81428745, 0.81036588,
        0.80644431],
       [0.68487569, 0.68879725, 0.69271882, ..., 0.81428745, 0.80869922,
        0.80477765],
       [0.68487569, 0.68879725, 0.68879725, ..., 0.81428745, 0.80869922,
        0.80477765]])
"""

sift = SIFT()
sift.detect_and_extract(gray_img1) # インスタンス作成

# キーポイント
kp1 = sift.keypoints

# 特徴量　取得
des1 = sift.descriptors

kp1
### 出力
"""
array([[37, 48],
       [43, 73],
       [54, 83],
       [59, 29],
       [62, 26],
       [68, 26],
       [71, 28],
       [83, 35],
       [84, 79],
       [87, 38],
       [90, 43],
       [94, 55],
       [78, 68],
       [66, 60],
       [99, 87],
       [62, 52]])
"""

des1.shape
### 出力
"""
(16, 128)
"""
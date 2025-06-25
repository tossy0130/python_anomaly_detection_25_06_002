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

### 出力
"""

"""

# 画像読み込み
img1 = cv2.imread('cookie1.jpg')
img2 = cv2.imread('senbei1.jpg')

#######################
######### ヒストグラム
#######################
fd, hog_image = hog(
    img1,
    orientations=9,
    pixels_per_cell=(16, 16),
    cells_per_block=(2, 2),
    visualize=True,
    channel_axis=-1,
)

fd.shape
### 出力
"""
(1764,)
"""

hog_image
### 出力
"""
array([[0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.]])
"""

######### 画像表示
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
plt.imshow(hog_image, cmap='gray')

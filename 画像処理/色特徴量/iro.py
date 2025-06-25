### コラボラトリー
# !pip install japanize_matplotlib

import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import cv2

# 画像読み込み
img1 = cv2.imread('cookie1.jpg')
img2 = cv2.imread('senbei1.jpg')

# 画像表示 
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))

# 画像表示
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

#######################
######### 色 特徴量 関数
########################
def analyze_rgb(image):
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  
  
  pixel_values = image_rgb.reshape(-1, 3)

  ###### RGB の集計 ,
  r_mean = np.mean(pixel_values[:, 0])
  g_mean = np.mean(pixel_values[:, 1])
  b_mean = np.mean(pixel_values[:, 2]) 

  r_variance = np.var(pixel_values[:, 0])
  g_variance = np.var(pixel_values[:, 1])
  b_variance = np.var(pixel_values[:, 2])

  r_median = np.median(pixel_values[:, 0])
  g_median = np.median(pixel_values[:, 1])
  b_median = np.median(pixel_values[:, 2])

  return {'mean': {'R': r_mean, 'G':g_mean, 'B':b_mean}, 
          'variance': {'R': r_variance, 'G':g_variance, 'B':b_variance}, 
          'median': {'R': r_median, 'G':g_median, 'B':b_median}
          }

### 関数実行
analysis_img1 = analyze_rgb(img1)
analysis_img2 = analyze_rgb(img2)

###########################
######### 色 特徴量の可視化 ※グラフで値表示
###########################
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

colors = ['red', 'green', 'blue']
labels = ['R', 'G', 'B']

x = np.arange(len(labels))
width = 0.35

label1 = "クッキー"
label2 = "おせんべい"

rects1 = axes[0].bar(x - width/2, [analysis_img1['mean'][color] for color in labels], width, label=label1)
rects2 = axes[0].bar(x + width/2, [analysis_img2['mean'][color] for color in labels], width, label=label2)

axes[0].set_ylabel('Mean')
axes[0].set_title('Mean Comparison')
axes[0].set_xticks(x, labels)
axes[0].legend()

rects1 = axes[1].bar(x - width/2, [analysis_img1['variance'][color] for color in labels], width, label=label1)
rects2 = axes[1].bar(x + width/2, [analysis_img2['variance'][color] for color in labels], width, label=label2)

axes[1].set_ylabel('Variance')
axes[1].set_title('Variance Comparison')
axes[1].set_xticks(x, labels)
axes[1].legend()

rects1 = axes[2].bar(x - width/2, [analysis_img1['median'][color] for color in labels], width, label=label1)
rects2 = axes[2].bar(x + width/2, [analysis_img2['median'][color] for color in labels], width, label=label2)

axes[2].set_ylabel('Median')
axes[2].set_title('Median Comparison')
axes[2].set_xticks(x, labels)
axes[2].legend()

plt.tight_layout()
plt.show()



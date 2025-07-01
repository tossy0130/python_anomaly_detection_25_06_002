"""
from google.colab import drive
drive.mount('/content/drive')

!pip install japanize_matplotlib
"""

######### 出力
""" 

"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns


from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray


### データの読み込み
image_dir = "/content/drive/MyDrive/画像解析_機械学習_udemy_25_06/"

images = []
labels = []

# おせんべいの画像の読み込み　＆　前処理
for filename in os.listdir(os.path.join(image_dir, "おせんべい")):
  if filename.endswith((".jpg", ".jpeg", ".png")) :
    img_path = os.path.join(image_dir, "おせんべい", filename)
    try:
      img = Image.open(img_path) # 画像を開く
      img = img.resize((64, 64)) # リザイズ
      img_array = np.array(img)
      images.append(img_array)
      labels.append(0) # おせんべい　ラベルは 0
    except Exception as e:
      print(f"Error processing image {img_path}: {e}")

# クッキーの画像の読み込み　＆　前処理
for filename in os.listdir(os.path.join(image_dir, "クッキー")):
  if filename.endswith((".jpg", ".jpeg", ".png")) :
    img_path = os.path.join(image_dir, "クッキー", filename)
    try:
      img = Image.open(img_path) # 画像を開く
      img = img.resize((64, 64)) # リザイズ
      img_array = np.array(img)
      images.append(img_array)
      labels.append(1) # クッキー　ラベルは 1
    except Exception as e:
      print(f"Error processing image {img_path}: {e}")

len(images)
######### 出力
""" 
109
"""

labels
######### 出力
""" 
[0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1]
"""

### numpy の配列へ変換
images = np.array(images)
labels = np.array(labels)

print(images.shape)

######### 出力
""" 
(109, 64, 64, 3)
"""

### 訓練データと、テストデータに分ける
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

### 特徴量をまとめて作成
### 特徴量をまとめて作成
def extract_features(image):
  
  ### 色の特徴量
  # RGB 平均
  r_mean = np.mean(image[:, :, 0])
  g_mean = np.mean(image[:, :, 1])
  b_mean = np.mean(image[:, :, 2])
  # RGB 分散
  r_var = np.var(image[:, :, 0])
  g_var = np.var(image[:, :, 1])
  b_var = np.var(image[:, :, 2])

  # グレースケール画像に変換
  gray_image = rgb2gray(image)

  # HOG 特徴量
  fd, hog_image = hog(gray_image, orientations=9, pixels_per_cell=(16, 16),
                      cells_per_block=(1, 1), visualize=True)

  # LBP 特徴量（チャンネル指定不要）
  radius = 3
  n_points = radius * 8
  lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')

  lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
  lbp_hist = lbp_hist.astype("float")
  lbp_hist /= (lbp_hist.sum() + 1e-7)

  # 全特徴量を結合
  features = np.concatenate(([r_mean, g_mean, b_mean, r_var, g_var, b_var], fd, lbp_hist))
  return features

x_train_features_list = [extract_features(image) for image in x_train]
x_test_features_list = [extract_features(image) for image in x_test]

feature_names = []
feature_names += ['r_mean', 'g_mean', 'b_mean', 'r_var', 'g_var', 'b_var']
feature_names += [f'hog_{i}' for i in range(144)]
feature_names += [f'ldp_{i}' for i in range(26)]

feature_names
######### 出力
""" 
'hog_1',
 'hog_2',
 'hog_3',
 'hog_4',
 'hog_5',
 'hog_6',
 'hog_7',
 'hog_8',
 'hog_9',
 'hog_10',
 'hog_11',
 'hog_12',
 'hog_13',
 'hog_14',
 'hog_15',
 'hog_16',
 'hog_17',
 'hog_18',
 'hog_19',
 'hog_20',
 'hog_21',
 'hog_22',
 'hog_23',
 'hog_24',
 'hog_25',
 'hog_26',
 'hog_27',
 'hog_28',
 'hog_29',
 'hog_30',
 'hog_31',
 'hog_32',
 'hog_33',
 'hog_34',
 'hog_35',
 'hog_36',
 'hog_37',
 'hog_38',
 'hog_39',
 'hog_40',
 'hog_41',
 'hog_42',
 'hog_43',
 'hog_44',
 'hog_45',
 'hog_46',
 'hog_47',
 'hog_48',
 'hog_49',
 'hog_50',
 'hog_51',
 'hog_52',
 'hog_53',
 'hog_54',
 'hog_55',
 'hog_56',
 'hog_57',
 'hog_58',
 'hog_59',
 'hog_60',
 'hog_61',
 'hog_62',
 'hog_63',
 'hog_64',
 'hog_65',
 'hog_66',
 'hog_67',
 'hog_68',
 'hog_69',
 'hog_70',
 'hog_71',
 'hog_72',
 'hog_73',
 'hog_74',
 'hog_75',
 'hog_76',
 'hog_77',
 'hog_78',
 'hog_79',
 'hog_80',
 'hog_81',
 'hog_82',
 'hog_83',
 'hog_84',
 'hog_85',
 'hog_86',
 'hog_87',
 'hog_88',
 'hog_89',
 'hog_90',
 'hog_91',
 'hog_92',
 'hog_93',
 'hog_94',
 'hog_95',
 'hog_96',
 'hog_97',
 'hog_98',
 'hog_99',
 'hog_100',
 'hog_101',
 'hog_102',
 'hog_103',
 'hog_104',
 'hog_105',
 'hog_106',
 'hog_107',
 'hog_108',
 'hog_109',
 'hog_110',
 'hog_111',
 'hog_112',
 'hog_113',
 'hog_114',
 'hog_115',
 'hog_116',
 'hog_117',
 'hog_118',
 'hog_119',
 'hog_120',
 'hog_121',
 'hog_122',
 'hog_123',
 'hog_124',
 'hog_125',
 'hog_126',
 'hog_127',
 'hog_128',
 'hog_129',
 'hog_130',
 'hog_131',
 'hog_132',
 'hog_133',
 'hog_134',
 'hog_135',
 'hog_136',
 'hog_137',
 'hog_138',
 'hog_139',
 'hog_140',
 'hog_141',
 'hog_142',
 'hog_143',
 'ldp_0',
 'ldp_1',
 'ldp_2',
 'ldp_3',
 'ldp_4',
 'ldp_5',
 'ldp_6',
 'ldp_7',
 'ldp_8',
 'ldp_9',
 'ldp_10',
 'ldp_11',
 'ldp_12',
 'ldp_13',
 'ldp_14',
 'ldp_15',
 'ldp_16',
 'ldp_17',
 'ldp_18',
 'ldp_19',
 'ldp_20',
 'ldp_21',
 'ldp_22',
 'ldp_23',
 'ldp_24',
 'ldp_25']
"""

######### データフレームに入れる
x_train_df = pd.DataFrame(x_train_features_list, columns=feature_names)
x_test_df = pd.DataFrame(x_test_features_list, columns=feature_names)

x_train_df.head()

######### 出力
""" 
r_mean	g_mean	b_mean	r_var	g_var	b_var	hog_0	hog_1	hog_2	hog_3	...	ldp_16	ldp_17	ldp_18	ldp_19	ldp_20	ldp_21	ldp_22	ldp_23	ldp_24	ldp_25
0	175.503662	172.221680	161.182617	681.942369	1109.611503	2838.292823	0.231253	0.370006	0.497375	0.497375	...	0.019043	0.010254	0.006104	0.004639	0.004150	0.004883	0.005859	0.010742	0.018799	0.190186
1	179.228516	175.165283	164.315430	1149.592312	1456.156519	2588.556266	0.362449	0.362449	0.362449	0.362449	...	0.020752	0.017578	0.011963	0.008789	0.013428	0.015137	0.014893	0.011230	0.010986	0.223877
2	176.804932	172.027100	159.880127	1205.084263	1534.907713	2867.028355	0.362323	0.362323	0.362323	0.362323	...	0.026855	0.017578	0.017090	0.015625	0.007568	0.011230	0.013916	0.009277	0.013184	0.259766
3	176.538574	172.627686	160.117676	1208.484840	1365.884575	2307.913399	0.364846	0.364846	0.364846	0.158799	...	0.029541	0.018555	0.013184	0.012207	0.013184	0.013184	0.011719	0.005127	0.015869	0.242676
4	176.281738	174.002930	163.934082	1215.999237	1379.198234	2166.745655	0.379786	0.379786	0.237518	0.379786	...	0.027344	0.021729	0.013184	0.009521	0.008789	0.014160	0.009766	0.004883	0.014648	0.237305
5 rows × 176 columns

"""
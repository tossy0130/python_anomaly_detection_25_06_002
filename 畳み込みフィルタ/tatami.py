import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models

def edge_filter(shape=None, dtype=None):
    filter0 = np.array(
        [[2, 1, 0, -1, -2],
        [3, 2, 0, -2, -3],
        [4, 3, 0, -3, -4],
        [3, 2, 0, -2, -3],
        [2, 1, 0, -1, -2]]) / 23.0 # Added an extra row to make the shape (5, 5)

    filter1 = np.array(
        [[2, 3, 4, 3, 2],
         [1, 2, 3, 2, 1],
         [0, 0, 0, 0, 0],
         [-1, -2, -3, -2, -1],
         [-2, -3, -4, -3, -2]]) / 23.0

    filter_array = np.zeros([5, 5, 1, 2])
    filter_array[:, :, 0, 0] = filter0
    filter_array[:, :, 0, 1] = filter1

    return filter_array

base_repo = 'https://github.com/enakai00/colab_GenAI_lecture'

### Googleコラボ
# !curl -LO {base_repo}/raw/main/Part01/ORENIST.data
with open('ORENIST.data', 'rb') as file:
    images, labels = pickle.load(file)
images = np.array(images)
labels = np.array(labels)

fig = plt.figure(figsize=(5, 2))
for i in range(10):
  subplot = fig.add_subplot(2, 5, i+1)
  subplot.set_xticks([])
  subplot.set_yticks([])
  subplot.imshow(images[i].reshape(28, 28), interpolation='none',
                 vmin=0, vmax=1, cmap=plt.cm.gray_r)
  

### フィルターを適用する入力画像
model1 = models.Sequential(name='conv_filter_model1')
model1.add(layers.Input(shape=(784,), name='input'))
# １次元リストを、 28, 28, 1 の ３次元リストに変換　28 ✖️ 28 ピクセル １レイヤー
model1.add(layers.Reshape((28, 28, 1), name='reshape'))

# layers.Conv2D(2, (5, 5) => 畳み込みフィルターを適用するレイヤー。
# kernel_initializer=edge_filter, =>  適用するフィルター。（上の関数）
# padding='same' => はみ出た部分は、０のピクセルにする。（０パディング）
# bias_initializer=tf.constant_initializer(-0.2),  activation='relu', => 「定数を加えて、活性化関数を適用する」

model1.add(layers.Conv2D(2, (5, 5), padding='same',
                         kernel_initializer=edge_filter,
                         bias_initializer=tf.constant_initializer(-0.2),
                         activation='relu', name='conv_filter'))

model1.summary()

#################
### 結果出力 ####
#################
conv_output1 = model1.predict(images[:9], verbose=0)

# 畳み込み層の出力（フィルタ数, 高さ, 幅）を可視化
fig, axes = plt.subplots(nrows=9, ncols=2, figsize=(3, 10))
fig.suptitle('Feature maps: Filter 0 (left), Filter 1 (right)', fontsize=16)

for n in range(9):
    # フィルタ0の出力（1番目のフィルタ）
    feature_map_0 = conv_output1[n, :, :, 0]
    # フィルタ1の出力（2番目のフィルタ）
    feature_map_1 = conv_output1[n, :, :, 1]
    
    # 左にフィルタ0の出力
    axes[n, 0].imshow(feature_map_0, cmap='gray')
    axes[n, 0].set_title(f'Image {n} - Filter 0')
    axes[n, 0].axis('off')
    
    # 右にフィルタ1の出力
    axes[n, 1].imshow(feature_map_1, cmap='gray')
    axes[n, 1].set_title(f'Image {n} - Filter 1')
    axes[n, 1].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
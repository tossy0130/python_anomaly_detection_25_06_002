import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv('/content/Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values # 独立変数
y = dataset.iloc[:, -1].values # 従属変数

# 出力 01
print(x)
print(y)

# n_estimators = 10, 木の数。random_state = 0, モデル作成時のオプション
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)

# 出力 02
regressor.fit(x, y)

# 出力 03
regressor.predict([[6.5]])

### 結果表示 04
#  min(x) => 最小値と、, max(x) => 最大値を, 0.1 => 0.1ずつ刻んでいく)
X_grid = np.arange(min(x), max(x), 0.1)
# 2Darray 変換用
X_grid = X_grid.reshape((len(X_grid), 1))
# 散布図作成
plt.scatter(x, y, color = 'red')
# 直線作成
### ランダムフォレストが刻んできた値を直線で、繋げていく
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')

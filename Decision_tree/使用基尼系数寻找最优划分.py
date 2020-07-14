import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from collections import Counter
from math import log

# 决策边界，axis是划分边界的坐标
def plot_decision_boundary(model, axis):
    x0, x1=np.meshgrid(
        # axis[0]: 左边界， axis[1]：右边界
        np.linspace(axis[0],axis[1], int((axis[1]-axis[0])*100)),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100))
    )
    X_new = np.c_[x0.ravel(),x1.ravel()]
    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A','#FFF59D', '#90CAF9'])
    plt.contourf(x0, x1, zz, cmap=custom_cmap)



iris = datasets.load_iris()
X = iris.data[:, 2:]    # 保留后两个特征
y = iris.target

from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier(max_depth=2, criterion='gini')
dt_clf.fit(X, y)
# 绘制决策边界
plot_decision_boundary(dt_clf, axis=[0.5, 7.5, 0, 3])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.scatter(X[y==2,0], X[y==2,1])
plt.show()

# 模拟使用基尼系数划分
# 训练集x, y，维度， 值
def split(X, y, d, value):
    index_a = (X[:,d] <= value) # bool值的索引
    index_b = (X[:,d] > value)
    return X[index_a], X[index_b], y[index_a], y[index_b]

# 信息熵，y表示代表的分类
def gini(y):
    counter =  Counter(y)   # 键值对
    res = 0.0
    for num in counter.values():
        p = num / len(y)
        res -= p**2
    return res

# 寻找函数,最低的信息熵
def try_split(X, y):
    best_gini = float('inf')    # 初始化成正无穷的值
    best_d, best_v = -1, -1
    for d in range(X.shape[1]):
        sorted_index = np.argsort(X[:,d])
        for i in range(1, len(X)):
            if X[sorted_index[i-1], d] != X[sorted_index[i], d]:    # 判断相邻的值
                v = (X[sorted_index[i-1], d] + X[sorted_index[i], d]) / 2
                X_l, X_r, y_l, y_r = split(X, y, d, v)
                g = gini(y_l) + gini(y_r)
                if g < best_gini:
                    best_gini, best_d, best_v = g, d, v

    return best_gini, best_d, best_v



best_gini, best_d, best_v = try_split(X, y)
print('best_gini=', best_gini)
print('best_d=', best_d)
print('best_v=', best_v)
x1_l, x1_r, y1_l, y1_r = split(X, y, best_d, best_v)
print('左侧的信息熵=', gini(y1_l))
print('右侧的信息熵=', gini(y1_r))










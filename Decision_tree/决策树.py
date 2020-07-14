from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data[:, 2:]    # 保留后两个特征
y = iris.target

# plt.scatter(X[y==0,0], X[y==0,1])
# plt.scatter(X[y==1,0], X[y==1,1])
# plt.scatter(X[y==2,0], X[y==2,1])
# plt.show()

from sklearn.tree import DecisionTreeClassifier
# 传入最大深度，熵
dt_clf = DecisionTreeClassifier(max_depth=2, criterion="entropy")
dt_clf.fit(X, y)

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

# 绘制决策边界
plot_decision_boundary(dt_clf, axis=[0.5, 7.5, 0, 3])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.scatter(X[y==2,0], X[y==2,1])
plt.show()


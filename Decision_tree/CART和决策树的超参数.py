from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

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

# 构建数据集
X, y = datasets.make_moons(noise=0.25, random_state=666)
# plt.scatter(X[y==0,0], X[y==0,1])
# plt.scatter(X[y==1,0], X[y==1,1])
# plt.show()

from sklearn.tree import DecisionTreeClassifier
# 训练数据集
dt_clf = DecisionTreeClassifier()   # 不传参，就是基尼系数，一直向下划分，基尼系数达到0
dt_clf.fit(X, y)

plot_decision_boundary(dt_clf, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()

# 参数max——depth
dt_clf2 = DecisionTreeClassifier(max_depth=2)
dt_clf2.fit(X, y)
plot_decision_boundary(dt_clf2, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()
print(dt_clf2.score(X, y))

# 参数min_sample_split 值越高，越不容易过拟合，表示最小的划分数据为10
dt_clf3 = DecisionTreeClassifier(min_samples_split=10)
dt_clf3.fit(X, y)
plot_decision_boundary(dt_clf3, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()

# 参数min_samplt_split 最小的叶子数，越大越不容易过拟合
dt_clf4 = DecisionTreeClassifier(min_samples_leaf=6)
dt_clf4.fit(X, y)
plot_decision_boundary(dt_clf4, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()
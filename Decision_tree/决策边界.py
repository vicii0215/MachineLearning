import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target
# 只选取两个特征
X = X[y<2, :2]
y = y[y<2]

from Machine_Learning.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,seed=666)
from Linear_And_Logistic.LogisticRegression import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
print('准确率：', log_reg.score(X_test, y_test))
print(log_reg.coef_)

# def x2(x1):
#     return (-log_reg.coef_[0] * x1 - log_reg.interception_) / log_reg.coef_[1]
#
# x1_plot = np.linspace(4, 8, 1000)
# x2_plot = x2(x1_plot)
# plt.scatter(X[y==0,0], X[y==0,1], color='r')
# plt.scatter(X[y==1,0], X[y==1,1], color='green')
# plt.plot(x1_plot, x2_plot)
# plt.show()


'''决策边界，axis是划分边界的坐标'''
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
plot_decision_boundary(log_reg, axis=[4, 7.5, 1.5, 4.5])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()





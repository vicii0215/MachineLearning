import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target
X = X[y<2, :2]
y = y[y<2]


# 标准化处理
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
standardScaler.fit(X)
X_standard = standardScaler.transform(X)

# 导入线性svm的模块，使用支撑向量机完成分类问题
from sklearn.svm import LinearSVC


'''决策边界，axis是划分边界的坐标'''
def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
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

svc = LinearSVC(C=1e9)  # c越大，容错越小
svc.fit(X_standard, y)
plot_decision_boundary(svc, axis=[-3, 3, -3, 3])
plt.scatter(X_standard[y==0,0], X_standard[y==0,1])
plt.scatter(X_standard[y==1,0], X_standard[y==1,1])
plt.show()

# 调整参数c
svc1 = LinearSVC(C=0.01)  # c越大，容错越小
svc1.fit(X_standard, y)
plot_decision_boundary(svc1, axis=[-3, 3, -3, 3])
plt.scatter(X_standard[y==0,0], X_standard[y==0,1])
plt.scatter(X_standard[y==1,0], X_standard[y==1,1])
plt.show()


# svc的决策边界
def plot_svc_decision_boundary(model, axis):
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
    # 绘制出margin的两条线
    w = model.coef_[0]
    b = model.intercept_[0]

    # w0*x0 + w1*x1 + b = 0 最优边界
    # => x1 = -w0/w1 * x0 - b/w1
    plot_x = np.linspace(axis[0], axis[1], 200)
    # 上面的线
    up_y = -w[0] / w[1] * plot_x - b / w[1] + 1 / w[1]
    # 下面的线
    down_y = -w[0] / w[1] * plot_x - b / w[1] - 1 / w[1]
    # 剔除超过边界的y的值 bool值
    up_index = (up_y >= axis[2]) & (up_y <= axis[3])
    down_index = (down_y >= axis[2]) & (down_y <= axis[3])

    plt.plot(plot_x[up_index], up_y[up_index], color='black')
    plt.plot(plot_x[down_index], down_y[down_index], color='black')

plot_svc_decision_boundary(svc, axis=[-3, 3,-3, 3])
plt.scatter(X_standard[y==0,0], X_standard[y==0,1])
plt.scatter(X_standard[y==1,0], X_standard[y==1,1])
plt.show()

plot_svc_decision_boundary(svc1, axis=[-3, 3,-3, 3])
plt.scatter(X_standard[y==0,0], X_standard[y==0,1])
plt.scatter(X_standard[y==1,0], X_standard[y==1,1])
plt.show()





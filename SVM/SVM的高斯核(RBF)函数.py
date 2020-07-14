import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-4, 5, 1)
y = np.array((x >= -2) & (x <= 2), dtype='int')

# plt.scatter(x[y==0], [0]*len(x[y==0]))
# plt.scatter(x[y==1], [0]*len(x[y==1]))
# plt.show()

def gaussian(x, l):
    # 此处直接将超参数 γ 设定为 1.0；
    # 此处 x 表示一维的样本，也就是一个具体的值，l 相应的也是一个具体的数，因为 l 和 x 一样，从特征空间中选定；
    gamma = 1.0
    # 此处因为 x 和 l 都只是一个数，不需要再计算模，可以直接平方；
    return np.exp(-gamma * (x-l)**2)

# 设定地标 l1、l2 为 -1和1
l1, l2 = -1, 1
x_new = np.empty((len(x), 2))

for i, data in enumerate(x):
    x_new[i, 0] = gaussian(data, l1)
    x_new[i, 1] = gaussian(data, l2)

# plt.scatter(x_new[y==0, 0], x_new[y==0, 1])
# plt.scatter(x_new[y==1, 0], x_new[y==1, 1])
# plt.show()


from sklearn import datasets
# sklearn中的RBF核
X, y = datasets.make_moons(noise=0.15, random_state=666)
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

def RBFkernelSVC(gamma=1.0):
    return Pipeline([
        ('std_scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', gamma=gamma))
    ])

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

# gamma = 1.0
svc = RBFkernelSVC(gamma=1.0)
svc.fit(X, y)
print(svc.score(X, y))

plot_decision_boundary(svc, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.title('gamma=1.0')
plt.show()

# gamma = 100
svc_gamma100 = RBFkernelSVC(gamma=100)
svc_gamma100.fit(X, y)
print(svc_gamma100.score(X, y))

plot_decision_boundary(svc_gamma100, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.title('gamma=100')
plt.show()


# gamma = 0.5
svc_gamma05 = RBFkernelSVC(gamma=3)
svc_gamma05.fit(X, y)
print(svc_gamma05.score(X, y))

plot_decision_boundary(svc_gamma05, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y == 0,0], X[y == 0,1])
plt.scatter(X[y == 1,0], X[y == 1,1])
plt.title('gamma=0.5')
plt.show()


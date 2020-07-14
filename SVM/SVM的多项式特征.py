import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

X, y = datasets.make_moons(noise=0.15, random_state=666)
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()

'''多项式的svm'''
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

def polynomialSVC(degree, C=1.0):
    return Pipeline(
        [
            ('poly',PolynomialFeatures(degree=degree)),
            ('std_scaler', StandardScaler()),
            ('linearSVC', LinearSVC(C=C))
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


poly_svc = polynomialSVC(degree=3)
poly_svc.fit(X,y)
print(poly_svc.score(X,y))
plot_decision_boundary(poly_svc, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()


'''使用多项式核函数的svm'''
from sklearn.svm import SVC
def polynomialKernelSVC(degree, C=1.0):
    return Pipeline(
        [
            ('std_scaler', StandardScaler()),
            ('kernelSVC', SVC(kernel='poly', degree=degree, C=C))
        ])
poly_kernel_svc = polynomialKernelSVC(degree=0.001)
poly_kernel_svc.fit(X,y)
print(poly_kernel_svc.score(X,y))

plot_decision_boundary(poly_kernel_svc, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()




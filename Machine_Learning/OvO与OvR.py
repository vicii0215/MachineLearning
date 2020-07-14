import numpy as np
import matplotlib.pyplot as plt
from sklearn import  datasets
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

iris = datasets.load_iris()
X = iris.data[:,:2]
y = iris.target

from Machine_Learning.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,seed=666)

from sklearn.linear_model import LogisticRegression
# 默认的方式为OvR
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
print(log_reg.score(X_test, y_test))

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

plot_decision_boundary(log_reg, axis=[4, 8.5, 1.5, 4.5])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.scatter(X[y==2,0], X[y==2,1])
plt.title('三个类型的OvR决策边界')
plt.show()

# 使用OvO方法
log_reg2 = LogisticRegression(multi_class="multinomial", solver="newton-cg")
log_reg2.fit(X_train, y_train)
print(log_reg2.score(X_test, y_test))

plot_decision_boundary(log_reg2, axis=[4, 8.5, 1.5, 4.5])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.scatter(X[y==2,0], X[y==2,1])
plt.title('三个类型的OvO决策边界')
plt.show()



'''使用所有数据'''
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y,seed=666)
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
print(log_reg.score(X_test, y_test))

log_reg2 = LogisticRegression(multi_class="multinomial", solver="newton-cg")
log_reg2.fit(X_train, y_train)
print(log_reg2.score(X_test, y_test))


'''sklearn的两个类完成ovo和ovr'''
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier

ovr = OneVsRestClassifier(log_reg)
ovr.fit(X_train, y_train)
print(ovr.score(X_test, y_test))

ovo = OneVsOneClassifier(log_reg)
ovo.fit(X_train, y_train)
print(ovo.score(X_test, y_test))
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

np.random.seed(666)
X = np.random.normal(0,1,size=(200,2))
y = np.array(X[:,0]**2 + X[:,1]**2 <1.5, dtype='int')

'''如果只使用逻辑回归'''
from Linear_And_Logistic.LogisticRegression import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X, y)
print('只是用逻辑回归的准确率=',log_reg.score(X,y))

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

# plot_decision_boundary(log_reg, axis=[-4, 4,-4, 4])
# plt.scatter(X[y==0,0], X[y==0,1])
# plt.scatter(X[y==1,0], X[y==1,1])
# plt.title('单一逻辑回归的分类边界')
# plt.show()

# 向逻辑回归添加多项式回归
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler    # 归一化
def PolynomialLogisticRegression(degree):
    return Pipeline([
        ('poly',PolynomialFeatures(degree=degree)),
        ('std_scaler',StandardScaler()),
        ('log_reg', LogisticRegression())
    ])
ploy_log_reg = PolynomialLogisticRegression(degree=2)
ploy_log_reg.fit(X, y)
print('逻辑回归+多项式回归的准确度=', ploy_log_reg.score(X,y))

plot_decision_boundary(ploy_log_reg, axis=[-4, 4,-4, 4])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.title('逻辑回归+多项式回归的分类边界')
plt.show()


# 使用sklearn
def PolynomialLogisticRegression(degree, C):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('log_reg', LogisticRegression())
    ])
from Meachine_Learning.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,seed=666)

ploy_log_reg3 = PolynomialLogisticRegression(degree=20, C=0.1)
ploy_log_reg3.fit(X_train, y_train)
print(ploy_log_reg3.score(X_train, y_train))
print(ploy_log_reg3.score(X_test, y_test))







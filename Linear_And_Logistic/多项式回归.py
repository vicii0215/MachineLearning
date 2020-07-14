import numpy as np
import matplotlib.pyplot as plt

# x = np.random.uniform(-3,3, size=100)
# X = x.reshape(-1, 1)
# y = 0.5*x**2 + x + 2 + np.random.normal(0, 1, size=100)
'''使用线性回归拟合'''
from sklearn.linear_model import LinearRegression
# lin_reg = LinearRegression()
# lin_reg.fit(X, y)
# y_predict = lin_reg.predict(X)
# # 多项式拟合
# # 添加一个特征
# X2 = np.hstack([X, X**2])       # 将X(100,1) -> X2(100,2)
#
# lin_reg2 = LinearRegression()
# lin_reg2.fit(X2, y)
# y_predict2 = lin_reg2.predict(X2)
# plt.scatter(X, y)
# plt.plot(np.sort(x), y_predict2[np.argsort(x)], color='r')  # 将x从小打大排序， 将y——predict2安装返回的下标排序
# plt.show()


'''使用sklearn中的多项式回归'''
from sklearn.preprocessing import PolynomialFeatures
# poly = PolynomialFeatures(degree=2)     # 表示多项式回归的几次幂
# poly.fit(X)
# X2 = poly.transform(X)
#
# lin_reg2 = LinearRegression()
# lin_reg2.fit(X2, y)
# y_predict2 = lin_reg2.predict(X2)
# plt.scatter(X, y)
# plt.plot(np.sort(x), y_predict2[np.argsort(x)], color='r')  # 将x从小打大排序， 将y——predict2安装返回的下标排序
# plt.show()


'''当x是多维'''
# x = np.arange(1, 11).reshape(-1, 2)
# poly = PolynomialFeatures(degree=3)
# poly.fit(x)
# x3 = poly.transform(x)
# print(x3)


'''在使用degree=n时候，要进行线性归一化'''
x = np.random.uniform(-3,3, size=100)
X = x.reshape(-1, 1)
y = 0.5*x**2 + x + 2 + np.random.normal(0, 1, size=100)

from sklearn.pipeline import Pipeline   # 对回归方程管道处理
from sklearn.preprocessing import StandardScaler
# poly_reg = Pipeline(
#     [
#         ("poly", PolynomialFeatures(degree=500)), # 建立多项式
#         ("std_scaler", StandardScaler()),   # 归一化处理
#         ("lin_reg", LinearRegression())     # 回归方程
#     ])
# poly_reg.fit(X, y)
# y_predict = poly_reg.predict(X)
# plt.scatter(X, y)
# plt.plot(np.sort(x), y_predict[np.argsort(x)], color='red')  # 将x从小打大排序， 将y——predict2安装返回的下标排序
# plt.show()


from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.model_selection import train_test_split

def plot_learning_curve(algo, X_train, X_test, y_train, y_test):
    train_score = []
    test_score = []
    for i in range(1, len(X_train)+1):
        algo.fit(X_train[:i], y_train[:i])  # 只用train的前i个数据拟合

        y_train_predict = algo.predict(X_train[:i])
        train_score.append(mean_squared_error(y_train[:i], y_train_predict))

        y_test_predict = algo.predict(X_test)
        test_score.append(mean_squared_error(y_test, y_test_predict))
    plt.plot([i for i in range(1, len(X_train)+1)], np.sqrt(train_score), label='train')
    plt.plot([i for i in range(1, len(X_train)+1)], np.sqrt(test_score), label='test')
    plt.legend()
    plt.axis([0, len(X_train)+1, 0, 4])
    plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
# plot_learning_curve(LinearRegression(), X_train, X_test, y_train, y_test )

# 多项式回归
def PolynomialRegression(degree):
    return Pipeline(
    [
        ("poly", PolynomialFeatures(degree=degree)), # 建立多项式
        ("std_scaler", StandardScaler()),   # 归一化处理
        ("lin_reg", LinearRegression())     # 回归方程
    ])
poly2_reg = PolynomialRegression(degree=20)
plot_learning_curve(poly2_reg, X_train, X_test, y_train, y_test)
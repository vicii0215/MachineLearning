import numpy as np
import matplotlib.pyplot as plt

m = 100000

x = np.random.normal(size=m)
X = x.reshape(-1, 1)
y = 4.*x +3. + np.random.normal(0,3,size=m)

'''批量梯度下降'''
# def J(theta, X_b, y):
#     try:
#         return np.sum((y - X_b.dot(theta)) ** 2) / len(X_b)
#     except:
#         return float('inf')
# def dJ(theta, X_b, y):
#     # 向量梯度下降
#     return X_b.T.dot(X_b.dot(theta) - y) * 2. / len(X_b)
# def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
#     theta = initial_theta
#     i_iter = 0
#
#     '''最多执行n_iters次数'''
#     while i_iter < n_iters:
#         gradient = dJ(theta, X_b, y)
#         last_theta = theta
#         theta = theta - gradient * eta
#
#         if abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon:
#             break
#         i_iter += 1
#     return theta
# X_b = np.hstack([np.ones((len(X), 1)), X])
# initial_theta = np.zeros(X_b.shape[1])
# eta = 0.01
# theta = gradient_descent(X_b, y, initial_theta, eta)
# print(theta)


'''随机梯度下降'''
# def dJ_sgd(theta, X_b_i, y_i):
#     # 向量梯度下降
#     return X_b_i.T.dot(X_b_i.dot(theta) - y_i) * 2.
#
# def sgd(X_b, y, initial_theta, n_iters):
#     t0 = 5
#     t1 = 50
#     # 计算学习率
#     def learning_rate(t):
#         return t0 / (t + t1)
#
#     theta = initial_theta
#     for cur_iter in range(n_iters):
#         rand_i = np.random.randint(len(X_b))
#         gradient = dJ_sgd(theta, X_b[rand_i], y[rand_i])
#         theta = theta - learning_rate(cur_iter) * gradient
#     return theta
# X_b = np.hstack([np.ones((len(X), 1)), X])
# initial_theta = np.zeros(X_b.shape[1])
# theta = sgd(X_b, y, initial_theta, n_iters=len(X_b)//3)
# print(theta)

'''使用LinearRegression中的fit_sgd'''
# from LinearRegression import LinearRegression
# lin_reg = LinearRegression()
# lin_reg.fit_sgd(X, y, n_iters=2)
# print(lin_reg.coef_)
# print(lin_reg.interception_)

'''真实数据使用sgd'''
from sklearn import datasets
from Machine_Learning.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Linear_And_Logistic.LinearRegression import LinearRegression

boston = datasets.load_boston()
X = boston.data    # 取出房间数这个特征
y = boston.target
X = X[y<50.0]
y = y[y<50.0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2, seed=666)
# 数据归一化处理
standardScaler = StandardScaler()
standardScaler.fit(X_train)
X_train_standard = standardScaler.transform(X_train)
X_test_standard = standardScaler.transform(X_test)

lin_reg = LinearRegression()
lin_reg.fit_sgd(X_train_standard, y_train, n_iters=1000)
print(lin_reg.score(X_test_standard, y_test))



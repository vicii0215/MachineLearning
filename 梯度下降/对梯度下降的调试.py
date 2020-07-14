import numpy as np
import  matplotlib.pyplot as plt

np.random.seed(666)
X = np.random.random(size=(1000, 10))   # 1000个样本，10个维度

true_theta = np.arange(1, 12, dtype=float)  # 11个特征，包括截距

X_b = np.hstack([np.ones((len(X), 1)), X])
y = X_b.dot(true_theta) + np.random.normal(size=1000)


def J(theta, X_b, y):
    try:
        return np.sum((y - X_b.dot(theta)) ** 2) / len(X_b)
    except:
        return float('inf')



def dJ_math(theta, X_b, y):
    # 向量梯度下降
    return X_b.T.dot(X_b.dot(theta) - y) *2./len(X_b)

def dJ_debug(theta, X_b, y, epsilon=0.01):
    res = np.empty(len(theta))
    for i in range(len(theta)):
        theta_1 = theta.copy()
        theta_1[i] += epsilon
        theta_2 = theta.copy()
        theta_2[i] -= epsilon
        res[i] = (J(theta_1, X_b, y) - J(theta_2, X_b, y)) / (2*epsilon)        # 在第i个维度上的倒数值
    return res


def gradient_descent(dJ, X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
    theta = initial_theta
    i_iter = 0

    '''最多执行n_iters次数'''
    while i_iter < n_iters:
        gradient = dJ(theta, X_b, y)
        last_theta = theta
        theta = theta - gradient * eta

        if abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon:
            break
        i_iter += 1
    return theta

X_b = np.hstack([np.ones((len(X), 1)), X])
initial_theta = np.zeros(X_b.shape[1])
eta = 0.01

theta = gradient_descent(dJ_debug, X_b, y, initial_theta, eta)
print(theta)
theta1 = gradient_descent(dJ_math, X_b, y, initial_theta, eta)
print(theta1)




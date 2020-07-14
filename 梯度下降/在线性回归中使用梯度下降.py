import numpy as np
import matplotlib.pyplot as plt

np.random.seed(666)
x = 2*np.random.random(size=100)
y = x*3. + 4. + np.random.normal(size=100)

X = x.reshape(-1, 1)    # 100行1列
# plt.scatter(x, y)
# plt.show()

def J(theta, X_b, y):
    try:
        return np.sum((y - X_b.dot(theta)) ** 2) / len(X_b)
    except:
        return float('inf')

def dJ(theta, X_b, y):
    res = np.empty((len(theta)))
    res[0] = np.sum(X_b.dot(theta) - y)
    for i in range(1, len(theta)):
        res[i] = (X_b.dot(theta) - y).dot(X_b[:,i])

    return res*2/len(X_b)

def gradient_descent(X_b, y, initial_theta, eta, n_iters= 1e4, epsilon=1e-8):
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

X_b = np.hstack([np.ones((len(X),1)), x.reshape(-1,1)])
initial_theta = np.zeros(X_b.shape[1])
eta = 0.01
theta = gradient_descent(X_b, y, initial_theta, eta)

print(theta)
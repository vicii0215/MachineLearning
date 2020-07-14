import numpy as np
import matplotlib.pyplot as plt

np.random.seed(666)
'''两个维度'''
X = np.empty((100, 2 ))
X[:,0] = np.random.uniform(0., 100., size=100)
X[:,1] = 0.75*X[:,0] + 3.+ np.random.normal(0, 10., size=100)

'''demean'''
def demean(X):
    return X - np.mean(X, axis=0)   # 在X的行，求均值，结果是(1,n)的向量

X_demean = demean(X)        # 均值归零操作

# plt.scatter(X_demean[:,0], X_demean[:,1])

'''梯度上升'''
def f(w, X):
    return np.sum((X.dot(w)**2)) / len(X)

def df_math(w, X):
    return X.T.dot(X.dot(w)) * 2. / len(X)

def df_debug(w, X, epsilon=0.0001):
    res = np.empty(len(w))
    for i in range(len(w)):
        w_1 = w.copy()
        w_1[i] += epsilon
        w_2 = w.copy()
        w_2[i] -= epsilon
        res[i] = (f(w_1, X) - f(w_2, X)) / (2*epsilon)
    return res

def direction(w):
    return w/ np.linalg.norm(w)     # w除以w的model

# eta步长设置较小
def gradient_ascent(df, X, initial_w, eta, n_iters=1e4, epsilon=1e-8):
    w = direction(initial_w)
    cur_iter = 0

    while cur_iter < n_iters:
        gradient = df(w, X)
        last_w = w
        w = w + eta*gradient    # 梯度上升计算新的w
        w = direction(initial_w)        # 每次求一个单位方向
        if abs(f(w,X) - f(last_w, X)) < epsilon:
            break
        cur_iter += 1
    return w

# 初始化一个0向量
initial_w = np.random.random(X.shape[1])
print(initial_w)
eta = 0.001
# 不能使用standardScaler处理数据，归一化处理会使方差为零
w = gradient_ascent(df_debug, X_demean, initial_w, eta)
print(gradient_ascent(df_math, X_demean, initial_w, eta))

# plt.scatter(X_demean[:,0],X_demean[:,1])
# plt.plot([0,w[0]*30], [0,w[1]*30], color='r')
# plt.show()

# 使用极端从测试用例
X2 = np.empty((100, 2 ))
X2[:,0] = np.random.uniform(0., 100., size=100)
X2[:,1] = 0.75 * X2[:,0] + 3.
X2_demean = demean(X2)
w2 = gradient_ascent(df_debug, X2_demean, initial_w, eta)
print(w2)
plt.scatter(X2_demean[:,0],X2_demean[:,1])
plt.plot([0,w2[0]*30], [0,w2[1]*30], color='r')
plt.show()

# plt.scatter(X2[:,0],X2[:,1])
# plt.show()
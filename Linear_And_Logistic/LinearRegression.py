import numpy as np
from Machine_Learning.metrices import r2_score



class LinearRegression:
    def __init__(self):
        '''初始化模型'''
        self.coef_ = None           # 系数
        self.interception_ = None   # 截距
        self._theta = None  # 向量,多维系数

    def fit_normal(self, X_train, y_train):     # 输入的X——train是矩阵
        assert X_train.shape[0] == y_train.shape[0],\
            "x—train和y—train大小不一致"

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])      # 构造含有第一列是全1的矩阵
        '''求出theta， 包含截距和系数'''
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)         # np.linalg.inv为求逆矩阵
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    '''线性回归中的批量梯度下降'''
    def fit_gd(self, X_train, y_train, eta = 0.01, n_iters=1e4):
        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(X_b)
            except:
                return float('inf')

        def dJ(theta, X_b, y):
            # 向量梯度下降
            return X_b.T.dot(X_b.dot(theta) - y) *2./len(X_b)

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
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

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])

        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1]

    '''随机梯度下降 n_iters表示循环多少次随机train'''
    def fit_sgd(self,X_train, y_train, n_iters=5, t0=5, t1=50):
        def dJ_sgd(theta, X_b_i, y_i):
            # 向量梯度下降
            return X_b_i.T.dot(X_b_i.dot(theta) - y_i) * 2.

        def sgd(X_b, y, initial_theta, n_iters, t0=5, t1=50):
            # 计算学习率
            def learning_rate(t):
                return t0 / (t + t1)

            theta = initial_theta
            m = len(X_b)

            for cur_iter in range(n_iters):
                indexes = np.random.permutation(m)  # 从0---m-1取随机的排列
                # 乱序化处理
                X_b_new = X_b[indexes]
                y_new = y[indexes]
                for i in range(m):  # 遍历所有样本
                    gradient = dJ_sgd(theta, X_b_new[i], y_new[i])
                    theta = theta - learning_rate(cur_iter*m + i) * gradient
            return theta
        X_b = np.hstack([np.ones((len(X_train),1)), X_train])
        initial_theta = np.random.randn(X_b.shape[1])

        self._theta = sgd(X_b, y_train, initial_theta, n_iters, t0, t1)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1]

    '''预测'''
    def predict(self, X_predict):
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])  # 构造含有第一列是全1的矩阵
        return X_b.dot(self._theta)

    '''R2评估'''
    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)


    def __repr__(self):
        return 'LinearRegression()'
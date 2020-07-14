import numpy as np
from Machine_Learning.metrices import r2_score
from Machine_Learning.metrices import accuracy_score  # 分类问题使用accuracy——score

class LogisticRegression:
    def __init__(self):
        '''初始化模型'''
        self.coef_ = None           # 系数
        self.interception_ = None   # 截距
        self._theta = None  # 向量,多维系数

    def _sigmoid(self, t):
        return 1./(1. + np.exp(-t))


    '''逻辑回归中的批量梯度下降'''
    def fit(self, X_train, y_train, eta = 0.01, n_iters=1e4):
        def J(theta, X_b, y):
            y_hat = self._sigmoid(X_b.dot(theta))
            try:
                return -np.sum(y*np.log(y_hat) + (1-y)*np.log(1-y_hat)) / len(y)
            except:
                return float('inf')

        # 求梯度
        def dJ(theta, X_b, y):
            return X_b.T.dot(self._sigmoid(X_b.dot(theta)) - y) /len(X_b)

        # 梯度下降
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
        return self

    '''预测概率'''
    def predict_proba(self, X_predict):
        '''返回表示X_predict的结果向量概率'''
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])  # 构造含有第一列是全1的矩阵
        return self._sigmoid(X_b.dot(self._theta))

    '''预测'''
    def predict(self, X_predict):
        '''返回表示X_predict的结果向量'''
        proba = self.predict_proba(X_predict)
        return np.array(proba >= 0.5, dtype='int')  # 返回一个0,1的向量，表示概率

    '''分类准确度'''
    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return 'LinearRegression()'
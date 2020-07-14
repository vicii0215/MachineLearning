import numpy as np
import time
from Machine_Learning.metrices import r2_score
# 普通的运算
class SimpleLinearRegression1:
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        '''使用X_TREIN, Y_TRAIN建立拟合模型'''
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        num = 0.0
        d = 0.0
        for x, y in zip(x_train, y_train):
            num += (x-x_mean) * (y-y_mean)
            d += (x-x_mean) ** 2

        self.a_ = num / d
        self.b_ = y_mean- self.a_ * x_mean
        return self

    def predict(self, x_predict):
        '''输入数据集x_predict，返回结果向量'''
        assert x_predict.ndim == 1, "输入量必须是一个简单的线性向量"
        assert self.a_ is not None and self.b_ is not None, "调用函数前，必须先进行训练集的拟合"

        return [self._predict(x) for x in x_predict]

    def _predict(self, x_single):
        return self.a_ * x_single + self.b_

    def score(self, x_test, y_test):
        '''测试准确度'''
        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return 'SimpleLinearRegression1'


# 向量计算，时间更短
class SimpleLinearRegression2:
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        '''使用X_TREIN, Y_TRAIN建立拟合模型'''
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        num = (x_train - x_mean).dot(y_train - y_mean)  # 分子部分
        d = (x_train - x_mean).dot(x_train - x_mean)    # 分母部分

        self.a_ = num / d
        self.b_ = y_mean- self.a_ * x_mean
        return self

    def predict(self, x_predict):
        '''输入数据集x_predict，返回结果向量'''
        assert x_predict.ndim == 1, "输入量必须是一个简单的线性向量"
        assert self.a_ is not None and self.b_ is not None, "调用函数前，必须先进行训练集的拟合"

        return [self._predict(x) for x in x_predict]

    def _predict(self, x_single):
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return 'SimpleLinearRegression2'
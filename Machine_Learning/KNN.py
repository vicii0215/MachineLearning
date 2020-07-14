import numpy as np
from math import sqrt
from collections import Counter
from Machine_Learning.metrices import accuracy_score

# def KNN_classfiy(k, X_train, y_train, x):
#     distance = [sqrt(np.sum((x_train - x)**2)) for x_train in X_train]
#     nearest = np.argsort(distance)
#
#     topK_y = [y_train[i] for i in nearest[:k]]
#     votes = Counter(topK_y)
#
#     return votes.most_common(1)[0][0]

class KNNclassifier:
    def __init__(self, k):
        # 初始化knn分类器
        assert k>=1, "k must be valid"
        self.k = k
        self._X_train = None
        self._Y_train = None

    def fit(self, X_train, y_train):
        # 根据训练集xtrain y_train 训练knn
        self._X_train = X_train
        self._Y_train = y_train
        return self

    def predict(self, X_predict):
        # 给定等待预测的x——predict， 返回表示x——predict的结果向量
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        # 给定单个的待预测的数据x， 返回x的预测结果值
        distance = [sqrt(np.sum((x_train - x)**2)) for x_train in self._X_train]
        nearest = np.argsort(distance)
        topK_y = [self._Y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)
        return votes.most_common(1)[0][0]

    # 计算准确度
    '''根据X_test和y_test，确定当前模型的准确度'''
    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)
    def __repr__(self):
        return "KNN(k=%d)" %self.k




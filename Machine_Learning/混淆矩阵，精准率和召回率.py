# 处理偏斜数据
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

digits = datasets.load_digits()
X = digits.data
y = digits.target.copy()
# 构建偏斜数据
y[digits.target == 9] = 1
y[digits.target != 9] = 0

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
print(log_reg.score(X_test, y_test))

y_log_predict = log_reg.predict(X_test)

# 00
def TN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 0))

# 01
def FP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 1))

# 10
def FN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 0))

# 11
def TP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 1))

# 混淆函数
def confuison_Matrix(y_true, y_predict):
    return np.array([
        [TN(y_true, y_predict), FP(y_true, y_predict)],
        [FN(y_true, y_predict), TP(y_true, y_predict)]
    ])

print(confuison_Matrix(y_test, y_log_predict))
# 精准率
def precision_score(y_true, y_predict):
    tp = TP(y_test, y_predict)
    fp = FP(y_test, y_predict)
    try:
        return tp / (tp + fp)
    except:
        return 0.0
print('精准率=', precision_score(y_test, y_log_predict))
# 召回率
def recall_score(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fn = FN(y_true, y_predict)
    try:
        return tp / (tp + fn)
    except:
        return 0.0
print('召回率=',recall_score(y_test, y_log_predict))

# 机器学习中的混淆函数,精准率，召回率
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print(confuison_Matrix(y_true=y_test, y_predict=y_log_predict))
print(precision_score(y_test, y_log_predict))
print(recall_score(y_test, y_log_predict))


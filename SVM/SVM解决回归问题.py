import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

bostan = datasets.load_boston()
X = bostan.data
y = bostan.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)


# LinearSVR 处理回归问题
from sklearn.svm import LinearSVR
from sklearn.svm import SVR # 核函数
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def standardLinearSVR(epsilon=0.1):
    return Pipeline(
        [
            ("std_scaler", StandardScaler()),
            ("linearSVR", LinearSVR(epsilon=epsilon))
        ])
svr = standardLinearSVR()
svr.fit(X_train, y_train)
print(svr.score(X_test, y_test))        # R2 的值
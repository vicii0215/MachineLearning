import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target
# 只选取两个特征
X = X[y<2, :2]
y = y[y<2]

# plt.scatter(X[y==0,0], X[y==0,1], color='r')
# plt.scatter(X[y==1,0], X[y==1,1], color='green')
# plt.show()

from Machine_Learning.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,seed=666)
from Linear_And_Logistic.LogisticRegression import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
print(log_reg.score(X_test,  y_test))
print(log_reg.predict_proba(X_test))    # 得到分类概率
print(log_reg.predict(X_test))      # 得到分类结果



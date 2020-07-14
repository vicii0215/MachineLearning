from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
import numpy as np
import matplotlib.pyplot as plt
from time import time

from sklearn import datasets
X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)


'''使用oob'''
# 使用bagging
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
# bootstrap表达是否为放回取样
bagging_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, max_samples=100, bootstrap=True, oob_score=True)
bagging_clf.fit(X, y)
print(bagging_clf.oob_score_)

# 使用n_jobs ，多核处理
bagging_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, max_samples=100, bootstrap=True, oob_score=True, n_jobs=-1)
bagging_clf.fit(X, y)
print(bagging_clf.oob_score_)
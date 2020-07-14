import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)
# plt.scatter(X[y==0,0], X[y==0,1])
# plt.scatter(X[y==1,0], X[y==1,1])
# plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

# 使用bagging
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
# bootstrap表达是否为放回取样
bagging_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, max_samples=100, bootstrap=True)
bagging_clf.fit(X_train, y_train)
print(bagging_clf.score(X_test, y_test))

bagging_clf1 = BaggingClassifier(DecisionTreeClassifier(), n_estimators=5000, max_samples=50, bootstrap=True)
bagging_clf1.fit(X_train, y_train)
print(bagging_clf1.score(X_test, y_test))

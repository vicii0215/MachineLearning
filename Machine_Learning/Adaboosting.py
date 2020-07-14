from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

'''AdaBoosting: 集成方式'''
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=500)
ada_clf.fit(X_train, y_train)
print(ada_clf.score(X_test, y_test))

'''Gradient Boosting'''
from sklearn.ensemble import GradientBoostingClassifier
# n_estimator表示集成数
gb_clf = GradientBoostingClassifier(max_depth=2, n_estimators=500)
gb_clf.fit(X_train, y_train)
print(gb_clf.score(X_test, y_test))
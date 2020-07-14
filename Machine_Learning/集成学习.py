import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)
# plt.scatter(X[y==0,0], X[y==0,1])
# plt.scatter(X[y==1,0], X[y==1,1])
# plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
# 逻辑回归模型
from sklearn.linear_model import LogisticRegression
log_clf = LogisticRegression()
log_clf.fit(X_train, y_train)
print('逻辑回归=',log_clf.score(X_test, y_test))
# svc
from sklearn.svm import SVC
svm_clf = SVC()
svm_clf.fit(X_train, y_train)
print('SVM=',svm_clf.score(X_test, y_test))
# 决策树
from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
print('决策树=',dt_clf.score(X_test, y_test))


y_predict1 = log_clf.predict(X_test)
y_predict2 = svm_clf.predict(X_test)
y_predict3 = dt_clf.predict(X_test)

y_predict = np.array((y_predict1 + y_predict2 + y_predict3) >= 2, dtype='int')
print(y_predict[:10])

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_predict))


'''sklearn中的集成学习'''
from sklearn.ensemble import VotingClassifier
# estimators 传入的方法
voting_clf = VotingClassifier(estimators=[
    ('log_clf', LogisticRegression()),
    ('svm_clf', SVC()),
    ('dt_clf', DecisionTreeClassifier())
        ], voting='hard')
voting_clf.fit(X_train, y_train)
print('集成学习=',voting_clf.score(X_test, y_test))





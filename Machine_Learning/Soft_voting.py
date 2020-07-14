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

'''hard_voting'''
# estimators 传入的方法
voting_clf = VotingClassifier(estimators=[
    ('log_clf', LogisticRegression()),
    ('svm_clf', SVC()),
    ('dt_clf', DecisionTreeClassifier(random_state=666))
        ], voting='hard')
voting_clf.fit(X_train, y_train)
print('hard_voting.score=',voting_clf.score(X_test, y_test))


'''soft-voting classifer'''
voting_clf2 = VotingClassifier(estimators=[
    ('log_clf', LogisticRegression()),
    ('svm_clf', SVC(probability=True)), # soft 要计算概率
    ('dt_clf', DecisionTreeClassifier(random_state=666))
        ], voting='soft')
voting_clf2.fit(X_train, y_train)
print('soft_voting.score=',voting_clf2.score(X_test, y_test))





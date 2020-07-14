import numpy as np
from sklearn import  datasets

digits = datasets.load_digits()
X = digits.data
y = digits.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=666)
from sklearn.neighbors import KNeighborsClassifier
# 寻找超参数k 和p(距离)
best_score, best_p, best_k = 0, 0, 0
# for k in range(2, 11)
#     for p in range(1, 6):
#         knn_clf = KNeighborsClassifier(weights='distance', n_neighbors=k, p=p)
#         knn_clf.fit(X_train, y_train)
#         score = knn_clf.score(X_test, y_test)
#         if score > best_score:
#             best_score, best_p, best_k = score, p, k    # 找到最佳的p,k
#         print("K = %d, p = %d, Score = %d" % (k, p, score))
#
# print("bestK = %d, bestp = %d, bestScore = %d" %(k, p, score))

# sklearn的交叉验证
from sklearn.model_selection import cross_val_score
# knn_clf = KNeighborsClassifier()
# print(cross_val_score(knn_clf, X_train, y_train))
best_score, best_p, best_k = 0, 0, 0
for k in range(2, 11):
    for p in range(1, 6):
        knn_clf = KNeighborsClassifier(weights='distance', n_neighbors=k, p=p)
        # knn_clf.fit(X_train, y_train)
        scores = cross_val_score(knn_clf, X_train, y_train)
        score = np.mean(scores)
        if score > best_score:
            best_score, best_p, best_k = score, p, k    # 找到最佳的p,k
        print("K = %d, p = %d, Score = %d" % (k, p, score))

print("bestK = %d, bestp = %d, bestScore = %d" %(k, p, score))
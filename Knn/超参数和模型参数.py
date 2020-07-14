'''
超参数： 在算法运行前需要决定的参数
模型参数：算法过程中学习的参数

KNN算法没有模型参数，但是k是典型的超参数
'''

from sklearn import datasets


digits = datasets.load_digits()
X = digits.data
y = digits.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier
# knn_lcf = KNeighborsClassifier(n_neighbors=3)
# knn_lcf.fit(X_train, y_train)
# print(knn_lcf.score(X_test, y_test))

'''find the best K, 调参'''
# best_method = ''
# best_score = 0.0
# best_k = -1
# for method in ['uniform', 'distance']:  # 是否考虑权重问题
#     for k in range(1, 11):
#         knn_lcf = KNeighborsClassifier(n_neighbors=k, weights=method)
#         knn_lcf.fit(X_train, y_train)
#         score = knn_lcf.score(X_test, y_test)
#         if score > best_score:
#             best_k = k
#             best_score = score
#             best_method = method
# print('best_method = ', best_method)
# print('best_k = ', best_k)
# print('best_score = ', best_score)



'''明可夫斯基距离对应的p'''
# best_p = -1
# best_score = 0.0
# best_k = -1
# for k in range(1, 11):
#     for p in range(1, 6):
#         knn_lcf = KNeighborsClassifier(n_neighbors=k, weights="distance")
#         knn_lcf.fit(X_train, y_train)
#         score = knn_lcf.score(X_test, y_test)
#         if score > best_score:
#             best_k = k
#             best_score = score
#             best_p = p
# print('best_p = ', best_p)
# print('best_k = ', best_k)
# print('best_score = ', best_score)


'''网格搜索'''
param_grid = [
    {
        'weights': ['uniform'],
        'n_neighbors': [i for i in range(1, 11)]
    },
    {
        'weights': ['distance'],
        'n_neighbors': [i for i in range(1, 11)],
        'p': [i for i in range(1, 6)]
    }
]
knn_clf = KNeighborsClassifier()
from sklearn.model_selection import GridSearchCV        # 导入网格搜索的库
# grid_search = GridSearchCV(knn_clf, param_grid)
# print(grid_search.fit(X_train, y_train))

grid_search = GridSearchCV(knn_clf, param_grid, n_jobs=-1, verbose=2)
print(grid_search.fit(X_train, y_train))
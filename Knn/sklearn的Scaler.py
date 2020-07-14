import numpy as np
from matplotlib.pyplot import plot
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data
y = iris.target
# print(X[:10,:]

'''归一化处理'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=666)
from sklearn.preprocessing import StandardScaler # 导入scaler类
standardscaler = StandardScaler()
standardscaler.fit(X_train)
# print(standardscaler.mean_)     # 均值
# print(standardscaler.scale_)    # 方差
X_train = standardscaler.transform(X_train)     # 数据归一化处理
X_test_standard = standardscaler.transform(X_test)
# print(X_train)

from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)
print(knn_clf.score(X_test_standard, y_test))
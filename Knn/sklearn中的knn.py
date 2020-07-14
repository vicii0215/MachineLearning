from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

raw_dataX = [
    [3.16515, 2.33546],
    [3.11654, 1.78546],
    [1.34444, 3.36545],
    [3.58221, 4.68798],
    [2.28654, 2.86546],
    [7.42654, 4.68465],
    [5.74654, 3.53987],
    [9.17654, 2.51336],
    [7.79265, 3.42654],
    [7.93959, 0.79845]
]

raw_dataY = [0,0,0,0,0,1,1,1,1,1]

X_train = np.array(raw_dataX)
y_train = np.array(raw_dataY)
x = np.array([8.09321, 3.36465])

KNN_classifier = KNeighborsClassifier(n_neighbors=6)
print(KNN_classifier.fit(X_train, y_train))
# print(KNN_classifier.predict(x))
x_predict = x.reshape(1, -1)
# print(x_predict)
# KNN_classifier.predict(x_predict)
y_predict = KNN_classifier.predict(x_predict)
print(y_predict[0])







from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

boston = datasets.load_boston()
X = boston.data
y = boston.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

# Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor()
dt_reg.fit(X_train, y_train)
print(dt_reg.score(X_test, y_test))     # = 0.58
print(dt_reg.score(X_train, y_train))   # = 1.0 表示过拟合
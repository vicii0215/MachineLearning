import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

boston = datasets.load_boston()

X = boston.data
y = boston.target

X = X[y<50.0]
y = y[y<50.0]


# 使用linearRegression
# from model_selection import train_test_split
from Machine_Learning.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2,seed=666)
from Linear_And_Logistic import LinearRegression
reg = LinearRegression.LinearRegression()
reg.fit_normal(X_train, y_train)
# print(reg.coef_)        # 系数
# print(reg.interception_)        # 截距
# print(reg.score(X_test, y_test))


# 使用sklearn 可解释性
# from sklearn.linear_model import LinearRegression
# lin_reg = LinearRegression()
# lin_reg.fit(X, y)
#
# print(lin_reg.coef_)
# print(np.argsort(lin_reg.coef_))    # 负相关到正相关排序
# print(boston.feature_names)     # 所有的有关变量
# print(boston.feature_names[np.argsort(lin_reg.coef_)])  # 相关变量从负到正排序

# 普通线性回归预测
lin_reg1 = LinearRegression.LinearRegression()
lin_reg1.fit_normal(X_train, y_train)
print(lin_reg1.score(X_test, y_test))

# 使用向量，没有数据预处理（归一化处理）
lin_reg2 = LinearRegression.LinearRegression()
lin_reg2.fit_gd(X_train, y_train, eta=0.000001)
print(lin_reg2.score(X_test, y_test))

# 归一化处理数据
from sklearn.preprocessing import StandardScaler
standard = StandardScaler()
standard.fit(X_train)
X_train_standard = standard.transform(X_train)
lin_reg3 = LinearRegression.LinearRegression()
lin_reg3.fit_gd(X_train_standard, y_train)
X_test_standard = standard.transform(X_test)
print(lin_reg3.score(X_test_standard, y_test))


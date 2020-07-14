import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号


np.random.seed(42)
x = np.random.uniform(-3.0, 3.0, size=100)
X = x.reshape(-1, 1)
y = 0.5*x + 3+ np.random.normal(0, 1, size=100)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


from sklearn.model_selection import train_test_split
np.random.seed(666)
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.metrics import mean_squared_error
def plot_model(model, title, degree, alpha=None):
    X_plot = np.linspace(-3, 3, 100).reshape(100, 1)
    y_plot = model.predict(X_plot)      # 训练好的模型
    plt.scatter(x, y)
    plt.plot(X_plot[:, 0], y_plot, color='r')
    plt.axis([-3, 3, 0, 6])
    plt.title(title+' degree='+str(degree)+' alpha='+str(alpha))
    plt.show()

'''普通多项式'''
def Polynomial_Regression(degree):
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree)),  # 建立多项式
            ("std_scaler", StandardScaler()),  # 归一化处理
            ("lin_reg", LinearRegression())  # 回归方程
        ])
degree = 20
poly10_reg = Polynomial_Regression(degree=degree)       # 回归指数
poly10_reg.fit(X_train, y_train)
y10_predict = poly10_reg.predict(X_test)
error_ratio = mean_squared_error(y_test, y10_predict)
print('普通多项式均方误差 = '+ str(error_ratio))
# plot_model(poly10_reg, '普通多项式拟合', degree)


'''使用岭回归'''
from sklearn.linear_model import Ridge  # 岭
def Ridge_Regression(degree, alpha):
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree)),  # 建立多项式
            ("std_scaler", StandardScaler()),  # 归一化处理
            ("ridge_reg", Ridge(alpha=alpha))  # 岭回归方程
        ])
alpha = 1
ridge_reg = Ridge_Regression(degree, alpha)
ridge_reg.fit(X_train, y_train)
y_predict = ridge_reg.predict(X_test)
mean_error = mean_squared_error(y_test, y_predict)
print('岭回归多项式均方误差 = '+ str(mean_error))
plot_model(ridge_reg, '岭回归多项式拟合',degree, alpha)


'''LASSO对多项式回归正则化处理'''
from sklearn.linear_model import Lasso
def Lasso_Regression(degree, alpha):
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree)),  # 建立多项式
            ("std_scaler", StandardScaler()),  # 归一化处理
            ("ridge_reg", Lasso(alpha=alpha))  # 岭回归方程
        ])
alpha = 0.1
lasso_reg = Lasso_Regression(degree, alpha)
lasso_reg.fit(X_train, y_train)
y_predict = lasso_reg.predict(X_test)
mean_error = mean_squared_error(y_test, y_predict)
print('lasso多项式均方误差 = '+str(mean_error))
plot_model(lasso_reg, 'lasso模型正则化', degree, alpha)

















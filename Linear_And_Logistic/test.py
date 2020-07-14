from Linear_And_Logistic import SimpleLinearRegression
import numpy as np
from Linear_And_Logistic import SimpleLinearRegression
import matplotlib.pyplot as plt


x = np.array([1., 2., 3., 4., 5.,])
y = np.array([1., 3., 2., 3., 5.,])
x_predict = 9

reg1 = SimpleLinearRegression.SimpleLinearRegression1()
# reg1.fit(x, y)
# y_predcit = reg1.predict(np.array([x_predict]))
# print(y_predcit)
# print(reg1.a_)
# print(reg1.b_)


reg2 = SimpleLinearRegression.SimpleLinearRegression2()
# reg2.fit(x, y)
# y_predcit = reg2.predict(np.array([x_predict]))
# print(y_predcit)
# print(reg2.a_)
# print(reg2.b_)
#
# y_hat1 = reg1.predict(x)
# plt.scatter(x, y)
# plt.plot(x, y_hat1, color='blue')
# plt.show()

'''性能测试'''
m = 100
big_x = np.random.random(size=m)
big_y = big_x *2.0 + 3.0 + np.random.normal(size=m) # 加上随机噪声



print(reg1.fit(big_x, big_y))
print(reg2.fit(big_x, big_y))

y_hat1 = reg1.predict(big_x)
plt.scatter(big_x, big_y)
plt.plot(big_x, y_hat1, color='blue')
plt.show()

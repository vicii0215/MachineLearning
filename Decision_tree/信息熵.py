import numpy as np
import matplotlib.pyplot as plt

def entropy(p):
    return -p * np.log(p) - (1-p)*np.log(1-p)


# 绘制信息熵entropy的曲线
x = np.linspace(0.01, 0.99, 200)
plt.plot(x, entropy(x), color='r')
plt.show()
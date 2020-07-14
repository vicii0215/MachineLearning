import numpy as np
import matplotlib.pyplot as plt

x = np.random.randint(0,100,100)
'''最值归一化'''
# print((x-np.min(x))/(np.max(x) - np.min(x)))
X = np.random.randint(0,100,(50,2))
X = np.array(X, dtype=float)
# print(X[:10,:])
X[:,0] = (X[:,0] - np.min(X[:,0])) / (np.max(X[:,0]) - np.min(X[:,0]))
X[:,1] = (X[:,1] - np.min(X[:,1])) / (np.max(X[:,1]) - np.min(X[:,1]))
# plt.scatter(X[:,0], X[:,1])
# plt.show()


'''均值方差归一化'''
X2 = np.random.randint(0,100,(50,2))
X2 = np.array(X2, dtype=float)
X2[:,0] = (X2[:,0] - np.mean(X2[:,0]))/ np.std(X2[:,0]) # 第一列
X2[:,1] = (X2[:,1] - np.mean(X2[:,1]))/ np.std(X2[:,1]) # 第二列
plt.scatter(X2[:,0], X2[:,1])
plt.show()
print(np.mean(X2[:,0]))
print(np.std(X2[:,0]))
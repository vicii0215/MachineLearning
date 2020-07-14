import numpy as np

# 检测准确度
def accuracy_score(y_true, y_predict):
    return sum(y_predict == y_true)/len(y_true)

'''计算MSE'''
def mean_squared_error(y_true, y_predict):
    return np.sum((y_true-y_predict)**2) / len(y_true)

'''计算RMSE'''
def root_mean_squared_error(y_true, y_predict):
    return np.sqrt(mean_squared_error(y_true, y_predict))

'''计算MAE'''
def mean_absolution_error(y_true, y_predict):
    return np.sum(np.absolute(y_true - y_predict)) / len(y_true)

'''计算R2'''
def r2_score(y_true, y_predict):
    return 1-mean_squared_error(y_true, y_predict) / np.var(y_true)
import numpy as np

# 数据集的->训练集和测试集
def train_test_split(X, y, test_ratio=0.2, seed=None):
    assert X.shape[0] == y.shape[0], "x大小必须等于y的大小"
    assert 0.0 <= test_ratio <= 1.0, "测试集的比例必须大于0小于1"

    # 随机种子
    if seed:
        np.random.seed(seed)

    # 乱序处理
    shuffle_indexes = np.random.permutation(len(X))

    # 建立训练集和测试集
    test_ratio = 0.2
    test_size = int(len(X) * test_ratio)
    test_indexes = shuffle_indexes[:test_size]
    train_indexes = shuffle_indexes[test_size:]

    X_train = X[train_indexes]  # 训练集数据
    y_train = y[train_indexes]  # 训练集结果
    X_test = X[test_indexes]  # 测试集数据
    y_test = y[test_indexes]  # 测试集结果

    return X_train, X_test, y_train, y_test
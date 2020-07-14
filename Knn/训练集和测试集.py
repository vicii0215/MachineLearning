from sklearn import datasets
'''导入自己写的KNN和分类库'''
'''sklearn的库'''
from sklearn.model_selection import train_test_split

# 导入鸢尾花的数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target     # 结果标签0,1,2

'''使用model_selection自己的函数接口'''
# # 乱序处理
# shuffle_indexes = np.random.permutation(len(X))
#
# # 建立训练集和测试集
# test_ratio = 0.2
# test_size = int(len(X) * test_ratio)
# test_indexes = shuffle_indexes[:test_size]
# train_indexes = shuffle_indexes[test_size:]
#
# X_train = X[train_indexes]  # 训练集数据
# y_train = y[train_indexes]  # 训练集结果
# X_test = X[test_indexes]    # 测试集数据
# y_test = y[test_indexes]    # 测试集结果

# X_train, X_test, y_train, y_test = train_test_split(X, y)
# my_knn_clf = KNNclassifier(k=3)
# print(my_knn_clf.fit(X_train, y_train))
# y_predict = my_knn_clf.predict(X_test)  # 获取测试集的结果
# # 比较测试结果和真实结果的差距
# print(sum(y_predict == y_test)/len(y_test))


'''use sklearn's train_test_split'''
X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.3)








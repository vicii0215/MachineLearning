from sklearn import datasets
from Meachine_Learning import model_selection
# 导入自己写的knn算法和分类算法
from Knn.KNN import KNNclassifier  # 实现KNN计算

# from metrices import accuracy_score     # 检查准确度

# 手写数据的库
digits = datasets.load_digits()

# print(digits.keys())
# 输出->  dict_keys(['data', 'target', 'target_names', 'images', 'DESCR'])

X = digits.data
y = digits.target
# print(y[:100])
# 查看某个具体数据
# some_digit = X[666]
# some_digit_image = some_digit.reshape(8,8) # 8*8的矩阵
# plt.imshow(some_digit_image, cmap=matplotlib.cm.binary)
# plt.show()
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_ratio=0.2)
my_knn_clf = KNNclassifier(k=3)
my_knn_clf.fit(X_train, y_train)
y_predict = my_knn_clf.predict(X_test)
print(y_predict)
print(sum(y_predict == y_test)/len(y_test))
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


digits = datasets.load_digits()
X = digits.data
y = digits.target.copy()
# 构建偏斜数据
y[digits.target == 9] = 1
y[digits.target != 9] = 0


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_predict = log_reg.predict(X_test)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_predict))

# 计算f1 值
from sklearn.metrics import f1_score
print(f1_score(y_test, y_predict))
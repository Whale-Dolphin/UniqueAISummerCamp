import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据集
data = pd.read_csv('titanic.csv')

# 数据预处理
data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
data = data.dropna()
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
X = data.drop('Survived', axis=1).values
y = data['Survived'].values

# 定义高斯核函数
def gaussian_kernel(x1, x2, sigma):
    return np.exp(-np.sum((x1 - x2) ** 2) / (2 * (sigma ** 2)))

# 计算Gram矩阵
def gram_matrix(X, sigma):
    m = X.shape[0]
    K = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            K[i, j] = gaussian_kernel(X[i], X[j], sigma)
    return K

# 训练SVM模型
def svm_train(X, y, C, sigma):
    m, n = X.shape
    K = gram_matrix(X, sigma)
    P = np.outer(y, y) * K
    q = -np.ones((m, 1))
    G = np.vstack((-np.eye(m), np.eye(m)))
    h = np.vstack((np.zeros((m, 1)), C * np.ones((m, 1))))
    A = y.reshape(1, -1)
    b = np.zeros((1, 1))
    alpha = np.linalg.solve(P, q)
    return alpha

# 预测函数
def svm_predict(X_train, y_train, X_test, alpha, sigma):
    m_train = X_train.shape[0]
    m_test = X_test.shape[0]
    K = np.zeros((m_test, m_train))
    for i in range(m_test):
        for j in range(m_train):
            K[i, j] = gaussian_kernel(X_test[i], X_train[j], sigma)
    y_pred = K.dot(alpha * y_train)
    y_pred[y_pred > 0] = 1
    y_pred[y_pred <= 0] = 0
    return y_pred

# 划分训练集和测试集
m = X.shape[0]
m_train = int(m * 0.8)
m_test = m - m_train
X_train = X[:m_train]
y_train = y[:m_train]
X_test = X[m_train:]
y_test = y[m_train:]

# 训练模型并预测
C = 1
sigma = 0.1
alpha = svm_train(X_train, y_train, C, sigma)
y_pred = svm_predict(X_train, y_train, X_test, alpha, sigma)

# 计算准确率
accuracy = np.mean(y_pred == y_test)
print('Accuracy:', accuracy)

# 可视化
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
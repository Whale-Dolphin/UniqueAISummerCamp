import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

data = pd.read_csv('titanic.csv')

# 对 Age 特征进行平均值填补
data['Age'].fillna(data['Age'].mean(), inplace=True)

# 将 Cabin 特征转换为二元特征
data['Cabin'] = data['Cabin'].notnull().astype(int)

# 将 Sex 特征转换为二元特征
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# 将 Embarked 特征转换为数值特征
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# 划分训练集和测试集
np.random.seed(1)
indices = np.random.permutation(len(data))
train_size = int(len(data) * 0.7)
train_indices, test_indices = indices[:train_size], indices[train_size:]
x_train, x_test = data.iloc[train_indices, 1:], data.iloc[test_indices, 1:]
y_train, y_test = data.iloc[train_indices, 0], data.iloc[test_indices, 0]

# 训练模型（使用 SGD）
x_train = np.hstack((x_train.values, np.ones((len(x_train), 1))))  # 添加偏置项
w = np.zeros((x_train.shape[1], 1))  # 初始化权重
lr = 0.01  # 学习率
epochs = 1000  # 迭代次数
for i in range(epochs):
    idx = np.random.randint(len(x_train))  # 随机选择一个样本
    y_pred = 1 / (1 + np.exp(-np.dot(x_train[idx], w)))  # sigmoid 函数
    grad = x_train[idx].reshape(-1, 1) * (y_pred - y_train[idx])  # 计算梯度
    w -= lr * grad  # 更新权重

# 预测并评估模型
x_test = np.hstack((x_test.values, np.ones((len(x_test), 1))))  # 添加偏置项
y_pred = np.round(1 / (1 + np.exp(-np.dot(x_test, w))))  # 预测标签
print('Logistic Regression with SGD:')
print(classification_report(y_test, y_pred))

# 训练模型（使用 Adam）
x_train = np.hstack((x_train.values, np.ones((len(x_train), 1))))  # 添加偏置项
w = np.zeros((x_train.shape[1], 1))  # 初始化权重
lr = 0.01  # 学习率
beta1 = 0.9  # 一阶矩估计的指数衰减率
beta2 = 0.999  # 二阶矩估计的指数衰减率
eps = 1e-8  # 避免除以零
m = np.zeros_like(w)  # 一阶矩估计
v = np.zeros_like(w)  # 二阶矩估计
for i in range(epochs):
    idx = np.random.randint(len(x_train))  # 随机选择一个样本
    y_pred = 1 / (1 + np.exp(-np.dot(x_train[idx], w)))  # sigmoid 函数
    grad = x_train[idx].reshape(-1, 1) * (y_pred - y_train[idx])  # 计算梯度
    m = beta1 * m + (1 - beta1) * grad  # 更新一阶矩估计
    v = beta2 * v + (1 - beta2) * grad ** 2  # 更新二阶矩估计
    m_hat = m / (1 - beta1 ** (i+1))  # 计算偏差修正后的一阶矩估计
    v_hat = v / (1 - beta2 ** (i+1))  # 计算偏差修正后的二阶矩估计
    w -= lr * m_hat / (np.sqrt(v_hat) + eps)  # 更新权重

# 预测并评估模型
x_test = np.hstack((x_test.values, np.ones((len(x_test), 1))))  # 添加偏置项
y_pred = np.round(1 / (1 + np.exp(-np.dot(x_test, w))))  # 预测标签
print('Logistic Regression with Adam:')
print(classification_report(y_test, y_pred))
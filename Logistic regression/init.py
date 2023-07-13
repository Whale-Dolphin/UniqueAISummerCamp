import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

data = pd.read_csv('titanic.csv')

data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Cabin'] = data['Cabin'].notnull().astype(int)
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

np.random.seed(1)
indices = np.random.permutation(len(data))
train_size = int(len(data) * 0.7)
train_indices, test_indices = indices[:train_size], indices[train_size:]
x_train, x_test = data.iloc[train_indices, 2:], data.iloc[test_indices, 2:]
y_train, y_test = data.iloc[train_indices, 1], data.iloc[test_indices,1]

# SGD
x_train = np.hstack((x_train.values, np.ones((len(x_train), 1)))) 
w = np.zeros((x_train.shape[1], 1))  
lr = 0.01  
epochs = 1000  
for i in range(epochs):
    idx = np.random.randint(len(x_train))  
    y_pred = 1 / (1 + np.exp(-np.dot(x_train[idx], w)))  
    grad = x_train[idx].reshape(-1, 1) * (y_pred - y_train[idx])  
    w -= lr * grad  

    y_pred_all = 1 / (1 + np.exp(-np.dot(x_train, w)))
    loss = -np.mean(y_train * np.log(y_pred_all) + (1 - y_train) * np.log(1 - y_pred_all))
    print(f"Epoch {i+1}: Loss = {loss:.4f}")

x_test = np.hstack((x_test.values, np.ones((len(x_test), 1))))  # 添加偏置项
y_pred = np.round(1 / (1 + np.exp(-np.dot(x_test, w))))  # 预测标签
print('Logistic Regression with SGD:')
print(classification_report(y_test, y_pred))

# # 训练模型（使用 Adam）
# x_train = np.hstack((x_train.values, np.ones((len(x_train), 1))))  # 添加偏置项
# w = np.zeros((x_train.shape[1], 1))  # 初始化权重
# lr = 0.01  # 学习率
# beta1 = 0.9  # 一阶矩估计的指数衰减率
# beta2 = 0.999  # 二阶矩估计的指数衰减率
# eps = 1e-8  # 避免除以零
# m = np.zeros_like(w)  # 一阶矩估计
# v = np.zeros_like(w)  # 二阶矩估计
# for i in range(epochs):
#     idx = np.random.randint(len(x_train))  # 随机选择一个样本
#     y_pred = 1 / (1 + np.exp(-np.dot(x_train[idx], w)))  # sigmoid 函数
#     grad = x_train[idx].reshape(-1, 1) * (y_pred - y_train[idx])  # 计算梯度
#     m = beta1 * m + (1 - beta1) * grad  # 更新一阶矩估计
#     v = beta2 * v + (1 - beta2) * grad ** 2  # 更新二阶矩估计
#     m_hat = m / (1 - beta1 ** (i+1))  # 计算偏差修正后的一阶矩估计
#     v_hat = v / (1 - beta2 ** (i+1))  # 计算偏差修正后的二阶矩估计
#     w -= lr * m_hat / (np.sqrt(v_hat) + eps)  # 更新权重

# # 预测并评估模型
# x_test = np.hstack((x_test.values, np.ones((len(x_test), 1))))  # 添加偏置项
# y_pred = np.round(1 / (1 + np.exp(-np.dot(x_test, w))))  # 预测标签
# print('Logistic Regression with Adam:')
# print(classification_report(y_test, y_pred))
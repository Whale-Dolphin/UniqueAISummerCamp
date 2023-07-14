import pandas as pd
import numpy as np
# from sklearn.metrics import classification_report

data_1 = pd.read_csv('titanic.csv')
data = data_1.drop(['Name', 'Ticket'], axis=1) 
# 舱室和人名理论上来说是可以分下类的但是我不会分，干脆去掉算了，也不影响我学算法，阿巴阿巴
# 终于知道问题在哪了还是数据类型的问题，python可以随便写数据类型导致经常在这个地方出问题，以后要多加注意

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

# # SGD
# x_train = np.hstack((x_train.values, np.ones((len(x_train), 1)))) 
# w = np.zeros((x_train.shape[1], 1))  
# lr = 0.01  # 确定学习率
# epochs = 175    # 确定迭代次数
# alpha = 0.01    # 确定正则化系数
# for i in range(epochs):
#     idx = np.random.randint(len(x_train))
#     y_pred = 1 / (1 + np.exp(-np.dot(x_train[idx], w)))
#     grad = x_train[idx].reshape(-1, 1) * (y_pred - y_train.iloc[idx])+alpha*np.sign(w)
#     w -= lr * grad

#     y_pred_all = 1 / (1 + np.exp(-np.dot(x_train[idx], w)))
#     # print(y_pred_all.shape)
#     loss = -np.mean(y_train.iloc[idx]* np.log(y_pred_all) + (1 - y_train.iloc[idx])* np.log(1 - y_pred_all))
#     print(f"Epoch {i+1}: Loss = {loss:.4f}")

# x_test = np.hstack((x_test.values, np.ones((len(x_test), 1))))  # 添加偏置项
# y_pred = np.round(1 / (1 + np.exp(-np.dot(x_test, w))))  # 预测标签
# acc = np.mean(y_pred.ravel() == y_test)
# print(acc)

# # 动量法
# x_train = np.hstack((x_train.values, np.ones((len(x_train), 1))))  
# w = np.zeros((x_train.shape[1], 1))  

# lr = 0.01  # 确定学习率
# epochs = 175    # 确定迭代次数
# alpha = 0.01    # 确定正则化系数
# beta = 0.9      # 确定动量系数
# v = np.zeros_like(w)

# for i in range(epochs):
#     idx = np.random.randint(len(x_train))
#     y_pred = 1 / (1 + np.exp(-np.dot(x_train[idx], w)))
#     grad = x_train[idx].reshape(-1, 1) * (y_pred - y_train.iloc[idx])+alpha*np.sign(w)
    
#     v = beta * v + (1 - beta) * grad
#     w -= lr * v

#     y_pred_all = 1 / (1 + np.exp(-np.dot(x_train[idx], w)))
#     loss = -np.mean(y_train.iloc[idx]* np.log(y_pred_all) + (1 - y_train.iloc[idx])* np.log(1 - y_pred_all))
#     print(f"Epoch {i+1}: Loss = {loss:.4f}")

# x_test = np.hstack((x_test.values, np.ones((len(x_test), 1))))  # 添加偏置项
# y_pred = np.round(1 / (1 + np.exp(-np.dot(x_test, w))))  # 预测标签
# acc = np.mean(y_pred.ravel() == y_test)
# print(acc)

# # adam
# x_train = np.hstack((x_train.values, np.ones((len(x_train), 1))))  # 添加偏置项
# w = np.zeros((x_train.shape[1], 1))  # 初始化权重
# lr = 0.01  # 学习率
# epochs = 175  # 迭代次数
# alpha = 0.01    # 确定正则化系数
# beta1 = 0.9  # 一阶矩估计的指数衰减率
# beta2 = 0.999  # 二阶矩估计的指数衰减率
# eps = 1e-8 
# m = np.zeros_like(w)
# v = np.zeros_like(w) 
# for i in range(epochs):
#     idx = np.random.randint(len(x_train)) 
#     y_pred = 1 / (1 + np.exp(-np.dot(x_train[idx], w)))
#     grad = x_train[idx].reshape(-1, 1) * (y_pred - y_train.iloc[idx]) +alpha*np.sign(w)
#     m = beta1 * m + (1 - beta1) * grad
#     v = beta2 * v + (1 - beta2) * grad ** 2
#     m_hat = m / (1 - beta1 ** (i+1))
#     v_hat = v / (1 - beta2 ** (i+1))
#     w -= lr * m_hat / (np.sqrt(v_hat) + eps)

# x_test = np.hstack((x_test.values, np.ones((len(x_test), 1))))
# y_pred = np.round(1 / (1 + np.exp(-np.dot(x_test, w)))) 
# acc = np.mean(y_pred.ravel() == y_test)
# print(acc)

# 后记
# 发现迭代次数不能很大，否则会出现过拟合现象，导致测试集上的准确率下降
# 另外一个是迭代次数太大会导致权重参数w直接全部变成NaN，原因不明
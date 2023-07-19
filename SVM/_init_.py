import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import copy

# 读取数据集
data = pd.read_csv('titanic.csv')

# 数据预处理
data['Age'].fillna(data['Age'].mean(), inplace=True)
data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
data = data.dropna()
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# 划分训练集和测试集
np.random.seed(1)
indices = np.random.permutation(len(data))
train_size = int(len(data) * 0.7)
train_indices, test_indices = indices[:train_size], indices[train_size:]
x_train, x_test = data.iloc[train_indices, 2:], data.iloc[test_indices, 2:]
y_train, y_test = data.iloc[train_indices, 1], data.iloc[test_indices,1]
for i in range(len(y_train)):
    if y_train.iloc[i] == 0:
        y_train.iloc[i] = -1
for i in range(len(y_test)):
    if y_test.iloc[i] == 0:
        y_test.iloc[i] = -1

# 定义高斯核函数
def gaussian_kernel(x1, x2):
    gamma = 0.5
    return np.exp(gamma*((x1 - x2) ** 2))

# 定义多项式核函数
def polynomial_kernel(x1, x2):
    c = 0.5
    d = 2
    return (np.dot(x1, x2) + c) ** d

# 定义sigmoid核函数
def sigmoid_kernel(x1, x2):
    sigma = 0.1
    return np.tanh(sigma * np.dot(x1, x2) + 1)

# 选取alpha_j
def select_j_rand(i ,m):
    j = i
    while j == i:
        j = int(np.random.uniform(0, m))
    return j

# 修剪alpha
def clip_alpha(alpha, L, H):
    if alpha > H:
        alpha = H
    elif alpha < L:
        alpha = L
    return alpha

# SMO算法
def smo(data_mat_In, class_label, C, toler, epochs):
    data_matrix = np.mat(data_mat_In)
    label_mat = np.mat(class_label).transpose()
    b = 0
    m, n = np.shape(data_matrix)
    alphas = np.mat(np.zeros((m, 1)))
    iter_num = 0
    alpha_pairs_changed = 0
    for iter_num in range(epochs):
        for i in range(m):
            fxi = float(np.multiply(alphas, label_mat).T*(data_matrix* data_matrix[i, :].T)) + b
            Ei = fxi - float(label_mat[i])
            if (label_mat[i]*Ei < -toler and alphas[i] < C) or (label_mat[i]*Ei > toler and alphas[i] > 0):
                j = select_j_rand(i, m)
                fxj = float(np.multiply(alphas, label_mat).T*(data_matrix* data_matrix[j, :].T)) + b
                Ej = fxj - float(label_mat[j])
                alpha_i_old = copy.deepcopy(alphas[i])
                alpha_j_old = copy.deepcopy(alphas[j])
                if label_mat[i] != label_mat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L == H")
                    continue
                eta = 2.0 * polynomial_kernel(data_matrix[i, :], data_matrix[j, :].T) - polynomial_kernel(data_matrix[i, :], data_matrix[i, :].T) - polynomial_kernel(data_matrix[j, :],data_matrix[j, :].T)
                # if eta >= 0:
                #     print("eta >= 0")
                #     continue
                alphas[j] -= label_mat[j]*(Ei - Ej)/eta
                alphas[j] = clip_alpha(alphas[j], H, L)
                if abs(alphas[j] - alphas[i]) < 0.001:
                    print("alpha_j变化太小")
                    continue
                alphas[i] += label_mat[j]*label_mat[i]*(alpha_j_old - alphas[j])
                b_1 = b - Ei - label_mat[i]*(alphas[i] - alpha_i_old)*polynomial_kernel(data_matrix[i, :], data_matrix[i, :].T) - label_mat[j]*(alphas[j] - alpha_j_old)*polynomial_kernel(data_matrix[i, :], data_matrix[j, :].T)
                b_2 = b - Ej - label_mat[i]*(alphas[i] - alpha_i_old)*polynomial_kernel(data_matrix[i, :], data_matrix[j, :].T) - label_mat[j]*(alphas[j] - alpha_j_old)*polynomial_kernel(data_matrix[j, :], data_matrix[j, :].T)
                if 0 < alphas[i] and C > alphas[i]:
                    b = b_1
                elif 0 < alphas[j] and C > alphas[j]:
                    b = b_2
                else:
                    b = (b_1 + b_2)/2
                alpha_pairs_changed += 1
                print("第%d次迭代 样本：%d , alpha优化次数：%d" % (iter_num, i, alpha_pairs_changed))
                # print(alphas)
        # print(epochs)
        # if alpha_pairs_changed == 0:
        #     iter_num += 1
        print("迭代次数：%d" % iter_num)
    return b, alphas

# 计算w
def caluelate_w(data_mat, label_mat, alphas):
    alphas = np.array(alphas)
    data_mat = np.array(data_mat)
    label_mat = np.array(label_mat)
    w = np.dot((np.tile(label_mat.reshape(1, -1).T, (1, 6))*data_mat).T, alphas)
    return w.tolist()

# 预测
def prediction(x_test, y_test, w, b):
    test = np.mat(x_test)
    y_pred = test*w + b
    y_pred = np.ravel(y_pred)  # 将y_pred转换为一维数组
    y_pred[y_pred >= 0] = 1
    y_pred[y_pred < 0] = -1
    acc = np.mean(y_pred == y_test)
    print("准确率为：%f" % acc)

# 主函数
b, alphas = smo(x_train, y_train, 0.6, 0.001, 10)
w = caluelate_w(x_train, y_train, alphas)
print(w)
prediction(x_test, y_test, w, b)

# # 可视化
# plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
# plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, marker='x')
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.title('SVM Training Result')
# plt.show()
import pandas as pd
import numpy as np
import matplotlib
# import sklearn

df = pd.read_csv('titanic.csv')

# 众数填补
df_copy = df.copy()
mode = df_copy.mode().iloc[0]
df_copy.fillna(mode, inplace=True)
df_copy.to_csv('mode.csv', index=False)

# 均值填补
df_copy_1 = df.copy()
# numeric_cols = df_copy_1.select_dtypes(include=['float64']).columns
# df_copy_1[numeric_cols] = df_copy_1[numeric_cols].apply(pd.to_numeric)
df_drop = df_copy_1.drop(['Name', 'Ticket', 'Cabin', 'Sex', 'Embarked'], axis=1)
mean = df_drop.mean()
df_copy_1.fillna(mean, inplace=True)
df_copy_1.to_csv('mean.csv', index=False)

# 中位数填补
df_copy_2 = df.copy()
df_drop = df_copy_2.drop(['Name', 'Ticket', 'Cabin', 'Sex', 'Embarked'], axis=1)
median = df_drop.median()
df_copy_2.fillna(median, inplace=True)
df_copy_2.to_csv('median.csv', index=False)

# 补0
df_copy_3 = df.copy()
df_copy_3.fillna(0, inplace=True)
df_copy_3.to_csv('zero.csv', index=False)

# knn填补
df_copy_4 = df.copy()
numeric_cols = df_copy_4.select_dtypes(include=['float64']).columns

def distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn_impute(df, idx, k):
    non_missing = df.dropna()
    distances = non_missing.apply(lambda x: distance(x, df.loc[idx]))
    nearest = non_missing.iloc[distances.argsort()[:k]]
    impute_value = nearest.mean()
    return impute_value

for col in numeric_cols:
    missing_idx = df_copy_4[col][df_copy_4[col].isnull()].index
    for idx in missing_idx:
        impute_value = knn_impute(df_copy_4[col], idx, k=5)
        df_copy_4[col][idx] = impute_value
df_copy_4.to_csv('knn.csv', index=False)

# 0-1标准化
df_copy_5 = df_copy_4[numeric_cols].copy()
for col in df_copy_5.columns:
    min_val = df_copy_5[col].min()
    max_val = df_copy_5[col].max()
    df_copy_5[col] = (df_copy_5[col] - min_val) / (max_val - min_val)
# df_copy_5 = MinMaxScaler().fit_transform(df_copy_5)
# df_copy_5 = pd.DataFrame(df_copy_5, columns=df.columns)
df_copy_5.to_csv('normalized_data.csv', index=False)

# z-scroe标准化
df_copy_6 = df[numeric_cols].copy()
for col in df_copy_6.columns:
    mean = df_copy_6[col].mean()
    std = df_copy_6[col].std()
    df_copy_6[col] = (df[col] - mean) / std
# df_copy_6 = StandardScaler().fit_transform(df_copy_6)
# df_copy_6 = pd.DataFrame(df_copy_6, columns=df.columns)
df_copy_6.to_csv('standardized_data.csv', index=False)

#小数标定标准化
df_copy_7 = df[numeric_cols].copy()
for col in df[numeric_cols].columns:
    max_abs = 10**np.ceil(np.log10(df_copy_7.abs().max()))
    df_copy_7[numeric_cols] = df[numeric_cols] / max_abs
# df_copy_7 = MaxAbsScaler().fit_transform(df_copy_7)
# df_copy_7 = pd.DataFrame(df_copy_7, columns=df.columns)
df_copy_7.to_csv('decimal_scaling.csv', index=False)

# logistic回归标准化
df_copy_8 = df[numeric_cols].copy()
for col in df[numeric_cols].columns:
    df_copy_8[numeric_cols] = 1 / (1 + np.exp(-df_copy_8[numeric_cols]))
# df_copy_8 = LogisticScaler().fit_transform(df_copy_8)
# df_copy_8 = pd.DataFrame(df_copy_8, columns=df.columns)
df_copy_8.to_csv('logistic.csv', index=False)

# 哑变量编码
df_copy_9 = df.copy()
df_copy_9 = pd.get_dummies(df_copy_7)
df_copy_9.to_csv('one_hot.csv', index=False)

# one-hot编码
df_copy_10 = df.copy()
# 看了下name、ticket和cabin等列不太适合one-hot编码，这几列就不参与编码了阿巴阿巴
encoded_data = pd.get_dummies(df_copy_10[['Sex', 'Embarked']])
df_encode = pd.concat([df_copy_10, encoded_data], axis=1)
df_encode.to_csv('encode.csv', index=False)
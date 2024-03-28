import numpy as np
import pandas as pd


def load_data():
    x = pd.read_excel('X(2).xlsx', header=None, sheet_name='Sheet1').values
    y = pd.read_excel('Y(2).xlsx', header=None, sheet_name='Sheet1').values
    return x, y


def train_valid_split(X, y, val_indices = [3, 7,11, 16, 19]):
    # 定义验证集索引
      # 验证集索引
    val_indices = np.array(val_indices)-1
    # 生成训练集索引
    train_indices = np.setdiff1d(np.arange(len(X)), val_indices)

    # 根据索引划分训练集和验证集
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]

    return X_train, y_train, X_val, y_val

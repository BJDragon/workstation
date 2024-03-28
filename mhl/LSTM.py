from keras.src.callbacks import ReduceLROnPlateau
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVC

from util import load_data, train_valid_split

# 创建示例数据集
X, y = load_data()
X_train, y_train, X_val, y_val = train_valid_split(X, y, val_indices=[4, 8, 12, 16, 18])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, LSTM
from tensorflow.keras.optimizers import Adam

# 定义RNN模型
model = Sequential([
    LSTM(32, input_shape=(None, 2), activation='tanh', recurrent_activation='tanh'), # 输入形状为(None, 2)，表示可变长度的序列，每个时间步有两个特征
    # Dense(8, activation='relu'),
    # LSTM(64, input_shape=(None, 32), activation='sigmoid', recurrent_activation='tanh'),
    Dense(2)  # 输出两个值
])

# 手动设置学习率
learning_rate = 0.001

# 创建Adam优化器并设置学习率
optimizer = Adam(learning_rate=learning_rate)
# 编译模型
model.compile(optimizer=optimizer, loss='mse')  # 使用均方误差作为损失函数

# 输出模型摘要
model.summary()
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

# 创建ReduceLROnPlateau回调函数
reduce_lr = ReduceLROnPlateau(monitor='val_loss',  # 监视的指标，可以是'val_loss'或'val_accuracy'等
                              factor=0.1,         # 学习率减小的因子，新学习率 = 学习率 * factor
                              patience=50,         # 在多少个epoch内没有改善时减小学习率
                              min_lr=1e-5)        # 学习率的下限

history = model.fit(X_train, y_train, epochs=300, batch_size=4, validation_data=(X_val, y_val),callbacks=[reduce_lr])
y_val_pred = model.predict(X_val)  # 预测验证集数据
# print(history.history)

# 可视化训练过程中的损失变化
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# 绘制预测和真实数据
plt.figure(figsize=(10, 5))

# 预测数据
plt.plot(y_val_pred[:, 0], label='Predicted Value 1', linestyle='--', marker='o')
plt.plot(y_val_pred[:, 1], label='Predicted Value 2', linestyle='--', marker='o')

# 真实数据
plt.plot(y_val[:, 0], label='True Value 1', linestyle='-', marker='x')
plt.plot(y_val[:, 1], label='True Value 2', linestyle='-', marker='x')

plt.title('Predicted vs True Values')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()



y_train_pred = model.predict(X_train)  # 预测验证集数据

plt.figure(figsize=(10, 5))

# 预测数据
plt.plot(y_train_pred[:, 0], label='Predicted Value 1', linestyle='--', marker='o')
plt.plot(y_train_pred[:, 1], label='Predicted Value 2', linestyle='--', marker='o')

# 真实数据
plt.plot(y_train[:, 0], label='True Value 1', linestyle='-', marker='x')
plt.plot(y_train[:, 1], label='True Value 2', linestyle='-', marker='x')

plt.title('Predicted vs True Values (Train)')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()


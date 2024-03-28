from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from util import load_data, train_valid_split
from sklearn.ensemble import RandomForestRegressor

# 创建示例数据集
X, y = load_data()
X_train, y_train, X_val, y_val = train_valid_split(X, y, val_indices=[6, 12, 16, 18])

# 构建多输出决策树模型
decision_tree = RandomForestRegressor(n_estimators=500)
multi_output_tree = MultiOutputRegressor(decision_tree)

# 训练模型
multi_output_tree.fit(X, y)

# 预测
predictions = multi_output_tree.predict(X_val)

# 输出预测结果
print("Actual:", y_val)
print("Predictions:", predictions)
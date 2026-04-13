import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ================= 加载特征和标签 =================
data = np.load('shl_all_features.npz')
X_train = data['X_train_feat']
y_train = data['y_train_enc']
X_val = data['X_val_feat']
y_val = data['y_val_enc']
X_test = data['X_test_feat']
y_test = data['y_test_enc']

# ================= 训练初始模型并获取特征重要性 =================
model_init = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
model_init.fit(X_train, y_train)

# 获取特征重要性
importance = model_init.feature_importances_

# 按重要性降序排序
sorted_idx = np.argsort(importance)[::-1]

# ================= 对不同 K 值进行特征选择与评估 =================
K_values = [100, 200, 300, 400, 500]
val_accuracies = []
test_accuracies = []

for K in K_values:
    print(f"\n--- 选取前 {K} 个特征 ---")
    selected_features = sorted_idx[:K]
    X_train_sel = X_train[:, selected_features]
    X_val_sel = X_val[:, selected_features]
    X_test_sel = X_test[:, selected_features]

    # 训练 XGBoost 模型
    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=5,
        learning_rate=0.3,
        objective='multi:softmax',
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X_train_sel, y_train)

    # 验证集准确率
    y_val_pred = model.predict(X_val_sel)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_accuracies.append(val_acc)

    # 测试集准确率
    y_test_pred = model.predict(X_test_sel)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_accuracies.append(test_acc)

    print(f"验证集准确率: {val_acc:.4f}, 测试集准确率: {test_acc:.4f}")

# ================= 绘制准确率随 K 值变化的折线图 =================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(10, 6))
plt.plot(K_values, val_accuracies, marker='o', label='验证集准确率', linewidth=2)
plt.plot(K_values, test_accuracies, marker='s', label='测试集准确率', linewidth=2)
plt.ylim(0.8, 1.0)
plt.xlabel('选取的特征数量 K', fontsize=14)
plt.ylabel('准确率', fontsize=14)
# plt.title('XGBoost 准确率随特征数量 K 的变化', fontsize=16)
plt.xticks(K_values, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('feature_selection_accuracy.png', dpi=150)
plt.show()

# 打印最佳 K 值
best_k_idx = np.argmax(test_accuracies)
print(f"\n最佳测试准确率 {test_accuracies[best_k_idx]:.4f} 对应的 K = {K_values[best_k_idx]}")
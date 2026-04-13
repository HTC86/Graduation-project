import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 加载特征文件
data = np.load('shl_features.npz')
X_train_feat = data['X_train_feat']
y_train_enc = data['y_train_enc']
X_val_feat = data['X_val_feat']
y_val_enc = data['y_val_enc']
X_test_feat = data['X_test_feat']
y_test_enc = data['y_test_enc']

# 加载标签编码器
label_encoder = joblib.load('label_encoder.pkl')
class_names = [str(c) for c in label_encoder.classes_]  # 转换为字符串列表

# 缺失值处理
all_nan_train = np.isnan(X_train_feat).all(axis=0)
if np.any(all_nan_train):
    print(f"删除训练集中全NaN的列，索引: {np.where(all_nan_train)[0]}")
    X_train_feat = X_train_feat[:, ~all_nan_train]
    X_val_feat = X_val_feat[:, ~all_nan_train]
    X_test_feat = X_test_feat[:, ~all_nan_train]
    print(f"删除后特征维度: {X_train_feat.shape[1]}")

imputer = SimpleImputer(strategy='mean')
X_train_feat = imputer.fit_transform(X_train_feat)
X_val_feat = imputer.transform(X_val_feat)
X_test_feat = imputer.transform(X_test_feat)

joblib.dump(imputer, 'imputer.pkl')

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_feat)
X_val_scaled = scaler.transform(X_val_feat)
X_test_scaled = scaler.transform(X_test_feat)
joblib.dump(scaler, 'feature_scaler.pkl')
print("特征标准化完成。")

# 定义评估函数
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{model_name} 测试准确率: {acc:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name}混淆矩阵')
    plt.xlabel('预测')
    plt.ylabel('真实')
    plt.show()
    return y_pred

# 合并训练集+验证集
X_train_full = np.vstack([X_train_scaled, X_val_scaled])
y_train_full = np.concatenate([y_train_enc, y_val_enc])

# 训练随机森林
print("\n开始训练随机森林...")
rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
rf.fit(X_train_full, y_train_full)
evaluate_model(rf, X_test_scaled, y_test_enc, "随机森林")
joblib.dump(rf, 'random_forest.pkl')

# 训练SVM
print("\n开始训练SVM...")
svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
svm.fit(X_train_full, y_train_full)
evaluate_model(svm, X_test_scaled, y_test_enc, "SVM")
joblib.dump(svm, 'svm.pkl')

# 训练XGBoost
print("\n开始训练XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    random_state=42, use_label_encoder=False, eval_metric='mlogloss'
)
xgb_model.fit(X_train_full, y_train_full)
evaluate_model(xgb_model, X_test_scaled, y_test_enc, "XGBoost")
joblib.dump(xgb_model, 'xgb.pkl')

print("\n所有模型训练完成并已保存。")
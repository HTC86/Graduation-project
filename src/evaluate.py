import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer

# 加载数据
print("加载数据...")
label_encoder = joblib.load('label_encoder.pkl')
class_names = label_encoder.classes_

feat_data = np.load('shl_features.npz')
X_test_feat = feat_data['X_test_feat']
y_test_enc = feat_data['y_test_enc']

# 检查并处理 NaN
print("检查特征数据中的 NaN...")
nan_count = np.isnan(X_test_feat).sum()
print(f"NaN 总数: {nan_count}, 含 NaN 的样本数: {(np.isnan(X_test_feat).any(axis=1)).sum()}")

if nan_count > 0:
    print("使用中位数填充 NaN...")
    imputer = SimpleImputer(strategy='median')
    X_test_feat = imputer.fit_transform(X_test_feat)
    # 可选：保存 imputer 以便后续复用
    # joblib.dump(imputer, 'svm_imputer.pkl')
    print("填充完成，再次检查 NaN:", np.isnan(X_test_feat).sum())
else:
    print("无 NaN，直接使用原始特征。")

# 加载 SVM 模型并预测
print("加载 SVM 模型...")
svm_model = joblib.load('svm.pkl')
y_pred = svm_model.predict(X_test_feat)

# 评估
acc = accuracy_score(y_test_enc, y_pred)
report = classification_report(y_test_enc, y_pred, target_names=class_names)
cm = confusion_matrix(y_test_enc, y_pred)

print(f"\nSVM 准确率: {acc:.4f}")
print(report)

# 绘图保存
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 混淆矩阵
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=ax1, cbar=False)
ax1.set_title('Confusion Matrix')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('True')

# 分类报告文本
ax2.axis('off')
report_lines = report.split('\n')
text = f"Model: SVM\nAccuracy: {acc:.4f}\n\nClassification Report:\n"
text += '\n'.join(report_lines)
ax2.text(0, 1, text, family='monospace', fontsize=10,
         verticalalignment='top', transform=ax2.transAxes)

plt.suptitle(f'SVM - Accuracy: {acc:.4f}', fontsize=14)
plt.tight_layout()
plt.savefig('evaluation_SVM.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig)

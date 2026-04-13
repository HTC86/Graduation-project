import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import RidgeClassifier
from sktime.transformations.panel.rocket import MiniRocket
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.pipeline import Pipeline

# 加载数据集
data = np.load('shl_balanced_motion_location.npz')
X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']
y_test = data['y_test']

# 标签编码
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_val_enc = label_encoder.transform(y_val)
y_test_enc = label_encoder.transform(y_test)
class_names = label_encoder.classes_
num_classes = len(class_names)

# 数据格式转换
X_train = X_train.transpose(0, 2, 1)
X_val = X_val.transpose(0, 2, 1)
X_test = X_test.transpose(0, 2, 1)

# 构建 pipeline
minirocket = MiniRocket(random_state=42)
classifier = RidgeClassifier(alpha=1.0)

pipeline = Pipeline([
    ('minirocket', minirocket),
    ('ridge', classifier)
])

# 训练
print("\n开始训练 MiniRocket (Motion+Location)...")
pipeline.fit(X_train, y_train_enc)

y_pred = pipeline.predict(X_test)
accuracy = np.mean(y_pred == y_test_enc)
print(f"\nMiniRocket (Motion+Location) 测试准确率: {accuracy:.4f}")

print("\n分类报告:")
print(classification_report(y_test_enc, y_pred, target_names=class_names))

# 混淆矩阵
cm = confusion_matrix(y_test_enc, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('MiniRocket (Motion+Location) 混淆矩阵')
plt.xlabel('预测')
plt.ylabel('真实')
plt.show()

joblib.dump(pipeline, 'mini_rocket_pipeline.pkl')
print("模型已保存为 mini_rocket_pipeline.pkl")
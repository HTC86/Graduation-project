import numpy as np
import pandas as pd
import os
import time
import joblib
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# 配置参数
WINDOW_DATA_FILE = 'shl_balanced_motion_location.npz'
FEATURE_DATA_FILE = 'shl_features.npz'
LABEL_ENCODER_FILE = 'label_encoder.pkl'

models_info = [
    # Keras 模型
    {'name': '1D-CNN', 'path': '1d_cnn.h5', 'type': 'keras', 'scaler': 'scaler.pkl'},
    {'name': 'LSTM', 'path': 'lstm.h5', 'type': 'keras', 'scaler': 'scaler.pkl'},
    {'name': 'CNN+LSTM', 'path': 'cnn_lstm.h5', 'type': 'keras', 'scaler': 'scaler.pkl'},
    {'name': 'InceptionTime', 'path': 'inception_time.h5', 'type': 'keras', 'scaler': 'scaler_inception.pkl'},
    # MiniRocket
    {'name': 'MiniRocket', 'path': 'mini_rocket_pipeline.pkl', 'type': 'minirocket', 'scaler': None},
    # 传统机器学习模型
    {'name': '随机森林', 'path': 'random_forest.pkl', 'type': 'sklearn', 'scaler': 'feature_scaler.pkl', 'imputer': 'imputer.pkl'},
    {'name': 'SVM', 'path': 'svm.pkl', 'type': 'sklearn', 'scaler': 'feature_scaler.pkl', 'imputer': 'imputer.pkl'},
    {'name': 'XGBoost', 'path': 'xgb.pkl', 'type': 'sklearn', 'scaler': 'feature_scaler.pkl', 'imputer': 'imputer.pkl'},
]

# 加载数据
print("加载测试数据...")
if not os.path.exists(WINDOW_DATA_FILE):
    raise FileNotFoundError(f"找不到窗口数据文件: {WINDOW_DATA_FILE}")
window_data = np.load(WINDOW_DATA_FILE)
X_test_window = window_data['X_test']
y_test_str = window_data['y_test']

if not os.path.exists(FEATURE_DATA_FILE):
    raise FileNotFoundError(f"找不到特征数据文件: {FEATURE_DATA_FILE}")
feat_data = np.load(FEATURE_DATA_FILE)
X_test_feat = feat_data['X_test_feat']
y_test_enc_feat = feat_data['y_test_enc']

# 加载标签编码器
if not os.path.exists(LABEL_ENCODER_FILE):
    raise FileNotFoundError(f"找不到标签编码器文件: {LABEL_ENCODER_FILE}")
label_encoder = joblib.load(LABEL_ENCODER_FILE)
class_names = label_encoder.classes_
y_test_enc = label_encoder.transform(y_test_str)

# 验证两个测试集标签一致
if not np.array_equal(y_test_enc, y_test_enc_feat):
    raise ValueError("窗口数据和特征数据的测试集标签编码不一致！")

print(f"测试集样本数: {len(y_test_enc)}")
print(f"类别: {class_names}")


# 预处理加载函数
def load_scaler(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到标准化器文件: {path}")
    obj = joblib.load(path)
    if not isinstance(obj, StandardScaler):
        raise TypeError(f"{path} 的内容不是 StandardScaler，实际类型: {type(obj)}")
    print(f"加载标准化器: {path}")
    return obj

# 加载插补器
def load_imputer(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到插补器文件: {path}")
    obj = joblib.load(path)
    if not isinstance(obj, SimpleImputer):
        raise TypeError(f"{path} 的内容不是 SimpleImputer，实际类型: {type(obj)}")
    print(f"加载插补器: {path}")
    return obj


# 评估函数
def evaluate_model(name, y_true, y_pred, model_path, inference_time=None):
    acc = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    size_mb = os.path.getsize(model_path) / (1024*1024) if os.path.exists(model_path) else None

    return {
        '模型名称': name,
        '准确率': f"{acc:.4f}",
        '精确率(宏)': f"{precision_macro:.4f}",
        '召回率(宏)': f"{recall_macro:.4f}",
        'F1(宏)': f"{f1_macro:.4f}",
        '精确率(加权)': f"{precision_weighted:.4f}",
        '召回率(加权)': f"{recall_weighted:.4f}",
        'F1(加权)': f"{f1_weighted:.4f}",
        '模型大小(MB)': f"{size_mb:.2f}" if size_mb else 'N/A',
        '推理时间(ms)': f"{inference_time:.2f}" if inference_time else 'N/A'
    }

# 逐个模型预测
results = []

for info in models_info:
    name = info['name']
    mtype = info['type']
    path = info['path']
    print(f"\n正在评估 {name}  ...")

    if mtype == 'keras':
        if not os.path.exists(path):
            print(f"警告: 模型文件 {path} 不存在，跳过")
            continue
        model = tf.keras.models.load_model(path)
        scaler = load_scaler(info['scaler'])

        n_samples, n_timesteps, n_features = X_test_window.shape
        X_test_2d = X_test_window.reshape(-1, n_features)
        X_test_scaled_2d = scaler.transform(X_test_2d)
        X_test_scaled = X_test_scaled_2d.reshape(n_samples, n_timesteps, n_features)

        start = time.time()
        y_pred_prob = model.predict(X_test_scaled, verbose=0, batch_size=64)
        end = time.time()
        inference_time = (end - start) * 1000 / n_samples
        y_pred = np.argmax(y_pred_prob, axis=1)

    elif mtype == 'minirocket':
        if not os.path.exists(path):
            print(f"警告: 模型文件 {path} 不存在，跳过")
            continue
        pipeline = joblib.load(path)
        X_test_mini = X_test_window.transpose(0, 2, 1)
        start = time.time()
        y_pred = pipeline.predict(X_test_mini)
        end = time.time()
        inference_time = (end - start) * 1000 / len(y_test_enc)

    elif mtype == 'sklearn':
        if not os.path.exists(path):
            print(f"警告: 模型文件 {path} 不存在，跳过")
            continue
        model = joblib.load(path)
        imputer = load_imputer(info['imputer'])
        scaler = load_scaler(info['scaler'])

        X_test_imp = imputer.transform(X_test_feat)
        X_test_scaled = scaler.transform(X_test_imp)

        start = time.time()
        y_pred = model.predict(X_test_scaled)
        end = time.time()
        inference_time = (end - start) * 1000 / len(y_test_enc)

    else:
        continue

    res = evaluate_model(name, y_test_enc, y_pred, path, inference_time)
    results.append(res)

# 生成对比表格
if not results:
    print("没有成功评估任何模型，请检查文件是否存在。")
    exit(1)

df = pd.DataFrame(results)
df['准确率数值'] = df['准确率'].astype(float)
df = df.sort_values('准确率数值', ascending=False).drop(columns='准确率数值')

print("\n\n========== 模型性能对比 ==========")
print(df.to_string(index=False))

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
plt.title('模型性能对比', fontsize=16, pad=20)
plt.tight_layout()
img_file = 'model_comparison.png'
plt.savefig(img_file, dpi=150, bbox_inches='tight')
plt.show()
print(f"表格图片已保存到 {img_file}")
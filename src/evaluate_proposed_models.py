import numpy as np
import os
import time
import joblib
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd

# 加载测试数据
window_data = np.load('shl_balanced_motion_location.npz')
X_test_window = window_data['X_test']
y_test_str = window_data['y_test']

feat_data = np.load('shl_features.npz')
X_test_feat = feat_data['X_test_feat']
y_test_enc_feat = feat_data['y_test_enc']

label_encoder = joblib.load('label_encoder.pkl')
y_test_enc = label_encoder.transform(y_test_str)

if not np.array_equal(y_test_enc, y_test_enc_feat):
    raise ValueError("标签不一致")

class_names = label_encoder.classes_

# 多阶段 XGBoost 模型
def load_multi_stage_xgb():
    sel_C1 = np.load('sel_C1.npy')
    sel_C2 = np.load('sel_C2.npy')
    sel_C3 = np.load('sel_C3.npy')
    sel_C4 = np.load('sel_C4.npy')
    model_C1 = joblib.load('xgb_C1.pkl')
    model_C2 = joblib.load('xgb_C2.pkl')
    model_C3 = joblib.load('xgb_C3.pkl')
    model_C4 = joblib.load('xgb_C4.pkl')
    return sel_C1, sel_C2, sel_C3, sel_C4, model_C1, model_C2, model_C3, model_C4

def multi_stage_predict(x_feat, sel_C1, sel_C2, sel_C3, sel_C4,
                        model_C1, model_C2, model_C3, model_C4):
    X_C1 = x_feat[:, sel_C1]
    pred_C1 = model_C1.predict(X_C1)
    final = np.zeros(len(x_feat), dtype=int)

    dyn_idx = np.where(pred_C1 == 1)[0]
    if len(dyn_idx) > 0:
        X_C4 = x_feat[dyn_idx][:, sel_C4]
        pred_C4 = model_C4.predict(X_C4)
        still_from_dyn = np.where(pred_C4 == 1)[0]
        if len(still_from_dyn) > 0:
            still_global = dyn_idx[still_from_dyn]
            final[still_global] = 4
        remain_dyn = np.where(pred_C4 == 0)[0]
        if len(remain_dyn) > 0:
            remain_global = dyn_idx[remain_dyn]
            X_C2 = x_feat[remain_global][:, sel_C2]
            pred_C2 = model_C2.predict(X_C2)
            mapping_dyn = {0:0, 1:3, 2:7}
            final[remain_global] = [mapping_dyn[p] for p in pred_C2]

    sta_idx = np.where(pred_C1 == 0)[0]
    if len(sta_idx) > 0:
        X_C3 = x_feat[sta_idx][:, sel_C3]
        pred_C3 = model_C3.predict(X_C3)
        mapping_static = {0:4, 1:2, 2:1, 3:6, 4:5}
        for i, idx_in_sta in enumerate(sta_idx):
            final[idx_in_sta] = mapping_static[pred_C3[i]]
    return final

def multi_stage_model_size():
    files = ['xgb_C1.pkl', 'xgb_C2.pkl', 'xgb_C3.pkl', 'xgb_C4.pkl',
             'sel_C1.npy', 'sel_C2.npy', 'sel_C3.npy', 'sel_C4.npy']
    total = 0
    for f in files:
        if os.path.exists(f):
            total += os.path.getsize(f)
    return total / (1024*1024)  # MB

# 评估多阶段 XGBoost
print("评估多阶段 XGBoost...")
sel_C1, sel_C2, sel_C3, sel_C4, m1, m2, m3, m4 = load_multi_stage_xgb()
start = time.time()
y_pred_xgb = multi_stage_predict(X_test_feat, sel_C1, sel_C2, sel_C3, sel_C4, m1, m2, m3, m4)
end = time.time()
inf_time_xgb = (end - start) * 1000 / len(y_test_enc)
acc_xgb = accuracy_score(y_test_enc, y_pred_xgb)
f1_xgb = f1_score(y_test_enc, y_pred_xgb, average='weighted')
size_xgb = multi_stage_model_size()

print(f"准确率: {acc_xgb:.4f}, 加权F1: {f1_xgb:.4f}, 推理时间: {inf_time_xgb:.2f} ms/样本, 模型大小: {size_xgb:.2f} MB")

# 手工特征辅助 CNN 混合模型
# 需要加载时序标准化器
scaler_time = joblib.load('scaler.pkl')
n_features = X_test_window.shape[2]
def scale_time(X):
    orig_shape = X.shape
    X_2d = X.reshape(-1, n_features)
    X_scaled = scaler_time.transform(X_2d)
    return X_scaled.reshape(orig_shape)
X_test_window_scaled = scale_time(X_test_window)

# 加载增强特征
if os.path.exists('shl_features_enhanced.npz'):
    feat_enhanced = np.load('shl_features_enhanced.npz')
    X_test_feat_enhanced = feat_enhanced['X_test_feat']
else:
    X_test_feat_enhanced = X_test_feat  # 回退

# 特征选择索引
selected_indices = joblib.load('feature_indices.pkl')
X_test_feat_sel = X_test_feat_enhanced[:, selected_indices]

# 加载模型
model_cnn = tf.keras.models.load_model('multi_input_rf_selected.h5')

start = time.time()
y_pred_prob = model_cnn.predict({'time_series': X_test_window_scaled, 'handcrafted': X_test_feat_sel}, verbose=0)
y_pred_cnn = np.argmax(y_pred_prob, axis=1)
end = time.time()
inf_time_cnn = (end - start) * 1000 / len(y_test_enc)
acc_cnn = accuracy_score(y_test_enc, y_pred_cnn)
f1_cnn = f1_score(y_test_enc, y_pred_cnn, average='weighted')
size_cnn = os.path.getsize('multi_input_rf_selected.h5') / (1024*1024)

print(f"手工特征辅助 CNN 准确率: {acc_cnn:.4f}, 加权F1: {f1_cnn:.4f}, 推理时间: {inf_time_cnn:.2f} ms/样本, 模型大小: {size_cnn:.2f} MB")

# 输出表格
results = [
    {'模型名称': '多阶段 XGBoost', '准确率': f'{acc_xgb:.4f}', '加权F1': f'{f1_xgb:.4f}', '推理时间(ms)': f'{inf_time_xgb:.2f}', '模型大小(MB)': f'{size_xgb:.2f}'},
    {'模型名称': '手工特征辅助 CNN', '准确率': f'{acc_cnn:.4f}', '加权F1': f'{f1_cnn:.4f}', '推理时间(ms)': f'{inf_time_cnn:.2f}', '模型大小(MB)': f'{size_cnn:.2f}'}
]

df = pd.DataFrame(results)
print("\n========== 本文提出模型性能 ==========")
print(df.to_string(index=False))

# 可选：绘制表格图片
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(8, 2))
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.5)
plt.title('本文提出模型性能对比', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('proposed_models_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
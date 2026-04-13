import os
import numpy as np
import shap
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import joblib

# ================= 配置 =================
MODEL_PATH = 'multi_input_rf_selected.h5'
FEATURE_INDICES_PATH = 'feature_indices.pkl'
RAW_DATA_FILE = 'shl_balanced_motion_location.npz'
FEAT_FILE = 'shl_all_features.npz'
LABEL_ENCODER_FILE = 'label_encoder.pkl'
SCALER_TIME_FILE = 'scaler_time.pkl'
OUTPUT_DIR = './shap_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_BG_SAMPLES = 100
N_TEST_SAMPLES = 100

# ================= 动态生成完整的特征名称=================
def generate_full_feature_names(expected_length):
    names = []
    channel_names = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    motion_feat_names = [
        'mean', 'std', 'var', 'rms', 'max', 'min', 'ptp', 'skew', 'kurtosis',
        'q25', 'q50', 'q75', 'iqr', 'zero_cross', 'sma', 'crest_factor',
        'shape_factor', 'impulse_factor', 'margin_factor', 'lag1_corr',
        'spectral_centroid', 'spectral_energy', 'spectral_entropy', 'dominant_freq',
        'dominant_mag', 'spectral_skew', 'spectral_kurtosis'
    ]
    time_feat_names = [
        'mean', 'std', 'var', 'rms', 'max', 'min', 'ptp', 'skew', 'kurtosis',
        'q25', 'q50', 'q75', 'iqr', 'zero_cross', 'sma', 'crest_factor',
        'shape_factor', 'impulse_factor', 'margin_factor', 'lag1_corr'
    ]
    freq_feat_names = [
        'spectral_centroid', 'spectral_energy', 'spectral_entropy', 'dominant_freq',
        'dominant_mag', 'spectral_skew', 'spectral_kurtosis'
    ]
    # 子带特征
    N_SUBBANDS = 10
    subband_energy_names = [f'subband_{i}_energy' for i in range(N_SUBBANDS)]
    subband_ratio_names = [f'subband_{i}_energy_ratio' for i in range(N_SUBBANDS)]
    subband_entropy_names = [f'subband_{i}_entropy' for i in range(N_SUBBANDS)]
    # 倒谱系数
    N_CEPSTRAL = 12
    ceps_names = [f'ceps_{i}' for i in range(N_CEPSTRAL)]
    diff_ceps_names = [f'diff_ceps_{i}' for i in range(N_CEPSTRAL)]
    freq_feat_names.extend(subband_energy_names + subband_ratio_names + subband_entropy_names +
                           ceps_names + diff_ceps_names)
    print(f"频域特征数: {len(freq_feat_names)}")
    motion_feat_names = time_feat_names + freq_feat_names
    print(f"每个运动通道特征数: {len(motion_feat_names)}")

    for ch in channel_names:
        for f in motion_feat_names:
            names.append(f'{ch}_{f}')

    # 位置特征
    loc_basic = ['accuracy', 'latitude', 'longitude', 'altitude']
    for lb in loc_basic:
        for stat in ['mean', 'std', 'var', 'min', 'max', 'ptp', 'q25', 'q50', 'q75', 'iqr']:
            names.append(f'{lb}_{stat}')
    for coord in ['lat_diff', 'lon_diff', 'alt_diff']:
        for stat in ['mean', 'std', 'max', 'min', 'change_ratio']:
            names.append(f'{coord}_{stat}')
    names.extend(['total_distance', 'avg_speed', 'bearing_std', 'alt_ptp',
                  'weighted_lat', 'weighted_lon', 'weighted_alt'])
    names.extend(['motion_energy_loc_change', 'motion_energy_div_loc_change'])

    # 如果长度不足，用数字填充
    if len(names) < expected_length:
        print(f"警告: 动态生成的特征名长度 {len(names)} < 实际特征数 {expected_length}，将用数字补全")
        for i in range(len(names), expected_length):
            names.append(f'feat_{i}')
    elif len(names) > expected_length:
        names = names[:expected_length]
    return names

# ================= 加载数据并获取实际特征维度 =================
print("加载数据...")
feat_data = np.load(FEAT_FILE)
X_train_feat = feat_data['X_train_feat']
X_val_feat = feat_data['X_val_feat']
X_test_feat = feat_data['X_test_feat']
y_train_enc = feat_data['y_train_enc']
y_val_enc = feat_data['y_val_enc']
y_test_enc = feat_data['y_test_enc']

# 合并所有特征
X_feat_full = np.vstack([X_train_feat, X_val_feat])
actual_feat_dim = X_feat_full.shape[1]

# 生成特征名称
feature_names = generate_full_feature_names(actual_feat_dim)

# 加载特征选择索引
selected_indices = joblib.load(FEATURE_INDICES_PATH)

if max(selected_indices) >= len(feature_names):
    raise ValueError(f"特征索引超出范围: max index {max(selected_indices)} >= {len(feature_names)}")

X_feat_full_sel = X_feat_full[:, selected_indices]
X_test_feat_sel = X_test_feat[:, selected_indices]
selected_feature_names = [feature_names[i] for i in selected_indices]

# ================= 加载原始时序数据 =================
raw_data = np.load(RAW_DATA_FILE)
X_train_raw = raw_data['X_train']
X_val_raw = raw_data['X_val']
X_test_raw = raw_data['X_test']

label_encoder = joblib.load(LABEL_ENCODER_FILE)
class_names = label_encoder.classes_
num_classes = len(class_names)

scaler_time = joblib.load(SCALER_TIME_FILE)
n_features_raw = X_train_raw.shape[2]

def scale_raw_data(X, scaler, n_features):
    orig_shape = X.shape
    X_2d = X.reshape(-1, n_features)
    X_scaled = scaler.transform(X_2d)
    return X_scaled.reshape(orig_shape)

X_train_raw_scaled = scale_raw_data(X_train_raw, scaler_time, n_features_raw)
X_val_raw_scaled = scale_raw_data(X_val_raw, scaler_time, n_features_raw)
X_test_raw_scaled = scale_raw_data(X_test_raw, scaler_time, n_features_raw)

X_raw_full = np.vstack([X_train_raw_scaled, X_val_raw_scaled])

# ================= 加载模型 =================
print("加载混合模型...")
model = keras.models.load_model(MODEL_PATH)
handcrafted_dim = model.inputs[1].shape[-1]

if len(selected_feature_names) != handcrafted_dim:
    raise ValueError(f"维度不匹配！特征选择后维度为 {len(selected_feature_names)}, 模型期望 {handcrafted_dim}")

# ================= 选择背景数据和测试样本 =================
np.random.seed(42)
bg_indices = np.random.choice(len(X_raw_full), min(N_BG_SAMPLES, len(X_raw_full)), replace=False)
bg_raw = X_raw_full[bg_indices]
bg_feat = X_feat_full_sel[bg_indices]
bg_raw_constant = np.mean(bg_raw, axis=0, keepdims=True)

test_indices = np.random.choice(len(X_test_raw_scaled), N_TEST_SAMPLES, replace=False)
test_raw = X_test_raw_scaled[test_indices]
test_feat = X_test_feat_sel[test_indices]
test_labels = y_test_enc[test_indices]

# ================= Kernel SHAP =================
print("\n计算 Kernel SHAP...")

def predict_wrapper(feat_input):
    n = feat_input.shape[0]
    raw_input = np.tile(bg_raw_constant, (n, 1, 1))
    return model.predict([raw_input, feat_input], verbose=0)

explainer = shap.KernelExplainer(predict_wrapper, bg_feat, link="logit")
shap_values = explainer.shap_values(test_feat, nsamples=100)

print(f"shap_values 列表长度: {len(shap_values)}")
print(f"shap_values[0].shape: {shap_values[0].shape}")

n_samples = len(shap_values)
n_features = shap_values[0].shape[0]
n_classes = shap_values[0].shape[1]
print(f"样本数: {n_samples}, 特征数: {n_features}, 类别数: {n_classes}")

# ================= 全局特征重要性 =================
global_importance = np.zeros(n_features)
for sv in shap_values:
    global_importance += np.mean(np.abs(sv), axis=1)
global_importance /= n_samples

top_k = min(20, n_features)
top_indices = np.argsort(global_importance)[-top_k:][::-1]
top_importance = global_importance[top_indices]
top_feature_names = [selected_feature_names[i] for i in top_indices]

plt.figure(figsize=(10, 8))
plt.barh(range(top_k), top_importance, color='steelblue')
plt.yticks(range(top_k), top_feature_names)
plt.xlabel('Mean |SHAP value|')
plt.title('Top Handcrafted Features - Global Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'handcrafted_shap_global.png'), dpi=150)
plt.show()

# ================= 时序梯度热力图 =================
print("计算时序梯度热力图...")
sample_indices = []
for c in range(num_classes):
    idx = np.where(y_test_enc == c)[0]
    if len(idx) > 0:
        sample_indices.append(idx[0])

sample_raw = X_test_raw_scaled[sample_indices]
sample_feat = X_test_feat_sel[sample_indices]
sample_labels = y_test_enc[sample_indices]
sample_class_names = [class_names[l] for l in sample_labels]

def compute_gradient_importance(raw_input, feat_input, label):
    raw_tf = tf.convert_to_tensor(raw_input, dtype=tf.float32)
    feat_tf = tf.convert_to_tensor(feat_input, dtype=tf.float32)
    label_tf = tf.convert_to_tensor([label], dtype=tf.int32)
    with tf.GradientTape() as tape:
        tape.watch(raw_tf)
        pred = model([raw_tf, feat_tf], training=False)
        loss = tf.keras.losses.sparse_categorical_crossentropy(label_tf, pred)
    grad = tape.gradient(loss, raw_tf)
    if grad is not None:
        importance = np.abs(grad.numpy()[0] * raw_input[0])
    else:
        importance = np.zeros_like(raw_input[0])
    return importance

importance_maps = []
for i, idx in enumerate(sample_indices):
    raw = X_test_raw_scaled[idx:idx+1]
    feat = X_test_feat_sel[idx:idx+1]
    label = y_test_enc[idx]
    imp = compute_gradient_importance(raw, feat, label)
    importance_maps.append(imp)

channel_labels = [
    'AX', 'AY', 'AZ',      # 加速度 X, Y, Z
    'GX', 'GY', 'GZ',      # 陀螺仪 X, Y, Z
    'Acc',                 # 定位精度
    'Lat',                 # 纬度
    'Lon',                 # 经度
    'Alt'                  # 海拔
]
fig, axes = plt.subplots(len(sample_indices), 1, figsize=(12, 3*len(sample_indices)))
if len(sample_indices) == 1:
    axes = [axes]
for i, (imp, cls_name) in enumerate(zip(importance_maps, sample_class_names)):
    im = axes[i].imshow(imp.T, aspect='auto', cmap='hot', interpolation='nearest')
    axes[i].set_title(f'Class: {cls_name} - Gradient×Input Importance')
    axes[i].set_xlabel('Time step')
    axes[i].set_ylabel('Channels', fontsize=10)
    axes[i].set_yticks(range(len(channel_labels)))
    axes[i].set_yticklabels(channel_labels, fontsize=8)
    plt.colorbar(im, ax=axes[i])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'temporal_importance_heatmap.png'), dpi=150)
plt.show()

# ================= 单个样本瀑布图 =================
sample_idx = 0
feat_single = test_feat[sample_idx:sample_idx+1]
label_single = test_labels[sample_idx]
pred_proba = model.predict([test_raw[sample_idx:sample_idx+1], feat_single], verbose=0)[0]
pred_class = np.argmax(pred_proba)
pred_class_name = class_names[pred_class]
true_class_name = class_names[label_single]

shap_single = shap_values[sample_idx][:, pred_class]
base_value = explainer.expected_value[pred_class] if hasattr(explainer.expected_value, '__len__') else explainer.expected_value

plt.figure(figsize=(12, 8))
shap.plots.waterfall(
    shap.Explanation(
        values=shap_single,
        base_values=base_value,
        data=feat_single[0],
        feature_names=selected_feature_names
    ),
    max_display=15, show=False
)
plt.title(f'Waterfall Plot (True: {true_class_name}, Pred: {pred_class_name})')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'waterfall_sample.png'), dpi=150)
plt.show()

print(f"\n所有结果已保存至 {OUTPUT_DIR}")
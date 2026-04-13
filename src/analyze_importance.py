import numpy as np
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd

# ================= 配置 =================
MULTI_PREFIX = 'xgb_v5_'
SINGLE_MODEL_FILE = 'xgb.pkl'
OUTPUT_DIR = './picture/'

# ================= 生成特征名称 =================
def generate_feature_names():
    names = []
    # 运动通道名称
    channel_names = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    # 每个通道的26个特征
    motion_feat_names = [
        'mean', 'std', 'var', 'rms', 'max', 'min', 'ptp', 'skew', 'kurtosis',
        'q25', 'q50', 'q75', 'iqr',
        'zero_cross', 'sma', 'crest_factor', 'shape_factor', 'impulse_factor', 'margin_factor',
        'spectral_centroid', 'spectral_energy', 'spectral_entropy', 'dominant_freq', 'dominant_mag',
        'spectral_skew', 'spectral_kurtosis'
    ]
    for ch in channel_names:
        for f in motion_feat_names:
            names.append(f'{ch}_{f}')

    # 位置特征
    loc_basic = ['accuracy', 'latitude', 'longitude', 'altitude']
    for lb in loc_basic:
        for stat in ['mean', 'std', 'var', 'min', 'max', 'ptp', 'q25', 'q50', 'q75', 'iqr']:
            names.append(f'{lb}_{stat}')

    # 差分特征
    for coord in ['lat_diff', 'lon_diff', 'alt_diff']:
        for stat in ['mean', 'std', 'max', 'min', 'change_ratio']:
            names.append(f'{coord}_{stat}')

    # 其他位置特征
    names.extend(['total_distance', 'avg_speed', 'bearing_std', 'alt_ptp',
                  'weighted_lat', 'weighted_lon', 'weighted_alt'])

    # 联合特征
    names.extend(['motion_energy_loc_change', 'motion_energy_div_loc_change'])

    return names

feature_names = generate_feature_names()

# ================= 辅助函数：获取Top K重要性 =================
def get_top_features(importances, indices, feature_names, top_n=20):

    if indices is not None:
        # 多阶段模型
        sorted_idx = np.argsort(importances)[::-1]
        top_orig_indices = indices[sorted_idx[:top_n]]
        top_names = [feature_names[i] for i in top_orig_indices]
        top_scores = importances[sorted_idx[:top_n]]
    else:
        # 单阶段模型
        sorted_idx = np.argsort(importances)[::-1]
        top_orig_indices = sorted_idx[:top_n]
        top_names = [feature_names[i] for i in top_orig_indices]
        top_scores = importances[sorted_idx[:top_n]]
    return top_names, top_scores

def plot_importance(names, scores, title, save_path):
    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(names))
    plt.barh(y_pos, scores[::-1])
    plt.yticks(y_pos, names[::-1])
    plt.xlabel('Feature Importance')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# ================= 加载模型 =================
sel_C1 = np.load('sel_C1.npy')
sel_C2 = np.load('sel_C2.npy')
sel_C3 = np.load('sel_C3.npy')
sel_C4 = np.load('sel_C4.npy')

model_C1 = joblib.load(f'{MULTI_PREFIX}C1.pkl')
model_C2 = joblib.load(f'{MULTI_PREFIX}C2.pkl')
model_C3 = joblib.load(f'{MULTI_PREFIX}C3.pkl')
model_C4 = joblib.load(f'{MULTI_PREFIX}C4.pkl')

model_single = joblib.load(SINGLE_MODEL_FILE)

# ================= 分析 C1 =================
top_names, top_scores = get_top_features(model_C1.feature_importances_, sel_C1, feature_names, 20)
print("\n=== C1 (动态/静态) 特征重要性 Top 20 ===")
for name, score in zip(top_names, top_scores):
    print(f"{name}: {score:.6f}")
plot_importance(top_names, top_scores, 'C1 - Top 20 Features', f'{OUTPUT_DIR}importance_C1.png')

# ================= 分析 C2 =================
top_names, top_scores = get_top_features(model_C2.feature_importances_, sel_C2, feature_names, 20)
print("\n=== C2 (动态细分类) 特征重要性 Top 20 ===")
for name, score in zip(top_names, top_scores):
    print(f"{name}: {score:.6f}")
plot_importance(top_names, top_scores, 'C2 - Top 20 Features', f'{OUTPUT_DIR}importance_C2.png')

# ================= 分析 C3 =================
top_names, top_scores = get_top_features(model_C3.feature_importances_, sel_C3, feature_names, 20)
print("\n=== C3 (静态五分类) 特征重要性 Top 20 ===")
for name, score in zip(top_names, top_scores):
    print(f"{name}: {score:.6f}")
plot_importance(top_names, top_scores, 'C3 - Top 20 Features', f'{OUTPUT_DIR}importance_C3.png')

# ================= 分析 C4 =================
top_names, top_scores = get_top_features(model_C4.feature_importances_, sel_C4, feature_names, 20)
print("\n=== C4 (Still验证器) 特征重要性 Top 20 ===")
for name, score in zip(top_names, top_scores):
    print(f"{name}: {score:.6f}")
plot_importance(top_names, top_scores, 'C4 - Top 20 Features', f'{OUTPUT_DIR}importance_C4.png')

# ================= 分析单阶段模型 =================
top_names, top_scores = get_top_features(model_single.feature_importances_, None, feature_names, 20)
print("\n=== 单阶段模型 (8分类) 特征重要性 Top 20 ===")
for name, score in zip(top_names, top_scores):
    print(f"{name}: {score:.6f}")
plot_importance(top_names, top_scores, 'Single Stage - Top 20 Features', f'{OUTPUT_DIR}importance_single.png')

# =================保存所有重要性到CSV =================
def save_all_importances_to_csv():
    stages = ['C1', 'C2', 'C3', 'C4', 'Single']
    models = [model_C1, model_C2, model_C3, model_C4, model_single]
    indices = [sel_C1, sel_C2, sel_C3, sel_C4, None]

    df = pd.DataFrame(index=feature_names)
    for stage, model, idx in zip(stages, models, indices):
        imp = model.feature_importances_
        if idx is not None:
            full_imp = np.zeros(len(feature_names))
            full_imp[idx] = imp
            df[stage] = full_imp
        else:
            df[stage] = imp
    df.to_csv('feature_importances_all.csv')
    print("已保存所有特征重要性至 feature_importances_all.csv")

save_all_importances_to_csv()
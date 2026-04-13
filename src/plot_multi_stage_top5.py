import numpy as np
import joblib
import matplotlib.pyplot as plt

# ================= 配置 =================
MULTI_PREFIX = 'xgb_v5_'
MULTI_PREFIX = 'xgb_'

SEL_FILES = {
    'C1': 'sel_C1.npy',
    'C2': 'sel_C2.npy',
    'C3': 'sel_C3.npy',
    'C4': 'sel_C4.npy'
}

MODEL_FILES = {
    'C1': f'{MULTI_PREFIX}C1.pkl',
    'C2': f'{MULTI_PREFIX}C2.pkl',
    'C3': f'{MULTI_PREFIX}C3.pkl',
    'C4': f'{MULTI_PREFIX}C4.pkl'
}

OUTPUT_IMG = 'multi_stage_top5_features.png'

# ================= 生成特征名称=================
def generate_feature_names():
    names = []
    # 运动通道名称
    channel_names = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
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
print(f"总特征数: {len(feature_names)}")

# ================= 辅助函数：获取 Top K 特征 =================
def get_top_features(model, sel_indices, top_n=5):
    importances = model.feature_importances_
    # 按重要性降序排序
    sorted_idx = np.argsort(importances)[::-1]
    top_idx_in_selected = sorted_idx[:top_n]
    top_orig_indices = sel_indices[top_idx_in_selected]
    top_names = [feature_names[i] for i in top_orig_indices]
    top_scores = importances[top_idx_in_selected]
    return top_names, top_scores

# ================= 加载模型和特征索引 =================
models = {}
sel_indices_dict = {}
for stage in ['C1', 'C2', 'C3', 'C4']:
    try:
        models[stage] = joblib.load(MODEL_FILES[stage])
        sel_indices_dict[stage] = np.load(SEL_FILES[stage])
        print(f"成功加载 {stage} 模型和特征索引")
    except Exception as e:
        print(f"加载 {stage} 失败: {e}")
        exit(1)

# ================= 获取各阶段 Top5 =================
top5_data = {}
for stage in ['C1', 'C2', 'C3', 'C4']:
    top5_data[stage] = get_top_features(models[stage], sel_indices_dict[stage], top_n=5)
    print(f"\n{stage} Top5:")
    for name, score in zip(*top5_data[stage]):
        print(f"  {name}: {score:.6f}")

# ================= 绘制子图 =================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

stage_names = ['C1 (动态/静态)', 'C2 (动态细分类)', 'C3 (静态五分类)', 'C4 (Still验证器)']

for idx, stage in enumerate(['C1', 'C2', 'C3', 'C4']):
    ax = axes[idx]
    names, scores = top5_data[stage]
    y_pos = np.arange(len(names))
    ax.barh(y_pos, scores, color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance', fontsize=10)
    ax.set_title(stage_names[idx], fontsize=12)
    for i, (name, score) in enumerate(zip(names, scores)):
        ax.text(score + 0.001, i, f'{score:.4f}', va='center', fontsize=9)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

plt.suptitle('Multi-Stage XGBoost - Top 5 Features per Stage', fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
plt.show()
print(f"\n图片已保存至 {OUTPUT_IMG}")
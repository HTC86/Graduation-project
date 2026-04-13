import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.fft import fft, fftfreq
import os

# ================= 配置 =================
DATA_FILE = 'shl_balanced_motion_location.npz'
FEAT_FILE = 'shl_all_features.npz'
OUTPUT_DIR = './visualization'
os.makedirs(OUTPUT_DIR, exist_ok=True)
FS = 100
CLASS_NAMES = ['Still', 'Walking', 'Run', 'Bike', 'Car', 'Bus', 'Train', 'Subway']


def compute_simple_features(window):
    motion = window[:, :6]  # 加速度+陀螺仪
    acc = motion[:, :3]  # 三轴加速度
    acc_mag = np.sqrt(np.sum(acc ** 2, axis=1))  # 加速度模值

    features = {}
    # 加速度模值的统计量
    features['acc_mag_mean'] = np.mean(acc_mag)
    features['acc_mag_std'] = np.std(acc_mag)
    features['acc_mag_max'] = np.max(acc_mag)
    features['acc_mag_ptp'] = np.ptp(acc_mag)

    # 频域特征：加速度模值的主频能量占比
    n = len(acc_mag)
    if n > 1:
        fft_vals = fft(acc_mag)
        fft_mag = np.abs(fft_vals[:n // 2])
        freqs = fftfreq(n, d=1 / FS)[:n // 2]
        total_energy = np.sum(fft_mag ** 2)
        if total_energy > 0:
            # 主频（最大幅值对应的频率）
            dom_freq = freqs[np.argmax(fft_mag)]
            dom_energy_ratio = np.max(fft_mag ** 2) / total_energy
        else:
            dom_freq, dom_energy_ratio = 0, 0
        features['dom_freq'] = dom_freq
        features['dom_energy_ratio'] = dom_energy_ratio

        # 频域熵
        p = fft_mag / (np.sum(fft_mag) + 1e-10)
        spectral_entropy = -np.sum(p * np.log2(p + 1e-10))
        features['spectral_entropy'] = spectral_entropy
    else:
        features['dom_freq'] = 0
        features['dom_energy_ratio'] = 0
        features['spectral_entropy'] = 0

    # 过零率
    zero_cross = ((acc_mag[:-1] * acc_mag[1:]) < 0).sum() / len(acc_mag)
    features['zero_cross_rate'] = zero_cross

    return features


# ================= 类别分布图 =================
print("加载原始数据...")
data = np.load(DATA_FILE)
X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']
y_test = data['y_test']

# 统计各集合的类别分布
train_counts = Counter(y_train)
val_counts = Counter(y_val)
test_counts = Counter(y_test)

# 绘制堆叠柱状图或分组柱状图
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(CLASS_NAMES))
width = 0.25

train_vals = [train_counts.get(cls, 0) for cls in CLASS_NAMES]
val_vals = [val_counts.get(cls, 0) for cls in CLASS_NAMES]
test_vals = [test_counts.get(cls, 0) for cls in CLASS_NAMES]

ax.bar(x - width, train_vals, width, label='Train', color='skyblue')
ax.bar(x, val_vals, width, label='Validation', color='lightgreen')
ax.bar(x + width, test_vals, width, label='Test', color='salmon')

ax.set_xlabel('Activity Class')
ax.set_ylabel('Number of Samples')
ax.set_title('Class Distribution in Train/Validation/Test Sets')
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'class_distribution.png'), dpi=150)
plt.show()
print(f"类别分布图已保存至 {OUTPUT_DIR}/class_distribution.png")

# ================= 特征分布可视化 =================
print("\n计算特征分布（基于原始数据，计算简单特征）...")
# 随机采样部分样本（避免计算量过大）
sample_frac = 0.3
np.random.seed(42)
n_train = len(X_train)
idx_sample = np.random.choice(n_train, int(n_train * sample_frac), replace=False)
X_sample = X_train[idx_sample]
y_sample = y_train[idx_sample]

# 为每个样本计算特征
feature_list = []
labels_list = []
for i, window in enumerate(X_sample):
    feats = compute_simple_features(window)
    feature_list.append(feats)
    labels_list.append(y_sample[i])

# 转换为DataFrame便于绘图
import pandas as pd

df = pd.DataFrame(feature_list)
df['class'] = labels_list

# 选择几个特征绘制箱线图
features_to_plot = ['acc_mag_mean', 'acc_mag_std', 'dom_energy_ratio', 'spectral_entropy', 'zero_cross_rate']
feature_names_display = {
    'acc_mag_mean': 'Mean of Acc Magnitude',
    'acc_mag_std': 'Std of Acc Magnitude',
    'dom_energy_ratio': 'Dominant Frequency Energy Ratio',
    'spectral_entropy': 'Spectral Entropy',
    'zero_cross_rate': 'Zero Crossing Rate'
}

for feat in features_to_plot:
    plt.figure(figsize=(10, 6))
    # 使用小提琴图+箱线图
    sns.violinplot(x='class', y=feat, data=df, order=CLASS_NAMES, palette='Set2', inner='quartile')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Distribution of {feature_names_display[feat]} by Activity Class')
    plt.xlabel('Activity Class')
    plt.ylabel(feature_names_display[feat])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'feature_dist_{feat}.png'), dpi=150)
    plt.show()
    print(f"特征分布图已保存: {feat}")

# 额外绘制动态vs静态的对比图
df['super_class'] = df['class'].apply(lambda x: 'Dynamic' if x in ['Walking', 'Run', 'Bike'] else 'Static')
plt.figure(figsize=(8, 5))
sns.boxplot(x='super_class', y='acc_mag_std', data=df, palette='Set1')
plt.title('Acceleration Magnitude Std: Dynamic vs Static')
plt.savefig(os.path.join(OUTPUT_DIR, 'dynamic_vs_static_std.png'), dpi=150)
plt.show()

# ================= 时序信号可视化 =================
print("\n绘制各类别原始加速度时序信号...")
# 为每个类别随机选取2个样本，绘制三轴加速度时序图
n_samples_per_class = 2
fig, axes = plt.subplots(len(CLASS_NAMES), n_samples_per_class, figsize=(12, 2.5 * len(CLASS_NAMES)))
if len(CLASS_NAMES) == 1:
    axes = axes.reshape(1, -1)
for i, cls in enumerate(CLASS_NAMES):
    # 找出该类别的所有样本索引
    cls_indices = np.where(y_train == cls)[0]
    if len(cls_indices) == 0:
        print(f"警告：类别 {cls} 在训练集中无样本，跳过")
        continue
    # 随机选择
    chosen = np.random.choice(cls_indices, min(n_samples_per_class, len(cls_indices)), replace=False)
    for j, idx in enumerate(chosen):
        window = X_train[idx]
        acc = window[:, :3]  # 三轴加速度
        time = np.arange(acc.shape[0]) / FS
        ax = axes[i, j] if n_samples_per_class > 1 else axes[i]
        ax.plot(time, acc[:, 0], label='X', alpha=0.7)
        ax.plot(time, acc[:, 1], label='Y', alpha=0.7)
        ax.plot(time, acc[:, 2], label='Z', alpha=0.7)
        ax.set_title(f'{cls} - Sample {j + 1}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration (m/s²)')
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'time_series_signals.png'), dpi=150)
plt.show()
print(f"时序信号图已保存至 {OUTPUT_DIR}/time_series_signals.png")

# 绘制典型动态和静态的对比
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
still_idx = np.random.choice(np.where(y_train == 'Still')[0], 1)[0]
run_idx = np.random.choice(np.where(y_train == 'Run')[0], 1)[0]
still_acc = X_train[still_idx][:, :3]
run_acc = X_train[run_idx][:, :3]
time = np.arange(still_acc.shape[0]) / FS
ax1.plot(time, still_acc[:, 0], label='X')
ax1.plot(time, still_acc[:, 1], label='Y')
ax1.plot(time, still_acc[:, 2], label='Z')
ax1.set_title('Still (Static)')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Acceleration')
ax1.legend()
ax2.plot(time, run_acc[:, 0], label='X')
ax2.plot(time, run_acc[:, 1], label='Y')
ax2.plot(time, run_acc[:, 2], label='Z')
ax2.set_title('Run (Dynamic)')
ax2.set_xlabel('Time (s)')
ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'still_vs_run_example.png'), dpi=150)
plt.show()

print("\n所有可视化完成！")
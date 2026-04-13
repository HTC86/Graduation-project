import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import interpolate

# 配置参数
DATA_ROOT = "SHLDataset_preview_v2"
USERS = ["User1", "User2", "User3"]
POSITIONS = ["Hand", "Bag", "Hips", "Torso"]
WINDOW_SIZE = 200
WINDOW_STEP = 100
TARGET_CLASSES = ['Still', 'Walking', 'Run', 'Bike', 'Car', 'Bus', 'Train', 'Subway']
SENSOR_COLS = [1, 2, 3, 4, 5, 6]       # 加速度 X,Y,Z；陀螺仪 X,Y,Z
LOCATION_COLS = [3, 4, 5, 6]            # accuracy, latitude, longitude, altitude

# 标签映射
LABEL_MAP = {
    0: 'Still', 1: 'Still', 2: 'Still', 3: 'Still',
    4: 'Walking', 5: 'Walking',
    6: 'Run',
    7: 'Bike',
    8: 'Car', 9: 'Car',
    10: 'Bus', 11: 'Bus', 12: 'Bus', 13: 'Bus',
    14: 'Train', 15: 'Train',
    16: 'Subway', 17: 'Subway'
}


# 数据读取
def read_motion_file(filepath):
    df = pd.read_csv(filepath, sep='\s+', header=None)
    df.interpolate(method='linear', limit_direction='both', inplace=True)
    df.fillna(0, inplace=True)
    data = df.values
    return data[:, 0], data[:, SENSOR_COLS]


def read_location_file(filepath):
    df = pd.read_csv(filepath, sep='\s+', header=None)
    df.interpolate(method='linear', limit_direction='both', inplace=True)
    df.fillna(0, inplace=True)
    data = df.values
    return data[:, 0], data[:, LOCATION_COLS]


def read_label_file(filepath):
    df = pd.read_csv(filepath, sep='\s+', header=None)
    df.fillna(0, inplace=True)
    data = df.values
    return data[:, 0], data[:, 2]


def map_fine_to_target(fine_label):
    cls = LABEL_MAP.get(int(fine_label), None)
    return cls if cls in TARGET_CLASSES else None


# 滑动窗口
def sliding_window(features, labels, window_size, step_size):
    n = features.shape[0]
    windows, win_labels = [], []
    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        win_feat = features[start:end, :]
        win_lab = labels[start:end]
        label_mode = np.bincount(win_lab.astype(int)).argmax()
        windows.append(win_feat)
        win_labels.append(label_mode)
    return np.array(windows), np.array(win_labels)


# 处理单个位置
def process_position(record_path, position):
    motion_file = os.path.join(record_path, f"{position}_Motion.txt")
    location_file = os.path.join(record_path, f"{position}_Location.txt")
    label_file = os.path.join(record_path, "Label.txt")

    if not (os.path.exists(motion_file) and os.path.exists(location_file) and os.path.exists(label_file)):
        return [], []

    ts_m, motion = read_motion_file(motion_file)
    ts_l, location = read_location_file(location_file)

    n_loc_features = location.shape[1]
    location_interp = np.zeros((len(ts_m), n_loc_features))
    for col in range(n_loc_features):
        f = interpolate.interp1d(ts_l, location[:, col],
                                  kind='linear',
                                  bounds_error=False,
                                  fill_value='extrapolate')
        location_interp[:, col] = f(ts_m)

    # 合并Motion和Location
    features = np.hstack((motion, location_interp))

    ts_label, fine_labels = read_label_file(label_file)

    if not np.array_equal(ts_m, ts_label):
        f_labels = interpolate.interp1d(ts_label, fine_labels,
                                        kind='nearest',
                                        bounds_error=False,
                                        fill_value='extrapolate')
        fine_labels_aligned = f_labels(ts_m).astype(int)
    else:
        fine_labels_aligned = fine_labels

    windows, win_fine_labels = sliding_window(features, fine_labels_aligned, WINDOW_SIZE, WINDOW_STEP)

    # 映射并过滤目标类别
    valid_windows, valid_labels = [], []
    for w, fl in zip(windows, win_fine_labels):
        target = map_fine_to_target(fl)
        if target is not None:
            valid_windows.append(w)
            valid_labels.append(target)

    return valid_windows, valid_labels


# 主流程
def main():
    all_windows = []
    all_labels = []

    for user in USERS:
        user_path = os.path.join(DATA_ROOT, user)
        if not os.path.isdir(user_path):
            print(f"跳过不存在用户: {user_path}")
            continue

        records = [d for d in os.listdir(user_path)
                   if os.path.isdir(os.path.join(user_path, d)) and not d.startswith('.')]
        for record in tqdm(records, desc=f"处理用户 {user}"):
            record_path = os.path.join(user_path, record)
            for pos in POSITIONS:
                windows, labels = process_position(record_path, pos)
                if windows:
                    all_windows.extend(windows)
                    all_labels.extend(labels)

    print(f"原始样本总数: {len(all_windows)}")

    X = np.array(all_windows)
    y = np.array(all_labels)

    # 检查并删除包含 NaN 的样本
    nan_samples = np.isnan(X).any(axis=(1, 2))
    if nan_samples.any():
        print(f"发现 {nan_samples.sum()} 个样本含 NaN，已删除")
        X = X[~nan_samples]
        y = y[~nan_samples]

    print("原始类别分布:")
    counter = Counter(y)
    for cls in TARGET_CLASSES:
        print(f"  {cls}: {counter.get(cls, 0)}")

    # 可视化原始分布
    plt.figure(figsize=(10, 5))
    plt.bar(counter.keys(), counter.values())
    plt.title("原始类别分布")
    plt.show()

    # 类别平衡采样
    target_per_class = 3000
    classes = np.unique(y)
    indices_per_class = {cls: np.where(y == cls)[0] for cls in classes}

    sampled_indices = []
    for cls in classes:
        cls_indices = indices_per_class[cls]
        n_current = len(cls_indices)
        if n_current >= target_per_class:
            chosen = np.random.choice(cls_indices, target_per_class, replace=False)
        else:
            chosen = np.random.choice(cls_indices, target_per_class, replace=True)
        sampled_indices.extend(chosen)

    X_sampled = X[sampled_indices]
    y_sampled = y[sampled_indices]

    shuffle_idx = np.random.permutation(len(X_sampled))
    X_sampled = X_sampled[shuffle_idx]
    y_sampled = y_sampled[shuffle_idx]

    print("\n采样后类别分布:")
    print(Counter(y_sampled))

    # 训练/验证/测试集 (70% / 15% / 15%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_sampled, y_sampled, test_size=0.3, stratify=y_sampled, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    print(f"\n训练集大小: {len(X_train)}")
    print(f"验证集大小: {len(X_val)}")
    print(f"测试集大小: {len(X_test)}")

    np.savez_compressed('shl_balanced_motion_location.npz',
                        X_train=X_train, y_train=y_train,
                        X_val=X_val, y_val=y_val,
                        X_test=X_test, y_test=y_test)
    print("数据已保存至 shl_balanced_motion_location.npz")


if __name__ == "__main__":
    main()
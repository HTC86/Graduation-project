import numpy as np
from scipy import stats
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm

# 配置参数
DATA_FILE = 'shl_balanced_motion_location.npz'
OUTPUT_FILE = 'shl_features.npz'
FS = 100
WINDOW_SIZE = 200
MOTION_CHANNELS = 6
LOCATION_CHANNELS = 4
EARTH_RADIUS = 6371000

# 加载数据
data = np.load(DATA_FILE)
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

# 特征提取函数

# 计算两点间的大圆距离
def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS * c

# 运动通道特征：时域 + 频域
def extract_motion_features(x):
    feats = []
    feats.append(np.mean(x))
    feats.append(np.std(x))
    feats.append(np.var(x))
    feats.append(np.sqrt(np.mean(x**2)))               # RMS
    feats.append(np.max(x))
    feats.append(np.min(x))
    feats.append(np.ptp(x))                             # 峰峰值
    feats.append(stats.skew(x))
    feats.append(stats.kurtosis(x))
    # 四分位数
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    feats.extend([q25, q50, q75, q75 - q25])
    # 过零率
    zero_cross = ((x[:-1] * x[1:]) < 0).sum() / len(x)
    feats.append(zero_cross)
    # 信号幅度面积 (SMA)
    feats.append(np.sum(np.abs(x)) / len(x))
    # 峰值因子 (Crest factor) = max / RMS
    feats.append(np.max(x) / (np.sqrt(np.mean(x**2)) + 1e-10))
    # 波形因子 = RMS / 平均绝对值
    feats.append(np.sqrt(np.mean(x**2)) / (np.mean(np.abs(x)) + 1e-10))
    # 脉冲因子 = max / 平均绝对值
    feats.append(np.max(x) / (np.mean(np.abs(x)) + 1e-10))
    # 裕度因子 = max / ( (sum(sqrt(|x|))/N )^2 )
    margin = np.max(x) / ( (np.sum(np.sqrt(np.abs(x)))/len(x))**2 + 1e-10)
    feats.append(margin)

    # 频域特征（使用FFT）
    n = len(x)
    fft_vals = fft(x)
    fft_mag = np.abs(fft_vals[:n//2])
    freqs = np.fft.fftfreq(n, d=1/FS)[:n//2]
    if len(fft_mag) > 0:
        # 频谱质心
        spectral_centroid = np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-10)
        feats.append(spectral_centroid)
        # 频谱能量
        spectral_energy = np.sum(fft_mag**2)
        feats.append(spectral_energy)
        # 频谱熵
        p = fft_mag / (np.sum(fft_mag) + 1e-10)
        spectral_entropy = -np.sum(p * np.log2(p + 1e-10))
        feats.append(spectral_entropy)
        # 主频
        dominant_freq = freqs[np.argmax(fft_mag)]
        feats.append(dominant_freq)
        # 主频幅度
        feats.append(np.max(fft_mag))
        # 频谱偏度、峰度
        feats.append(stats.skew(fft_mag))
        feats.append(stats.kurtosis(fft_mag))
    else:
        feats.extend([0]*7)  # 占位

    return feats


# 位置通道特征
def extract_location_features(ch_data, channel_names):
    acc = ch_data[:, 0]
    lat = ch_data[:, 1]
    lon = ch_data[:, 2]
    alt = ch_data[:, 3]

    feats = []

    for ch, name in zip([acc, lat, lon, alt], ['acc','lat','lon','alt']):
        feats.append(np.mean(ch))
        feats.append(np.std(ch))
        feats.append(np.var(ch))
        feats.append(np.min(ch))
        feats.append(np.max(ch))
        feats.append(np.ptp(ch))
        # 四分位数
        q25, q50, q75 = np.percentile(ch, [25,50,75])
        feats.extend([q25, q50, q75, q75-q25])

    # 变化率
    for ch in [lat, lon, alt]:
        diff = np.diff(ch)
        feats.append(np.mean(diff))
        feats.append(np.std(diff))
        feats.append(np.max(diff))
        feats.append(np.min(diff))
        # 变化点数比例
        change_ratio = np.sum(np.abs(diff) > 1e-5) / len(diff)
        feats.append(change_ratio)

    # 位移特征
    total_distance = 0.0
    for i in range(len(lat)-1):
        total_distance += haversine_distance(lat[i], lon[i], lat[i+1], lon[i+1])
    feats.append(total_distance)

    # 平均速度
    time_span = (len(lat)-1) / FS
    avg_speed = total_distance / time_span if time_span > 0 else 0
    feats.append(avg_speed)

    # 位移方向变化标准差
    bearings = []
    for i in range(len(lat)-1):
        y = np.sin(lon[i+1]-lon[i]) * np.cos(lat[i+1])
        x = np.cos(lat[i])*np.sin(lat[i+1]) - np.sin(lat[i])*np.cos(lat[i+1])*np.cos(lon[i+1]-lon[i])
        bearing = np.arctan2(y, x)
        bearings.append(bearing)
    if len(bearings) > 1:
        feats.append(np.std(bearings))
    else:
        feats.append(0)

    # 海拔变化幅度
    feats.append(np.ptp(alt))

    # 精度加权的位置特征，用 accuracy 的倒数作为权重
    w = 1.0 / (acc + 1e-10)
    w = w / np.sum(w)  # 归一化
    weighted_lat = np.sum(w * lat)
    weighted_lon = np.sum(w * lon)
    weighted_alt = np.sum(w * alt)
    feats.extend([weighted_lat, weighted_lon, weighted_alt])

    return feats


# 运动与位置的联合特征
def extract_joint_features(motion_data, location_data):
    # 运动能量（各通道RMS的均值）
    motion_energy = np.mean(np.sqrt(np.mean(motion_data**2, axis=0)))
    # 位置变化率（经纬度差分的均值）
    lat_diff = np.mean(np.abs(np.diff(location_data[:, 1])))
    lon_diff = np.mean(np.abs(np.diff(location_data[:, 2])))
    loc_change = (lat_diff + lon_diff) / 2

    feats = []
    feats.append(motion_energy * loc_change)          # 交互项
    feats.append(motion_energy / (loc_change + 1e-10)) # 比值
    return feats


# 主特征提取函数
def extract_features(signal):
    motion = signal[:, :MOTION_CHANNELS]
    location = signal[:, MOTION_CHANNELS:]

    motion_feats = []
    for ch in range(MOTION_CHANNELS):
        motion_feats.extend(extract_motion_features(motion[:, ch]))

    location_feats = extract_location_features(location, None)

    joint_feats = extract_joint_features(motion, location)

    return np.array(motion_feats + location_feats + joint_feats)

def extract_features_all(X, desc="Extracting"):
    n = X.shape[0]
    feats = []
    for i in tqdm(range(n), desc=desc):
        feats.append(extract_features(X[i]))
    return np.array(feats)

# 执行特征提取
X_train_feat = extract_features_all(X_train, "Train")
X_val_feat = extract_features_all(X_val, "Val")
X_test_feat = extract_features_all(X_test, "Test")

# 标准化
scaler = StandardScaler()
X_train_feat = scaler.fit_transform(X_train_feat)
X_val_feat = scaler.transform(X_val_feat)
X_test_feat = scaler.transform(X_test_feat)

# 保存
np.savez_compressed(OUTPUT_FILE,
                    X_train_feat=X_train_feat, y_train_enc=y_train_enc,
                    X_val_feat=X_val_feat, y_val_enc=y_val_enc,
                    X_test_feat=X_test_feat, y_test_enc=y_test_enc)
print(f"特征已保存至 {OUTPUT_FILE}")
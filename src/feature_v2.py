import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from scipy import stats, signal
from scipy.fft import fft, fftfreq, dct
from scipy.signal.windows import gaussian
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ================= 配置参数 =================
DATA_FILE = 'shl_balanced_motion_location.npz'
OUTPUT_FEAT_FILE = 'shl_all_features.npz'
FS = 100
N_SUBBANDS = 10
USE_FILTERING = True
USE_CEPSTRAL = True
N_CEPSTRAL = 12
MOTION_CHANNELS = 6
LOCATION_CHANNELS = 4
EARTH_RADIUS = 6371000

# 数据加载
print("加载数据...")
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

# 保存标签编码器
joblib.dump(label_encoder, 'label_encoder.pkl')

# 特征提取函数

def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS * c

def bandpass_filter(signal_data, lowcut=1.0, highcut=47.5, fs=FS, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    if len(signal_data) > 3 * order:
        return signal.filtfilt(b, a, signal_data)
    else:
        return signal.lfilter(b, a, signal_data)

def extract_motion_time_features(x):
    feats = []
    feats.append(np.mean(x))
    feats.append(np.std(x))
    feats.append(np.var(x))
    feats.append(np.sqrt(np.mean(x**2)))               # RMS
    feats.append(np.max(x))
    feats.append(np.min(x))
    feats.append(np.ptp(x))
    feats.append(stats.skew(x))
    feats.append(stats.kurtosis(x))
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    feats.extend([q25, q50, q75, q75 - q25])
    zero_cross = ((x[:-1] * x[1:]) < 0).sum() / len(x)
    feats.append(zero_cross)
    feats.append(np.sum(np.abs(x)) / len(x))
    feats.append(np.max(x) / (np.sqrt(np.mean(x**2)) + 1e-10))
    feats.append(np.sqrt(np.mean(x**2)) / (np.mean(np.abs(x)) + 1e-10))
    feats.append(np.max(x) / (np.mean(np.abs(x)) + 1e-10))
    margin = np.max(x) / ( (np.sum(np.sqrt(np.abs(x)))/len(x))**2 + 1e-10)
    feats.append(margin)
    if len(x) > 1:
        x_demean = x - np.mean(x)
        auto_corr = np.correlate(x_demean, x_demean, mode='full')
        if auto_corr[len(x)-1] != 0:
            lag1_corr = auto_corr[len(x)] / auto_corr[len(x)-1]
        else:
            lag1_corr = 0
        feats.append(lag1_corr)
    else:
        feats.append(0)
    return feats

def extract_motion_frequency_features(fft_mag, freqs, fs=FS):
    feats = []
    if len(fft_mag) == 0:
        return [0] * (7 + 3*N_SUBBANDS + 2*N_CEPSTRAL)

    spectral_centroid = np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-10)
    feats.append(spectral_centroid)
    spectral_energy = np.sum(fft_mag**2)
    feats.append(spectral_energy)
    p = fft_mag / (np.sum(fft_mag) + 1e-10)
    spectral_entropy = -np.sum(p * np.log2(p + 1e-10))
    feats.append(spectral_entropy)
    dominant_freq = freqs[np.argmax(fft_mag)]
    feats.append(dominant_freq)
    feats.append(np.max(fft_mag))
    feats.append(stats.skew(fft_mag))
    feats.append(stats.kurtosis(fft_mag))

    max_freq = fs / 2
    subband_width = max_freq / N_SUBBANDS
    total_energy = spectral_energy + 1e-10
    for i in range(N_SUBBANDS):
        low = i * subband_width
        high = (i+1) * subband_width
        mask = (freqs >= low) & (freqs < high)
        sub_mag = fft_mag[mask]
        sub_energy = np.sum(sub_mag**2) if len(sub_mag) > 0 else 0
        feats.append(sub_energy)
        feats.append(sub_energy / total_energy)
        if sub_energy > 0:
            p_sub = sub_mag / (np.sum(sub_mag) + 1e-10)
            sub_entropy = -np.sum(p_sub * np.log2(p_sub + 1e-10))
        else:
            sub_entropy = 0
        feats.append(sub_entropy)

    if USE_CEPSTRAL:
        log_mag = np.log(fft_mag + 1e-10)
        ceps = dct(log_mag, type=2, norm='ortho')[:N_CEPSTRAL]
        if len(ceps) < N_CEPSTRAL:
            ceps = np.pad(ceps, (0, N_CEPSTRAL - len(ceps)), 'constant')
        feats.extend(ceps)
        diff_ceps = np.diff(ceps, prepend=ceps[0])
        feats.extend(diff_ceps)
    else:
        feats.extend([0] * (2 * N_CEPSTRAL))

    return feats

def extract_motion_channel_features(x, fs=FS):
    if USE_FILTERING:
        x_filt = bandpass_filter(x)
    else:
        x_filt = x

    time_feats = extract_motion_time_features(x_filt)

    n = len(x_filt)
    fft_vals = fft(x_filt)
    fft_mag = np.abs(fft_vals[:n//2])
    freqs = fftfreq(n, d=1/fs)[:n//2]

    if USE_FILTERING and len(fft_mag) > 3:
        fft_mag = signal.medfilt(fft_mag, kernel_size=3)
        gauss_win = gaussian(5, std=1.2)
        gauss_win /= gauss_win.sum()
        fft_mag = signal.convolve(fft_mag, gauss_win, mode='same')

    freq_feats = extract_motion_frequency_features(fft_mag, freqs, fs)
    return time_feats + freq_feats

def extract_location_features(loc_data):
    acc = loc_data[:, 0]
    lat = loc_data[:, 1]
    lon = loc_data[:, 2]
    alt = loc_data[:, 3]

    feats = []
    for ch in [acc, lat, lon, alt]:
        feats.append(np.mean(ch))
        feats.append(np.std(ch))
        feats.append(np.var(ch))
        feats.append(np.min(ch))
        feats.append(np.max(ch))
        feats.append(np.ptp(ch))
        q25, q50, q75 = np.percentile(ch, [25,50,75])
        feats.extend([q25, q50, q75, q75 - q25])

    for ch in [lat, lon, alt]:
        diff = np.diff(ch)
        feats.append(np.mean(diff) if len(diff)>0 else 0)
        feats.append(np.std(diff) if len(diff)>0 else 0)
        feats.append(np.max(diff) if len(diff)>0 else 0)
        feats.append(np.min(diff) if len(diff)>0 else 0)
        change_ratio = np.sum(np.abs(diff) > 1e-5) / len(diff) if len(diff)>0 else 0
        feats.append(change_ratio)

    total_distance = 0.0
    for i in range(len(lat)-1):
        total_distance += haversine_distance(lat[i], lon[i], lat[i+1], lon[i+1])
    feats.append(total_distance)

    time_span = (len(lat)-1) / FS
    avg_speed = total_distance / time_span if time_span > 0 else 0
    feats.append(avg_speed)

    speeds = []
    for i in range(len(lat)-1):
        dist = haversine_distance(lat[i], lon[i], lat[i+1], lon[i+1])
        speeds.append(dist * FS)
    if len(speeds) > 0:
        feats.append(np.mean(speeds))
        feats.append(np.std(speeds))
        feats.append(np.max(speeds))
        feats.append(np.min(speeds))
    else:
        feats.extend([0,0,0,0])

    if len(speeds) > 1:
        accel = np.diff(speeds) * FS
        feats.append(np.mean(accel))
        feats.append(np.std(accel))
        feats.append(np.max(accel))
        feats.append(np.min(accel))
    else:
        feats.extend([0,0,0,0])

    bearings = []
    for i in range(len(lat)-1):
        y = np.sin(lon[i+1]-lon[i]) * np.cos(lat[i+1])
        x = np.cos(lat[i])*np.sin(lat[i+1]) - np.sin(lat[i])*np.cos(lat[i+1])*np.cos(lon[i+1]-lon[i])
        bearing = np.arctan2(y, x)
        bearings.append(bearing)
    feats.append(np.std(bearings) if len(bearings)>1 else 0)

    feats.append(np.ptp(alt))

    poor_gps_ratio = np.sum(acc > 20) / len(acc)
    feats.append(poor_gps_ratio)

    w = 1.0 / (acc + 1e-10)
    w = w / np.sum(w)
    weighted_lat = np.sum(w * lat)
    weighted_lon = np.sum(w * lon)
    weighted_alt = np.sum(w * alt)
    feats.extend([weighted_lat, weighted_lon, weighted_alt])

    return feats

def extract_joint_features(motion_data, location_data):
    motion_energy = np.mean(np.sqrt(np.mean(motion_data**2, axis=0)))
    lat_diff = np.mean(np.abs(np.diff(location_data[:, 1]))) if len(location_data) > 1 else 0
    lon_diff = np.mean(np.abs(np.diff(location_data[:, 2]))) if len(location_data) > 1 else 0
    loc_change = (lat_diff + lon_diff) / 2
    feats = []
    feats.append(motion_energy * loc_change)
    feats.append(motion_energy / (loc_change + 1e-10))
    return feats

def extract_all_features(sample):
    motion = sample[:, :MOTION_CHANNELS]
    location = sample[:, MOTION_CHANNELS:]

    motion_feats = []
    for ch in range(MOTION_CHANNELS):
        ch_data = motion[:, ch]
        ch_feats = extract_motion_channel_features(ch_data, FS)
        motion_feats.extend(ch_feats)

    location_feats = extract_location_features(location)
    joint_feats = extract_joint_features(motion, location)

    return np.concatenate([motion_feats, location_feats, joint_feats])

def extract_features_for_dataset(X, desc="Extracting"):
    features = []
    for i in tqdm(range(len(X)), desc=desc):
        feats = extract_all_features(X[i])
        features.append(feats)
    return np.array(features)

# 提取所有特征
X_train_feat = extract_features_for_dataset(X_train, "Train")
X_val_feat = extract_features_for_dataset(X_val, "Val")
X_test_feat = extract_features_for_dataset(X_test, "Test")

# 检查NaN/Inf
if np.isnan(X_train_feat).any() or np.isinf(X_train_feat).any():
    print("警告：特征中存在NaN或Inf，将替换为0")
    X_train_feat = np.nan_to_num(X_train_feat)
    X_val_feat = np.nan_to_num(X_val_feat)
    X_test_feat = np.nan_to_num(X_test_feat)

# 标准化特征
scaler_feat = StandardScaler()
X_train_feat = scaler_feat.fit_transform(X_train_feat)
X_val_feat = scaler_feat.transform(X_val_feat)
X_test_feat = scaler_feat.transform(X_test_feat)
joblib.dump(scaler_feat, 'scaler_feat.pkl')

# 保存所有特征
np.savez_compressed(OUTPUT_FEAT_FILE,
                    X_train_feat=X_train_feat,
                    X_val_feat=X_val_feat,
                    X_test_feat=X_test_feat,
                    y_train_enc=y_train_enc,
                    y_val_enc=y_val_enc,
                    y_test_enc=y_test_enc)
print(f"\n所有特征已保存至 {OUTPUT_FEAT_FILE}")
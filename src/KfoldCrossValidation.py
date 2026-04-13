import os
import numpy as np
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import warnings
warnings.filterwarnings('ignore')

# ================= 配置参数 =================
RANDOM_SEED = 42
N_FOLDS = 5
CONFIDENCE_THRESHOLD = 0.6
OUTPUT_DIR = './cv_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 多阶段XGBoost参数
K_C1, K_C2, K_C3, K_C4 = 300, 300, 300, 300
STILL_WEIGHT_C1, STILL_WEIGHT_C3, STILL_WEIGHT_C4 = 3.0, 5.0, 5.0
XGB_PARAMS_MULTI = {
    'n_estimators': 300,
    'max_depth': 8,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_SEED,
    'n_jobs': -1
}

# 单阶段XGBoost参数
XGB_PARAMS_SINGLE = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
    'objective': 'multi:softmax',
    'num_class': 8
}

# CNN/混合模型训练参数
CNN_EPOCHS = 100
CNN_BATCH_SIZE = 128
CNN_EARLY_STOP_PATIENCE = 10
CNN_REDUCE_LR_PATIENCE = 5
CNN_REDUCE_LR_FACTOR = 0.5

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# 加载数据
feat_data = np.load('shl_all_features.npz')
X_feat = np.vstack([feat_data['X_train_feat'], feat_data['X_val_feat'], feat_data['X_test_feat']])
y_enc = np.concatenate([feat_data['y_train_enc'], feat_data['y_val_enc'], feat_data['y_test_enc']])

raw_data = np.load('shl_balanced_motion_location.npz')
X_raw_train = raw_data['X_train']
X_raw_val = raw_data['X_val']
X_raw_test = raw_data['X_test']
y_raw_train = raw_data['y_train']
y_raw_val = raw_data['y_val']
y_raw_test = raw_data['y_test']

# 标签编码器
label_encoder = joblib.load('label_encoder.pkl')
class_names = label_encoder.classes_

# 合并所有原始数据
X_raw = np.vstack([X_raw_train, X_raw_val, X_raw_test])
y_raw = np.concatenate([y_raw_train, y_raw_val, y_raw_test])
y_raw_enc = label_encoder.transform(y_raw)

# 加载预保存的标准化器并标准化原始时序数据
print("加载 scaler.pkl 并标准化原始时序数据...")
scaler_raw = joblib.load('scaler.pkl')
n_samples, time_steps, n_features_raw = X_raw.shape
X_raw_2d = X_raw.reshape(-1, n_features_raw)
X_raw_scaled_2d = scaler_raw.transform(X_raw_2d)
X_raw_scaled = X_raw_scaled_2d.reshape(n_samples, time_steps, n_features_raw)

# ================= 定义模型类 =================
# 单阶段XGBoost
class SingleStageXGB:
    def __init__(self):
        self.model = None
    def fit(self, X_train, y_train):
        self.model = xgb.XGBClassifier(**XGB_PARAMS_SINGLE)
        self.model.fit(X_train, y_train)
    def predict(self, X_test):
        return self.model.predict(X_test)

# 多阶段XGBoost
class MultiStageXGBoost:
    def __init__(self, conf_threshold=0.6):
        self.conf_threshold = conf_threshold
        self.model_C1 = None
        self.model_C2 = None
        self.model_C3 = None
        self.model_C4 = None
        self.sel_C1 = None
        self.sel_C2 = None
        self.sel_C3 = None
        self.sel_C4 = None
        self.dyn_mapping = None
        self.inv_dyn_mapping = None
        self.static_mapping = None
        self.inv_static_mapping = None

    def _get_mappings(self):
        self.dyn_mapping = {0: 0, 3: 1, 7: 2}
        self.inv_dyn_mapping = {0: 0, 1: 3, 2: 7}
        self.static_mapping = {4: 0, 2: 1, 1: 2, 6: 3, 5: 4}
        self.inv_static_mapping = {0: 4, 1: 2, 2: 1, 3: 6, 4: 5}

    def _feature_selection_train(self, X, y, params, sample_weight=None, k=300):
        temp_model = xgb.XGBClassifier(**params)
        if sample_weight is not None:
            temp_model.fit(X, y, sample_weight=sample_weight)
        else:
            temp_model.fit(X, y)
        importances = temp_model.feature_importances_
        k = min(k, X.shape[1])
        selected = np.argsort(importances)[::-1][:k]
        X_sel = X[:, selected]
        opt_model = xgb.XGBClassifier(**params)
        if sample_weight is not None:
            opt_model.fit(X_sel, y, sample_weight=sample_weight)
        else:
            opt_model.fit(X_sel, y)
        return opt_model, selected

    def fit(self, X, y):
        self._get_mappings()
        dynamic_set = [0, 3, 7]
        y_c1 = np.array([1 if yi in dynamic_set else 0 for yi in y])
        sample_weight_c1 = np.ones_like(y_c1, dtype=float)
        sample_weight_c1[y == 4] = STILL_WEIGHT_C1

        dyn_mask = np.isin(y, dynamic_set)
        X_c2 = X[dyn_mask]
        y_c2 = np.array([self.dyn_mapping[yi] for yi in y[dyn_mask]])

        static_set = [1, 2, 4, 5, 6]
        static_mask = np.isin(y, static_set)
        X_c3 = X[static_mask]
        y_c3 = np.array([self.static_mapping[yi] for yi in y[static_mask]])
        sample_weight_c3 = np.ones_like(y_c3, dtype=float)
        sample_weight_c3[y_c3 == 0] = STILL_WEIGHT_C3

        c4_mask = np.isin(y, [0, 3, 4, 7])
        X_c4 = X[c4_mask]
        y_c4 = np.array([1 if yi == 4 else 0 for yi in y[c4_mask]])
        sample_weight_c4 = np.ones_like(y_c4, dtype=float)
        sample_weight_c4[y_c4 == 1] = STILL_WEIGHT_C4

        params_c1 = XGB_PARAMS_MULTI.copy()
        params_c1['objective'] = 'binary:logistic'
        self.model_C1, self.sel_C1 = self._feature_selection_train(X, y_c1, params_c1,
                                                                   sample_weight_c1, k=K_C1)

        params_c2 = XGB_PARAMS_MULTI.copy()
        params_c2['objective'] = 'multi:softmax'
        params_c2['num_class'] = 3
        self.model_C2, self.sel_C2 = self._feature_selection_train(X_c2, y_c2, params_c2,
                                                                   k=K_C2)

        params_c3 = XGB_PARAMS_MULTI.copy()
        params_c3['objective'] = 'multi:softmax'
        params_c3['num_class'] = 5
        self.model_C3, self.sel_C3 = self._feature_selection_train(X_c3, y_c3, params_c3,
                                                                   sample_weight_c3, k=K_C3)

        params_c4 = XGB_PARAMS_MULTI.copy()
        params_c4['objective'] = 'binary:logistic'
        self.model_C4, self.sel_C4 = self._feature_selection_train(X_c4, y_c4, params_c4,
                                                                   sample_weight_c4, k=K_C4)

    def predict(self, X):
        X_c1 = X[:, self.sel_C1]
        pred_c1 = self.model_C1.predict(X_c1)
        final = np.zeros(len(X), dtype=int)

        static_idx = np.where(pred_c1 == 0)[0]
        if len(static_idx) > 0:
            X_c3 = X[static_idx][:, self.sel_C3]
            pred_c3 = self.model_C3.predict(X_c3)
            for i, idx in enumerate(static_idx):
                final[idx] = self.inv_static_mapping[pred_c3[i]]

        dyn_idx = np.where(pred_c1 == 1)[0]
        if len(dyn_idx) > 0:
            X_dyn = X[dyn_idx]
            X_c2 = X_dyn[:, self.sel_C2]
            proba_c2 = self.model_C2.predict_proba(X_c2)
            max_conf = np.max(proba_c2, axis=1)
            pred_c2 = self.model_C2.predict(X_c2)
            high_conf_mask = max_conf >= self.conf_threshold
            low_conf_mask = ~high_conf_mask

            if np.any(high_conf_mask):
                high_global = dyn_idx[high_conf_mask]
                for i, idx in enumerate(high_global):
                    final[idx] = self.inv_dyn_mapping[pred_c2[high_conf_mask][i]]

            if np.any(low_conf_mask):
                low_global = dyn_idx[low_conf_mask]
                X_low = X_dyn[low_conf_mask]
                X_c4 = X_low[:, self.sel_C4]
                pred_c4 = self.model_C4.predict(X_c4)
                for i, idx in enumerate(low_global):
                    if pred_c4[i] == 1:
                        final[idx] = 4
                    else:
                        final[idx] = self.inv_dyn_mapping[pred_c2[low_conf_mask][i]]
        return final

# CNN模型
def create_cnn_model(input_shape, num_classes=8):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 混合模型
def create_hybrid_model(input_shape_raw, input_shape_feat, num_classes=8):
    raw_input = layers.Input(shape=input_shape_raw, name='raw')
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(raw_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(256, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    feat_input = layers.Input(shape=(input_shape_feat,), name='features')
    y = layers.Dense(256, activation='relu')(feat_input)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.5)(y)

    combined = layers.concatenate([x, y])
    z = layers.Dense(256, activation='relu')(combined)
    z = layers.Dropout(0.5)(z)
    output = layers.Dense(num_classes, activation='softmax')(z)

    model = models.Model(inputs=[raw_input, feat_input], outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# =================交叉验证主循环 =================
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
results = {'SingleXGB': [], 'MultiXGB': [], 'CNN': [], 'Hybrid': []}
fold_details = []

fold = 1
for train_idx, test_idx in skf.split(X_feat, y_enc):
    print(f"\n{'='*50} Fold {fold}/{N_FOLDS} {'='*50}")

    # 划分手工特征数据
    X_feat_train = X_feat[train_idx]
    X_feat_test = X_feat[test_idx]
    y_train = y_enc[train_idx]
    y_test = y_enc[test_idx]

    # 划分原始时序数据
    X_raw_train = X_raw_scaled[train_idx]
    X_raw_test = X_raw_scaled[test_idx]

    # 确保标签一致
    assert np.array_equal(y_train, y_raw_enc[train_idx])
    assert np.array_equal(y_test, y_raw_enc[test_idx])

    # 从训练集中划分验证集
    val_split = 0.15
    val_size = int(len(X_raw_train) * val_split)
    indices = np.random.permutation(len(X_raw_train))
    val_idx = indices[:val_size]
    train_idx_cnn = indices[val_size:]

    X_raw_train_sub = X_raw_train[train_idx_cnn]
    y_train_sub = y_train[train_idx_cnn]
    X_raw_val = X_raw_train[val_idx]
    y_val = y_train[val_idx]

    # 对应的手工特征子集
    X_feat_train_sub = X_feat_train[train_idx_cnn]
    X_feat_val = X_feat_train[val_idx]

    # 单阶段XGBoost
    print("训练单阶段XGBoost...")
    model_sxgb = SingleStageXGB()
    model_sxgb.fit(X_feat_train, y_train)
    pred_sxgb = model_sxgb.predict(X_feat_test)
    acc_sxgb = accuracy_score(y_test, pred_sxgb)
    results['SingleXGB'].append(acc_sxgb)
    print(f"准确率: {acc_sxgb:.4f}")

    # 多阶段XGBoost
    print("训练多阶段XGBoost...")
    model_mxgb = MultiStageXGBoost(conf_threshold=CONFIDENCE_THRESHOLD)
    model_mxgb.fit(X_feat_train, y_train)
    pred_mxgb = model_mxgb.predict(X_feat_test)
    acc_mxgb = accuracy_score(y_test, pred_mxgb)
    results['MultiXGB'].append(acc_mxgb)
    print(f"准确率: {acc_mxgb:.4f}")

    # CNN模型
    print("训练CNN...")
    input_shape_raw = (X_raw_train.shape[1], X_raw_train.shape[2])
    model_cnn = create_cnn_model(input_shape_raw, num_classes=8)

    callbacks_cnn = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(f'cnn_fold_{fold}.h5', monitor='val_accuracy', save_best_only=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]

    history_cnn = model_cnn.fit(
        X_raw_train_sub, y_train_sub,
        validation_data=(X_raw_val, y_val),
        epochs=CNN_EPOCHS, batch_size=CNN_BATCH_SIZE,
        callbacks=callbacks_cnn, verbose=0
    )
    pred_cnn = np.argmax(model_cnn.predict(X_raw_test, verbose=0), axis=1)
    acc_cnn = accuracy_score(y_test, pred_cnn)
    results['CNN'].append(acc_cnn)
    print(f"准确率: {acc_cnn:.4f}")

    # 混合模型
    print("训练混合模型...")
    model_hybrid = create_hybrid_model(input_shape_raw, X_feat_train.shape[1], num_classes=8)

    callbacks_hybrid = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(f'hybrid_fold_{fold}.h5', monitor='val_accuracy', save_best_only=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]

    history_hybrid = model_hybrid.fit(
        [X_raw_train_sub, X_feat_train_sub], y_train_sub,
        validation_data=([X_raw_val, X_feat_val], y_val),
        epochs=CNN_EPOCHS, batch_size=CNN_BATCH_SIZE,
        callbacks=callbacks_hybrid, verbose=0
    )
    pred_hybrid = np.argmax(model_hybrid.predict([X_raw_test, X_feat_test], verbose=0), axis=1)
    acc_hybrid = accuracy_score(y_test, pred_hybrid)
    results['Hybrid'].append(acc_hybrid)
    print(f"准确率: {acc_hybrid:.4f}")

    fold_details.append({
        'fold': fold,
        'SingleXGB': acc_sxgb,
        'MultiXGB': acc_mxgb,
        'CNN': acc_cnn,
        'Hybrid': acc_hybrid
    })
    fold += 1

# ================= 结果汇总与可视化 =================
df_details = pd.DataFrame(fold_details)
df_melted = df_details.melt(id_vars='fold', var_name='Model', value_name='Accuracy')

mean_acc = {m: np.mean(accs) for m, accs in results.items()}
std_acc = {m: np.std(accs) for m, accs in results.items()}

print("\n" + "="*60)
print("5-Fold Cross-Validation Summary")
print("="*60)
summary_df = pd.DataFrame({
    'Model': list(mean_acc.keys()),
    'Mean Accuracy': [f"{mean_acc[m]:.4f}" for m in mean_acc],
    'Std Dev': [f"{std_acc[m]:.4f}" for m in std_acc],
    'Min': [f"{np.min(results[m]):.4f}" for m in mean_acc],
    'Max': [f"{np.max(results[m]):.4f}" for m in mean_acc]
})
print(summary_df.to_string(index=False))

summary_df.to_csv(os.path.join(OUTPUT_DIR, 'cv_summary.csv'), index=False)
df_details.to_csv(os.path.join(OUTPUT_DIR, 'per_fold_results.csv'), index=False)

sns.set_style('whitegrid')
plt.rcParams['font.size'] = 12

# 箱线图
plt.figure(figsize=(10, 6))
sns.boxplot(x='Model', y='Accuracy', data=df_melted, palette='Set2')
plt.title('5-Fold CV Accuracy Distribution', fontsize=14)
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.ylim(0.8, 1.0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'boxplot.png'), dpi=150)
plt.show()

# 折线图
plt.figure(figsize=(10, 6))
for model in results.keys():
    plt.plot(range(1, N_FOLDS+1), results[model], marker='o', label=model)
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Per-Fold Accuracy Comparison')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(0.7, 1.0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'lineplot.png'), dpi=150)
plt.show()

# 条形图
models = list(mean_acc.keys())
means = [mean_acc[m] for m in models]
stds = [std_acc[m] for m in models]
plt.figure(figsize=(10, 6))
bars = plt.bar(models, means, yerr=stds, capsize=10, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.ylabel('Accuracy')
plt.title('Mean Accuracy ± Std Dev over 5 Folds')
plt.ylim(0.7, 1.0)
for bar, mean_val in zip(bars, means):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'barplot_with_error.png'), dpi=150)
plt.show()

# 热力图
heatmap_data = df_details.set_index('fold').T
plt.figure(figsize=(8, 5))
sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlGnBu', cbar_kws={'label': 'Accuracy'})
plt.title('Per-Fold Accuracy Heatmap')
plt.xlabel('Fold')
plt.ylabel('Model')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'heatmap.png'), dpi=150)
plt.show()

print(f"\n所有结果已保存至: {OUTPUT_DIR}")
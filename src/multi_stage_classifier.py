import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 配置参数 =================
FEATURE_FILE = 'shl_features.npz'
LABEL_ENCODER_FILE = 'label_encoder.pkl'
OUTPUT_MODEL_PREFIX = 'xgb_'
RANDOM_SEED = 42
USE_POSTPROCESSING = False
POSTPROCESS_WINDOW = 10

# 置信度阈值（C2低置信度样本触发C4）
CONFIDENCE_THRESHOLD = 0.6

# 各阶段特征数
K_C1 = 300
K_C2 = 300
K_C3 = 300
K_C4 = 300

# 权重因子
STILL_WEIGHT_C1 = 3.0
STILL_WEIGHT_C3 = 5.0
STILL_WEIGHT_C4 = 5.0

# XGBoost基础参数
XGB_PARAMS = {
    'n_estimators': 300,
    'max_depth': 8,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_SEED,
    'n_jobs': -1
}

# ================= 加载数据 =================
print("加载特征数据...")
data = np.load(FEATURE_FILE)
X_train = data['X_train_feat']
y_train_enc = data['y_train_enc']
X_val = data['X_val_feat']
y_val_enc = data['y_val_enc']
X_test = data['X_test_feat']
y_test_enc = data['y_test_enc']

X_train_full = np.vstack([X_train, X_val])
y_train_full = np.concatenate([y_train_enc, y_val_enc])

label_encoder = joblib.load(LABEL_ENCODER_FILE)
class_names = [str(c) for c in label_encoder.classes_]

dynamic_labels = [0, 3, 7]
static_labels = [1, 2, 4, 5, 6]

# ================= 生成各阶段训练标签 =================
# C1: 动态/静态
y_train_C1 = np.array([1 if v in dynamic_labels else 0 for v in y_train_full])
sample_weight_C1 = np.ones_like(y_train_C1, dtype=float)
still_mask_C1 = (y_train_full == 4)
sample_weight_C1[still_mask_C1] = STILL_WEIGHT_C1

# C2: 动态细分类
train_dynamic_mask = np.isin(y_train_full, dynamic_labels)
X_train_C2 = X_train_full[train_dynamic_mask]
y_train_C2 = np.array([{0:0, 3:1, 7:2}[v] for v in y_train_full[train_dynamic_mask]])

# C3: 静态五分类
train_static_mask = np.isin(y_train_full, static_labels)
X_train_C3 = X_train_full[train_static_mask]
y_train_static = y_train_full[train_static_mask]

def label_C3_five(y):
    mapping = {4:0, 2:1, 1:2, 6:3, 5:4}
    return np.array([mapping[v] for v in y])

y_train_C3 = label_C3_five(y_train_static)

sample_weight_C3 = np.ones_like(y_train_C3, dtype=float)
still_mask_C3 = (y_train_static == 4)
sample_weight_C3[still_mask_C3] = STILL_WEIGHT_C3

# C4: 动态分支Still验证器
train_still_dyn_mask = np.isin(y_train_full, [0,3,4,7])
X_train_C4 = X_train_full[train_still_dyn_mask]
y_train_C4 = np.array([1 if v==4 else 0 for v in y_train_full[train_still_dyn_mask]])

sample_weight_C4 = np.ones_like(y_train_C4, dtype=float)
still_mask_C4 = (y_train_full[train_still_dyn_mask] == 4)
sample_weight_C4[still_mask_C4] = STILL_WEIGHT_C4


# ================= 辅助函数 =================
def train_with_feature_selection_xgb(X_train, y_train, K, model_name, base_params, sample_weight=None):
    num_classes = len(np.unique(y_train))
    params = base_params.copy()
    if num_classes > 2:
        params['objective'] = 'multi:softprob'
        params['num_class'] = num_classes
    else:
        params['objective'] = 'binary:logistic'

    temp_model = xgb.XGBClassifier(**params)
    if sample_weight is not None:
        temp_model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        temp_model.fit(X_train, y_train)
    importances = temp_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    K = min(K, X_train.shape[1])
    selected = indices[:K]
    print(f"  {model_name} 选中特征数: {K}")

    X_train_sel = X_train[:, selected]
    opt_model = xgb.XGBClassifier(**params)
    if sample_weight is not None:
        opt_model.fit(X_train_sel, y_train, sample_weight=sample_weight)
    else:
        opt_model.fit(X_train_sel, y_train)
    return opt_model, selected

# ================= 训练各阶段 =================
print("\n开始训练优化后的各阶段分类器...")

model_C1, sel_C1 = train_with_feature_selection_xgb(X_train_full, y_train_C1, K_C1, "C1", XGB_PARAMS, sample_weight=sample_weight_C1)
np.save('sel_C1.npy', sel_C1)
joblib.dump(model_C1, f'{OUTPUT_MODEL_PREFIX}C1.pkl')

model_C2, sel_C2 = train_with_feature_selection_xgb(X_train_C2, y_train_C2, K_C2, "C2", XGB_PARAMS)
np.save('sel_C2.npy', sel_C2)
joblib.dump(model_C2, f'{OUTPUT_MODEL_PREFIX}C2.pkl')

model_C3, sel_C3 = train_with_feature_selection_xgb(X_train_C3, y_train_C3, K_C3, "C3", XGB_PARAMS, sample_weight=sample_weight_C3)
np.save('sel_C3.npy', sel_C3)
joblib.dump(model_C3, f'{OUTPUT_MODEL_PREFIX}C3.pkl')

model_C4, sel_C4 = train_with_feature_selection_xgb(X_train_C4, y_train_C4, K_C4, "C4", XGB_PARAMS, sample_weight=sample_weight_C4)
np.save('sel_C4.npy', sel_C4)
joblib.dump(model_C4, f'{OUTPUT_MODEL_PREFIX}C4.pkl')

print("所有优化模型已保存。")

# ================= 预测函数 =================
def multi_stage_predict(X):
    # C1 预测
    X_C1 = X[:, sel_C1]
    pred_C1 = model_C1.predict(X_C1)
    final = np.zeros(len(X), dtype=int)

    # 静态样本
    sta_idx = np.where(pred_C1 == 0)[0]
    if len(sta_idx) > 0:
        X_C3 = X[sta_idx][:, sel_C3]
        pred_C3 = model_C3.predict(X_C3)
        mapping_static = {0:4, 1:2, 2:1, 3:6, 4:5}
        final[sta_idx] = [mapping_static[p] for p in pred_C3]

    # 动态样本
    dyn_idx = np.where(pred_C1 == 1)[0]
    if len(dyn_idx) > 0:
        X_dyn = X[dyn_idx]
        X_C2 = X_dyn[:, sel_C2]

        # C2 预测概率
        proba_C2 = model_C2.predict_proba(X_C2)
        max_conf = np.max(proba_C2, axis=1)
        pred_C2 = model_C2.predict(X_C2)

        # 高置信度样本直接采纳
        high_conf_mask = max_conf >= CONFIDENCE_THRESHOLD
        low_conf_mask = ~high_conf_mask
        mapping_dyn = {0:0, 1:3, 2:7}
        if np.any(high_conf_mask):
            high_conf_global = dyn_idx[high_conf_mask]
            high_pred_original = [mapping_dyn[p] for p in pred_C2[high_conf_mask]]
            final[high_conf_global] = high_pred_original

        # 低置信度样本：进入 C4 验证器
        if np.any(low_conf_mask):
            low_conf_global = dyn_idx[low_conf_mask]
            X_low = X_dyn[low_conf_mask]
            X_C4 = X_low[:, sel_C4]
            pred_C4 = model_C4.predict(X_C4)

            # 对于低置信度样本，若 C4 判定为 Still 则输出 Still(4)，否则保留 C2 结果
            for i, global_idx in enumerate(low_conf_global):
                if pred_C4[i] == 1:
                    final[global_idx] = 4
                else:
                    final[global_idx] = mapping_dyn[pred_C2[low_conf_mask][i]]

    return final

# ================= 后处理平滑 =================
def smooth_predictions(preds, w):
    smoothed = np.copy(preds)
    half = w // 2
    for i in range(len(preds)):
        start = max(0, i - half)
        end = min(len(preds), i + half + 1)
        window = preds[start:end]
        counts = np.bincount(window)
        smoothed[i] = np.argmax(counts)
    return smoothed

# ================= 测试集评估 =================
print("\n=== 测试集性能 ===")
y_test_pred = multi_stage_predict(X_test)

if USE_POSTPROCESSING:
    print("应用多数投票后处理...")
    y_test_pred = smooth_predictions(y_test_pred, POSTPROCESS_WINDOW)

test_acc = accuracy_score(y_test_enc, y_test_pred)
print(f"测试准确率: {test_acc:.4f}")
print("\n分类报告:")
print(classification_report(y_test_enc, y_test_pred, target_names=class_names))

# 混淆矩阵
cm = confusion_matrix(y_test_enc, y_test_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'XGBoost Multi-Stage with Confidence Threshold {CONFIDENCE_THRESHOLD}')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('confusion_matrix_xgb_confidence.png')
plt.show()

# ================= 各阶段性能评估 =================
print("\n=== 各阶段分类器性能 ===")

# C1
y_test_C1 = np.array([1 if v in dynamic_labels else 0 for v in y_test_enc])
acc_C1 = accuracy_score(y_test_C1, model_C1.predict(X_test[:, sel_C1]))
print(f"C1 (动态/静态) 准确率: {acc_C1:.4f}")

# C2 (仅在真实动态样本上评估)
test_dyn_mask = np.isin(y_test_enc, dynamic_labels)
if test_dyn_mask.any():
    y_test_C2 = np.array([{0:0,3:1,7:2}[v] for v in y_test_enc[test_dyn_mask]])
    X_test_C2 = X_test[test_dyn_mask][:, sel_C2]
    acc_C2 = accuracy_score(y_test_C2, model_C2.predict(X_test_C2))
    print(f"C2 (动态细分类) 准确率: {acc_C2:.4f} (基于 {len(y_test_C2)} 个动态样本)")

# C3
test_static_mask = np.isin(y_test_enc, static_labels)
if test_static_mask.any():
    y_test_C3 = label_C3_five(y_test_enc[test_static_mask])
    X_test_C3 = X_test[test_static_mask][:, sel_C3]
    acc_C3 = accuracy_score(y_test_C3, model_C3.predict(X_test_C3))
    print(f"C3 (静态五分类) 准确率: {acc_C3:.4f} (基于 {len(y_test_C3)} 个静态样本)")

# C4
test_still_dyn_mask = np.isin(y_test_enc, [0,3,4,7])
if test_still_dyn_mask.any():
    y_test_C4 = np.array([1 if v==4 else 0 for v in y_test_enc[test_still_dyn_mask]])
    X_test_C4 = X_test[test_still_dyn_mask][:, sel_C4]
    acc_C4 = accuracy_score(y_test_C4, model_C4.predict(X_test_C4))
    print(f"C4 (动态分支Still验证器) 准确率: {acc_C4:.4f} (基于 {len(y_test_C4)} 个样本)")

print("\n所有评估完成。")
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 加载原始数据
data = np.load('shl_balanced_motion_location.npz')
X_train_raw = data['X_train']
X_val_raw = data['X_val']
X_test_raw = data['X_test']

# 加载标签编码器
label_encoder = joblib.load('label_encoder.pkl')
class_names = label_encoder.classes_
num_classes = len(class_names)

# 加载时间序列标准化器
scaler_time = joblib.load('scaler_time.pkl')
n_features = X_train_raw.shape[2]

def scale_data(X, scaler):
    orig_shape = X.shape
    X_2d = X.reshape(-1, n_features)
    X_scaled = scaler.transform(X_2d)
    return X_scaled.reshape(orig_shape)

X_train_raw = scale_data(X_train_raw, scaler_time)
X_val_raw = scale_data(X_val_raw, scaler_time)
X_test_raw = scale_data(X_test_raw, scaler_time)

# 加载特征
feat_data = np.load('shl_all_features.npz')
X_train_feat = feat_data['X_train_feat']
X_val_feat = feat_data['X_val_feat']
X_test_feat = feat_data['X_test_feat']
y_train_enc = feat_data['y_train_enc']
y_val_enc = feat_data['y_val_enc']
y_test_enc = feat_data['y_test_enc']

print("数据加载完成。")

# 基于随机森林特征重要性
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_feat, y_train_enc)

# 获取特征重要性
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# 选择前 k 个特征
k = 256
selected_indices = indices[:k]
print(f"选择了 {k} 个最重要的特征，索引为: {selected_indices[:10]} ...")

# 应用特征选择
X_train_feat_sel = X_train_feat[:, selected_indices]
X_val_feat_sel = X_val_feat[:, selected_indices]
X_test_feat_sel = X_test_feat[:, selected_indices]
print(f"选择后特征形状: {X_train_feat_sel.shape}")

# 保存特征选择器
joblib.dump(selected_indices, 'feature_indices.pkl')
print("特征选择索引已保存。")


# 模型构建
def create_multi_input_model(input_shape_time, num_handcrafted, num_classes):
    # 时间序列分支
    time_input = layers.Input(shape=input_shape_time, name='time_series')
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(time_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(256, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    # 手工特征分支
    feat_input = layers.Input(shape=(num_handcrafted,), name='handcrafted')
    y = layers.Dense(256, activation='relu')(feat_input)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.5)(y)

    # 融合
    combined = layers.concatenate([x, y])
    z = layers.Dense(256, activation='relu')(combined)
    z = layers.Dropout(0.5)(z)
    output = layers.Dense(num_classes, activation='softmax')(z)

    model = models.Model(inputs=[time_input, feat_input], outputs=output)
    return model

input_shape_time = (X_train_raw.shape[1], X_train_raw.shape[2])
num_handcrafted = X_train_feat_sel.shape[1]
model = create_multi_input_model(input_shape_time, num_handcrafted, num_classes)
model.summary()


# 训练
def compile_and_train(model, X_train_time, X_train_feat, y_train,
                      X_val_time, X_val_feat, y_val, model_name,
                      epochs=100, batch_size=128):
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(f'{model_name}.h5', monitor='val_accuracy', save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6)
    ]
    history = model.fit(
        {'time_series': X_train_time, 'handcrafted': X_train_feat}, y_train,
        validation_data=({'time_series': X_val_time, 'handcrafted': X_val_feat}, y_val),
        epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1
    )
    return history

print("\n开始训练...")
history = compile_and_train(model, X_train_raw, X_train_feat_sel, y_train_enc,
                            X_val_raw, X_val_feat_sel, y_val_enc,
                            'multi_input_rf_selected')

# 评估
def evaluate_model(model, X_test_time, X_test_feat, y_test_enc, class_names, model_name):
    y_pred_prob = model.predict({'time_series': X_test_time, 'handcrafted': X_test_feat})
    y_pred = np.argmax(y_pred_prob, axis=1)
    loss, acc = model.evaluate({'time_series': X_test_time, 'handcrafted': X_test_feat},
                               y_test_enc, verbose=0)
    print(f"\n{model_name} 测试准确率: {acc:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test_enc, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test_enc, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} 混淆矩阵')
    plt.xlabel('预测')
    plt.ylabel('真实')
    plt.show()
    return y_pred

y_pred = evaluate_model(model, X_test_raw, X_test_feat_sel, y_test_enc, class_names,
                        'Multi-Input with RF Selected Features')

# 训练曲线
def plot_training_history(history, model_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['loss'], label='train_loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.set_title(f'{model_name} - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(history.history['accuracy'], label='train_acc')
    ax2.plot(history.history['val_accuracy'], label='val_acc')
    ax2.set_title(f'{model_name} - Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    plt.tight_layout()
    plt.show()

plot_training_history(history, 'Multi-Input with RF Selected Features')
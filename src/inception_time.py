import numpy as np
import tensorflow as tf
import keras
from keras import layers, Model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 加载数据集
data = np.load('shl_balanced_motion_location.npz')
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
num_classes = len(class_names)

# 数据标准化
n_samples_train, time_steps, n_features = X_train.shape
X_train_2d = X_train.reshape(-1, n_features)
scaler = StandardScaler()
scaler.fit(X_train_2d)


def scale_data(X, scaler):
    original_shape = X.shape
    X_2d = X.reshape(-1, n_features)
    X_scaled_2d = scaler.transform(X_2d)
    return X_scaled_2d.reshape(original_shape)


X_train = scale_data(X_train, scaler)
X_val = scale_data(X_val, scaler)
X_test = scale_data(X_test, scaler)

joblib.dump(scaler, 'scaler_inception.pkl')


# InceptionTime
def inception_module(x, filters):
    # 分支1：1x1卷积
    branch1 = layers.Conv1D(filters, kernel_size=1, padding='same', activation='relu')(x)

    # 分支2：3x1卷积
    branch2 = layers.Conv1D(filters, kernel_size=3, padding='same', activation='relu')(x)

    # 分支3：5x1卷积
    branch3 = layers.Conv1D(filters, kernel_size=5, padding='same', activation='relu')(x)

    # 分支4：最大池化 + 1x1卷积
    branch4 = layers.MaxPooling1D(pool_size=3, strides=1, padding='same')(x)
    branch4 = layers.Conv1D(filters, kernel_size=1, padding='same', activation='relu')(branch4)

    # 合并所有分支
    x = layers.Concatenate(axis=-1)([branch1, branch2, branch3, branch4])
    return x


def shortcut_layer(x, out_shape):
    if x.shape[-1] != out_shape:
        x = layers.Conv1D(out_shape, kernel_size=1, padding='same')(x)
    return x


def inception_block(x, filters, residual=True):
    inception_out = inception_module(x, filters)
    inception_out = inception_module(inception_out, filters)

    if residual:
        # 残差连接
        shortcut = shortcut_layer(x, inception_out.shape[-1])
        x = layers.Add()([inception_out, shortcut])
        x = layers.ReLU()(x)
    else:
        x = inception_out
    return x


def build_inception_time(input_shape, num_classes, nb_filters=16, depth=3, use_residual=True):
    inputs = layers.Input(shape=input_shape)
    x = inputs

    for d in range(depth):
        filters = nb_filters * (2 ** (d // 2))
        x = inception_block(x, filters, residual=use_residual)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model


input_shape = (X_train.shape[1], X_train.shape[2])
model = build_inception_time(input_shape, num_classes, nb_filters=16, depth=4, use_residual=True)
model.summary()

# 编译与训练
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.002),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=16, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint('inception_time.h5', monitor='val_accuracy', save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6)
]

print("\n开始训练 InceptionTime ...")
history = model.fit(
    X_train, y_train_enc,
    validation_data=(X_val, y_val_enc),
    epochs=200,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)

# 评估
model.load_weights('inception_time.h5')

y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

loss, acc = model.evaluate(X_test, y_test_enc, verbose=0)
print(f"\nInceptionTime 测试准确率: {acc:.4f}")

print("\n分类报告:")
print(classification_report(y_test_enc, y_pred, target_names=class_names))

# 混淆矩阵
cm = confusion_matrix(y_test_enc, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('InceptionTime 混淆矩阵')
plt.xlabel('预测')
plt.ylabel('真实')
plt.show()

# 绘制训练曲线
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history['loss'], label='train')
ax1.plot(history.history['val_loss'], label='val')
ax1.set_title('InceptionTime Loss')
ax1.legend()

ax2.plot(history.history['accuracy'], label='train')
ax2.plot(history.history['val_accuracy'], label='val')
ax2.set_title('InceptionTime Accuracy')
ax2.legend()
plt.show()